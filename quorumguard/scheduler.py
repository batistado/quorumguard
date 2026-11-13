"""
Event-Driven QuorumGuard Adaptive Consensus Controller & Priority-Budget Scheduler.
===================================================================================
Implements Theorem 3's adaptive stopping time:
    tau_QG = inf { t : N_eff(t, G) >= k*(H_t) }
coupled with Priority-Budget Dissent Boosting for in-flight deep verification
stragglers whenever fast arrivals exhibit genealogy correlation, disagreement,
or elevated semantic entropy. Includes vLLM priority-scheduling request hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .entropy_gate import EpistemicSignals, compute_epistemic_weight
from .genealogy import ProvenanceDAG


# Standard vLLM priority levels (lower integer = higher scheduling priority in vLLM Scheduler)
VLLM_PRIORITY_NORMAL: int = 0
VLLM_PRIORITY_DISSENT_BOOST: int = -10
VLLM_PRIORITY_BACKGROUND_SPECULATIVE: int = 5


@dataclass(frozen=True)
class QuorumGuardConfig:
    min_independent_roots: int = 3
    tau_effective_quorum: float = 2.22
    min_lead_margin: float = 0.95
    max_fast_normalized_entropy: float = 0.22
    dissent_trigger_min_arrivals: int = 3
    priority_boost_speedup: float = 3.2
    max_boost_slots_per_stage: int = 2


@dataclass
class AgentVote:
    agent_id: str
    claim: str
    arrival_time_s: float
    signals: EpistemicSignals
    is_deep_verifier: bool = False
    irreducible_exec_floor_s: float = 0.0


@dataclass
class StageCommitResult:
    committed_claim: str
    commit_time_s: float
    agents_observed: int
    effective_quorum_weight: float
    independent_roots: int
    dissent_boost_triggered: bool
    boosted_agent_ids: Tuple[str, ...]
    early_closed: bool


class PriorityBudgetPool:
    """
    Token-bucket priority budget bounding cluster-wide straggler priority boosts
    so priority elevation cannot cause priority-inversion storms under heavy load.
    """

    def __init__(self, max_budget_tokens: float = 10.0, refill_per_stage: float = 1.2) -> None:
        self.max_budget = max_budget_tokens
        self.refill_per_stage = refill_per_stage
        self.available = max_budget_tokens

    def try_acquire(self, slots: int = 1) -> int:
        granted = min(slots, int(self.available))
        self.available -= float(granted)
        return max(0, granted)

    def on_stage_complete(self) -> None:
        self.available = min(self.max_budget, self.available + self.refill_per_stage)


class QuorumGuardController:
    """
    Stateful online quorum gate + straggler boost controller for a single stage.
    """

    def __init__(
        self,
        dag: ProvenanceDAG,
        config: Optional[QuorumGuardConfig] = None,
        budget_pool: Optional[PriorityBudgetPool] = None,
        on_vllm_priority_update: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        self.dag = dag
        self.config = config or QuorumGuardConfig()
        self.budget_pool = budget_pool
        self.on_vllm_priority_update = on_vllm_priority_update

    def evaluate_asynchronous_pool(
        self,
        pool_votes: Sequence[AgentVote],
    ) -> StageCommitResult:
        if not pool_votes:
            raise ValueError("pool_votes must be non-empty")

        n_total = len(pool_votes)
        cfg = self.config
        min_roots = cfg.min_independent_roots if n_total >= 6 else max(2, cfg.min_independent_roots - 1)
        eff_tau = cfg.tau_effective_quorum if n_total >= 6 else max(1.50, cfg.tau_effective_quorum * 0.70)

        dyn_arrival: Dict[str, float] = {v.agent_id: v.arrival_time_s for v in pool_votes}
        vote_map: Dict[str, AgentVote] = {v.agent_id: v for v in pool_votes}
        observed_ids: List[str] = []
        boost_triggered = False
        boosted_ids: List[str] = []
        early_closed = False
        commit_claim = ""
        commit_time = 0.0
        commit_weight = 0.0
        commit_roots = 0

        while len(observed_ids) < n_total:
            remaining = [aid for aid in dyn_arrival if aid not in observed_ids]
            next_aid = min(remaining, key=lambda aid: dyn_arrival[aid])
            t_now = dyn_arrival[next_aid]
            observed_ids.append(next_aid)

            weighted_tuples = [
                (
                    aid,
                    vote_map[aid].claim,
                    compute_epistemic_weight(vote_map[aid].signals),
                )
                for aid in observed_ids
            ]
            claim_weights, claim_roots = self.dag.compute_effective_support(weighted_tuples)
            ranked = sorted(claim_weights.items(), key=lambda kv: kv[1], reverse=True)
            lead_claim, lead_w = ranked[0]
            runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
            lead_r = claim_roots[lead_claim]

            supporters = [vote_map[aid] for aid in observed_ids if vote_map[aid].claim == lead_claim]
            mean_ent = sum(s.signals.normalized_option_entropy for s in supporters) / max(1, len(supporters))
            has_deep_cert = any(s.signals.tool_verified or s.is_deep_verifier for s in supporters)

            clean_fast_pass = (
                lead_r >= min_roots
                and lead_w >= eff_tau
                and (lead_w - runner_w) >= cfg.min_lead_margin
                and mean_ent <= cfg.max_fast_normalized_entropy
            )
            verified_pass = (
                has_deep_cert
                and lead_r >= 2
                and lead_w >= (eff_tau - 0.18)
                and (lead_w - runner_w) >= 0.65
            )

            if clean_fast_pass or verified_pass:
                commit_claim = lead_claim
                commit_time = t_now
                commit_weight = lead_w
                commit_roots = lead_r
                early_closed = (len(observed_ids) < n_total)
                break

            if (not boost_triggered) and len(observed_ids) >= cfg.dissent_trigger_min_arrivals:
                boost_triggered = True
                in_flight_verifiers = [
                    aid
                    for aid in remaining
                    if aid != next_aid and vote_map[aid].is_deep_verifier and dyn_arrival[aid] > t_now
                ]
                in_flight_verifiers.sort(key=lambda aid: dyn_arrival[aid])
                desired = min(cfg.max_boost_slots_per_stage, len(in_flight_verifiers))
                granted = self.budget_pool.try_acquire(desired) if self.budget_pool else desired

                for aid_boost in in_flight_verifiers[:granted]:
                    boosted_ids.append(aid_boost)
                    if self.on_vllm_priority_update is not None:
                        self.on_vllm_priority_update(aid_boost, VLLM_PRIORITY_DISSENT_BOOST)
                    floor_t = t_now + 0.45 * vote_map[aid_boost].irreducible_exec_floor_s
                    comp_t = t_now + (dyn_arrival[aid_boost] - t_now) / max(1.0, cfg.priority_boost_speedup)
                    dyn_arrival[aid_boost] = max(floor_t, comp_t)
        else:
            weighted_tuples = [
                (aid, vote_map[aid].claim, compute_epistemic_weight(vote_map[aid].signals))
                for aid in observed_ids
            ]
            claim_weights, claim_roots = self.dag.compute_effective_support(weighted_tuples)
            commit_claim, commit_weight = max(claim_weights.items(), key=lambda kv: kv[1])
            commit_roots = claim_roots[commit_claim]
            commit_time = max(dyn_arrival.values())

        if self.budget_pool is not None:
            self.budget_pool.on_stage_complete()

        return StageCommitResult(
            committed_claim=commit_claim,
            commit_time_s=round(commit_time, 5),
            agents_observed=len(observed_ids),
            effective_quorum_weight=round(commit_weight, 5),
            independent_roots=commit_roots,
            dissent_boost_triggered=boost_triggered,
            boosted_agent_ids=tuple(boosted_ids),
            early_closed=early_closed,
        )

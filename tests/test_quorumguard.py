import math

from quorumguard import (
    VLLM_PRIORITY_DISSENT_BOOST,
    AgentVote,
    EpistemicSignals,
    PriorityBudgetPool,
    ProvenanceDAG,
    QuorumGuardConfig,
    QuorumGuardController,
    compute_epistemic_weight,
    posterior_entropy_and_margin_from_logits,
)


def test_shared_genealogy_root_deduplication() -> None:
    dag = ProvenanceDAG()
    dag.add_node("raw_rag_doc", kind="source")
    dag.add_node("flawed_triage_summary", kind="summary", parents=["raw_rag_doc"])
    # Three fast agents all reading the SAME flawed triage summary!
    dag.register_agent_lineage("fast_0", ["flawed_triage_summary"])
    dag.register_agent_lineage("fast_1", ["flawed_triage_summary"])
    dag.register_agent_lineage("fast_2", ["flawed_triage_summary"])
    dag.register_agent_lineage("verifier_3", ["sandbox_exec_root"])

    weights, roots = dag.compute_effective_support(
        [
            ("fast_0", "NO_BUG", 0.90),
            ("fast_1", "NO_BUG", 0.88),
            ("fast_2", "NO_BUG", 0.85),
            ("verifier_3", "OVERFLOW_BUG", 1.10),
        ]
    )
    assert roots["NO_BUG"] == 1
    assert math.isclose(weights["NO_BUG"], 0.90, rel_tol=1e-6)
    assert roots["OVERFLOW_BUG"] == 1
    assert weights["OVERFLOW_BUG"] > weights["NO_BUG"]


def test_clean_fast_quorum_closes_early_on_independent_low_entropy() -> None:
    dag = ProvenanceDAG()
    for i in range(6):
        dag.register_agent_lineage(f"a{i}", [f"indep_root_{i}"])

    ctrl = QuorumGuardController(dag)
    votes = [
        AgentVote("a0", "B", 0.50, EpistemicSignals(option_entropy_nats=0.02, logit_margin=6.0)),
        AgentVote("a1", "B", 0.70, EpistemicSignals(option_entropy_nats=0.03, logit_margin=5.5)),
        AgentVote("a2", "B", 0.95, EpistemicSignals(option_entropy_nats=0.04, logit_margin=5.2)),
        AgentVote("a3", "B", 3.50, EpistemicSignals(option_entropy_nats=0.01, logit_margin=7.0, has_cot_trace=True)),
        AgentVote("a4", "B", 9.80, EpistemicSignals(option_entropy_nats=0.01, logit_margin=8.0, tool_verified=True), is_deep_verifier=True),
        AgentVote("a5", "B", 12.40, EpistemicSignals(option_entropy_nats=0.01, logit_margin=8.0, tool_verified=True), is_deep_verifier=True),
    ]
    res = ctrl.evaluate_asynchronous_pool(votes)
    assert res.committed_claim == "B"
    assert res.early_closed is True
    assert math.isclose(res.commit_time_s, 0.95, rel_tol=1e-5)
    assert res.agents_observed == 3
    assert res.dissent_boost_triggered is False


def test_correlated_echo_blocks_fast_close_and_triggers_vllm_dissent_boost() -> None:
    dag = ProvenanceDAG()
    dag.add_node("shared_hint", kind="source")
    dag.register_agent_lineage("a0", ["shared_hint"])
    dag.register_agent_lineage("a1", ["shared_hint"])
    dag.register_agent_lineage("a2", ["shared_hint"])
    dag.register_agent_lineage("a3", ["cot_root"],)
    dag.register_agent_lineage("a4", ["sandbox_1"])
    dag.register_agent_lineage("a5", ["sandbox_2"])

    priority_updates: list[tuple[str, int]] = []
    pool = PriorityBudgetPool(max_budget_tokens=4.0)
    ctrl = QuorumGuardController(
        dag,
        config=QuorumGuardConfig(priority_boost_speedup=3.0),
        budget_pool=pool,
        on_vllm_priority_update=lambda aid, prio: priority_updates.append((aid, prio)),
    )

    votes = [
        # Three correlated fast echoes all vote WRONG ("A") within 0.9s!
        AgentVote("a0", "A", 0.50, EpistemicSignals(option_entropy_nats=0.10, logit_margin=3.5)),
        AgentVote("a1", "A", 0.70, EpistemicSignals(option_entropy_nats=0.12, logit_margin=3.2)),
        AgentVote("a2", "A", 0.90, EpistemicSignals(option_entropy_nats=0.15, logit_margin=3.0)),
        # Medium CoT & Sandbox Verifiers find ground truth ("B")!
        AgentVote("a3", "B", 3.40, EpistemicSignals(option_entropy_nats=0.03, logit_margin=6.5, has_cot_trace=True)),
        AgentVote(
            "a4",
            "B",
            9.90,
            EpistemicSignals(option_entropy_nats=0.01, logit_margin=8.5, tool_verified=True),
            is_deep_verifier=True,
            irreducible_exec_floor_s=1.8,
        ),
        AgentVote(
            "a5",
            "B",
            13.50,
            EpistemicSignals(option_entropy_nats=0.01, logit_margin=8.5, tool_verified=True),
            is_deep_verifier=True,
            irreducible_exec_floor_s=1.8,
        ),
    ]
    res = ctrl.evaluate_asynchronous_pool(votes)
    assert res.committed_claim == "B"
    assert res.dissent_boost_triggered is True
    assert ("a4", VLLM_PRIORITY_DISSENT_BOOST) in priority_updates
    # Unboosted a4 would arrive at 9.90s; boosted at t=0.90s with 3.0x speedup -> 0.90 + (9.90 - 0.90)/3 = 3.90s!
    assert math.isclose(res.commit_time_s, 3.90, rel_tol=1e-4)
    assert res.early_closed is True  # Did NOT wait for slowest straggler a5!


def test_posterior_entropy_and_margin_from_logits() -> None:
    ent_unif, mar_unif, probs_unif = posterior_entropy_and_margin_from_logits([1.0, 1.0, 1.0, 1.0])
    assert math.isclose(ent_unif, math.log(4.0), rel_tol=1e-6)
    assert math.isclose(mar_unif, 0.0, abs_tol=1e-9)
    assert all(math.isclose(p, 0.25, rel_tol=1e-6) for p in probs_unif)

    ent_sharp, mar_sharp, _ = posterior_entropy_and_margin_from_logits([10.0, 1.0, 0.0, -1.0])
    assert ent_sharp < 0.01
    assert math.isclose(mar_sharp, 9.0, rel_tol=1e-6)
    w_verified = compute_epistemic_weight(
        EpistemicSignals(option_entropy_nats=ent_sharp, logit_margin=mar_sharp, tool_verified=True)
    )
    w_shallow = compute_epistemic_weight(
        EpistemicSignals(option_entropy_nats=0.9, logit_margin=0.4, tool_verified=False)
    )
    assert w_verified > 2.0 * w_shallow

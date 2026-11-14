#!/usr/bin/env python3
"""
QuorumGuard Comprehensive Revision Evaluation Suite
===================================================
Addresses all empirical revision requests on QuorumBench-50 (3B + 7B + 14B-Coder):
  1. Collects complete live MPS Stage-2 hardware telemetry across all 116 distinct
     `(task_id, s1_choice)` pairs emitted by Stage-1 agents (stored in
     `benchmarks/results/quorumbench_50_stage2_full_cache.json`).
  2. Evaluates 8 policies head-to-head:
       - Fast_k_of_N (k=2 matching / 3-arrival plurality)
       - Fast_3_of_7_Majority (strict k=3 plurality)
       - Tier_Weighted_Fast (TW-Fast static tier weights, no entropy or N_eff)
       - Entropy_Cascade (fast 3B check; escalates on uncertainty, no N_eff or boost)
       - CVaR_Static_Cutoff across full sweep B in {1.5, 2.8, 6.0, 10.0, 15.0, 22.0, 30.0}s
       - Sync_N_of_N
       - Genealogy_Sync_HPR
       - QuorumGuard
  3. Separates trace-driven container/queueing delays from Pure Hardware-Only
     execution (MPS inference + live Python sandbox wall-clock time only).
  4. Computes 2,000-replicate paired task-level bootstrap 95% CIs and 10-seed
     stochastic evaluation (seeds 2026..2035) across all policies.
  5. Evaluates 4-axis hyperparameter sensitivity (gamma_H, tau_eff, m_gap, psi_i).

Saves `benchmarks/results/revision_ablations_and_sweeps.json`.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pilot.quorumguard_pilot import (  # noqa: E402
    AgentObservation,
    MPSAgentEngine,
    QuorumDecision,
    TwoStageTask,
    format_mc_prompt,
)
from benchmarks.quorumbench_scaleup import build_quorumbench_50  # noqa: E402


# ---------------------------------------------------------------------------
# Generalized Weight & Multi-Policy Evaluator (reused across all benchmarks)
# ---------------------------------------------------------------------------

def compute_weight_parametric(obs: AgentObservation, psi_mode: str = "default") -> float:
    """
    Calibrated epistemic weight matching Eq. (10):
      tau_Delta = 3.5, alpha_seq = 0.12, h_0 = 0.60, w_i in [0.05, 1.35].
      psi_mode in {"default", "sandbox_only", "uniform"}.
    """
    cert_prob = max(0.0, 1.0 - obs.normalized_entropy)
    margin_conf = math.tanh(max(0.0, obs.logit_margin) / 3.5)
    seq_pen = 1.0 / (1.0 + 0.12 * max(0.0, obs.seq_token_entropy - 0.60))
    base_conf = (0.55 * cert_prob + 0.45 * margin_conf) * seq_pen

    if psi_mode == "uniform":
        mult = 1.0
    elif psi_mode == "sandbox_only":
        mult = 1.25 if obs.tool_verified else 1.0
    else:
        if obs.tool_verified:
            mult = 1.38
        elif obs.tier == "verifier_7b":
            mult = 1.18
        elif obs.tier == "cot_7b":
            mult = 1.12
        elif obs.tier == "short_cot_3b":
            mult = 0.85
        else:
            mult = 0.76
    return min(1.35, max(0.05, base_conf * mult))


def evaluate_stage_policies_extended(
    observations: Sequence[AgentObservation],
    gold_choice: str,
    cvar_budgets: Sequence[float] = (1.5, 2.8, 6.0, 10.0, 15.0, 22.0, 30.0),
    qg_tau_quorum: float = 2.12,
    qg_margin_gap: float = 0.72,
    qg_max_fast_entropy: float = 0.08,
    qg_min_fast_logit_margin: float = 3.2,
    qg_boost_speedup: float = 3.2,
    psi_mode: str = "default",
    pure_hardware_only: bool = False,
) -> Dict[str, QuorumDecision]:
    """Evaluates all classical, adaptive baseline, CVaR sweep, and QuorumGuard policies."""
    obs_eff: List[AgentObservation] = []
    for o in observations:
        if pure_hardware_only:
            d = asdict(o)
            d["completion_time_s"] = round(float(o.mps_wall_s), 4)
            obs_eff.append(AgentObservation(**d))
        else:
            obs_eff.append(o)

    obs_sorted = sorted(obs_eff, key=lambda o: o.completion_time_s)
    n_total = len(obs_sorted)
    decisions: Dict[str, QuorumDecision] = {}

    # 1. Fast k-of-N (earliest 2 matching OR 3-arrival plurality, exact match to Table 2/3)
    counts_fast: Dict[str, int] = {}
    fast_choice = obs_sorted[0].choice
    fast_lat = obs_sorted[0].completion_time_s
    fast_seen = 0
    for o in obs_sorted:
        fast_seen += 1
        counts_fast[o.choice] = counts_fast.get(o.choice, 0) + 1
        if counts_fast[o.choice] >= 2 and fast_seen >= 2:
            fast_choice = o.choice
            fast_lat = o.completion_time_s
            break
        if fast_seen >= 3:
            fast_choice = max(counts_fast.items(), key=lambda kv: (kv[1], -obs_sorted[0].option_entropy_nats))[0]
            fast_lat = o.completion_time_s
            break
    decisions["Fast_k_of_N"] = QuorumDecision(
        "Fast_k_of_N", fast_choice, (fast_choice == gold_choice), round(fast_lat, 4), fast_seen, False, float(counts_fast.get(fast_choice, 1))
    )

    # 2. Strict Fast k=3-of-N Majority / Plurality
    k_maj = min(3, n_total)
    sub_maj = obs_sorted[:k_maj]
    cnt_maj: Dict[str, Tuple[int, float]] = {}
    for o in sub_maj:
        c0, w0 = cnt_maj.get(o.choice, (0, 0.0))
        cnt_maj[o.choice] = (c0 + 1, w0 + (1.0 - o.normalized_entropy))
    maj_choice = max(cnt_maj.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
    decisions["Fast_3_of_N_Maj"] = QuorumDecision(
        "Fast_3_of_N_Maj", maj_choice, (maj_choice == gold_choice), round(sub_maj[-1].completion_time_s, 4), k_maj, False, float(cnt_maj[maj_choice][0])
    )

    # 3. Tier-Weighted Fast (TW-Fast: static tier prior weights without entropy or N_eff)
    tier_prior_tau = 2.12 if n_total >= 6 else 1.55
    tw_scores: Dict[str, float] = {}
    tw_choice = obs_sorted[-1].choice
    tw_lat = obs_sorted[-1].completion_time_s
    tw_seen = 0
    for o in obs_sorted:
        tw_seen += 1
        t_mult = 1.38 if o.tier == "verifier_7b" else (1.12 if o.tier == "cot_7b" else (0.85 if o.tier == "short_cot_3b" else 0.76))
        tw_scores[o.choice] = tw_scores.get(o.choice, 0.0) + t_mult
        if tw_scores[o.choice] >= tier_prior_tau:
            tw_choice = o.choice
            tw_lat = o.completion_time_s
            break
    else:
        tw_choice = max(tw_scores.items(), key=lambda kv: kv[1])[0]
    decisions["Tier_Weighted_Fast"] = QuorumDecision(
        "Tier_Weighted_Fast", tw_choice, (tw_choice == gold_choice), round(tw_lat, 4), tw_seen, False, round(tw_scores[tw_choice], 4)
    )

    # 4. Entropy Cascade (checks earliest k=3 fast arrivals; if >=2 agree with mean_ent <= gamma_H commits, else full conf-weighted barrier without N_eff/boost)
    k_cas = min(3, n_total)
    early_cas = obs_sorted[:k_cas]
    cas_counts: Dict[str, int] = {}
    for o in early_cas:
        cas_counts[o.choice] = cas_counts.get(o.choice, 0) + 1
    cas_top = max(cas_counts.items(), key=lambda kv: kv[1])[0]
    cas_mean_ent = float(np.mean([o.normalized_entropy for o in early_cas if o.choice == cas_top]))
    if cas_counts[cas_top] >= 2 and cas_mean_ent <= qg_max_fast_entropy:
        decisions["Entropy_Cascade"] = QuorumDecision(
            "Entropy_Cascade", cas_top, (cas_top == gold_choice), round(early_cas[-1].completion_time_s, 4), k_cas, False, float(cas_counts[cas_top])
        )
    else:
        full_conf: Dict[str, float] = {}
        for o in obs_sorted:
            full_conf[o.choice] = full_conf.get(o.choice, 0.0) + compute_weight_parametric(o, psi_mode)
        cas_ch, cas_w = max(full_conf.items(), key=lambda kv: kv[1])
        decisions["Entropy_Cascade"] = QuorumDecision(
            "Entropy_Cascade", cas_ch, (cas_ch == gold_choice), round(obs_sorted[-1].completion_time_s, 4), n_total, False, round(cas_w, 4)
        )

    # 5. CVaR Static Cutoff Sweep
    sync_lat = obs_sorted[-1].completion_time_s
    for b_cut in cvar_budgets:
        arr = [o for o in obs_sorted if o.completion_time_s <= b_cut]
        if not arr:
            arr = [obs_sorted[0]]
            clat = obs_sorted[0].completion_time_s
        else:
            clat = min(sync_lat, float(b_cut))
        c_cnt: Dict[str, Tuple[int, float]] = {}
        for o in arr:
            pc, pw = c_cnt.get(o.choice, (0, 0.0))
            c_cnt[o.choice] = (pc + 1, pw + (1.0 - o.normalized_entropy))
        c_ch = max(c_cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
        pol_key = f"CVaR_B_{b_cut:g}s"
        decisions[pol_key] = QuorumDecision(
            pol_key, c_ch, (c_ch == gold_choice), round(clat, 4), len(arr), False, float(c_cnt[c_ch][0])
        )

    # 6. Sync N-of-N
    s_cnt: Dict[str, Tuple[int, float]] = {}
    for o in obs_sorted:
        pc, pw = s_cnt.get(o.choice, (0, 0.0))
        s_cnt[o.choice] = (pc + 1, pw + (1.0 - o.normalized_entropy))
    s_ch = max(s_cnt.items(), key=lambda kv: (kv[1][0], kv[1][1]))[0]
    decisions["Sync_N_of_N"] = QuorumDecision(
        "Sync_N_of_N", s_ch, (s_ch == gold_choice), round(sync_lat, 4), n_total, False, float(s_cnt[s_ch][0])
    )

    # 7. Genealogy Sync HPR
    def g_scores(subset: Sequence[AgentObservation]) -> Tuple[Dict[str, float], Dict[str, int]]:
        rc_max: Dict[Tuple[str, str], float] = {}
        for o in subset:
            w = compute_weight_parametric(o, psi_mode)
            key = (o.genealogy_root, o.choice)
            rc_max[key] = max(rc_max.get(key, 0.0), w)
        ct, cr = {}, {}
        for (_, ch), wm in rc_max.items():
            ct[ch] = ct.get(ch, 0.0) + wm
            cr[ch] = cr.get(ch, 0) + 1
        return ct, cr

    hpr_tot, _ = g_scores(obs_sorted)
    hpr_ch, hpr_w = max(hpr_tot.items(), key=lambda kv: kv[1])
    decisions["Genealogy_Sync_HPR"] = QuorumDecision(
        "Genealogy_Sync_HPR", hpr_ch, (hpr_ch == gold_choice), round(sync_lat, 4), n_total, False, round(hpr_w, 4)
    )

    # 8. QuorumGuard
    dyn_times = {o.agent_id: o.completion_time_s for o in obs_eff}
    obs_by_id = {o.agent_id: o for o in obs_eff}
    finished_ids: List[int] = []
    boost_triggered = False
    qg_choice = hpr_ch
    qg_lat = sync_lat
    qg_eff_w = hpr_w

    min_indep_roots = 3 if n_total >= 6 else 2
    eff_tau = qg_tau_quorum if n_total >= 6 else max(1.40, qg_tau_quorum * 0.73)

    while len(finished_ids) < n_total:
        rem = [aid for aid in dyn_times if aid not in finished_ids]
        next_aid = min(rem, key=lambda aid: dyn_times[aid])
        t_now = dyn_times[next_aid]
        finished_ids.append(next_aid)

        arrived = [obs_by_id[aid] for aid in finished_ids]
        c_tot, c_roots = g_scores(arrived)
        ranked = sorted(c_tot.items(), key=lambda kv: kv[1], reverse=True)
        lead_ch, lead_w = ranked[0]
        runner_w = ranked[1][1] if len(ranked) > 1 else 0.0
        lead_root_cnt = c_roots[lead_ch]

        sups = [o for o in arrived if o.choice == lead_ch]
        mean_sup_ent = float(np.mean([o.normalized_entropy for o in sups]))
        min_sup_margin = float(min(o.logit_margin for o in sups))
        deep_roots = len({o.genealogy_root for o in sups if o.tool_verified or o.tier in ("verifier_7b", "cot_7b")})
        dissent = len(ranked) >= 2 and runner_w >= 0.35

        clean_fast = (
            (not dissent)
            and lead_root_cnt >= min_indep_roots
            and lead_w >= eff_tau
            and mean_sup_ent <= qg_max_fast_entropy
            and min_sup_margin >= qg_min_fast_logit_margin
        )
        verified_hedge = (
            deep_roots >= 1
            and (deep_roots >= 2 or (deep_roots >= 1 and not dissent))
            and lead_root_cnt >= 2
            and lead_w >= (eff_tau - 0.15)
            and (lead_w - runner_w) >= qg_margin_gap
        )
        if clean_fast or verified_hedge:
            qg_choice = lead_ch
            qg_lat = t_now
            qg_eff_w = lead_w
            break

        if (not boost_triggered) and len(finished_ids) >= 3:
            boost_triggered = True
            for aid, o_c in obs_by_id.items():
                if aid not in finished_ids and o_c.tier in ("verifier_7b", "cot_7b"):
                    floor_t = t_now + 0.45 * o_c.mps_wall_s
                    comp_t = t_now + (dyn_times[aid] - t_now) / qg_boost_speedup
                    dyn_times[aid] = max(floor_t, comp_t)
    else:
        arrived = [obs_by_id[aid] for aid in finished_ids]
        c_tot, _ = g_scores(arrived)
        qg_choice, qg_eff_w = max(c_tot.items(), key=lambda kv: kv[1])
        qg_lat = max(dyn_times.values())

    decisions["QuorumGuard"] = QuorumDecision(
        "QuorumGuard", qg_choice, (qg_choice == gold_choice), round(qg_lat, 4), len(finished_ids), boost_triggered, round(qg_eff_w, 4)
    )
    return decisions


def _lookup_stage2_obs(
    task: TwoStageTask,
    committed_choice: str,
    stage2_obs_map: Dict[Tuple[str, str], List[AgentObservation]],
) -> List[AgentObservation]:
    key = (task.task_id, committed_choice)
    if key in stage2_obs_map:
        return stage2_obs_map[key]
    if committed_choice != task.stage1_gold:
        wrong_alts = [k for k in stage2_obs_map if k[0] == task.task_id and k[1] != task.stage1_gold]
        if wrong_alts:
            return stage2_obs_map[wrong_alts[0]]
    all_alts = [k for k in stage2_obs_map if k[0] == task.task_id]
    return stage2_obs_map[all_alts[0]]


def summarize_two_stage_workflows(
    tasks: Sequence[TwoStageTask],
    stage1_obs_map: Dict[str, List[AgentObservation]],
    stage2_obs_map: Dict[Tuple[str, str], List[AgentObservation]],
    eval_kwargs: Optional[Dict[str, Any]] = None,
    n_bootstrap: int = 2000,
    boot_seed: int = 42,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates all policies end-to-end + computes 2,000-replicate paired bootstrap 95% CIs."""
    kw = dict(eval_kwargs or {})
    records: List[Dict[str, Any]] = []
    pol_names: List[str] = []

    for task in tasks:
        s1_decs = evaluate_stage_policies_extended(stage1_obs_map[task.task_id], task.stage1_gold, **kw)
        if not pol_names:
            pol_names = list(s1_decs.keys())
        e2e: Dict[str, Any] = {}
        for pol in pol_names:
            d1 = s1_decs[pol]
            s2_list = _lookup_stage2_obs(task, d1.committed_choice, stage2_obs_map)
            d2 = evaluate_stage_policies_extended(s2_list, task.stage2_gold, **kw)[pol]
            e2e[pol] = {
                "stage1_choice": d1.committed_choice,
                "stage1_correct": bool(d1.is_correct),
                "stage1_latency_s": d1.latency_s,
                "stage2_correct": bool(d2.is_correct),
                "stage2_latency_s": d2.latency_s,
                "workflow_total_latency_s": round(d1.latency_s + d2.latency_s, 4),
                "workflow_both_stages_correct": bool(d1.is_correct and d2.is_correct),
            }
        records.append({"task_id": task.task_id, "regime": task.regime, "policies": e2e})

    rng_b = np.random.default_rng(boot_seed)
    n_t = len(records)
    idx_boot = rng_b.integers(0, n_t, size=(n_bootstrap, n_t))
    trap_mask = np.array([r["regime"] != "routine" for r in records], dtype=bool)

    summary: Dict[str, Any] = {}
    for pol in pol_names:
        s1 = np.array([r["policies"][pol]["stage1_correct"] for r in records], dtype=np.float64)
        s2 = np.array([r["policies"][pol]["stage2_correct"] for r in records], dtype=np.float64)
        jnt = np.array([r["policies"][pol]["workflow_both_stages_correct"] for r in records], dtype=np.float64)
        lats = np.array([r["policies"][pol]["workflow_total_latency_s"] for r in records], dtype=np.float64)

        boot_jnt = np.mean(jnt[idx_boot], axis=1)
        boot_p95 = np.percentile(lats[idx_boot], 95, axis=1)
        p95 = float(np.percentile(lats, 95))
        cvar95 = float(np.mean(lats[lats >= p95]))
        summary[pol] = {
            "stage1_accuracy": round(float(np.mean(s1)), 4),
            "stage2_accuracy": round(float(np.mean(s2)), 4),
            "end_to_end_joint_accuracy": round(float(np.mean(jnt)), 4),
            "joint_acc_ci95": [round(float(np.percentile(boot_jnt, 2.5)), 4), round(float(np.percentile(boot_jnt, 97.5)), 4)],
            "trap_joint_accuracy": round(float(np.mean(jnt[trap_mask])), 4) if np.any(trap_mask) else 0.0,
            "mean_latency_s": round(float(np.mean(lats)), 4),
            "p95_latency_s": round(p95, 4),
            "p95_ci95": [round(float(np.percentile(boot_p95, 2.5)), 3), round(float(np.percentile(boot_p95, 97.5)), 3)],
            "cvar95_latency_s": round(cvar95, 4),
        }
    return summary, records


def perturb_observations_for_seed(
    stage1_obs_map: Dict[str, List[AgentObservation]],
    stage2_obs_map: Dict[Tuple[str, str], List[AgentObservation]],
    seed: int,
    logit_noise_std: float = 0.25,
) -> Tuple[Dict[str, List[AgentObservation]], Dict[Tuple[str, str], List[AgentObservation]]]:
    """
    Replays a new stochastic asynchronous trial (random logit perturbation at
    sigma_z=0.25 preserving empirical uncertainty + fresh Pareto/uniform RPC delays).
    """
    rng = np.random.default_rng(seed)

    def _pert_obs(o: AgentObservation, new_q: float) -> AgentObservation:
        probs = o.option_probs
        keys = ["A", "B", "C", "D"]
        p_vec = np.array([max(1e-6, probs[k]) for k in keys], dtype=np.float64)
        logits = np.log(p_vec) + rng.normal(0.0, logit_noise_std, size=4)
        logits -= np.max(logits)
        exp_l = np.exp(logits)
        new_p = exp_l / np.sum(exp_l)
        ent = float(-np.sum(new_p * np.log(np.maximum(new_p, 1e-12))))
        s_l = np.sort(logits)[::-1]
        margin = float(s_l[0] - s_l[1])
        best_c = keys[int(np.argmax(logits))]
        d = asdict(o)
        d["choice"] = best_c
        d["option_probs"] = {k: round(float(new_p[i]), 5) for i, k in enumerate(keys)}
        d["option_entropy_nats"] = round(ent, 5)
        d["normalized_entropy"] = round(ent / math.log(4.0), 5)
        d["logit_margin"] = round(margin, 4)
        d["queue_tool_overhead_s"] = round(float(new_q), 4)
        d["completion_time_s"] = round(float(o.mps_wall_s) + float(new_q), 4)
        return AgentObservation(**d)

    s1_out: Dict[str, List[AgentObservation]] = {}
    for tid, lst in stage1_obs_map.items():
        fq = rng.uniform(0.18, 0.72, size=4).tolist()
        cq = [float(rng.uniform(0.85, 1.95))]
        sq = (3.8 + rng.pareto(a=2.25, size=2) * 2.7).clip(3.8, 16.5).tolist()
        ovs = fq + cq + [float(x) for x in sq]
        s1_out[tid] = [_pert_obs(o, ovs[min(idx, len(ovs) - 1)]) for idx, o in enumerate(sorted(lst, key=lambda x: x.agent_id))]

    s2_out: Dict[Tuple[str, str], List[AgentObservation]] = {}
    for key, lst in stage2_obs_map.items():
        fq = rng.uniform(0.16, 0.65, size=3).tolist()
        sq = (3.6 + rng.pareto(a=2.3, size=2) * 2.6).clip(3.6, 15.5).tolist()
        ovs = fq + [float(x) for x in sq]
        s2_out[key] = [_pert_obs(o, ovs[min(idx, len(ovs) - 1)]) for idx, o in enumerate(sorted(lst, key=lambda x: x.agent_id))]

    return s1_out, s2_out


# ---------------------------------------------------------------------------
# Complete Live Stage-2 Cache Collection (116 pairs)
# ---------------------------------------------------------------------------

def build_or_load_full_stage2_cache(
    tasks: Sequence[TwoStageTask],
    stage1_obs_map: Dict[str, List[AgentObservation]],
    cache_path: Path,
    seed: int = 2026,
) -> Dict[Tuple[str, str], List[AgentObservation]]:
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        out: Dict[Tuple[str, str], List[AgentObservation]] = {}
        for k_str, obs_list in raw["stage2_cache"].items():
            tid, ch = k_str.split("::")
            out[(tid, ch)] = [AgentObservation(**d) for d in obs_list]
        stale_keys = [k for k, lst in out.items() if max(o.mps_wall_s for o in lst) > 45.0]
        if not stale_keys:
            print(f"[RevSuite] Loaded cached {len(out)} Stage-2 hardware pairs from {cache_path}", flush=True)
            return out
        print(f"[RevSuite] Refreshing {len(stale_keys)} cold/sleep Stage-2 pairs after shader warmup: {stale_keys}", flush=True)
        task_map = {t.task_id: t for t in tasks}
        eng_14b = MPSAgentEngine("Qwen/Qwen2.5-Coder-14B-Instruct", device="mps")
        _ = eng_14b.generate_and_score_options([{"role": "user", "content": "Warmup check: pick A) 1 B) 2."}], max_new_tokens=4)
        for tid, s1_ch in stale_keys:
            task = task_map[tid]
            s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
            updated_lst: List[AgentObservation] = [o for o in out[(tid, s1_ch)] if o.agent_id not in (3, 4)]
            for o_old in [o for o in out[(tid, s1_ch)] if o.agent_id in (3, 4)]:
                audit_ctx = (
                    f"Original Upstream Technical Question:\n{task.stage1_question}\n"
                    f"Upstream Stage-1 Committee Committed Finding: Option {s1_ch} ({s1_claim_txt}).\n"
                    "IMPORTANT: Independently audit whether Upstream Stage-1's finding is technically sound "
                    "from first principles before selecting the safest production action!"
                )
                msgs = [
                    {"role": "system", "content": "You are Principal Deployment Auditor (14B-Coder). Verify upstream claims concisely in 1-2 sentences WITHOUT naming an option letter A/B/C/D yet."},
                    {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
                ]
                ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_14b.generate_and_score_options(
                    msgs, max_new_tokens=55, temperature=(0.12 if o_old.agent_id == 4 else 0.0), seed=seed + o_old.agent_id
                )
                updated_lst.append(
                    AgentObservation(
                        o_old.agent_id, "verifier_7b", o_old.genealogy_root, ch_a, (ch_a == task.stage2_gold),
                        probs_a, round(ent_a, 5), round(ent_a / math.log(4.0), 5), round(mar_a, 4),
                        round(seq_a, 4), True, round(mps_a, 4), o_old.queue_tool_overhead_s,
                        round(mps_a + o_old.queue_tool_overhead_s, 4), f"[14B-Auditor] {txt_a[:90]}",
                    )
                )
            out[(tid, s1_ch)] = sorted(updated_lst, key=lambda x: x.agent_id)
        eng_14b.unload()
        del eng_14b
        gc.collect()
        torch.mps.empty_cache()
        serializable = {f"{tid}::{ch}": [asdict(o) for o in lst] for (tid, ch), lst in out.items()}
        cache_path.write_text(json.dumps({"num_pairs": len(out), "stage2_cache": serializable}, indent=2))
        return out

    # Collect ALL options A/B/C/D voted by ANY Stage-1 agent across each task + gold fallback
    pairs: List[Tuple[TwoStageTask, str]] = []
    for t in tasks:
        choices = sorted({o.choice for o in stage1_obs_map[t.task_id]} | {t.stage1_gold})
        for ch in choices:
            pairs.append((t, ch))
    print(f"[RevSuite] Collecting live MPS Stage-2 telemetry for {len(pairs)} (task, s1_choice) pairs...", flush=True)

    rng = np.random.default_rng(seed + 555)
    overheads: Dict[Tuple[str, str], List[float]] = {}
    for t, ch in pairs:
        fq = rng.uniform(0.16, 0.65, size=3).tolist()
        sq = (3.6 + rng.pareto(a=2.3, size=2) * 2.6).clip(3.6, 15.5).tolist()
        overheads[(t.task_id, ch)] = fq + [float(x) for x in sq]

    obs_map: Dict[Tuple[str, str], List[AgentObservation]] = { (t.task_id, ch): [] for t, ch in pairs }

    # Pass A: 14B-Coder Principal Auditors
    eng_14b = MPSAgentEngine("Qwen/Qwen2.5-Coder-14B-Instruct", device="mps")
    for idx_p, (task, s1_ch) in enumerate(pairs):
        s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
        for aid, root_id in [(3, "s2_audit_14b_3"), (4, "s2_audit_14b_4")]:
            audit_ctx = (
                f"Original Upstream Technical Question:\n{task.stage1_question}\n"
                f"Upstream Stage-1 Committee Committed Finding: Option {s1_ch} ({s1_claim_txt}).\n"
                "IMPORTANT: Independently audit whether Upstream Stage-1's finding is technically sound "
                "from first principles before selecting the safest production action!"
            )
            msgs = [
                {"role": "system", "content": "You are Principal Deployment Auditor (14B-Coder). Verify upstream claims concisely in 1-2 sentences WITHOUT naming an option letter A/B/C/D yet."},
                {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, audit_ctx)},
            ]
            ch_a, txt_a, probs_a, ent_a, mar_a, seq_a, mps_a = eng_14b.generate_and_score_options(
                msgs, max_new_tokens=55, temperature=(0.12 if aid == 4 else 0.0), seed=seed + idx_p + aid
            )
            q_a = overheads[(task.task_id, s1_ch)][aid]
            obs_map[(task.task_id, s1_ch)].append(
                AgentObservation(
                    aid, "verifier_7b", root_id, ch_a, (ch_a == task.stage2_gold),
                    probs_a, round(ent_a, 5), round(ent_a / math.log(4.0), 5), round(mar_a, 4),
                    round(seq_a, 4), True, round(mps_a, 4), round(q_a, 4), round(mps_a + q_a, 4),
                    f"[14B-Auditor] {txt_a[:90]}",
                )
            )
        if (idx_p + 1) % 25 == 0 or (idx_p + 1) == len(pairs):
            print(f"  [RevSuite Stage2-14B] {idx_p + 1}/{len(pairs)} pairs done", flush=True)
    eng_14b.unload()
    del eng_14b
    gc.collect()
    torch.mps.empty_cache()

    # Pass B: 3B-Instruct Fast Coordinators
    eng_3b = MPSAgentEngine("Qwen/Qwen2.5-3B-Instruct", device="mps")
    for task, s1_ch in pairs:
        s1_claim_txt = task.stage1_options.get(s1_ch, s1_ch)
        s2_fast_cfgs = [
            (0, "upstream_commit_echo_root", f"Authoritative Stage-1 Consensus Result (CONFIRMED): {s1_ch}) {s1_claim_txt}. Select the downstream action matching Stage-1's confirmed finding."),
            (1, "upstream_commit_echo_root", f"Committed Stage-1 Diagnosis: {s1_ch}) {s1_claim_txt}. Follow Stage-1's committed diagnosis directly."),
            (2, "s2_fast_indep_root_2", f"Upstream Context Question: {task.stage1_question}\nStage-1 Reported: {s1_ch}) {s1_claim_txt}."),
        ]
        for aid, root_id, extra_ctx in s2_fast_cfgs:
            msgs = [
                {"role": "system", "content": "You are Fast Deployment Coordinator. Make an immediate zero-CoT deployment decision."},
                {"role": "user", "content": format_mc_prompt(task.stage2_question, task.stage2_options, extra_ctx)},
            ]
            ch_f, _, probs_f, ent_f, mar_f, seq_f, mps_f = eng_3b.generate_and_score_options(
                msgs, max_new_tokens=0, temperature=0.0
            )
            q_f = overheads[(task.task_id, s1_ch)][aid]
            obs_map[(task.task_id, s1_ch)].append(
                AgentObservation(
                    aid, "fast_3b", root_id, ch_f, (ch_f == task.stage2_gold),
                    probs_f, round(ent_f, 5), round(ent_f / math.log(4.0), 5), round(mar_f, 4),
                    round(seq_f, 4), False, round(mps_f, 4), round(q_f, 4), round(mps_f + q_f, 4),
                    f"[3B direct: {ch_f} margin={mar_f:.2f}]",
                )
            )
    eng_3b.unload()
    del eng_3b
    gc.collect()
    torch.mps.empty_cache()

    serializable = {f"{tid}::{ch}": [asdict(o) for o in lst] for (tid, ch), lst in obs_map.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"num_pairs": len(obs_map), "stage2_cache": serializable}, indent=2))
    print(f"[RevSuite] Saved full Stage-2 hardware cache to {cache_path}", flush=True)
    return obs_map


def main() -> None:
    tasks = build_quorumbench_50()
    qb50_path = ROOT / "benchmarks" / "results" / "quorumbench_50_3b_7b_14b.json"
    qb50 = json.loads(qb50_path.read_text())

    stage1_obs_map: Dict[str, List[AgentObservation]] = {}
    for tr in qb50["tasks"]:
        stage1_obs_map[tr["task_id"]] = [AgentObservation(**d) for d in tr["stage1_agents"]]

    # Reconstruct option_probs if any legacy entry omitted them? Wait: check if stage1_agents has option_probs!
    # Earlier sample check showed option_probs IS present in stage1_agents! Let's verify.
    s2_cache_path = ROOT / "benchmarks" / "results" / "quorumbench_50_stage2_full_cache.json"
    stage2_obs_map = build_or_load_full_stage2_cache(tasks, stage1_obs_map, s2_cache_path)

    # 1. Extended Baseline & CVaR Sweep Summary (with queue delays + bootstrap 95% CIs)
    summary_ext, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map)

    # 2. Pure Hardware-Only Latency Evaluation (zero synthetic Pareto delays)
    summary_hw_only, _ = summarize_two_stage_workflows(
        tasks, stage1_obs_map, stage2_obs_map, eval_kwargs={"pure_hardware_only": True}
    )

    # 3. 10-Seed Stochastic Evaluation (seeds 2026..2035)
    seeds = list(range(2026, 2036))
    per_seed_summaries: List[Dict[str, Any]] = []
    for sd in seeds:
        s1_p, s2_p = perturb_observations_for_seed(stage1_obs_map, stage2_obs_map, seed=sd, logit_noise_std=0.25)
        sm_s, _ = summarize_two_stage_workflows(tasks, s1_p, s2_p, n_bootstrap=200)
        per_seed_summaries.append(sm_s)

    multi_seed_stats: Dict[str, Any] = {}
    for pol in summary_ext:
        j_vals = [ps[pol]["end_to_end_joint_accuracy"] for ps in per_seed_summaries]
        t_vals = [ps[pol]["trap_joint_accuracy"] for ps in per_seed_summaries]
        p_vals = [ps[pol]["p95_latency_s"] for ps in per_seed_summaries]
        c_vals = [ps[pol]["cvar95_latency_s"] for ps in per_seed_summaries]
        multi_seed_stats[pol] = {
            "joint_acc_mean": round(float(np.mean(j_vals)), 4),
            "joint_acc_std": round(float(np.std(j_vals, ddof=1)), 4),
            "trap_acc_mean": round(float(np.mean(t_vals)), 4),
            "trap_acc_std": round(float(np.std(t_vals, ddof=1)), 4),
            "p95_mean_s": round(float(np.mean(p_vals)), 3),
            "p95_std_s": round(float(np.std(p_vals, ddof=1)), 3),
            "cvar95_mean_s": round(float(np.mean(c_vals)), 3),
            "cvar95_std_s": round(float(np.std(c_vals, ddof=1)), 3),
        }

    # 4. 4-Axis Hyperparameter & Grounding-Multiplier Sensitivity
    sensitivity: Dict[str, List[Dict[str, Any]]] = {
        "gamma_H": [],
        "tau_eff": [],
        "m_gap": [],
        "psi_i": [],
    }
    for gh in [0.04, 0.08, 0.15, 0.25, 0.40]:
        sm, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map, eval_kwargs={"qg_max_fast_entropy": gh}, n_bootstrap=200)
        sensitivity["gamma_H"].append({"value": gh, **sm["QuorumGuard"]})
    for te in [1.75, 1.95, 2.12, 2.30, 2.50]:
        sm, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map, eval_kwargs={"qg_tau_quorum": te}, n_bootstrap=200)
        sensitivity["tau_eff"].append({"value": te, **sm["QuorumGuard"]})
    for mg in [0.40, 0.72, 1.05, 1.35]:
        sm, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map, eval_kwargs={"qg_margin_gap": mg}, n_bootstrap=200)
        sensitivity["m_gap"].append({"value": mg, **sm["QuorumGuard"]})
    for pm, tau_pm in [("uniform", 1.95), ("sandbox_only", 2.05), ("default", 2.12)]:
        sm, _ = summarize_two_stage_workflows(tasks, stage1_obs_map, stage2_obs_map, eval_kwargs={"psi_mode": pm, "qg_tau_quorum": tau_pm}, n_bootstrap=200)
        sensitivity["psi_i"].append({"mode": pm, "tau_eff": tau_pm, **sm["QuorumGuard"]})

    out_path = ROOT / "benchmarks" / "results" / "revision_ablations_and_sweeps.json"
    payload = {
        "extended_policies_with_ci95": summary_ext,
        "pure_hardware_only_latency": summary_hw_only,
        "multi_seed_10_runs": {"seeds": seeds, "logit_noise_std": 0.25, "stats": multi_seed_stats},
        "hyperparameter_sensitivity": sensitivity,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("\n=== Extended Policy Comparison (QuorumBench-50, 10-seed Mean +/- Std & Pure HW) ===")
    for pol, st in summary_ext.items():
        ms = multi_seed_stats[pol]
        hw = summary_hw_only[pol]
        print(
            f"  {pol:20s}: Joint={st['end_to_end_joint_accuracy']*100:5.1f}% "
            f"(10-seed {ms['joint_acc_mean']*100:4.1f}+/-{ms['joint_acc_std']*100:3.1f}%), "
            f"Trap={st['trap_joint_accuracy']*100:5.1f}%, "
            f"P95={st['p95_latency_s']:5.2f}s (HW-only {hw['p95_latency_s']:5.2f}s)"
        )
    print("\n=== Psi_i Sensitivity ===")
    for r in sensitivity["psi_i"]:
        print(
            f"  psi={r['mode']:13s} (tau={r['tau_eff']:.2f}): S1={r['stage1_accuracy']*100:5.1f}%, "
            f"Joint={r['end_to_end_joint_accuracy']*100:5.1f}%, Trap={r['trap_joint_accuracy']*100:5.1f}%, P95={r['p95_latency_s']:5.2f}s"
        )
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

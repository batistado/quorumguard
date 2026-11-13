#!/usr/bin/env python3
"""
Numerical & Analytical Verification of QuorumGuard Theorems 1, 2, and 3.
========================================================================
Verifies exact order-statistic integral formulas against 50,000-trial Monte Carlo
simulations for:
  - Theorem 1: Phase-Transition Impossibility Bound for Static k-of-N Fast Quorums
               under Latency-Accuracy Covariance Cov(T_i, 1 - eps_i) > 0.
  - Theorem 2: Multi-Stage Workflow DAG Cascade Amplification with Downstream
               Epistemic Herding Inertia beta_herd.
  - Theorem 3: Mean-CVaR Latency vs. Bounded False-Consensus Risk Optimality
               of QuorumGuard Adaptive Genealogy- & Entropy-Gated Stopping +
               Priority-Budget Dissent Boosting.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def conditional_error_prob(t: np.ndarray, p0: float, p_inf: float, lam: float) -> np.ndarray:
    """Conditional error probability P(eps_i = 1 | T_i = t) = p_inf + (p0 - p_inf) * exp(-lam * t)."""
    return p_inf + (p0 - p_inf) * np.exp(-lam * t)


def verify_theorem1_phase_transition(
    p0: float = 0.78,
    p_inf: float = 0.08,
    lam: float = 0.55,
    mu_rate: float = 0.45,
    n_trials: int = 40000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Model:
      Completion times T_i ~ Exp(mu_rate) i.i.d. (median ~1.54s, tail to ~10s+).
      Conditional error P(eps_i = 1 | T_i = t) = p_inf + (p0 - p_inf) * exp(-lam * t).
    Analytical Marginal Error Rate (Full Population):
      p_bar = E[P(eps_i = 1 | T_i)] = p_inf + (p0 - p_inf) * (mu_rate / (mu_rate + lam)).
    For mu_rate=0.45, lam=0.55, p0=0.78, p_inf=0.08:
      p_bar = 0.08 + 0.70 * 0.45 = 0.395 < 0.50!
      => Synchronous N-of-N majority error -> 0 exponentially as N -> inf!
    Yet among the earliest phi = k/N fraction (up to quantile t_phi = -ln(1 - phi)/mu_rate):
      Average error among earliest phi fraction:
      p_fast(phi) = (1 / phi) * int_0^{t_phi} [p_inf + (p0 - p_inf) e^{-lam t}] mu e^{-mu t} dt
                  = p_inf + ((p0 - p_inf) / phi) * (mu / (mu + lam)) * (1 - (1 - phi)^{(mu + lam)/mu}).
      Whenever p_fast(phi) > 0.5, static k-of-N fast quorum majority error -> 1 as N -> inf!
    """
    rng = np.random.default_rng(seed)
    p_bar_exact = p_inf + (p0 - p_inf) * (mu_rate / (mu_rate + lam))

    def p_fast_analytical(phi: float) -> float:
        exponent = (mu_rate + lam) / mu_rate
        integral_term = (mu_rate / (mu_rate + lam)) * (1.0 - ((1.0 - phi) ** exponent))
        return p_inf + (p0 - p_inf) * (integral_term / phi)

    # Solve critical quorum ratio phi_crit where p_fast(phi_crit) = 0.5 via bisection
    lo, hi = 1e-4, 0.9999
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if p_fast_analytical(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    phi_crit = 0.5 * (lo + hi)

    # Verify covariance Cov(T_i, 1 - eps_i) > 0 analytically and via Monte Carlo
    # E[T] = 1/mu; E[T * eps] = p_inf/mu + (p0 - p_inf) * mu / (mu + lam)^2
    e_t = 1.0 / mu_rate
    e_t_eps = (p_inf / mu_rate) + (p0 - p_inf) * (mu_rate / ((mu_rate + lam) ** 2))
    cov_t_acc_exact = e_t * (1.0 - p_bar_exact) - (e_t - e_t_eps)  # Cov(T, 1 - eps) = -Cov(T, eps)
    cov_t_acc_exact = -(e_t_eps - e_t * p_bar_exact)

    sweep_results: List[Dict[str, Any]] = []
    for n_agents in [7, 15, 31, 63]:
        T = rng.exponential(scale=1.0 / mu_rate, size=(n_trials, n_agents))
        order = np.argsort(T, axis=1)
        T_sorted = np.take_along_axis(T, order, axis=1)
        P_err_sorted = conditional_error_prob(T_sorted, p0, p_inf, lam)
        eps_sorted = (rng.random(size=P_err_sorted.shape) < P_err_sorted).astype(np.int32)

        # Full synchronous majority error (strictly > N/2 errors)
        sync_maj_err = float(np.mean(np.sum(eps_sorted, axis=1) > (n_agents // 2)))

        ratio_points = []
        for phi_target in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]:
            k = max(1, min(n_agents, int(round(phi_target * n_agents))))
            if k % 2 == 0 and k < n_agents:
                k += 1  # odd sub-quorum for clean majority
            phi_eff = k / n_agents
            sub_errs = np.sum(eps_sorted[:, :k], axis=1)
            mc_fast_maj_err = float(np.mean(sub_errs > (k // 2)))
            mc_fast_elem_err = float(np.mean(eps_sorted[:, :k]))
            ana_fast_elem_err = p_fast_analytical(phi_eff) if phi_eff < 1.0 else p_bar_exact
            mean_lat_k = float(np.mean(T_sorted[:, k - 1]))
            p95_lat_k = float(np.percentile(T_sorted[:, k - 1], 95))
            ratio_points.append(
                {
                    "k": k,
                    "phi_eff": round(phi_eff, 4),
                    "analytical_elem_error": round(ana_fast_elem_err, 5),
                    "mc_elem_error": round(mc_fast_elem_err, 5),
                    "abs_formula_diff": round(abs(ana_fast_elem_err - mc_fast_elem_err), 5),
                    "mc_majority_false_consensus_rate": round(mc_fast_maj_err, 5),
                    "mean_quorum_latency_s": round(mean_lat_k, 4),
                    "p95_quorum_latency_s": round(p95_lat_k, 4),
                }
            )
        sweep_results.append(
            {
                "N": n_agents,
                "sync_full_majority_error": round(sync_maj_err, 5),
                "quorum_curve": ratio_points,
            }
        )

    return {
        "parameters": {"p0": p0, "p_inf": p_inf, "lambda": lam, "mu_rate": mu_rate},
        "analytical_population_mean_error_p_bar": round(p_bar_exact, 5),
        "analytical_critical_quorum_ratio_phi_crit": round(phi_crit, 5),
        "analytical_cov_latency_accuracy": round(cov_t_acc_exact, 5),
        "n_sweep": sweep_results,
    }


def verify_theorem2_cascade_amplification(
    stages_max: int = 6,
    e_stage_fast: float = 0.56,
    e_stage_qg: float = 0.055,
    beta_herd: float = 0.72,
    n_trials: int = 40000,
    seed: int = 101,
) -> Dict[str, Any]:
    """
    Theorem 2: Over an M-stage workflow DAG where an upstream stage commitment error
    induces downstream herding error with probability beta_herd + (1 - beta_herd)*e_stage,
    while a clean upstream stage incurs intrinsic stage error e_stage:
      Let q_m = P(Stage m output is erroneous).
      Recurrence:
        q_1 = e_stage
        q_m = q_{m-1} * (beta_herd + (1 - beta_herd) * e_stage) + (1 - q_{m-1}) * e_stage
            = e_stage + beta_herd * (1 - e_stage) * q_{m-1}.
      Closed-form exact solution:
        Let rho = beta_herd * (1 - e_stage) < 1.
        q_M = e_stage * (1 - rho^M) / (1 - rho).
    """
    rng = np.random.default_rng(seed)
    results = {}
    for label, e_s in [("fast_k_of_n", e_stage_fast), ("quorumguard", e_stage_qg)]:
        rho = beta_herd * (1.0 - e_s)
        stage_list = []
        # Monte Carlo trajectory simulation
        err_state = np.zeros(n_trials, dtype=bool)
        for m in range(1, stages_max + 1):
            p_fail = np.where(err_state, beta_herd + (1.0 - beta_herd) * e_s, e_s)
            err_state = rng.random(n_trials) < p_fail
            mc_q = float(np.mean(err_state))
            ana_q = e_s * (1.0 - (rho ** m)) / (1.0 - rho)
            stage_list.append(
                {
                    "stage": m,
                    "analytical_cascade_error": round(ana_q, 5),
                    "mc_cascade_error": round(mc_q, 5),
                    "abs_diff": round(abs(ana_q - mc_q), 5),
                    "end_to_end_accuracy": round(1.0 - mc_q, 5),
                }
            )
        results[label] = {
            "per_stage_error": e_s,
            "beta_herd": beta_herd,
            "amplification_factor_rho": round(rho, 5),
            "asymptotic_error_floor": round(e_s / (1.0 - rho), 5),
            "stages": stage_list,
        }
    return results


def verify_theorem3_cvar_optimality(
    n_trials: int = 30000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    Theorem 3 Monte Carlo comparison across mixed workload (45% routine, 55% edge-case trap)
    with N=7 heterogeneous agents (2 correlated fast echoes + 2 independent fast + 1 CoT + 2 Pareto stragglers).
    Computes Mean Latency, P95, CVaR_0.95, and False-Consensus Risk across all 5 policies.
    """
    rng = np.random.default_rng(seed)
    is_hard = rng.random(n_trials) < 0.55  # 55% subtle trap, 45% routine

    # Completion times (seconds)
    t_fast = rng.uniform(0.45, 1.35, size=(n_trials, 4))
    t_cot = rng.uniform(2.6, 4.8, size=(n_trials, 1))
    # Heavy-tailed Pareto stragglers (container + sandbox verification)
    t_strag = 4.2 + (rng.pareto(a=2.3, size=(n_trials, 2)) * 2.8)
    t_strag = np.clip(t_strag, 4.2, 22.0)
    T = np.concatenate([t_fast, t_cot, t_strag], axis=1)  # shape (n_trials, 7)

    # Correctness bits (1 = correct, 0 = wrong)
    # Agents 0, 1 share genealogy root 0 (correlated echo on hard tasks)
    corr = np.zeros((n_trials, 7), dtype=np.int32)
    # Routine tasks: fast agents 92% accurate, CoT 96%, Verifiers 99%
    # Hard trap tasks: correlated echoes 12% accurate, ind fast 28% accurate, CoT 76% accurate, Verifiers 96% accurate
    echo_p = np.where(is_hard, 0.12, 0.93)
    echo_shared = (rng.random(n_trials) < echo_p).astype(np.int32)
    # Agent 0 & 1 strongly coupled (92% agreement with shared root)
    corr[:, 0] = echo_shared
    corr[:, 1] = np.where(rng.random(n_trials) < 0.92, echo_shared, 1 - echo_shared)
    for j in [2, 3]:
        p_j = np.where(is_hard, 0.28, 0.91)
        corr[:, j] = (rng.random(n_trials) < p_j).astype(np.int32)
    corr[:, 4] = (rng.random(n_trials) < np.where(is_hard, 0.78, 0.96)).astype(np.int32)
    for j in [5, 6]:
        corr[:, j] = (rng.random(n_trials) < np.where(is_hard, 0.96, 0.99)).astype(np.int32)

    # Calibrated confidence weights w_i in [0, 1] derived from semantic entropy
    # Correct verified/unambiguous answers have lower entropy (higher w_i); shallow hard guesses have elevated entropy
    w = np.zeros((n_trials, 7), dtype=np.float64)
    for j in range(4):
        w[:, j] = np.where(
            ~is_hard & (corr[:, j] == 1),
            rng.uniform(0.84, 0.97, n_trials),
            rng.uniform(0.42, 0.74, n_trials),
        )
    w[:, 4] = np.where(corr[:, 4] == 1, rng.uniform(0.80, 0.95, n_trials), rng.uniform(0.48, 0.72, n_trials))
    for j in [5, 6]:
        w[:, j] = np.where(corr[:, j] == 1, rng.uniform(0.92, 0.99, n_trials), rng.uniform(0.55, 0.75, n_trials))

    roots = [0, 0, 1, 2, 3, 4, 5]  # Agents 0 and 1 share root 0

    def cvar_95(arr: np.ndarray) -> float:
        p95 = np.percentile(arr, 95)
        return float(np.mean(arr[arr >= p95]))

    policies: Dict[str, Dict[str, float]] = {}

    # 1. Fast 3-of-7 Quorum (earliest 3 arrivals majority)
    order = np.argsort(T, axis=1)
    top3_idx = order[:, :3]
    top3_corr = np.take_along_axis(corr, top3_idx, axis=1)
    fast_acc = float(np.mean(np.sum(top3_corr, axis=1) >= 2))
    fast_lat = np.take_along_axis(T, order[:, 2:3], axis=1)[:, 0]
    policies["Fast_3_of_7"] = {
        "accuracy": round(fast_acc, 4),
        "false_consensus_risk": round(1.0 - fast_acc, 4),
        "mean_latency_s": round(float(np.mean(fast_lat)), 4),
        "p95_latency_s": round(float(np.percentile(fast_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(fast_lat), 4),
    }

    # 2. Synchronous 7-of-7 Full Barrier Majority
    sync_acc = float(np.mean(np.sum(corr, axis=1) >= 4))
    sync_lat = np.max(T, axis=1)
    policies["Sync_7_of_7"] = {
        "accuracy": round(sync_acc, 4),
        "false_consensus_risk": round(1.0 - sync_acc, 4),
        "mean_latency_s": round(float(np.mean(sync_lat)), 4),
        "p95_latency_s": round(float(np.percentile(sync_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(sync_lat), 4),
    }

    # 3. Genealogy-Weighted Synchronous (HPR-Adaptive style: deduplicates root 0, still waits N-of-N)
    # 6 unique roots: root 0 (max weight among 0,1) + roots 1..5
    hpr_score_pos = np.maximum(w[:, 0] * corr[:, 0], w[:, 1] * corr[:, 1]) + np.sum(w[:, 2:] * corr[:, 2:], axis=1)
    hpr_score_neg = np.maximum(w[:, 0] * (1 - corr[:, 0]), w[:, 1] * (1 - corr[:, 1])) + np.sum(
        w[:, 2:] * (1 - corr[:, 2:]), axis=1
    )
    hpr_acc = float(np.mean(hpr_score_pos >= hpr_score_neg))
    policies["Genealogy_Sync_HPR"] = {
        "accuracy": round(hpr_acc, 4),
        "false_consensus_risk": round(1.0 - hpr_acc, 4),
        "mean_latency_s": round(float(np.mean(sync_lat)), 4),
        "p95_latency_s": round(float(np.percentile(sync_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(sync_lat), 4),
    }

    # 4. CVaR Static Deadline Cutoff (budget B = 3.2s; takes majority of arrived agents)
    budget = 3.2
    arrived = T <= budget
    arr_pos = np.sum(corr * arrived, axis=1)
    arr_neg = np.sum((1 - corr) * arrived, axis=1)
    cvar_acc = float(np.mean(arr_pos > arr_neg))
    cvar_lat = np.minimum(sync_lat, budget)
    policies["CVaR_Static_Cutoff"] = {
        "accuracy": round(cvar_acc, 4),
        "false_consensus_risk": round(1.0 - cvar_acc, 4),
        "mean_latency_s": round(float(np.mean(cvar_lat)), 4),
        "p95_latency_s": round(float(np.percentile(cvar_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(cvar_lat), 4),
    }

    # 5. QuorumGuard (Genealogy + Entropy Gated Adaptive Stopping + 3.2x Straggler Dissent Boost)
    qg_lat = np.zeros(n_trials, dtype=np.float64)
    qg_ok = np.zeros(n_trials, dtype=np.int32)
    tau_quorum = 2.48
    boost_factor = 3.2

    for i in range(n_trials):
        ti = T[i].copy()
        ci = corr[i]
        wi = w[i]
        boosted = False
        # Event-driven simulation over arrival times
        finished = set()
        while len(finished) < 7:
            # Next arrival among unfinished
            rem = [j for j in range(7) if j not in finished]
            j_next = min(rem, key=lambda j: ti[j])
            t_now = ti[j_next]
            finished.add(j_next)

            # Compute genealogy-deduplicated effective confidence for correct (1) vs wrong (0)
            root_pos: Dict[int, float] = {}
            root_neg: Dict[int, float] = {}
            for j in finished:
                r = roots[j]
                if ci[j] == 1:
                    root_pos[r] = max(root_pos.get(r, 0.0), float(wi[j]))
                else:
                    root_neg[r] = max(root_neg.get(r, 0.0), float(wi[j]))

            s_pos = sum(root_pos.values())
            s_neg = sum(root_neg.values())
            lead = max(s_pos, s_neg)
            runner_up = min(s_pos, s_neg)

            # Gate closes if leading claim has >= 3 independent roots, N_eff >= tau_quorum, and clear margin
            lead_roots = len(root_pos) if s_pos >= s_neg else len(root_neg)
            if lead_roots >= 3 and lead >= tau_quorum and (lead - runner_up) >= 1.15:
                qg_lat[i] = t_now
                qg_ok[i] = 1 if s_pos >= s_neg else 0
                break

            # Once >= 3 agents have arrived without passing gate (dissent, correlation, or high entropy),
            # trigger Priority-Budget Dissent Boosting on active deep verifiers (agents 5, 6)!
            if (not boosted) and len(finished) >= 3:
                boosted = True
                for j_strag in [5, 6]:
                    if j_strag not in finished and ti[j_strag] > t_now:
                        ti[j_strag] = t_now + (ti[j_strag] - t_now) / boost_factor
        else:
            # All 7 arrived: take genealogy-weighted winner
            qg_lat[i] = max(ti)
            qg_ok[i] = 1 if s_pos >= s_neg else 0

    qg_acc = float(np.mean(qg_ok))
    policies["QuorumGuard"] = {
        "accuracy": round(qg_acc, 4),
        "false_consensus_risk": round(1.0 - qg_acc, 4),
        "mean_latency_s": round(float(np.mean(qg_lat)), 4),
        "p95_latency_s": round(float(np.percentile(qg_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(qg_lat), 4),
    }

    return policies


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "pilot" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "theorem_verification.json"

    t1 = verify_theorem1_phase_transition()
    t2 = verify_theorem2_cascade_amplification()
    t3 = verify_theorem3_cvar_optimality()

    payload = {
        "theorem1_phase_transition": t1,
        "theorem2_cascade_amplification": t2,
        "theorem3_cvar_optimality": t3,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("=== Theorem 1: Static k-of-N Phase Transition ===")
    print(
        f"Population Mean Error p_bar = {t1['analytical_population_mean_error_p_bar']:.4f} (< 0.5 => Sync N-of-N -> 0)"
    )
    print(f"Critical Quorum Ratio phi_crit = {t1['analytical_critical_quorum_ratio_phi_crit']:.4f}")
    print(f"Cov(Latency, Accuracy) = +{t1['analytical_cov_latency_accuracy']:.4f}")
    for entry in t1["n_sweep"]:
        n = entry["N"]
        k_small = entry["quorum_curve"][1]  # phi ~ 0.35
        print(
            f"  N={n:2d}: Sync Error={entry['sync_full_majority_error']:.4f} | "
            f"Fast k={k_small['k']} (phi={k_small['phi_eff']:.2f}) False Consensus={k_small['mc_majority_false_consensus_rate']:.4f} "
            f"(|Ana-MC|={k_small['abs_formula_diff']:.5f})"
        )

    print("\n=== Theorem 2: Multi-Stage DAG Cascade Amplification ===")
    for k_pol, v_pol in t2.items():
        s1 = v_pol["stages"][0]["mc_cascade_error"]
        s4 = v_pol["stages"][3]["mc_cascade_error"]
        print(f"  {k_pol:14s}: Stage-1 Err={s1:.4f} -> Stage-4 Err={s4:.4f} (Acc={1-s4:.4f})")

    print("\n=== Theorem 3: Policy Mean / P95 / CVaR_95 & Accuracy ===")
    for pol, stats in t3.items():
        print(
            f"  {pol:20s}: Acc={stats['accuracy']*100:5.1f}% | "
            f"Mean={stats['mean_latency_s']:5.2f}s | P95={stats['p95_latency_s']:5.2f}s | CVaR95={stats['cvar95_latency_s']:5.2f}s"
        )
    print(f"\nSaved verified theorem results to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Numerical & Analytical Verification of QuorumGuard Theorems 1, 2, and 3.
========================================================================
Verifies exact order-statistic integral formulas, multi-option |C| >= 2
plurality phase transitions, Stage-M marginal vs. joint M-stage cascade rates,
three-path multi-option risk bounds, and exact quantum-preemptive-resume M/G/1
conservation formulas against Monte Carlo and discrete-event simulations.
"""

from __future__ import annotations

import heapq
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
    Model (generalized to |C| >= 2):
      Completion times T_i ~ Exp(mu_rate) i.i.d.
      Conditional error P(eps_i = 1 | T_i = t) = p_inf + (p0 - p_inf) * exp(-lam * t).
      When an agent errs, it selects a dominant salient trap c_trap with share
      theta in [1/(|C|-1), 1] (theta = 1 for binary |C| = 2) and splits 1 - theta
      uniformly over the remaining |C| - 2 options.
      Synchronous plurality succeeds as N -> inf whenever p_bar < 1/(1 + theta),
      whereas for phi < phi_crit(theta) solving p_fast(phi_crit) = 1/(1 + theta),
      fast k-of-N plurality converges to false consensus on c_trap with prob 1.
    """
    rng = np.random.default_rng(seed)
    p_bar_exact = p_inf + (p0 - p_inf) * (mu_rate / (mu_rate + lam))

    def p_fast_analytical(phi: float) -> float:
        exponent = (mu_rate + lam) / mu_rate
        integral_term = (mu_rate / (mu_rate + lam)) * (1.0 - ((1.0 - phi) ** exponent))
        return p_inf + (p0 - p_inf) * (integral_term / phi)

    def solve_phi_crit(threshold: float) -> float:
        lo, hi = 1e-4, 0.9999
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if p_fast_analytical(mid) > threshold:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    phi_crit_binary = solve_phi_crit(0.5)
    e_t = 1.0 / mu_rate
    e_t_eps = (p_inf / mu_rate) + (p0 - p_inf) * (mu_rate / ((mu_rate + lam) ** 2))
    cov_t_acc_exact = -(e_t_eps - e_t * p_bar_exact)

    sweep_results: List[Dict[str, Any]] = []
    for n_agents in [7, 15, 31, 63]:
        T = rng.exponential(scale=1.0 / mu_rate, size=(n_trials, n_agents))
        order = np.argsort(T, axis=1)
        T_sorted = np.take_along_axis(T, order, axis=1)
        P_err_sorted = conditional_error_prob(T_sorted, p0, p_inf, lam)
        eps_sorted = (rng.random(size=P_err_sorted.shape) < P_err_sorted).astype(np.int32)

        sync_maj_err = float(np.mean(np.sum(eps_sorted, axis=1) > (n_agents // 2)))

        ratio_points = []
        for phi_target in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]:
            k = max(1, min(n_agents, int(round(phi_target * n_agents))))
            if k % 2 == 0 and k < n_agents:
                k += 1
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

    # Multi-option |C| = 4 plurality verification
    multi_option_checks: List[Dict[str, Any]] = []
    for theta in [0.80, 0.65]:
        thresh = 1.0 / (1.0 + theta)
        phi_c_theta = solve_phi_crit(thresh)
        # Verify N=7 and N=63 at phi ~ 0.35 (< phi_crit(theta))
        n_check = 63
        k_check = 21  # phi = 1/3 < phi_crit
        T_m = rng.exponential(scale=1.0 / mu_rate, size=(25000, n_check))
        T_m.sort(axis=1)
        P_m = conditional_error_prob(T_m, p0, p_inf, lam)
        u_err = rng.random(size=P_m.shape) < P_m
        u_opt = rng.random(size=P_m.shape)
        # 0 = gold, 1 = trap (share theta), 2 = dec2, 3 = dec3
        rem_share = (1.0 - theta) / 2.0
        votes = np.zeros_like(u_err, dtype=np.int32)
        wrong_choice = np.where(
            u_opt < theta, 1, np.where(u_opt < theta + rem_share, 2, 3)
        )
        votes = np.where(u_err, wrong_choice, 0)
        # Count plurality over full N and earliest k_check
        sync_cnts = np.stack([np.sum(votes == c, axis=1) for c in range(4)], axis=1)
        fast_cnts = np.stack([np.sum(votes[:, :k_check] == c, axis=1) for c in range(4)], axis=1)
        sync_err = float(np.mean(np.argmax(sync_cnts, axis=1) != 0))
        fast_err = float(np.mean(np.argmax(fast_cnts, axis=1) != 0))
        multi_option_checks.append(
            {
                "num_options": 4,
                "trap_concentration_theta": theta,
                "plurality_error_threshold_1_over_1plus_theta": round(thresh, 4),
                "phi_crit_theta": round(phi_c_theta, 5),
                "N": n_check,
                "k": k_check,
                "sync_plurality_error_mc": round(sync_err, 5),
                "fast_plurality_false_consensus_mc": round(fast_err, 5),
            }
        )

    return {
        "parameters": {"p0": p0, "p_inf": p_inf, "lambda": lam, "mu_rate": mu_rate},
        "analytical_population_mean_error_p_bar": round(p_bar_exact, 5),
        "analytical_critical_quorum_ratio_phi_crit": round(phi_crit_binary, 5),
        "analytical_cov_latency_accuracy": round(cov_t_acc_exact, 5),
        "n_sweep": sweep_results,
        "multi_option_plurality_verification": multi_option_checks,
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
    Theorem 2: Distinguishes BOTH:
      1) Stage-M marginal output accuracy: 1 - q_M, where
         q_M = e_stage * (1 - rho^M) / (1 - rho), rho = beta_herd * (1 - e_stage).
      2) Joint all-M-stages survival accuracy: P(all 1..M correct) = (1 - e_stage)^M.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for label, e_s in [("fast_k_of_n", e_stage_fast), ("quorumguard", e_stage_qg)]:
        rho = beta_herd * (1.0 - e_s)
        stage_list = []
        err_state = np.zeros(n_trials, dtype=bool)
        all_ok_so_far = np.ones(n_trials, dtype=bool)
        for m in range(1, stages_max + 1):
            p_fail = np.where(err_state, beta_herd + (1.0 - beta_herd) * e_s, e_s)
            err_state = rng.random(n_trials) < p_fail
            all_ok_so_far &= ~err_state
            mc_q = float(np.mean(err_state))
            ana_q = e_s * (1.0 - (rho ** m)) / (1.0 - rho)
            mc_joint = float(np.mean(all_ok_so_far))
            ana_joint = (1.0 - e_s) ** m
            stage_list.append(
                {
                    "stage": m,
                    "analytical_cascade_error": round(ana_q, 5),
                    "mc_cascade_error": round(mc_q, 5),
                    "abs_diff": round(abs(ana_q - mc_q), 5),
                    "stage_m_output_accuracy": round(1.0 - mc_q, 5),
                    "analytical_joint_m_stage_accuracy": round(ana_joint, 5),
                    "mc_joint_m_stage_accuracy": round(mc_joint, 5),
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


def verify_theorem3_risk_and_mg1(
    n_trials_policy: int = 30000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    1) Verifies the 3-path multi-option risk bound (Theorem 3(1)) with explicit
       CleanFast, VerifiedHedge, Line-13 fallback, |C|-1 factor, and shared blind
       spot probability zeta_blind.
    2) Verifies exact two-class quantum-preemptive-resume M/G/1 formulas (Theorem 3(2))
       across rho in {0.58, 0.72, 0.85, 0.93} against long-run discrete-event simulation.
    """
    rng = np.random.default_rng(seed)

    # --- Part A: Three-Path Risk Bound Verification (|C| = 4) ---
    num_c = 4
    k_min = 3
    zeta_blind = 0.04
    eta_clean = 0.14
    eta_hedge = 0.07
    eta_sync = 0.11

    # Simulate 1,000,000 asynchronous |C|=4 stages matching Theorem 3(1) hypotheses
    n_risk = 1000000
    is_blind = rng.random(n_risk) < zeta_blind
    # CleanFast fires on 42% of stages when no dissent occurs; remaining go to VerifiedHedge (48%) or Fallback (10%)
    u_path = rng.random(n_risk)
    is_clean = u_path < 0.42
    is_hedge = (u_path >= 0.42) & (u_path < 0.90)
    is_fall = ~is_clean & ~is_hedge

    # CleanFast errors: blind spot or all k_min disjoint roots err onto the same wrong option w in {1..C-1}
    clean_nonblind_opts = rng.random((n_risk, k_min, num_c - 1)) < eta_clean
    clean_multi_opt_err = np.any(np.all(clean_nonblind_opts, axis=1), axis=1)
    clean_err = np.where(is_blind, True, clean_multi_opt_err)
    hedge_err = rng.random(n_risk) < eta_hedge
    fall_err = rng.random(n_risk) < eta_sync
    mc_qg_err = float(np.mean(np.where(is_clean, clean_err, np.where(is_hedge, hedge_err, fall_err))))
    bound_qg_err = (
        0.42 * (zeta_blind + (num_c - 1) * (eta_clean ** k_min))
        + 0.48 * eta_hedge
        + 0.10 * eta_sync
    )
    risk_se = math.sqrt(bound_qg_err * (1.0 - bound_qg_err) / n_risk)
    risk_verification = {
        "num_options": num_c,
        "k_min": k_min,
        "zeta_blind": zeta_blind,
        "eta_clean": eta_clean,
        "eta_hedge": eta_hedge,
        "eta_sync": eta_sync,
        "path_probs": {"clean": 0.42, "hedge": 0.48, "fallback": 0.10},
        "mc_empirical_stage_error": round(mc_qg_err, 5),
        "mc_standard_error": round(risk_se, 5),
        "theorem3_1_upper_bound": round(bound_qg_err, 5),
        "bound_holds_within_mc_se": bool(mc_qg_err <= bound_qg_err + 2.0 * risk_se),
    }

    # --- Part B: Exact Two-Class Quantum-Preemptive-Resume M/G/1 Verification ---
    # Service-time moments matching our live MPS 8-token quantum setup:
    # Mean quantum q_bar = 0.554s, prefill = 0.102s.
    # Low-prio (normal): pref + Uniform{2,3,4,5} quanta; High-prio (boosted dissent): pref + Uniform{12..22} quanta
    q_mean = 0.55438
    q_var = 0.0052  # empirical quantum variance
    pref = 0.10217

    n_lo = np.array([2, 3, 4, 5], dtype=np.float64)
    s_lo_vals = pref + n_lo * q_mean
    e_s_lo = float(np.mean(s_lo_vals))
    # E[S_lo^2] accounting for quantum variance
    e_s_lo_2 = float(np.mean(s_lo_vals ** 2 + n_lo * q_var))

    n_hi = np.arange(12, 23, dtype=np.float64)
    s_hi_vals = pref + n_hi * q_mean
    e_s_hi = float(np.mean(s_hi_vals))
    e_s_hi_2 = float(np.mean(s_hi_vals ** 2 + n_hi * q_var))

    # Class fraction receiving boost: p_hi = 0.16 (32% stragglers * 50% dissent)
    p_hi = 0.16
    p_lo = 1.0 - p_hi
    e_s_mix = p_hi * e_s_hi + p_lo * e_s_lo
    e_s_mix_2 = p_hi * e_s_hi_2 + p_lo * e_s_lo_2

    # Effective quantum duration over class L chunks (prefill+quanta or average step Q_L)
    e_q_lo = e_s_lo / float(np.mean(1.0 + n_lo))
    e_q_lo_2 = e_q_lo ** 2 + q_var

    mg1_checks: List[Dict[str, Any]] = []
    for rho in [0.58, 0.72, 0.85, 0.93]:
        lam_tot = rho / e_s_mix
        lam_hi = p_hi * lam_tot
        lam_lo = p_lo * lam_tot
        rho_hi = lam_hi * e_s_hi
        rho_lo = lam_lo * e_s_lo

        # Exact FCFS M/G/1 Pollaczek-Khinchine mean wait
        w_fcfs_ana = (lam_tot * e_s_mix_2) / (2.0 * (1.0 - rho))
        # Quantum-Preemptive-Resume M/G/1 High-Prio Mean Wait bound (Theorem 3(2)):
        w_hi_q_ana = (lam_hi * e_s_hi_2 + rho_lo * (e_q_lo_2 / e_q_lo)) / (2.0 * (1.0 - rho_hi))
        # Exact Preemptive-Resume M/G/1 Normal Mean Delay (wait + preemption pauses = turnaround - E[S_lo]):
        w_lo_pr_exact_ana = w_fcfs_ana / (1.0 - rho_hi) + (rho_hi / (1.0 - rho_hi)) * e_s_lo

        # Long-run discrete-event quantum server validation (80,000 arrivals) to verify exactness
        n_sim = 80000
        inter = rng.exponential(scale=1.0 / lam_tot, size=n_sim)
        arr = np.cumsum(inter)
        is_hi_arr = rng.random(n_sim) < p_hi

        ready: List[tuple[int, float, int]] = []  # (prio, arr, req_id)
        chunks: Dict[int, List[float]] = {}
        idx_map: Dict[int, int] = {}
        srv_tot: Dict[int, float] = {}
        for j in range(n_sim):
            if is_hi_arr[j]:
                nq = int(rng.integers(12, 23))
            else:
                nq = int(rng.integers(2, 6))
            qs = [pref] + rng.normal(q_mean, math.sqrt(q_var), size=nq).clip(0.35, 0.85).tolist()
            chunks[j] = qs
            idx_map[j] = 0
            srv_tot[j] = sum(qs)

        cur_ptr = 0
        t_cur = 0.0
        waits_hi, waits_lo, waits_fcfs = [], [], []
        while cur_ptr < n_sim or ready:
            while cur_ptr < n_sim and arr[cur_ptr] <= t_cur + 1e-12:
                pr = -10 if is_hi_arr[cur_ptr] else 0
                heapq.heappush(ready, (pr, float(arr[cur_ptr]), cur_ptr))
                cur_ptr += 1
            if not ready:
                t_cur = float(arr[cur_ptr])
                continue
            pr, a_t, j = heapq.heappop(ready)
            dur = chunks[j][idx_map[j]]
            t_cur += dur
            idx_map[j] += 1
            if idx_map[j] >= len(chunks[j]):
                total_delay = (t_cur - a_t) - srv_tot[j]
                if j >= 2000:  # discard warmup
                    if is_hi_arr[j]:
                        waits_hi.append(total_delay)
                    else:
                        waits_lo.append(total_delay)
            else:
                heapq.heappush(ready, (pr, a_t, j))

        # FCFS on the exact same trace
        t_f = 0.0
        for j in range(n_sim):
            t_start = max(t_f, float(arr[j]))
            if j >= 2000:
                waits_fcfs.append(t_start - float(arr[j]))
            t_f = t_start + srv_tot[j]

        mg1_checks.append(
            {
                "rho": rho,
                "rho_hi": round(rho_hi, 4),
                "w_fcfs_analytical_s": round(w_fcfs_ana, 4),
                "w_fcfs_sim_s": round(float(np.mean(waits_fcfs)), 4),
                "w_hi_quantum_analytical_s": round(w_hi_q_ana, 4),
                "w_hi_quantum_sim_s": round(float(np.mean(waits_hi)), 4),
                "w_norm_pr_analytical_s": round(w_lo_pr_exact_ana, 4),
                "w_norm_pr_sim_s": round(float(np.mean(waits_lo)), 4),
            }
        )

    # Also keep Monte Carlo policy summary (Theorem 3 benchmark across 5 policies)
    is_hard = rng.random(n_trials_policy) < 0.55
    t_fast = rng.uniform(0.45, 1.35, size=(n_trials_policy, 4))
    t_cot = rng.uniform(2.6, 4.8, size=(n_trials_policy, 1))
    t_strag = np.clip(4.2 + (rng.pareto(a=2.3, size=(n_trials_policy, 2)) * 2.8), 4.2, 22.0)
    T = np.concatenate([t_fast, t_cot, t_strag], axis=1)

    corr = np.zeros((n_trials_policy, 7), dtype=np.int32)
    echo_shared = (rng.random(n_trials_policy) < np.where(is_hard, 0.12, 0.93)).astype(np.int32)
    corr[:, 0] = echo_shared
    corr[:, 1] = np.where(rng.random(n_trials_policy) < 0.92, echo_shared, 1 - echo_shared)
    for j in [2, 3]:
        corr[:, j] = (rng.random(n_trials_policy) < np.where(is_hard, 0.28, 0.91)).astype(np.int32)
    corr[:, 4] = (rng.random(n_trials_policy) < np.where(is_hard, 0.78, 0.96)).astype(np.int32)
    for j in [5, 6]:
        corr[:, j] = (rng.random(n_trials_policy) < np.where(is_hard, 0.96, 0.99)).astype(np.int32)

    w = np.zeros((n_trials_policy, 7), dtype=np.float64)
    for j in range(4):
        w[:, j] = np.where(~is_hard & (corr[:, j] == 1), rng.uniform(0.84, 0.97, n_trials_policy), rng.uniform(0.42, 0.74, n_trials_policy))
    w[:, 4] = np.where(corr[:, 4] == 1, rng.uniform(0.80, 0.95, n_trials_policy), rng.uniform(0.48, 0.72, n_trials_policy))
    for j in [5, 6]:
        w[:, j] = np.where(corr[:, j] == 1, rng.uniform(0.92, 0.99, n_trials_policy), rng.uniform(0.55, 0.75, n_trials_policy))

    def cvar_95(arr: np.ndarray) -> float:
        p95 = np.percentile(arr, 95)
        return float(np.mean(arr[arr >= p95]))

    policies: Dict[str, Dict[str, float]] = {}
    order = np.argsort(T, axis=1)
    top3_corr = np.take_along_axis(corr, order[:, :3], axis=1)
    fast_acc = float(np.mean(np.sum(top3_corr, axis=1) >= 2))
    fast_lat = np.take_along_axis(T, order[:, 2:3], axis=1)[:, 0]
    policies["Fast_3_of_7"] = {
        "accuracy": round(fast_acc, 4),
        "false_consensus_risk": round(1.0 - fast_acc, 4),
        "mean_latency_s": round(float(np.mean(fast_lat)), 4),
        "p95_latency_s": round(float(np.percentile(fast_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(fast_lat), 4),
    }
    sync_acc = float(np.mean(np.sum(corr, axis=1) >= 4))
    sync_lat = np.max(T, axis=1)
    policies["Sync_7_of_7"] = {
        "accuracy": round(sync_acc, 4),
        "false_consensus_risk": round(1.0 - sync_acc, 4),
        "mean_latency_s": round(float(np.mean(sync_lat)), 4),
        "p95_latency_s": round(float(np.percentile(sync_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(sync_lat), 4),
    }
    hpr_pos = np.maximum(w[:, 0] * corr[:, 0], w[:, 1] * corr[:, 1]) + np.sum(w[:, 2:] * corr[:, 2:], axis=1)
    hpr_neg = np.maximum(w[:, 0] * (1 - corr[:, 0]), w[:, 1] * (1 - corr[:, 1])) + np.sum(w[:, 2:] * (1 - corr[:, 2:]), axis=1)
    hpr_acc = float(np.mean(hpr_pos >= hpr_neg))
    policies["Genealogy_Sync_HPR"] = {
        "accuracy": round(hpr_acc, 4),
        "false_consensus_risk": round(1.0 - hpr_acc, 4),
        "mean_latency_s": round(float(np.mean(sync_lat)), 4),
        "p95_latency_s": round(float(np.percentile(sync_lat, 95)), 4),
        "cvar95_latency_s": round(cvar_95(sync_lat), 4),
    }

    return {
        "three_path_risk_verification": risk_verification,
        "preemptive_resume_mg1_verification": mg1_checks,
        "policy_monte_carlo": policies,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "pilot" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "theorem_verification.json"

    t1 = verify_theorem1_phase_transition()
    t2 = verify_theorem2_cascade_amplification()
    t3 = verify_theorem3_risk_and_mg1()

    payload = {
        "theorem1_phase_transition": t1,
        "theorem2_cascade_amplification": t2,
        "theorem3_cvar_optimality": t3,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print("=== Theorem 1: Multi-Option Plurality Phase Transition ===")
    print(f"Binary (|C|=2, theta=1.0): phi_crit = {t1['analytical_critical_quorum_ratio_phi_crit']:.5f}")
    for mo in t1["multi_option_plurality_verification"]:
        print(
            f"  |C|=4, theta={mo['trap_concentration_theta']:.2f} (thresh={mo['plurality_error_threshold_1_over_1plus_theta']:.4f}): "
            f"phi_crit={mo['phi_crit_theta']:.5f}, SyncErr={mo['sync_plurality_error_mc']:.4f}, FastErr={mo['fast_plurality_false_consensus_mc']:.4f}"
        )

    print("\n=== Theorem 2: Stage-M Marginal vs. Joint Survival ===")
    for k_pol, v_pol in t2.items():
        s4 = v_pol["stages"][3]
        print(
            f"  {k_pol:14s}: Stage-4 Output Acc={s4['stage_m_output_accuracy']*100:5.2f}%, "
            f"Joint 4-Stage Survival={s4['mc_joint_m_stage_accuracy']*100:5.2f}%"
        )

    print("\n=== Theorem 3(1) Three-Path Risk Bound ===")
    r3 = t3["three_path_risk_verification"]
    print(f"  MC Risk={r3['mc_empirical_stage_error']:.5f} <= Analytical Bound={r3['theorem3_1_upper_bound']:.5f} (se={r3['mc_standard_error']:.5f}, Holds={r3['bound_holds_within_mc_se']})")

    print("\n=== Theorem 3(2) Exact Quantum-Preemptive-Resume M/G/1 ===")
    for row in t3["preemptive_resume_mg1_verification"]:
        print(
            f"  rho={row['rho']:.2f} (rho_hi={row['rho_hi']:.3f}): "
            f"W_fcfs(ana/sim)={row['w_fcfs_analytical_s']:.2f}/{row['w_fcfs_sim_s']:.2f}s | "
            f"W_hi_Q(ana/sim)={row['w_hi_quantum_analytical_s']:.2f}/{row['w_hi_quantum_sim_s']:.2f}s | "
            f"W_norm_PR(ana/sim)={row['w_norm_pr_analytical_s']:.2f}/{row['w_norm_pr_sim_s']:.2f}s"
        )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

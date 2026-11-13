#!/usr/bin/env python3
"""
Generate 5 Publication-Grade Vector PDF Figures for QuorumGuard (FGCS / TPDS).
==============================================================================
Reads `pilot/results/theorem_verification.json`, `pilot/results/quorumguard_pilot.json`,
and `pilot/results/ablation_results.json` to emit crisp IEEE/Elsevier vector PDFs:
  1. `fig1_phase_transition.pdf` — Theorem 1 Static k-of-N Order-Statistic Phase Transition
  2. `fig2_cascade_amplification.pdf` — Theorem 2 Multi-Stage DAG Error Cascade Growth
  3. `fig3_pareto_latency_accuracy.pdf` — Live MPS 2-Stage Accuracy vs. P95/CVaR95 Pareto Frontier
  4. `fig4_entropy_covariance_calibration.pdf` — Live Qwen2.5 Tier Latency-Accuracy Covariance & Entropy
  5. `fig5_ablation_breakdown.pdf` — QuorumGuard Orthogonal Component Ablation Study
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "legend.fontsize": 8.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    res_dir = root / "pilot" / "results"
    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    thm = json.loads((res_dir / "theorem_verification.json").read_text())
    pilot = json.loads((res_dir / "quorumguard_pilot.json").read_text())
    abl = json.loads((res_dir / "ablation_results.json").read_text())

    # --- Figure 1: Theorem 1 Order-Statistic Phase Transition ---
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    t1 = thm["theorem1_phase_transition"]
    phi_crit = t1["analytical_critical_quorum_ratio_phi_crit"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    markers = ["o", "s", "^", "D"]
    for idx, entry in enumerate(t1["n_sweep"]):
        n_val = entry["N"]
        phis = [pt["phi_eff"] for pt in entry["quorum_curve"]]
        errs = [pt["mc_majority_false_consensus_rate"] * 100.0 for pt in entry["quorum_curve"]]
        ax.plot(
            phis,
            errs,
            marker=markers[idx],
            linewidth=1.9,
            markersize=5,
            color=colors[idx],
            label=f"$N = {n_val}$ (Sync Err = {entry['sync_full_majority_error']*100:.1f}%)",
        )
    ax.axvline(phi_crit, color="#222222", linestyle=":", linewidth=1.6, label=f"$\\phi_{{\\mathrm{{crit}}}} = {phi_crit:.3f}$")
    ax.axhline(50.0, color="#888888", linestyle="-.", linewidth=1.0)
    ax.set_xlabel("Fast Quorum Ratio $\\phi = k / N$")
    ax.set_ylabel("False Consensus Rate (%)")
    ax.set_title("Theorem 1: Phase Transition of Static $k$-of-$N$ Fast Quorums")
    ax.set_ylim(-2, 102)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_phase_transition.pdf")
    plt.close(fig)

    # --- Figure 2: Theorem 2 Multi-Stage DAG Cascade Amplification ---
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    t2 = thm["theorem2_cascade_amplification"]
    st_fast = t2["fast_k_of_n"]["stages"]
    st_qg = t2["quorumguard"]["stages"]
    m_vals = [s["stage"] for s in st_fast]
    ax.plot(
        m_vals,
        [s["end_to_end_accuracy"] * 100.0 for s in st_qg],
        marker="o",
        linewidth=2.2,
        color="#1B7837",
        label="QuorumGuard (Empirical + Analytical)",
    )
    ax.plot(
        m_vals,
        [(1.0 - s["analytical_cascade_error"]) * 100.0 for s in st_qg],
        linestyle="--",
        linewidth=1.2,
        color="#1B7837",
    )
    ax.plot(
        m_vals,
        [s["end_to_end_accuracy"] * 100.0 for s in st_fast],
        marker="s",
        linewidth=2.2,
        color="#C44E52",
        label="Static Fast $k$-of-$N$ Quorum",
    )
    ax.set_xlabel("Workflow DAG Depth $M$ (Stages)")
    ax.set_ylabel("End-to-End Workflow Accuracy (%)")
    ax.set_title("Theorem 2: Multi-Stage Error Cascade & Herding Stability")
    ax.set_xticks(m_vals)
    ax.set_ylim(0, 102)
    ax.legend(loc="center right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_cascade_amplification.pdf")
    plt.close(fig)

    # --- Figure 3: Live MPS 2-Stage Accuracy vs. P95 Latency Pareto Frontier ---
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    psum = pilot["policy_summary"]
    pol_meta = [
        ("Fast_k_of_N", "Fast $k$-of-$N$", "#C44E52", "s"),
        ("CVaR_Static_Cutoff", "CVaR Static Cutoff", "#DD8452", "^"),
        ("Sync_N_of_N", "Sync $N$-of-$N$ Barrier", "#4C72B0", "D"),
        ("Genealogy_Sync_HPR", "Genealogy Sync (HPR)", "#8172B2", "P"),
        ("QuorumGuard", "QuorumGuard (Ours)", "#1B7837", "*"),
    ]
    for key, label, col, mk in pol_meta:
        st = psum[key]
        x = st["p95_workflow_latency_s"]
        y = st["end_to_end_joint_accuracy"] * 100.0
        sz = 190 if key == "QuorumGuard" else 95
        ax.scatter([x], [y], color=col, marker=mk, s=sz, zorder=4, label=f"{label} ({x:.1f}s, {y:.1f}%)")
    ax.set_xlabel("End-to-End 2-Stage P95 Tail Latency (s) [Lower is Better]")
    ax.set_ylabel("Joint 2-Stage Workflow Accuracy (%)")
    ax.set_title("Live Qwen2.5 MPS Benchmark: Tail Latency vs. Consensus Accuracy")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_pareto_latency_accuracy.pdf")
    plt.close(fig)

    # --- Figure 4: Empirical Tier Latency-Accuracy Covariance & Pre-Commit Entropy ---
    fig, ax1 = plt.subplots(figsize=(5.8, 3.6))
    tiers = ["fast_3b", "short_cot_3b", "cot_7b", "verifier_7b"]
    t_labels = ["Fast 3B\n(Direct)", "Short-CoT 3B\n(Quick Check)", "Medium 7B\n(CoT)", "Verifier 7B\n(Py Sandbox)"]
    tt = pilot["tier_telemetry"]
    x_pos = np.arange(len(tiers))
    width = 0.36
    trap_accs = [tt[k]["boundary_trap_accuracy"] * 100.0 for k in tiers]
    ents = [tt[k]["mean_normalized_entropy"] * 100.0 for k in tiers]

    ax1.bar(x_pos - width / 2, trap_accs, width, color="#2B6CB0", label="Boundary-Trap Accuracy (%)")
    ax1.set_ylabel("Boundary-Trap Accuracy (%)", color="#2B6CB0")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(t_labels)

    ax2 = ax1.twinx()
    ax2.bar(x_pos + width / 2, ents, width, color="#E53E3E", alpha=0.82, label="Normalized Option Entropy (%)")
    ax2.set_ylabel("Pre-Commitment Normalized Entropy (%)", color="#E53E3E")
    ax2.set_ylim(0, max(45.0, max(ents) * 1.25))
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", frameon=True)
    ax1.set_title("Empirical Latency–Accuracy Covariance & Epistemic Entropy Across Tiers")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_entropy_covariance_calibration.pdf")
    plt.close(fig)

    # --- Figure 5: Orthogonal Component Ablation Breakdown ---
    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    abl_keys = [
        "1_Fast_k_of_N",
        "2_Entropy_Gate_Only",
        "3_Genealogy_Only",
        "4_QuorumGuard_No_Boost",
        "5_QuorumGuard_Full",
    ]
    abl_labels = [
        "Fast\n$k$-of-$N$",
        "+ Entropy\nGate Only",
        "+ Genealogy\n$N_{\\mathrm{eff}}$ Only",
        "QuorumGuard\n(w/o Boost)",
        "QuorumGuard\n(Full)",
    ]
    xp = np.arange(len(abl_keys))
    s1_accs = [abl[k]["stage1_accuracy"] * 100.0 for k in abl_keys]
    p95_lats = [abl[k]["p95_stage1_latency_s"] for k in abl_keys]

    ax1.bar(xp - 0.18, s1_accs, 0.35, color="#1B7837", label="Stage-1 Accuracy (%)")
    ax1.set_ylabel("Stage-1 Consensus Accuracy (%)", color="#1B7837")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(xp)
    ax1.set_xticklabels(abl_labels)

    ax2 = ax1.twinx()
    ax2.plot(xp + 0.18, p95_lats, color="#C44E52", marker="o", linewidth=2.1, label="Stage-1 P95 Latency (s)")
    ax2.set_ylabel("Stage-1 P95 Latency (s)", color="#C44E52")
    ax2.set_ylim(0, max(p95_lats) * 1.2)
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)
    ax1.set_title("QuorumGuard Component Ablation: Accuracy vs. P95 Tail Latency")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig5_ablation_breakdown.pdf")
    plt.close(fig)

    print(f"Generated all 5 vector PDF figures in {fig_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate All 6 Publication-Grade Vector PDF Figures for QuorumGuard (FGCS / TPDS).
==================================================================================
Reads `pilot/results/*.json` and `benchmarks/results/*.json` to emit crisp vector PDFs:
  0. `fig0_architecture.pdf` — System Architecture & Straggler Selection-Bias Overview
  1. `fig1_phase_transition.pdf` — Theorem 1 Static k-of-N Order-Statistic Phase Transition
  2. `fig2_cascade_amplification.pdf` — Theorem 2 Multi-Stage DAG Error Cascade Growth
  3. `fig3_pareto_latency_accuracy.pdf` — Live MPS 2-Stage Accuracy vs. P95 Tail Latency Pareto
  4. `fig4_entropy_covariance_calibration.pdf` — Live Tier Latency-Accuracy Covariance & Entropy
  5. `fig5_ablation_breakdown.pdf` — QuorumGuard Orthogonal Component Ablation Study
  6. `fig6_priority_contention.pdf` — Physical MPS GPU Queue Contention & Starvation Protection
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "legend.fontsize": 8.6,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
    }
)


def draw_architecture_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 3.65))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.08, 3.62)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str) -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08", facecolor=fc, edgecolor=ec, linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1A202C")
        ax.text(x + w / 2, y + (h - 0.28) / 2, body, ha="center", va="center", fontsize=7.9, color="#2D3748")

    # Left column: Heterogeneous Asynchronous Agent Pool (x = 0.12 .. 2.74)
    box(0.12, 2.42, 2.62, 0.96, "Fast Shallow Tier (0.4s - 0.9s)", "Agents 0, 1 (Shared Triage Root r0)\nAgent 2 (Direct Scan Root r1)", "#FFF5F5", "#C53030")
    box(0.12, 1.30, 2.62, 0.94, "Medium CoT Verifiers (2.1s - 6.8s)", "Agent 3 (Short Check r2)\nAgent 4 (CoT Reviewer r3)", "#FEFCBF", "#B7791F")
    box(0.12, 0.18, 2.62, 0.94, "Deep Tool Stragglers (10s - 18s)", "Agents 5, 6 (Deep Model + Py Sandbox)\nIndependent Verified Roots r4, r5", "#E6FFFA", "#2C7A7B")

    # Center: QuorumGuard Online Controller (x = 3.92 .. 7.98) — ample open channel (2.74 .. 3.92) for feedback arrow!
    rect_qg = mpatches.FancyBboxPatch(
        (3.92, 0.18), 4.06, 3.20, boxstyle="round,pad=0.07,rounding_size=0.10", facecolor="#F0FFF4", edgecolor="#276749", linewidth=1.8
    )
    ax.add_patch(rect_qg)
    ax.text(5.95, 3.16, "QuorumGuard Adaptive Gate & Scheduler", ha="center", va="center", fontsize=10.0, fontweight="bold", color="#1C4532")

    box(4.12, 2.16, 3.66, 0.78, "1. Pre-Commitment Semantic Entropy Gate", "Probes 4-way posterior H_ans & margin Delta BEFORE\nletter emission; discounts high-entropy guesses.", "#FFFFFF", "#2F855A")
    box(4.12, 1.22, 3.66, 0.78, "2. Claim-Genealogy Independence (N_eff)", "Resolves provenance DAG G: shared echoes (r0)\ncollapse to <= 1 root; requires k* indep. roots.", "#FFFFFF", "#2F855A")
    box(4.12, 0.28, 3.66, 0.78, "3. Token-Bucket Priority Dissent Boosting", "On dissent/correlation, boosts active deep verifier\nto Priority = -10 at 8-token quantum boundaries.", "#FFFFFF", "#2C5282")

    # Right: Dual Regime Outcomes -> Stage 2 Downstream Committee (x = 8.58 .. 11.45)
    box(8.58, 2.06, 2.84, 1.28, "Clean Routine Path (< 2.0s)", "3 indep. roots agree, H_ans <= 0.08\n-> Immediate Fast Closure\nCancels unboosted tail stragglers", "#EBF8FF", "#2B6CB0")
    box(8.58, 0.22, 2.84, 1.52, "Dissent-Hedged Path (~4s - 6s)", "Blocks false fast-quorum herding\nBoosted Verifier + Sandbox confirms\n-> Prevents Stage-2 Error Cascade", "#FAF5FF", "#553C9A")

    arrow_kw = dict(arrowstyle="->,head_width=0.22,head_length=0.22", lw=1.4, color="#4A5568")
    ax.annotate("", xy=(3.88, 2.85), xytext=(2.78, 2.88), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.88, 1.76), xytext=(2.78, 1.76), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.88, 0.80), xytext=(2.78, 0.80), arrowprops=arrow_kw)
    # Non-overlapping feedback boost arrow in open inter-column channel (x = 2.78 .. 3.92)
    ax.annotate(
        "",
        xy=(2.78, 0.36),
        xytext=(3.88, 0.36),
        arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.22", lw=1.6, color="#C53030", ls="--"),
    )
    ax.text(
        3.33,
        0.10,
        "Priority = -10\nPreempt",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#C53030",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="#FFF5F5", edgecolor="#FEB2B2", lw=0.8),
    )
    ax.annotate("", xy=(8.54, 2.66), xytext=(8.02, 2.52), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.54, 1.00), xytext=(8.02, 1.15), arrowprops=arrow_kw)

    fig.tight_layout(pad=0.15)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    res_dir = root / "pilot" / "results"
    bench_res_dir = root / "benchmarks" / "results"
    fig_dir = root / "paper" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    draw_architecture_overview(fig_dir / "fig0_architecture.pdf")

    thm = json.loads((res_dir / "theorem_verification.json").read_text())
    scale_path = bench_res_dir / "quorumbench_50_3b_7b_14b.json"
    pilot = json.loads(scale_path.read_text()) if scale_path.exists() else json.loads((res_dir / "quorumguard_pilot.json").read_text())
    abl = json.loads((res_dir / "ablation_results.json").read_text())

    # --- Figure 1: Theorem 1 Order-Statistic Phase Transition ---
    fig, ax = plt.subplots(figsize=(5.6, 3.55))
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
    fig.savefig(fig_dir / "fig1_phase_transition.png", dpi=220)
    plt.close(fig)

    # --- Figure 2: Theorem 2 Multi-Stage DAG Cascade Amplification ---
    fig, ax = plt.subplots(figsize=(5.5, 3.55))
    t2 = thm["theorem2_cascade_amplification"]
    st_fast = t2["fast_k_of_n"]["stages"]
    st_qg = t2["quorumguard"]["stages"]
    m_vals = [s["stage"] for s in st_fast]
    ax.plot(
        m_vals,
        [s["stage_m_output_accuracy"] * 100.0 for s in st_qg],
        marker="o",
        linewidth=2.2,
        color="#1B7837",
        label="QuorumGuard Marginal Stage-$M$ ($1 - q_M$)",
    )
    ax.plot(
        m_vals,
        [s["mc_joint_m_stage_accuracy"] * 100.0 for s in st_qg],
        linestyle=":",
        marker=".",
        linewidth=1.6,
        color="#1B7837",
        label="QuorumGuard All-$M$ Joint Survival",
    )
    ax.plot(
        m_vals,
        [s["stage_m_output_accuracy"] * 100.0 for s in st_fast],
        marker="s",
        linewidth=2.2,
        color="#C44E52",
        label="Fast $k$-of-$N$ Marginal Stage-$M$ ($1 - q_M$)",
    )
    ax.plot(
        m_vals,
        [s["mc_joint_m_stage_accuracy"] * 100.0 for s in st_fast],
        linestyle=":",
        marker="x",
        linewidth=1.6,
        color="#C44E52",
        label="Fast $k$-of-$N$ All-$M$ Joint Survival",
    )
    ax.set_xlabel("Workflow DAG Depth $M$ (Stages)")
    ax.set_ylabel("Stage-$M$ Output & Joint Accuracy (%)")
    ax.set_title("Theorem 2: Marginal $1-q_M$ vs. Joint Survival")
    ax.set_xticks(m_vals)
    ax.set_ylim(0, 102)
    ax.legend(loc="center right", fontsize=7.6, frameon=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_cascade_amplification.pdf")
    fig.savefig(fig_dir / "fig2_cascade_amplification.png", dpi=220)
    plt.close(fig)

    # --- Figure 3: Live MPS 2-Stage Accuracy vs. P95 Latency Pareto Frontier ---
    fig, ax = plt.subplots(figsize=(5.9, 3.7))
    rev_path = bench_res_dir / "revision_ablations_and_sweeps.json"
    if rev_path.exists():
        rev = json.loads(rev_path.read_text())
        ext_sum = rev["extended_policies_with_ci95"]
        cvar_keys = [
            ("CVaR_B_1.5s", 1.5),
            ("CVaR_B_2.8s", 2.8),
            ("CVaR_B_6s", 6.0),
            ("CVaR_B_10s", 10.0),
            ("CVaR_B_15s", 15.0),
            ("CVaR_B_22s", 22.0),
            ("CVaR_B_30s", 30.0),
        ]
        cx = [ext_sum[k]["p95_latency_s"] for k, _ in cvar_keys if k in ext_sum]
        cy = [ext_sum[k]["end_to_end_joint_accuracy"] * 100.0 for k, _ in cvar_keys if k in ext_sum]
        ax.plot(cx, cy, linestyle="--", marker="^", markersize=6, linewidth=1.7, color="#DD8452", label="CVaR Cutoff Sweep ($B=1.5$\u2013$30$s)")
        pol_meta_rev = [
            ("Fast_k_of_N", "Fast $k$-of-$N$", "#C44E52", "s", 85),
            ("Tier_Weighted_Fast", "TW-Fast (Static Tier)", "#A0522D", "v", 85),
            ("Entropy_Cascade", "Entropy-Cascade", "#17BECF", "D", 85),
            ("Sync_N_of_N", "Sync $N$-of-$N$ Barrier", "#4C72B0", "o", 85),
            ("Genealogy_Sync_HPR", "Genealogy Sync (HPR)", "#8172B2", "P", 105),
            ("QuorumGuard", "QuorumGuard (Ours)", "#1B7837", "*", 210),
        ]
        for key, label, col, mk, sz in pol_meta_rev:
            st = ext_sum[key]
            x = st["p95_latency_s"]
            y = st["end_to_end_joint_accuracy"] * 100.0
            ax.scatter([x], [y], color=col, marker=mk, s=sz, zorder=5, label=f"{label} ({x:.1f}s, {y:.1f}%)")
    else:
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
            sz = 195 if key == "QuorumGuard" else 95
            ax.scatter([x], [y], color=col, marker=mk, s=sz, zorder=4, label=f"{label} ({x:.1f}s, {y:.1f}%)")
    ax.set_xlim(-1.0, 51.0)
    ax.set_ylim(22.0, 78.0)
    ax.set_xlabel("End-to-End 2-Stage P95 Tail Latency (s) [Lower is Better]")
    ax.set_ylabel("Joint 2-Stage Workflow Accuracy (%)")
    ax.set_title("QuorumBench-50: Tail Latency vs. Consensus Accuracy")
    ax.legend(loc="center left", bbox_to_anchor=(0.015, 0.56), fontsize=7.3, frameon=True, framealpha=0.94)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_pareto_latency_accuracy.pdf")
    fig.savefig(fig_dir / "fig3_pareto_latency_accuracy.png", dpi=220)
    plt.close(fig)

    # --- Figure 4: Empirical Tier Latency-Accuracy Covariance & Pre-Commit Entropy ---
    fig, ax1 = plt.subplots(figsize=(5.8, 3.6))
    tiers = ["fast_3b", "short_cot_3b", "cot_7b", "verifier_7b"]
    is_14b = scale_path.exists()
    t_labels = [
        "Fast 3B\n(Direct)",
        "Short-CoT 3B\n(Quick Check)",
        "Medium 7B\n(CoT)",
        "14B-Coder\n(+ Py Sandbox)" if is_14b else "Verifier 7B\n(+ Py Sandbox)",
    ]
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
    ax2.set_ylim(0, max(40.0, max(ents) * 1.25))
    ax2.grid(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", frameon=True)
    ax1.set_title("Empirical Latency–Accuracy Covariance & Epistemic Entropy Across Tiers")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_entropy_covariance_calibration.pdf")
    fig.savefig(fig_dir / "fig4_entropy_covariance_calibration.png", dpi=220)
    plt.close(fig)

    # --- Figure 5: Orthogonal Component Ablation Breakdown ---
    fig, ax1 = plt.subplots(figsize=(6.2, 3.6))
    abl_keys = [
        "1_Fast_k_of_N",
        "2_Entropy_Gate_Only",
        "3_Genealogy_Only",
        "4_QuorumGuard_No_Boost",
        "5_QuorumGuard_Full",
    ]
    abl_labels = [
        "Fast\n$k$-of-$N$",
        "+ Entropy\nOnly",
        "+ Genealogy\n$N_{\\mathrm{eff}}$ Only",
        "QG w/o\nBoost",
        "QG\n(Full)",
    ]
    xp = np.arange(len(abl_keys))
    s1_accs = [abl[k]["stage1_accuracy"] * 100.0 for k in abl_keys]
    p95_lats = [abl[k]["p95_stage1_latency_s"] for k in abl_keys]

    ax1.bar(xp - 0.17, s1_accs, 0.34, color="#1B7837", label="Stage-1 Accuracy (%)")
    ax1.set_ylabel("Stage-1 Consensus Accuracy (%)", color="#1B7837")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(xp)
    ax1.set_xticklabels(abl_labels, fontsize=8.6)

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
    fig.savefig(fig_dir / "fig5_ablation_breakdown.png", dpi=220)
    plt.close(fig)

    # --- Figure 6: Physical MPS GPU Queue Contention Across Load rho ---
    cont_path = bench_res_dir / "priority_contention_mps.json"
    if cont_path.exists():
        cont = json.loads(cont_path.read_text())
        loads = ["rho_0.58", "rho_0.72", "rho_0.85", "rho_0.93"]
        rho_x = [0.58, 0.72, 0.85, 0.93]
        fcfs_w = [cont["loads"][k]["FCFS_Default"]["dissent_straggler_mean_queue_wait_s"] for k in loads]
        unb_cvar = [cont["loads"][k]["Unbounded_Urgent"]["dissent_straggler_cvar95_turnaround_s"] for k in loads]
        fcfs_cvar = [cont["loads"][k]["FCFS_Default"]["dissent_straggler_cvar95_turnaround_s"] for k in loads]
        qg_cvar = [cont["loads"][k]["QuorumGuard_Budgeted"]["dissent_straggler_cvar95_turnaround_s"] for k in loads]
        qg_w = [cont["loads"][k]["QuorumGuard_Budgeted"]["dissent_straggler_mean_queue_wait_s"] for k in loads]

        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.2, 3.3))
        ax_l.plot(rho_x, fcfs_w, marker="s", linewidth=2.0, color="#4C72B0", label="FCFS Default")
        ax_l.plot(rho_x, qg_w, marker="o", linewidth=2.2, color="#1B7837", label="QuorumGuard Budgeted")
        ax_l.set_xlabel("GPU Utilization $\\rho$")
        ax_l.set_ylabel("Mean Straggler Queue Wait (s)")
        ax_l.set_title("(a) Queue-Wait Compression (4.3\u00d7\u20136.0\u00d7)")
        ax_l.legend(loc="upper left", frameon=True)

        ax_r.plot(rho_x, unb_cvar, marker="^", linewidth=1.9, linestyle="--", color="#C44E52", label="Unbounded Urgent")
        ax_r.plot(rho_x, fcfs_cvar, marker="s", linewidth=1.9, color="#4C72B0", label="FCFS Default")
        ax_r.plot(rho_x, qg_cvar, marker="*", markersize=8, linewidth=2.2, color="#1B7837", label="QuorumGuard")
        ax_r.set_xlabel("GPU Utilization $\\rho$")
        ax_r.set_ylabel("Straggler $\\mathrm{CVaR}_{0.95}$ Turnaround (s)")
        ax_r.set_title("(b) Priority Inflation vs. Token Budget")
        ax_r.legend(loc="upper left", frameon=True)

        fig.tight_layout()
        fig.savefig(fig_dir / "fig6_priority_contention.pdf")
        fig.savefig(fig_dir / "fig6_priority_contention.png", dpi=220)
        plt.close(fig)

    print(f"Generated all vector PDF figures in {fig_dir}")


if __name__ == "__main__":
    main()

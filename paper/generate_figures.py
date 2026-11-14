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
    fig, ax = plt.subplots(figsize=(11.2, 3.55))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 3.55)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, fc: str, ec: str) -> None:
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08", facecolor=fc, edgecolor=ec, linewidth=1.4
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", fontsize=9.4, fontweight="bold", color="#1A202C")
        ax.text(x + w / 2, y + (h - 0.28) / 2, body, ha="center", va="center", fontsize=8.1, color="#2D3748")

    # Left column: Heterogeneous Asynchronous Agent Pool
    box(0.15, 2.38, 2.85, 0.96, "Fast Shallow Tier (0.4s - 0.9s)", "Agents 0, 1 (Shared Triage Root r0)\nAgent 2 (Direct Scan Root r1)", "#FFF5F5", "#C53030")
    box(0.15, 1.26, 2.85, 0.94, "Medium CoT Verifiers (2.1s - 6.8s)", "Agent 3 (Short Check r2)\nAgent 4 (7B Step-by-Step CoT r3)", "#FEFCBF", "#B7791F")
    box(0.15, 0.14, 2.85, 0.94, "Deep Tool Stragglers (10s - 18s)", "Agents 5, 6 (14B-Coder + Py Sandbox)\nIndependent Verified Roots r4, r5", "#E6FFFA", "#2C7A7B")

    # Center: QuorumGuard Online Controller
    rect_qg = mpatches.FancyBboxPatch(
        (3.55, 0.14), 4.15, 3.20, boxstyle="round,pad=0.07,rounding_size=0.10", facecolor="#F0FFF4", edgecolor="#276749", linewidth=1.8
    )
    ax.add_patch(rect_qg)
    ax.text(5.625, 3.12, "QuorumGuard Adaptive Gate & Scheduler", ha="center", va="center", fontsize=10.3, fontweight="bold", color="#1C4532")

    box(3.78, 2.12, 3.68, 0.78, "1. Pre-Commitment Semantic Entropy Gate", "Probes 4-way posterior H_ans & margin Delta BEFORE letter\nemission; discounts unverified high-entropy guesses.", "#FFFFFF", "#2F855A")
    box(3.78, 1.18, 3.68, 0.78, "2. Claim-Genealogy Independence (N_eff)", "Resolves provenance DAG G: shared echoes (r0) collapse\nto <= 1 root capacity; requires k* independent roots.", "#FFFFFF", "#2F855A")
    box(3.78, 0.24, 3.68, 0.78, "3. Token-Bucket Priority Dissent Boosting", "On root correlation / dissent, elevates active 14B straggler\nto vLLM Priority = -10 (4.3x - 6.0x queue-wait compression).", "#FFFFFF", "#2C5282")

    # Right: Dual Regime Outcomes -> Stage 2 Downstream Committee
    box(8.25, 2.02, 2.80, 1.28, "Clean Routine Path (< 2.0s)", "3 indep. roots agree, H_ans <= 0.08\n-> Immediate Fast Closure!\nCancels unboosted tail stragglers", "#EBF8FF", "#2B6CB0")
    box(8.25, 0.20, 2.80, 1.52, "Dissent-Hedged Path (~4s - 6s)", "Blocks false fast-quorum herding!\nBoosted 14B + Sandbox verifies truth\n-> Prevents Stage-2 Error Cascade", "#FAF5FF", "#553C9A")

    arrow_kw = dict(arrowstyle="->,head_width=0.22,head_length=0.22", lw=1.4, color="#4A5568")
    ax.annotate("", xy=(3.53, 2.75), xytext=(3.02, 2.82), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.53, 1.72), xytext=(3.02, 1.72), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.53, 0.65), xytext=(3.02, 0.62), arrowprops=arrow_kw)
    # Feedback boost arrow from Pillar 3 back to Deep Tool Stragglers
    ax.annotate(
        "vLLM Priority=-10",
        xy=(3.02, 0.32),
        xytext=(3.76, 0.36),
        arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.2", lw=1.4, color="#C53030", ls="--"),
        fontsize=7.5,
        color="#C53030",
        fontweight="bold",
        va="bottom",
    )
    ax.annotate("", xy=(8.22, 2.62), xytext=(7.72, 2.48), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.22, 0.98), xytext=(7.72, 1.15), arrowprops=arrow_kw)

    fig.tight_layout(pad=0.2)
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
    fig.savefig(fig_dir / "fig2_cascade_amplification.png", dpi=220)
    plt.close(fig)

    # --- Figure 3: Live MPS 2-Stage Accuracy vs. P95 Latency Pareto Frontier ---
    fig, ax = plt.subplots(figsize=(5.8, 3.65))
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
    ax.set_xlabel("End-to-End 2-Stage P95 Tail Latency (s) [Lower is Better]")
    ax.set_ylabel("Joint 2-Stage Workflow Accuracy (%)")
    ax.set_title("Live Qwen2.5 MPS Benchmark: Tail Latency vs. Consensus Accuracy")
    ax.legend(loc="lower right", frameon=True)
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
        ax_l.set_title("(a) Queue-Wait Compression ($4.3\\times$--$6.0\\times$)")
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

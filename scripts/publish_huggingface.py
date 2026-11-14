#!/usr/bin/env python3
"""
Build and Publish QuorumGuard Hugging Face Suite (`batistado` Namespace)
========================================================================
Publishes three interlinked Hugging Face Hub repositories:

  1. Dataset : `https://huggingface.co/datasets/batistado/QuorumBench`
       4 Dataset-Viewer configs (`tasks`, `agent_observations`,
       `policy_decisions`, `gpu_contention`) + raw JSON telemetry + figures.
  2. Space   : `https://huggingface.co/spaces/batistado/QuorumGuard-Explorer`
       Static HTML5 Interactive Explorer (`sdk: static`) with a Theorem 1
       phase-transition calculator, live provenance-DAG N_eff simulator, and
       50-task QuorumBench trace browser.
  3. Hub Repo: `https://huggingface.co/batistado/QuorumGuard`
       Zero-training QuorumGuard consensus controller & vLLM priority-preemption
       scheduler linked to Qwen2.5-3B/7B/Coder-14B and QuorumBench.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.quorumbench_scaleup import build_quorumbench_50  # noqa: E402


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_dataset_staging(stage_dir: Path, username: str) -> Dict[str, Any]:
    tasks_spec = {t.task_id: t for t in build_quorumbench_50()}

    qb50 = json.loads((REPO_ROOT / "benchmarks" / "results" / "quorumbench_50_3b_7b_14b.json").read_text())
    pilot30 = json.loads((REPO_ROOT / "pilot" / "results" / "quorumguard_pilot.json").read_text())
    cont = json.loads((REPO_ROOT / "benchmarks" / "results" / "priority_contention_mps.json").read_text())

    # 1. Tasks config (all 50 two-stage workflows)
    task_rows: List[Dict[str, Any]] = []
    for idx, t in enumerate(build_quorumbench_50()):
        task_rows.append(
            {
                "task_index": idx,
                "task_id": t.task_id,
                "regime": t.regime,
                "category": t.category,
                "stage1_question": t.stage1_question,
                "stage1_options_json": json.dumps(t.stage1_options, ensure_ascii=False),
                "stage1_gold": t.stage1_gold,
                "upstream_triage_note": t.upstream_triage_note,
                "stage2_question": t.stage2_question,
                "stage2_options_json": json.dumps(t.stage2_options, ensure_ascii=False),
                "stage2_gold": t.stage2_gold,
            }
        )
    _write_jsonl(stage_dir / "data" / "tasks" / "train.jsonl", task_rows)

    # 2. Per-agent Stage-1 observations across both suites (350 + 210 = 560 rows)
    obs_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    for suite_name, payload in [
        ("quorumbench_50_3b_7b_14b", qb50),
        ("pilot_30_3b_7b", pilot30),
    ]:
        for tr in payload["tasks"]:
            tid = tr["task_id"]
            for ag in tr["stage1_agents"]:
                obs_rows.append(
                    {
                        "suite": suite_name,
                        "task_id": tid,
                        "regime": tr["regime"],
                        "category": tr["category"],
                        "agent_id": int(ag["agent_id"]),
                        "tier": str(ag["tier"]),
                        "genealogy_root": str(ag["genealogy_root"]),
                        "choice": str(ag["choice"]),
                        "stage1_gold": str(tr["stage1_gold"]),
                        "is_correct": bool(ag["is_correct"]),
                        "normalized_entropy": float(ag["normalized_entropy"]),
                        "option_entropy_nats": float(ag["option_entropy_nats"]),
                        "logit_margin": float(ag["logit_margin"]),
                        "seq_token_entropy": float(ag["seq_token_entropy"]),
                        "tool_verified": bool(ag["tool_verified"]),
                        "mps_wall_s": float(ag["mps_wall_s"]),
                        "queue_tool_overhead_s": float(ag["queue_tool_overhead_s"]),
                        "completion_time_s": float(ag["completion_time_s"]),
                        "snippet_preview": str(ag["snippet_preview"]),
                    }
                )
            for pol_name, pdec in tr["policies"].items():
                decision_rows.append(
                    {
                        "suite": suite_name,
                        "task_id": tid,
                        "regime": tr["regime"],
                        "category": tr["category"],
                        "policy": pol_name,
                        "stage1_choice": str(pdec["stage1_choice"]),
                        "stage1_correct": bool(pdec["stage1_correct"]),
                        "stage1_latency_s": float(pdec["stage1_latency_s"]),
                        "stage1_boost": bool(pdec["stage1_boost"]),
                        "stage2_choice": str(pdec["stage2_choice"]),
                        "stage2_correct": bool(pdec["stage2_correct"]),
                        "stage2_latency_s": float(pdec["stage2_latency_s"]),
                        "workflow_total_latency_s": float(pdec["workflow_total_latency_s"]),
                        "workflow_both_stages_correct": bool(pdec["workflow_both_stages_correct"]),
                    }
                )

    _write_jsonl(stage_dir / "data" / "agent_observations" / "train.jsonl", obs_rows)
    _write_jsonl(stage_dir / "data" / "policy_decisions" / "train.jsonl", decision_rows)

    # 3. GPU priority-contention rows
    cont_rows: List[Dict[str, Any]] = []
    for load_key, ldata in cont["loads"].items():
        rho = float(load_key.replace("rho_", ""))
        for pol in ("FCFS_Default", "Unbounded_Urgent", "QuorumGuard_Budgeted"):
            st = ldata[pol]
            cont_rows.append(
                {
                    "gpu_utilization_rho": rho,
                    "scheduler_policy": pol,
                    "dissent_straggler_mean_turnaround_s": float(st["dissent_straggler_mean_turnaround_s"]),
                    "dissent_straggler_p95_turnaround_s": float(st["dissent_straggler_p95_turnaround_s"]),
                    "dissent_straggler_cvar95_turnaround_s": float(st["dissent_straggler_cvar95_turnaround_s"]),
                    "dissent_straggler_mean_queue_wait_s": float(st["dissent_straggler_mean_queue_wait_s"]),
                    "foreground_normal_mean_turnaround_s": float(st["foreground_normal_mean_turnaround_s"]),
                    "foreground_normal_p95_turnaround_s": float(st["foreground_normal_p95_turnaround_s"]),
                }
            )
    _write_jsonl(stage_dir / "data" / "gpu_contention" / "train.jsonl", cont_rows)

    raw_dir = stage_dir / "raw_telemetry"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for src in list((REPO_ROOT / "pilot" / "results").glob("*.json")) + list(
        (REPO_ROOT / "benchmarks" / "results").glob("*.json")
    ):
        shutil.copy2(src, raw_dir / src.name)

    fig_dir = stage_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in (REPO_ROOT / "paper" / "figures").glob("*.png"):
        shutil.copy2(png, fig_dir / png.name)
    paper_dst = stage_dir / "paper"
    paper_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "paper" / "main.pdf", paper_dst / "main.pdf")
    shutil.copy2(REPO_ROOT / "paper" / "highlights.pdf", paper_dst / "highlights.pdf")

    dataset_card = f"""---
license: apache-2.0
pretty_name: "QuorumBench: Asynchronous Multi-Agent LLM Consensus & Straggler Herding Benchmark"
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - llm-agents
  - multi-agent-systems
  - consensus
  - tail-latency
  - semantic-entropy
  - vllm
size_categories:
  - n<1K
models:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-Coder-14B-Instruct
  - HuggingFaceTB/SmolLM2-1.7B-Instruct
  - ibm-granite/granite-3.1-8b-instruct
configs:
  - config_name: policy_decisions
    default: true
    data_files:
      - split: train
        path: data/policy_decisions/train.jsonl
  - config_name: tasks
    data_files:
      - split: train
        path: data/tasks/train.jsonl
  - config_name: agent_observations
    data_files:
      - split: train
        path: data/agent_observations/train.jsonl
  - config_name: gpu_contention
    data_files:
      - split: train
        path: data/gpu_contention/train.jsonl
---

# QuorumBench v1.0

Benchmark tasks and Apple M5 Pro (`48 GB` unified memory, `MPS` FP16) telemetry for **[QuorumGuard](https://github.com/batistado/quorumguard)**: asynchronous multi-agent consensus across `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-Coder-14B-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`, and `ibm-granite/granite-3.1-8b-instruct` with isolated Python sandbox verification on both `QuorumBench` and `CRUXEval-50`.

* **Interactive Explorer:** [`{username}/QuorumGuard-Explorer`](https://huggingface.co/spaces/{username}/QuorumGuard-Explorer)
* **Scheduler & Controller Package:** [`{username}/QuorumGuard`](https://huggingface.co/{username}/QuorumGuard)
* **Compiled Paper PDF (10 pages):** [`paper/main.pdf`](paper/main.pdf)
* **GitHub Repository:** [`github.com/batistado/quorumguard`](https://github.com/batistado/quorumguard)

## Dataset Configurations

1. `policy_decisions` (`400` rows): Stage-1, Stage-2, and end-to-end latency/accuracy outcomes across 5 core consensus policies (`Fast_k_of_N`, `CVaR_Static_Cutoff`, `Sync_N_of_N`, `Genealogy_Sync_HPR`, `QuorumGuard`) on both the 50-task 3-tier suite and 30-task 2-tier pilot (`raw/revision_ablations_and_sweeps.json` adds `Fast_3_of_N_Maj`, `TW-Fast`, `Entropy-Cascade`, 7-point CVaR sweep, 10-seed variances, 2,000-bootstrap 95% CIs, hardware-only latencies, and 4-axis/`psi_i` sensitivity).
2. `tasks` (`50` rows): Two-stage systems, numerical precision, SQL, and algorithmic verification workflows (`20` routine tasks and `30` boundary traps) with shared upstream triage notes (`raw/cruxeval50_crossfamily.json` adds external `CRUXEval-50` & `SmolLM2-1.7B + Granite-3.1-8B` runs).
3. `agent_observations` (`560` rows): Per-agent MPS wall-clock times, completion timestamps, pre-commitment 4-option Shannon entropies, top-2 logit margins, provenance roots, and Python sandbox execution receipts.
4. `gpu_contention` (`12` rows): Trace-driven 8-token KV-cached decode quantum contention on `Qwen2.5-7B-Instruct` across loads `rho in {{0.58, 0.72, 0.85, 0.93}}` (`raw/live_mps_quantum_server.json` adds the live 64-request concurrent multi-`DynamicCache` MPS server benchmark).

## Summary Results (50 Two-Stage Workflows, 3B + 7B + 14B-Coder)

| Policy | S1 / S2 Acc | Joint [95% CI] | 10-Seed (Trap) | Mean / P95 [95% CI] | HW-Only P95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Fast_k_of_N (k=3)` | 46.0% / 52.0% | 28.0% [16, 42] | 27.8+-0.6% (6.7%) | 1.18s / 1.49s [1.4, 1.6] | 0.32s |
| `Fast_3_of_N_Maj` | 46.0% / 52.0% | 28.0% [16, 42] | 28.0+-0.9% (6.7%) | 1.34s / 1.58s [1.5, 1.6] | 0.34s |
| `CVaR_Cutoff (B=2.8s)` | 46.0% / 52.0% | 28.0% [16, 42] | 26.2+-1.1% (6.7%) | 5.60s / 5.60s [5.6, 5.6] | 5.60s |
| `TW-Fast (Static Tier)` | 62.0% / 68.0% | 42.0% [28, 56] | 44.2+-3.0% (23.3%) | 9.70s / 27.07s [19.7, 35.0] | 16.05s |
| `Entropy-Cascade` | 68.0% / 84.0% | 58.0% [44, 72] | 57.4+-2.3% (43.3%) | 25.12s / 45.06s [40.5, 47.9] | 28.81s |
| `Sync_N_of_N` | 66.0% / 68.0% | 42.0% [28, 56] | 42.6+-1.9% (23.3%) | 35.64s / 46.83s [43.4, 53.6] | 30.19s |
| `Genealogy_Sync_HPR` | 84.0% / 86.0% | 72.0% [58, 84] | 72.0+-1.6% (63.3%) | 36.34s / 47.39s [43.7, 53.6] | 30.19s |
| **`QuorumGuard`** | **84.0% / 86.0%** | **72.0% [58, 84]** | **72.0+-1.6% (63.3%)** | **10.18s / 15.09s [13.8, 16.1]** | **13.24s** |

![Pareto Frontier](figures/fig3_pareto_latency_accuracy.png)

## Citation

```bibtex
@misc{{syed2026quorumbench,
  author       = {{Syed, Mohammed Kamran and Rentala, Anudeep and Rahul, Somesh and Pai, Raghuram Malpe and Dara, Krishna Suman}},
  title        = {{{{QuorumBench}}: Asynchronous Multi-Agent {{LLM}} Consensus and Straggler Herding Benchmark}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/datasets/{username}/QuorumBench}}}}
}}
```
"""
    (stage_dir / "README.md").write_text(dataset_card, encoding="utf-8")
    return {
        "summary_50": qb50["policy_summary"],
        "tiers_50": qb50["tier_telemetry"],
        "tasks_preview": [
            {
                "id": r["task_id"],
                "regime": r["regime"],
                "cat": r["category"],
                "q1": tasks_spec[r["task_id"]].stage1_question,
                "gold1": r["stage1_gold"],
                "fast_c": r["policies"]["Fast_k_of_N"]["stage1_choice"],
                "fast_t": r["policies"]["Fast_k_of_N"]["workflow_total_latency_s"],
                "fast_ok": r["policies"]["Fast_k_of_N"]["workflow_both_stages_correct"],
                "sync_t": r["policies"]["Genealogy_Sync_HPR"]["workflow_total_latency_s"],
                "sync_ok": r["policies"]["Genealogy_Sync_HPR"]["workflow_both_stages_correct"],
                "qg_c": r["policies"]["QuorumGuard"]["stage1_choice"],
                "qg_t": r["policies"]["QuorumGuard"]["workflow_total_latency_s"],
                "qg_ok": r["policies"]["QuorumGuard"]["workflow_both_stages_correct"],
            }
            for r in qb50["tasks"]
        ],
    }


def build_static_space_staging(space_dir: Path, bundles: Dict[str, Any], username: str) -> None:
    space_dir.mkdir(parents=True, exist_ok=True)
    for png in (REPO_ROOT / "paper" / "figures").glob("*.png"):
        shutil.copy2(png, space_dir / png.name)

    space_readme = f"""---
title: QuorumGuard & QuorumBench Explorer
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: static
pinned: true
license: apache-2.0
datasets:
  - {username}/QuorumBench
models:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-Coder-14B-Instruct
short_description: Asynchronous multi-agent consensus & straggler simulator
---

# QuorumGuard & QuorumBench Explorer

Interactive simulator and 50-workflow trace browser for **[QuorumGuard](https://github.com/batistado/quorumguard)** and **[QuorumBench](https://huggingface.co/datasets/{username}/QuorumBench)**.
"""
    (space_dir / "README.md").write_text(space_readme, encoding="utf-8")

    data_js = json.dumps(bundles, ensure_ascii=False)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>QuorumGuard & QuorumBench Explorer</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --border: #334155; --text: #f8fafc; --muted: #94a3b8; --accent: #10b981; --warn: #ef4444; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; line-height: 1.5; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ margin: 0 0 6px; font-size: 1.65rem; }}
  .sub {{ color: var(--muted); margin-bottom: 20px; font-size: 0.95rem; }}
  .sub a {{ color: #38bdf8; text-decoration: none; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px; margin-bottom: 20px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .card h2 {{ margin: 0 0 10px; font-size: 1.08rem; }}
  label {{ display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-top: 8px; }}
  input[type=range], select {{ width: 100%; margin-top: 4px; }}
  select {{ background: #0f172a; color: var(--text); border: 1px solid var(--border); padding: 7px; border-radius: 6px; }}
  .stat {{ font-size: 1.25rem; font-weight: 700; color: var(--accent); }}
  .bad {{ color: var(--warn); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; margin-top: 8px; }}
  th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 600; }}
  .badge {{ padding: 2px 7px; border-radius: 4px; font-size: 0.76rem; font-weight: 600; }}
  .ok {{ background: rgba(16,185,129,0.18); color: #34d399; }}
  .err {{ background: rgba(239,68,68,0.18); color: #f87171; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>QuorumGuard &amp; QuorumBench Interactive Explorer</h1>
  <div class="sub">
    Live MPS Telemetry across Qwen2.5-3B, 7B, and 14B-Coder &middot;
    <a href="https://huggingface.co/datasets/{username}/QuorumBench" target="_blank">Dataset</a> &middot;
    <a href="https://huggingface.co/{username}/QuorumGuard" target="_blank">Hub Package</a> &middot;
    <a href="https://github.com/batistado/quorumguard" target="_blank">GitHub</a>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Theorem 1 Phase-Transition Calculator</h2>
      <label><span>Shallow Error p0</span><span id="v_p0">0.78</span></label>
      <input type="range" id="p0" min="0.52" max="0.95" step="0.01" value="0.78" />
      <label><span>Asymptotic Deep Error p_inf</span><span id="v_pinf">0.08</span></label>
      <input type="range" id="pinf" min="0.01" max="0.30" step="0.01" value="0.08" />
      <label><span>Verification Rate lambda / mu</span><span id="v_rat">1.22</span></label>
      <input type="range" id="rat" min="0.30" max="3.00" step="0.05" value="1.22" />
      <label><span>Fast Quorum Ratio phi = k / N</span><span id="v_phi">0.35</span></label>
      <input type="range" id="phi" min="0.10" max="0.99" step="0.01" value="0.35" />
      <div style="margin-top:12px; font-size:0.88rem;">
        Population Mean Error p_bar: <strong id="out_pbar"></strong><br/>
        Critical Quorum Threshold phi_crit: <strong id="out_crit"></strong><br/>
        Fast-Quorum Member Error p_fast(phi): <span id="out_pfast" class="stat"></span>
      </div>
    </div>

    <div class="card">
      <h2>QuorumBench-50 Summary (3B + 7B + 14B-Coder)</h2>
      <table>
        <thead><tr><th>Policy</th><th>Joint Acc</th><th>Trap Acc</th><th>P95 (s)</th></tr></thead>
        <tbody>
          <tr><td>Fast k-of-N</td><td class="bad">28.0%</td><td class="bad">6.7%</td><td>1.46s</td></tr>
          <tr><td>CVaR Cutoff</td><td class="bad">28.0%</td><td class="bad">6.7%</td><td>5.60s</td></tr>
          <tr><td>Sync N-of-N</td><td>42.0%</td><td>23.3%</td><td>49.86s</td></tr>
          <tr><td>Genealogy Sync</td><td>72.0%</td><td>63.3%</td><td>48.86s</td></tr>
          <tr><td><strong>QuorumGuard</strong></td><td><strong style="color:#34d399">72.0%</strong></td><td><strong style="color:#34d399">63.3%</strong></td><td><strong style="color:#34d399">14.82s</strong></td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>50-Workflow Live MPS Trace Browser</h2>
    <select id="task_sel"></select>
    <div id="task_detail" style="margin-top:12px; font-size:0.9rem;"></div>
  </div>
</div>

<script>
const DATA = {data_js};
function pFast(phi, p0, pinf, ratio) {{
  if (phi >= 0.9999) return pinf + (p0 - pinf) * (1.0 / (1.0 + ratio));
  const exp = 1.0 + ratio;
  const term = (1.0 / (1.0 + ratio)) * (1.0 - Math.pow(1.0 - phi, exp));
  return pinf + (p0 - pinf) * (term / phi);
}}
function updateCalc() {{
  const p0 = parseFloat(document.getElementById('p0').value);
  const pinf = parseFloat(document.getElementById('pinf').value);
  const rat = parseFloat(document.getElementById('rat').value);
  const phi = parseFloat(document.getElementById('phi').value);
  document.getElementById('v_p0').textContent = p0.toFixed(2);
  document.getElementById('v_pinf').textContent = pinf.toFixed(2);
  document.getElementById('v_rat').textContent = rat.toFixed(2);
  document.getElementById('v_phi').textContent = phi.toFixed(2);
  const pbar = pinf + (p0 - pinf) * (1.0 / (1.0 + rat));
  let lo = 0.001, hi = 0.9999;
  for (let i = 0; i < 50; i++) {{
    const mid = 0.5 * (lo + hi);
    if (pFast(mid, p0, pinf, rat) > 0.5) lo = mid; else hi = mid;
  }}
  const crit = (pbar >= 0.5) ? 1.0 : 0.5 * (lo + hi);
  const pf = pFast(phi, p0, pinf, rat);
  document.getElementById('out_pbar').textContent = pbar.toFixed(3) + (pbar < 0.5 ? ' (Sync -> 0% err)' : ' (Unreliable)');
  document.getElementById('out_crit').textContent = crit.toFixed(3);
  const el = document.getElementById('out_pfast');
  el.textContent = pf.toFixed(3) + (pf > 0.5 ? ' -> FALSE CONSENSUS' : ' -> Stable Quorum');
  el.className = pf > 0.5 ? 'stat bad' : 'stat';
}}
['p0','pinf','rat','phi'].forEach(id => document.getElementById(id).addEventListener('input', updateCalc));
updateCalc();

const sel = document.getElementById('task_sel');
DATA.tasks_preview.forEach((t, idx) => {{
  const opt = document.createElement('option');
  opt.value = idx;
  opt.textContent = `[${{t.regime.toUpperCase()}}] ${{t.id}} (${{t.cat}})`;
  sel.appendChild(opt);
}});
function renderTask() {{
  const t = DATA.tasks_preview[parseInt(sel.value, 10)];
  const b = ok => `<span class="badge ${{ok ? 'ok' : 'err'}}">${{ok ? 'BOTH STAGES OK' : 'FAILED'}}</span>`;
  document.getElementById('task_detail').innerHTML = `
    <div style="color:#cbd5e1; margin-bottom:8px;"><strong>Stage-1 Prompt:</strong> ${{t.q1}} (Gold: <code>${{t.gold1}}</code>)</div>
    <table>
      <thead><tr><th>Policy</th><th>Stage-1 Commit</th><th>2-Stage Latency</th><th>Workflow Outcome</th></tr></thead>
      <tbody>
        <tr><td>Fast k-of-N</td><td><code>${{t.fast_c}}</code></td><td>${{t.fast_t.toFixed(2)}}s</td><td>${{b(t.fast_ok)}}</td></tr>
        <tr><td>Genealogy Sync (HPR)</td><td><code>${{t.qg_c}}</code></td><td>${{t.sync_t.toFixed(2)}}s</td><td>${{b(t.sync_ok)}}</td></tr>
        <tr><td><strong>QuorumGuard</strong></td><td><code>${{t.qg_c}}</code></td><td><strong>${{t.qg_t.toFixed(2)}}s</strong></td><td>${{b(t.qg_ok)}}</td></tr>
      </tbody>
    </table>`;
}}
sel.addEventListener('change', renderTask);
renderTask();
</script>
</body>
</html>
"""
    (space_dir / "index.html").write_text(html, encoding="utf-8")


def build_hub_repo_staging(hub_dir: Path, username: str) -> None:
    pkg_dst = hub_dir / "quorumguard"
    pkg_dst.mkdir(parents=True, exist_ok=True)
    for py in (REPO_ROOT / "quorumguard").glob("*.py"):
        shutil.copy2(py, pkg_dst / py.name)
    shutil.copy2(REPO_ROOT / "pyproject.toml", hub_dir / "pyproject.toml")
    shutil.copy2(REPO_ROOT / "LICENSE", hub_dir / "LICENSE")
    shutil.copy2(
        REPO_ROOT / "paper" / "figures" / "fig0_architecture.png",
        hub_dir / "fig0_architecture.png",
    )
    hub_paper = hub_dir / "paper"
    hub_paper.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "paper" / "main.pdf", hub_paper / "main.pdf")
    shutil.copy2(REPO_ROOT / "paper" / "highlights.pdf", hub_paper / "highlights.pdf")

    hub_card = f"""---
license: apache-2.0
library_name: transformers
tags:
  - multi-agent-systems
  - consensus
  - tail-latency
  - semantic-entropy
  - llm-agents
  - vllm
datasets:
  - {username}/QuorumBench
base_model:
  - Qwen/Qwen2.5-3B-Instruct
  - Qwen/Qwen2.5-7B-Instruct
  - Qwen/Qwen2.5-Coder-14B-Instruct
  - HuggingFaceTB/SmolLM2-1.7B-Instruct
  - ibm-granite/granite-3.1-8b-instruct
---

# QuorumGuard: Claim-Genealogy-Gated Hedged Quorums & Priority Dissent Preemption

* **Benchmark Dataset:** [`{username}/QuorumBench`](https://huggingface.co/datasets/{username}/QuorumBench)
* **Interactive Simulator Space:** [`{username}/QuorumGuard-Explorer`](https://huggingface.co/spaces/{username}/QuorumGuard-Explorer)
* **Paper PDF (10 pages):** [`paper/main.pdf`](paper/main.pdf)
* **Source Code & Paper:** [`github.com/batistado/quorumguard`](https://github.com/batistado/quorumguard)

`QuorumGuard` is a model-agnostic consensus controller and quantum-preemptive priority scheduler for asynchronous multi-agent LLM committees. It requires zero weight modifications and operates on pre-commitment option logits, provenance DAGs, and priority-preemptive decode quanta (`Priority = -10` in both our live PyTorch MPS `DynamicCache` server and `vLLM`-compatible priority schedulers).

![QuorumGuard Overview](fig0_architecture.png)

## Install

```bash
pip install git+https://github.com/batistado/quorumguard.git
```
"""
    (hub_dir / "README.md").write_text(hub_card, encoding="utf-8")


def main() -> int:
    api = HfApi()
    username = api.whoami()["name"]
    ds_repo_id = f"{username}/QuorumBench"
    space_repo_id = f"{username}/QuorumGuard-Explorer"
    hub_repo_id = f"{username}/QuorumGuard"

    with tempfile.TemporaryDirectory(prefix="quorumguard_hf_") as tmp:
        tmp_path = Path(tmp)
        ds_dir = tmp_path / "dataset"
        sp_dir = tmp_path / "space"
        hb_dir = tmp_path / "hub"

        bundles = build_dataset_staging(ds_dir, username=username)
        api.create_repo(repo_id=ds_repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=ds_repo_id,
            repo_type="dataset",
            folder_path=str(ds_dir),
            commit_message="Publish QuorumBench v1.0 multi-agent consensus & contention telemetry",
        )
        print(f"[1/3 SUCCESS] Dataset -> https://huggingface.co/datasets/{ds_repo_id}")

        build_static_space_staging(sp_dir, bundles, username=username)
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=space_repo_id,
            repo_type="space",
            folder_path=str(sp_dir),
            commit_message="Publish QuorumGuard & QuorumBench static HTML5 interactive explorer",
        )
        print(f"[2/3 SUCCESS] Space   -> https://huggingface.co/spaces/{space_repo_id}")

        build_hub_repo_staging(hb_dir, username=username)
        api.create_repo(repo_id=hub_repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            repo_id=hub_repo_id,
            repo_type="model",
            folder_path=str(hb_dir),
            commit_message="Publish QuorumGuard controller & vLLM priority-preemption package",
        )
        print(f"[3/3 SUCCESS] Hub     -> https://huggingface.co/{hub_repo_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

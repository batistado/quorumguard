# QuorumGuard

**Straggler-Induced Herding Bounds and Claim-Genealogy-Gated Hedged Quorums in Asynchronous Multi-Agent LLM Consensus**

Research code, order-statistic proofs, Apple Silicon MPS telemetry (`Qwen2.5-3B`, `Qwen2.5-7B`, and `Qwen2.5-Coder-14B`), GPU priority-contention benchmarks, and a 9-page manuscript (`paper/main.pdf`).

* **Dataset (`QuorumBench`):** https://huggingface.co/datasets/batistado/QuorumBench
* **Interactive Explorer Space:** https://huggingface.co/spaces/batistado/QuorumGuard-Explorer
* **Hugging Face Hub Package:** https://huggingface.co/batistado/QuorumGuard
* **Compiled Paper PDF (9 pages):** [`paper/main.pdf`](paper/main.pdf)

## 1. Problem and Approach

Classical distributed systems use asynchronous $k$-of-$N$ fast quorums under the assumption that replica completion time $T_i$ and correctness $1 - \varepsilon_i$ are statistically independent. In heterogeneous multi-agent LLM committees, completion time and accuracy covary positively ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) > 0$, including $+0.6142\text{ s}$ for hardware-only MPS execution before queueing delay). Zero-CoT guesses and shared triage echoes finish under one second, while multi-step reasoning and sandboxed code verification arrive between $7\text{s}$ and $19\text{s}$.

QuorumGuard combines three mechanisms in an event-driven consensus gate:
1. Pre-commitment semantic entropy gating, which scores 4-way option entropy $H_{\mathrm{ans}}$ and top-2 logit margins $\Delta$ before any answer letter is emitted, avoiding post-generation self-conditioning collapse.
2. Claim-genealogy quorum sizing ($N_{\mathrm{eff}}$), which resolves upstream provenance DAG roots so correlated context echoes contribute at most one root.
3. Token-bucket priority dissent boosting, which raises `vLLM` decode priority (`Priority=-10`) for active deep-verification stragglers only when early arrivals share roots, disagree, or exceed entropy bounds.

## 2. Scale-Up `QuorumBench-50` Results (`3B` + `7B` + `14B-Coder` + Python Sandbox)

Evaluated on Apple M5 Pro (`48 GB` unified memory, `MPS` FP16, `benchmarks/results/quorumbench_50_3b_7b_14b.json`) across 50 two-stage workflows (100 stage evaluations), with $\mathrm{Cov}(T_i, 1 - \varepsilon_i) = +0.9661\text{ s}$ overall ($+1.5853\text{ s}$, $r = +0.404$ on boundary traps; hardware-only MPS $\mathrm{Cov} = +0.6142\text{ s}$):

| Consensus Policy | Stage-1 Acc | Stage-2 Acc | Joint 2-Stage Acc | Trap Joint Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Fast k-of-N Quorum` | 46.0% | 56.0% | 28.0% | 6.7% | 1.11s | 1.46s | 1.49s |
| `CVaR Static Cutoff` (`arXiv:2609.10964`) | 46.0% | 52.0% | 28.0% | 6.7% | 5.60s | 5.60s | 5.60s |
| `Sync N-of-N Full Barrier` | 66.0% | 68.0% | 42.0% | 23.3% | 36.34s | 49.86s | 51.40s |
| `Genealogy Sync (HPR)` (`arXiv:2603.04474`) | 84.0% | 88.0% | 72.0% | 63.3% | 35.97s | 48.86s | 51.20s |
| `QuorumGuard` (Ours) | 84.0% | 88.0% | 72.0% | 63.3% | 10.10s | 14.82s | 16.11s |

QuorumGuard gains $+44.0$ percentage points ($2.57\times$) in joint two-stage accuracy over `Fast k-of-N` and `CVaR Static Cutoff` ($63.3\%$ vs. $6.7\%$ on boundary traps), beats unweighted `Sync N-of-N` by $+30.0$ percentage points, matches synchronous `Genealogy Sync` accuracy ($72.0\%$), and lowers `P95` latency by $3.30\times$ to $3.36\times$ ($3.67\times$ on the 2-tier `3B+7B` pool).

### Orthogonal Component Ablation (`pilot/results/ablation_results.json`)

| Variant | Stage-1 Acc | Boundary-Trap Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1. Fast k-of-N` | 43.3% | 5.6% | 0.56s | 0.76s | 0.77s |
| `2. + Entropy Gate Only` | 76.7% | 66.7% | 9.30s | 17.24s | 19.04s |
| `3. + Genealogy N_eff Only` | 60.0% | 38.9% | 7.08s | 17.34s | 19.12s |
| `4. QuorumGuard (w/o Dissent Boost)` | 80.0% | 72.2% | 9.31s | 16.20s | 18.20s |
| `5. QuorumGuard (Full)` | 80.0% | 72.2% | 3.70s | 5.93s | 6.35s |

### Physical MPS 8-Token Quantum GPU Contention (`benchmarks/results/priority_contention_mps.json`)

| GPU Utilization $\rho$ | FCFS Queue Wait | QuorumGuard Queue Wait | Wait Compression | Unbounded Urgent CVaR95 | QuorumGuard CVaR95 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.58` | 10.35s | 2.04s | 5.09x | 93.5s | 24.7s |
| `0.72` | 10.08s | 1.91s | 5.29x | 115.6s | 20.8s |
| `0.85` | 13.52s | 3.11s | 4.35x | 101.3s | 30.5s |
| `0.93` | 16.72s | 2.78s | 6.01x | 116.0s (+98.6% FG delay) | 23.2s |

## 3. Repository Structure

```text
quorumguard/
├── HANDOFF_CONTEXT.md                     # Problem formulation, Nov 2026 literature & design
├── quorumguard/                           # Core library: ProvenanceDAG, EpistemicSignals, QuorumGuardController
├── pilot/
│   ├── quorumguard_pilot.py               # Live 30-task 2-tier (3B + 7B) MPS benchmark
│   └── results/                           # Pilot, theorem verification & ablation JSONs
├── benchmarks/
│   ├── verify_theorems.py                 # Analytical + 40,000-trial MC verification of Theorems 1-3
│   ├── run_ablations.py                   # Orthogonal mechanism ablation evaluator
│   ├── bench_priority_contention.py       # Physical MPS 8-token decode-quantum contention benchmark
│   ├── quorumbench_scaleup.py             # Live 50-task 3-tier (3B + 7B + 14B-Coder) scale-up benchmark
│   └── results/                           # Scale-up & GPU priority-contention telemetry JSONs
├── scripts/
│   └── publish_huggingface.py             # Publishes dataset, interactive Space & Hub package
├── paper/
│   ├── main.tex / main.pdf                # Compiled 9-page manuscript (within 10-page free limit)
│   ├── highlights.tex / highlights.pdf    # Elsevier FGCS Research Highlights
│   ├── references.bib                     # 33 verified BibTeX references
│   ├── generate_figures.py / figures/     # 7 vector PDF + PNG figures (fig0..fig6)
│   ├── validate_latex.py                  # Static LaTeX, BibTeX & anti-slop prose validator
│   └── quorumguard_fgcs_overleaf.zip      # Overleaf submission bundle
└── tests/
    └── test_quorumguard.py                # Pytest suite for genealogy, entropy gating & vLLM hooks
```

# QuorumGuard

**Straggler-Induced Herding Bounds and Claim-Genealogy-Gated Hedged Quorums in Asynchronous Multi-Agent LLM Consensus**

Research code, order-statistic proofs, Apple Silicon MPS telemetry (`Qwen2.5-3B/7B/Coder-14B`, `SmolLM2-1.7B`, `Granite-3.1-8B`), external `CRUXEval-50` benchmarks, trace-driven & live concurrent KV-cache GPU contention suites, and a 10-page manuscript (`paper/main.pdf`).

* **Dataset (`QuorumBench`):** https://huggingface.co/datasets/batistado/QuorumBench
* **Interactive Explorer Space:** https://huggingface.co/spaces/batistado/QuorumGuard-Explorer
* **Hugging Face Hub Package:** https://huggingface.co/batistado/QuorumGuard
* **Compiled Paper PDF (10 pages):** [`paper/main.pdf`](paper/main.pdf)

## 1. Problem and Approach

Classical distributed systems use asynchronous $k$-of-$N$ fast quorums under the assumption that replica completion time $T_i$ and correctness $1 - \varepsilon_i$ are statistically independent. In heterogeneous multi-agent LLM committees, completion time and accuracy covary positively ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) > 0$, including $+0.6142\text{ s}$ for hardware-only MPS execution before queueing delay). Zero-CoT guesses and shared triage echoes finish under one second, while multi-step reasoning and sandboxed code verification arrive between $7\text{s}$ and $19\text{s}$. We prove this order-statistic phase transition for general $|\mathcal{C}| \ge 2$ options (Theorem 1), multi-stage DAG herding amplification vs. joint survival (Theorem 2), and three-path risk + quantum-preemptive-resume M/G/1 delay bounds (Theorem 3).

QuorumGuard combines three mechanisms in an event-driven consensus gate:
1. Pre-commitment semantic entropy gating ($\tau_\Delta = 3.5, \alpha_{\mathrm{seq}} = 0.12, h_0 = 0.60, \psi_i \in [0.76, 1.38]$), scoring 4-way option entropy $H_{\mathrm{ans}}$ and top-2 logit margins $\Delta$ before any answer letter is emitted to avoid a $70\times$ post-generation self-conditioning collapse.
2. Claim-genealogy quorum sizing ($N_{\mathrm{eff}}$), resolving upstream provenance DAG roots so correlated context echoes contribute at most one root.
3. Token-bucket priority dissent boosting ($k_{\min} = 3, \tau_{\mathrm{eff}} = 2.12, \gamma_H = 0.08, \Delta_{\min} = 3.2, m_{\mathrm{gap}} = 0.72, B_{\max} = 6.0, B_{\mathrm{stage}} = 2, r_{\mathrm{refill}} = 0.64$/stage), elevating decode priority (`Priority = -10`) at 8-token decode quanta in both our live PyTorch MPS KV-cache server and `vLLM`-compatible priority schedulers.

## 2. `QuorumBench-50` & Revision Benchmark Results

All workflow evaluations ran on a PyTorch `2.15.0` / `transformers 5.22.1` Apple M5 Pro (`48 GB` unified memory, FP16 MPS, `torch.mps.synchronize()` barriers) using isolated single-tier passes (`<= 28.5 GB` peak footprint, zero swapping) with both emulated `Pareto(alpha=2.25)` queue/container delays and pure hardware-only latencies (`HW-Only`), plus a `450`-request quantum simulator and a live `64`-request concurrent multi-KV-cache MPS server. On `QuorumBench-50` ($50$ workflows, $100$ stages; $\mathrm{Cov}(T_i, 1 - \varepsilon_i) = +0.9661\text{ s}$ overall, $+1.5853\text{ s}$ on boundary traps, $+0.6142\text{ s}$ hardware-only):

| Consensus Policy | S1 / S2 Acc | Joint [95% CI] | 10-Seed (Trap) | Mean / P95 [95% CI] | HW-Only P95 | CVaR95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Fast k-of-N (k=3)` | 46.0% / 52.0% | 28.0% [16, 42] | 27.8+-0.6% (6.7%) | 1.18s / 1.49s [1.4, 1.6] | 0.32s | 1.56s |
| `Fast 3-of-N Maj` | 46.0% / 52.0% | 28.0% [16, 42] | 28.0+-0.9% (6.7%) | 1.34s / 1.58s [1.5, 1.6] | 0.34s | 1.61s |
| `CVaR Cutoff (B=2.8s)` (`arXiv:2609.10964`) | 46.0% / 52.0% | 28.0% [16, 42] | 26.2+-1.1% (6.7%) | 5.60s / 5.60s [5.6, 5.6] | 5.60s | 5.60s |
| `CVaR Cutoff (B=15s)` (`arXiv:2609.10964`) | 50.0% / 64.0% | 36.0% [24, 50] | 41.2+-3.0% (16.7%) | 28.60s / 30.00s [30.0, 30.0] | 24.29s | 30.00s |
| `TW-Fast (Static Tier)` | 62.0% / 68.0% | 42.0% [28, 56] | 44.2+-3.0% (23.3%) | 9.70s / 27.07s [19.7, 35.0] | 16.05s | 32.78s |
| `Entropy-Cascade` | 68.0% / 84.0% | 58.0% [44, 72] | 57.4+-2.3% (43.3%) | 25.12s / 45.06s [40.5, 47.9] | 28.81s | 47.21s |
| `Sync N-of-N Barrier` | 66.0% / 68.0% | 42.0% [28, 56] | 42.6+-1.9% (23.3%) | 35.64s / 46.83s [43.4, 53.6] | 30.19s | 51.17s |
| `Genealogy Sync (HPR)` (`arXiv:2603.04474`) | 84.0% / 86.0% | 72.0% [58, 84] | 72.0+-1.6% (63.3%) | 36.34s / 47.39s [43.7, 53.6] | 30.19s | 51.23s |
| **`QuorumGuard` (Ours)** | **84.0% / 86.0%** | **72.0% [58, 84]** | **72.0+-1.6% (63.3%)** | **10.18s / 15.09s [13.8, 16.1]** | **13.24s** | **15.65s** |

QuorumGuard gains $+44.0$ percentage points ($2.57\times$) over `Fast k-of-N` and `CVaR (B=2.8s)` ($63.3\%$ vs. $6.7\%$ on boundary traps), beats unweighted `Sync N-of-N` by $+30.0$ points, matches `Genealogy Sync (HPR)` at $72.0\%$, and lowers `P95` tail latency by $3.10\times$--$3.14\times$ ($2.28\times$ hardware-only; $3.57\times$ mean speedup; $3.67\times$ on the 2-tier `3B+7B` pilot).

### External `CRUXEval-50` & Cross-Family Non-Qwen Suites (`benchmarks/results/cruxeval50_crossfamily.json`)

| Suite & Model Pool | `Fast k=3` | `TW-Fast` | `Cascade` | `Sync N-of-N` | `Gen. Sync` | `QuorumGuard` | P95 Speedup (HW) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CRUXEval-50` (`Qwen2.5-3B+7B`) | 36.0% (1.45s) | 56.0% (17.79s) | 58.0% (29.93s) | 62.0% (31.36s) | 66.0% (31.36s) | **66.0% (9.79s)** | **3.20x** (2.18x) |
| `CRUXEval-50` (`SmolLM2-1.7B+Granite-3.1-8B`) | 18.0% (1.36s) | 36.0% (20.18s) | 56.0% (33.35s) | 40.0% (33.08s) | 58.0% (33.35s) | **58.0% (10.25s)** | **3.25x** (2.14x) |
| `QuorumBench-30` (`SmolLM2-1.7B+Granite-3.1-8B`) | 16.7% (1.32s) | 30.0% (20.89s) | 63.3% (34.21s) | 33.3% (34.21s) | 66.7% (34.21s) | **63.3% (9.67s)** | **3.54x** (2.18x) |

### Orthogonal Component Ablation (`pilot/results/ablation_results.json`)

| Variant | Stage-1 Acc | Boundary-Trap Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1. Fast k-of-N` | 43.3% | 5.6% | 0.56s | 0.76s | 0.77s |
| `2. + Entropy Gate Only` | 76.7% | 66.7% | 9.30s | 17.24s | 19.04s |
| `3. + Genealogy N_eff Only` | 60.0% | 38.9% | 7.08s | 17.34s | 19.12s |
| `4. QuorumGuard (w/o Dissent Boost)` | 80.0% | 72.2% | 9.31s | 16.20s | 18.20s |
| `5. QuorumGuard (Full)` | 80.0% | 72.2% | 3.70s | 5.93s | 6.35s |

### Trace-Driven Simulator & Live Concurrent KV-Cache MPS Server

* **450-Request Trace-Driven Quantum Simulator (`benchmarks/results/priority_contention_mps.json`, `554.38 ms` 8-token quanta):** Compresses dissenting straggler queue wait by $4.35\times$--$6.01\times$ across $\rho \in [0.58, 0.93]$ while preventing priority inflation (`Unbounded Urgent` $\CVaR_{0.95} = 116.0\text{s}$ vs. `23.2s` under QuorumGuard at $\rho = 0.93$).
* **Live Concurrent Multi-KV-Cache MPS Server (`benchmarks/results/live_mps_quantum_server.json`, `Qwen2.5-7B` FP16, 64 req/policy, $\rho = 0.82$, `15.50 GB` peak):** Compresses dissenting straggler wait by **$11.74\times$** (`3.66s -> 0.31s`) and straggler `P95` by **$1.92\times$** (`26.57s -> 13.82s`; $\CVaR_{0.95}$ `13.91s` vs. `58.51s` unbounded urgent), transferring queueing delay to normal jobs (`48.69s` P95) during dense bursts as predicted by Theorem 3(2).

## 3. Repository Structure

```text
quorumguard/
├── HANDOFF_CONTEXT.md                          # Problem formulation, Nov 2026 literature & design
├── quorumguard/                                # Core library: ProvenanceDAG, EpistemicSignals, QuorumGuardController
├── pilot/
│   ├── quorumguard_pilot.py                    # 30-task 2-tier (3B + 7B) MPS benchmark
│   └── results/                                # Pilot, theorem verification (Theorems 1-3) & ablation JSONs
├── benchmarks/
│   ├── verify_theorems.py                      # Analytical + MC/DES verification of Theorems 1-3
│   ├── run_ablations.py                        # Orthogonal mechanism ablation evaluator
│   ├── run_revisions_suite.py                  # Full Stage-2 cache, TW-Fast, Cascade, CVaR sweep & sensitivity
│   ├── eval_external_cruxeval_crossfamily.py   # CRUXEval-50 + cross-family SmolLM2-1.7B / Granite-3.1-8B suite
│   ├── bench_priority_contention.py            # Trace-driven 8-token decode-quantum contention simulator
│   ├── bench_live_mps_server.py                # Live concurrent DynamicCache KV-preemptive MPS server
│   ├── quorumbench_scaleup.py                  # 50-task 3-tier (3B + 7B + 14B-Coder) scale-up benchmark
│   └── results/                                # Scale-up, external, live-server & contention JSONs
├── scripts/
│   └── publish_huggingface.py                  # Publishes dataset, interactive Space & Hub package
├── paper/
│   ├── main.tex / main.pdf                     # Compiled 10-page archival manuscript
│   ├── highlights.tex / highlights.pdf         # Elsevier FGCS Research Highlights
│   ├── references.bib                          # 37 verified BibTeX references
│   ├── generate_figures.py / figures/          # 7 vector PDF + PNG figures (fig0..fig6)
│   ├── validate_latex.py                       # Static LaTeX, BibTeX & anti-slop prose validator
│   └── quorumguard_fgcs_overleaf.zip           # Overleaf submission bundle
└── tests/
    └── test_quorumguard.py                     # Pytest suite (6 tests) for genealogy, entropy & scheduler
```

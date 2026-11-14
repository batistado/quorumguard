# QuorumGuard

**Straggler-Induced Herding Bounds and Claim-Genealogy-Gated Hedged Quorums in Asynchronous Multi-Agent LLM Consensus**

> Archival research codebase, analytical order-statistic proofs, live Apple Silicon MPS multi-agent telemetry (`Qwen2.5-3B` / `7B` / `14B-Coder`), GPU priority-contention harness, and compiled 11-page Q1 manuscript targeting *Elsevier Future Generation Computer Systems (FGCS)* / *IEEE Transactions on Parallel and Distributed Systems (TPDS)*.

## 1. Core Problem & Hypothesis

Classical distributed systems use asynchronous $k$-of-$N$ fast quorums under the assumption that replica completion latency $T_i$ and response correctness are statistically independent. In multi-agent LLM workflows, completion time and epistemic accuracy exhibit strong positive covariance ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) > 0$ — including pure hardware-native MPS execution covariance $+0.6142\text{ s}$ even before queueing delay): zero-CoT assertions and correlated context echoes finish sub-second, whereas multi-hop reasoning and sandboxed code verification arrive in the `7s–19s` straggler tail.

**QuorumGuard** bridges tail-aware workflow scheduling with multi-agent herding defense via three coupled mechanisms:
1. **Pre-Commitment Epistemic Confidence & Entropy Gating:** Probes 4-way posterior semantic entropy $H_{\mathrm{ans}}$ and top-2 logit margin $\Delta_{\mathrm{ans}}$ *prior* to answer-letter emission (preventing $70\times$ post-generation self-conditioning collapse).
2. **Claim-Genealogy Independence ($N_{\mathrm{eff}}$):** Resolves upstream provenance DAG roots so correlated context echoes collapse to unit root capacity rather than masquerading as $k$ independent votes.
3. **Token-Bucket Priority-Budget Dissent Boosting:** Dynamically elevates GPU decode priority (`vLLM` `Priority=-10` quantum preemption) for active deep-verification stragglers *only* when early arrivals dissent, share lineage, or fail epistemic gating.

---

## 2. Scale-Up `QuorumBench-50` Results (`3B` + `7B` + `14B-Coder` + Live Python Sandbox)

Measured live on Apple M5 Pro (`48 GB` Unified Memory, `MPS` FP16, `benchmarks/results/quorumbench_50_3b_7b_14b.json`) across **50 two-stage workflows (100 stage evaluations)** ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) = +0.9661\text{ s}$ overall; $+1.5853\text{ s}$, $r = +0.404$ on boundary traps; pure MPS hardware $\mathrm{Cov} = +0.6142\text{ s}$):

| Consensus Policy | Stage-1 Acc | Stage-2 Acc | Joint 2-Stage Acc | Trap Joint Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Fast k-of-N Quorum` | 46.0% | 56.0% | 28.0% | 6.7% | 1.11s | 1.46s | 1.49s |
| `CVaR Static Cutoff` (`arXiv:2609.10964`) | 46.0% | 52.0% | 28.0% | 6.7% | 5.60s | 5.60s | 5.60s |
| `Sync N-of-N Full Barrier` | 66.0% | 68.0% | 42.0% | 23.3% | 36.34s | 49.86s | 51.40s |
| `Genealogy Sync (HPR)` (`arXiv:2603.04474`) | **84.0%** | **88.0%** | **72.0%** | **63.3%** | 35.97s | 48.86s | 51.20s |
| **`QuorumGuard` (Ours)** | **84.0%** | **88.0%** | **72.0%** | **63.3%** | **10.10s** | **14.82s** | **16.11s** |

* **`+44.0%` (`2.57×`) Joint 2-Stage Accuracy** (`9.45×` higher on boundary-trap tasks: `63.3%` vs. `6.7%`) over `Fast k-of-N` and `CVaR Static Cutoff`, and **`+30.0%`** over unweighted `Sync N-of-N`.
* **`3.30× – 3.36×` `P95` Speedup** (`3.67×` on the 2-tier `3B+7B` pool) while matching full synchronous `Genealogy Sync` accuracy identically (`72.0%`).

### Orthogonal Component Ablation (`pilot/results/ablation_results.json`)

| Variant | Stage-1 Acc | Boundary-Trap Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1. Fast k-of-N` | 43.3% | 5.6% | 0.56s | 0.76s | 0.77s |
| `2. + Entropy Gate Only` | 76.7% | 66.7% | 9.30s | 17.24s | 19.04s |
| `3. + Genealogy N_eff Only` | 60.0% | 38.9% | 7.08s | 17.34s | 19.12s |
| `4. QuorumGuard (w/o Dissent Boost)` | **80.0%** | **72.2%** | 9.31s | 16.20s | 18.20s |
| **`5. QuorumGuard (Full)`** | **80.0%** | **72.2%** | **3.70s** | **5.93s** | **6.35s** |

### Physical MPS 8-Token Quantum GPU Contention (`benchmarks/results/priority_contention_mps.json`)

| GPU Utilization $\rho$ | FCFS Queue Wait | QuorumGuard Queue Wait | Queue-Wait Compression | Unbounded Urgent $\mathrm{CVaR}_{95}$ | QuorumGuard $\mathrm{CVaR}_{95}$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `0.58` | 10.35s | **2.04s** | **5.09×** | 93.5s | **24.7s** |
| `0.72` | 10.08s | **1.91s** | **5.29×** | 115.6s | **20.8s** |
| `0.85` | 13.52s | **3.11s** | **4.35×** | 101.3s | **30.5s** |
| `0.93` | 16.72s | **2.78s** | **6.01×** | 116.0s (+98.6% FG starve) | **23.2s** |

---

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
├── paper/
│   ├── main.tex / main.pdf                # Compiled 11-page Q1 archival manuscript (FGCS / TPDS)
│   ├── highlights.tex / highlights.pdf    # Official Elsevier FGCS Research Highlights
│   ├── references.bib                     # 32 verified BibTeX references
│   ├── generate_figures.py / figures/     # 7 publication-grade vector PDFs (fig0..fig6.pdf)
│   └── quorumguard_fgcs_overleaf.zip      # Self-contained 1-click Overleaf / go/pubapprove bundle
└── tests/
    └── test_quorumguard.py                # Pytest suite verifying genealogy, entropy gating & vLLM hooks
```

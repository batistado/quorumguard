# QuorumGuard

**Straggler-Induced Herding Bounds and Claim-Genealogy-Gated Hedged Quorums in Asynchronous Multi-Agent LLM Consensus**

> Archival research codebase, analytical order-statistic proofs, live Apple Silicon MPS multi-agent telemetry, and figure suite targeting *Elsevier Future Generation Computer Systems (FGCS)* / *IEEE Transactions on Parallel and Distributed Systems (TPDS)*.

## 1. Core Problem & Hypothesis

Classical distributed systems use asynchronous $k$-of-$N$ fast quorums under the assumption that replica completion latency $T_i$ and response correctness are statistically independent. In multi-agent LLM workflows, completion time and epistemic accuracy exhibit strong positive covariance ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) > 0$): zero-CoT guesses and correlated context echoes finish in `<1s`, whereas multi-hop reasoning and sandboxed tool verification arrive as `6s–18s` stragglers.

**QuorumGuard** unifies tail-aware workflow scheduling with multi-agent herding defense via three coupled mechanisms:
1. **Pre-Commitment Epistemic Confidence & Entropy Gating:** Evaluates 4-way posterior semantic entropy $H_{\mathrm{ans}}$ and top-2 logit margin $\Delta_{\mathrm{ans}}$ prior to answer commitment, blocking premature fast-quorum closure when fast arrivals are uncertain or ungrounded.
2. **Claim-Genealogy Independence ($N_{\mathrm{eff}}$):** Tracks upstream provenance DAG roots so correlated context echoes collapse to a single provenance root rather than masquerading as $k$ independent votes.
3. **Priority-Budget Dissent Boosting:** Dynamically elevates GPU scheduling priority (`vLLM` preemption priority) for active deep-verification stragglers *only* when early arrivals dissent, share lineage, or fail epistemic gating.

---

## 2. Live 30-Task 2-Stage MPS Benchmark (`Qwen2.5-3B` + `Qwen2.5-7B` + Python Sandbox)

Measured live on Apple M5 Pro (`MPS` FP16, `Seed=2026`, `pilot/results/quorumguard_pilot.json`) across 30 two-stage engineering, numerical precision, SQL, and algorithmic verification workflows ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) = +0.7185\text{ s}$ overall; $+1.4339\text{ s}$, $r = +0.520$ on boundary-trap tasks):

| Consensus Policy | Stage-1 Acc | Stage-2 Acc | Joint 2-Stage Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Fast k-of-N Quorum` | 43.3% | 43.3% | 26.7% | 1.07s | 1.41s | 1.43s |
| `CVaR Static Cutoff` (`arXiv:2609.10964`) | 43.3% | 40.0% | 26.7% | 5.60s | 5.60s | 5.60s |
| `Sync N-of-N Full Barrier` | 60.0% | 50.0% | 36.7% | 27.79s | 36.00s | 36.99s |
| `Genealogy Sync (HPR)` (`arXiv:2603.04474`) | **80.0%** | **70.0%** | **66.7%** | 28.30s | 36.73s | 36.99s |
| **`QuorumGuard` (Ours)** | **80.0%** | **70.0%** | **66.7%** | **7.87s** | **10.00s** | **10.59s** |

* **`+40.0%` (`2.50×`) Joint 2-Stage Accuracy** over `Fast k-of-N` and `CVaR Static Cutoff`, and **`+30.0%`** over unweighted `Sync N-of-N`.
* **`3.60× – 3.67×` `P95` Speedup** (`3.49×` `CVaR95` speedup) over `Sync N-of-N` and `Genealogy Sync` while matching `Genealogy Sync` accuracy identically (`66.7%`).

### Orthogonal Component Ablation (Stage-1 Pool)

| Variant | Stage-1 Acc | Boundary-Trap Acc | Mean Latency | P95 Latency | CVaR95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `1. Fast k-of-N` | 43.3% | 5.6% | 0.56s | 0.76s | 0.77s |
| `2. + Entropy Gate Only` | 76.7% | 66.7% | 9.30s | 17.24s | 19.04s |
| `3. + Genealogy N_eff Only` | 60.0% | 38.9% | 7.08s | 17.34s | 19.12s |
| `4. QuorumGuard (w/o Dissent Boost)` | **80.0%** | **72.2%** | 9.31s | 16.20s | 18.20s |
| **`5. QuorumGuard (Full)`** | **80.0%** | **72.2%** | **3.70s** | **5.93s** | **6.35s** |

---

## 3. Repository Structure

```text
quorumguard/
├── HANDOFF_CONTEXT.md          # Problem formulation, Nov 2026 literature & experimental design
├── quorumguard/                # Core library: ProvenanceDAG, EpistemicSignals, QuorumGuardController
├── pilot/
│   ├── quorumguard_pilot.py    # Live 30-task 2-stage Qwen2.5-3B/7B + Python sandbox MPS benchmark
│   └── results/                # Logged telemetry JSONs (pilot, theorems, component ablations)
├── benchmarks/
│   ├── verify_theorems.py      # Analytical + 40,000-trial Monte Carlo verification of Theorems 1-3
│   └── run_ablations.py        # Orthogonal mechanism ablation evaluator
├── paper/
│   ├── generate_figures.py     # Reproducible matplotlib vector PDF figure generator
│   └── figures/                # 5 publication-grade vector PDFs (fig1..fig5.pdf)
└── tests/
    └── test_quorumguard.py     # Pytest suite verifying genealogy, entropy gating & vLLM boost hooks
```

## 4. Reproducing Everything Locally

```bash
# 1. Run unit tests
PYTHONPATH=. pytest tests/test_quorumguard.py -v

# 2. Verify Theorems 1, 2, and 3 analytically & via Monte Carlo
python benchmarks/verify_theorems.py

# 3. Run the 30-task 2-stage live MPS benchmark (Qwen2.5-3B + Qwen2.5-7B)
python pilot/quorumguard_pilot.py --max-tasks 30

# 4. Run component ablations & regenerate all 5 vector PDF figures
python benchmarks/run_ablations.py
python paper/generate_figures.py
```

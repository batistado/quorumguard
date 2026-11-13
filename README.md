# QuorumGuard

**Straggler-Induced Herding Bounds and Claim-Genealogy-Gated Hedged Quorums in Asynchronous Multi-Agent LLM Consensus**

> Research codebase, analytical proofs, benchmark suites, and manuscript artifacts targeting *Elsevier Future Generation Computer Systems (FGCS)* / *IEEE Transactions on Parallel and Distributed Systems (TPDS)*.

## Overview

Classical distributed systems use asynchronous $k$-of-$N$ fast quorums under the assumption that replica latency $T_i$ and response correctness are statistically independent. In multi-agent LLM workflows, completion time and epistemic accuracy exhibit strong positive covariance ($\mathrm{Cov}(T_i, 1 - \varepsilon_i) > 0$): superficial guesses and correlated context echoes finish fast, whereas multi-hop reasoning and tool-verified refutations arrive as stragglers.

**QuorumGuard** bridges tail-aware agentic scheduling and multi-agent error-cascade mitigation via three coupled mechanisms:
1. **Epistemic Confidence & Predictive Entropy Gating:** Blocks premature quorum closure when early arrivals exhibit elevated token-level semantic entropy $\mathcal{H}$ or lack verified tool grounding.
2. **Claim-Genealogy Independence ($N_{\mathrm{eff}}$):** Discounts correlated upstream context echoes by counting epistemically independent provenance roots rather than raw headcount.
3. **Priority-Budget Dissent Hedging:** Dynamically reallocates GPU scheduling priority to active deep-verification stragglers only when early arrivals dissent or fail epistemic gating—achieving near-fast-quorum `P95`/`P99` latency with full-verification consensus accuracy.

## Repository Layout

```text
quorumguard/
├── HANDOFF_CONTEXT.md   # Problem formulation, literature verification & experimental plan
├── pilot/               # Phase 0/1 local MPS 2-stage multi-agent consensus & cascade pilot
├── quorumguard/         # Core QuorumGuard adaptive quorum controller, genealogy & scheduler library
├── benchmarks/          # Full evaluation suite (local MPS + CloudTop vLLM A100/L4 scale-up)
├── paper/               # Manuscript (.tex/.bib), Theorems 1-3 numerical verification & vector figures
└── tests/               # Unit tests for quorum stopping rules, genealogy graphs & priority budgets
```

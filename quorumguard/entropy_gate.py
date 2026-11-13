"""
Epistemic Confidence & Token-Level Semantic Entropy Gating.
===========================================================
Evaluates calibrated per-agent epistemic confidence weights w_i from:
  1. Candidate-hypothesis posterior Shannon entropy H_ans,
  2. Top-2 decision logit margin Delta = z_(1) - z_(2),
  3. Reasoning-trace token predictive entropy H_seq, and
  4. Deterministic tool-execution verification certificates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EpistemicSignals:
    option_entropy_nats: float
    num_options: int = 4
    logit_margin: float = 2.0
    seq_token_entropy: float = 0.25
    tool_verified: bool = False
    has_cot_trace: bool = False

    @property
    def normalized_option_entropy(self) -> float:
        denom = math.log(max(2, self.num_options))
        return min(1.0, max(0.0, self.option_entropy_nats / denom))


def compute_epistemic_weight(
    signals: EpistemicSignals,
    margin_scale: float = 3.0,
    unverified_shallow_discount: float = 0.84,
    tool_verified_bonus: float = 1.25,
    max_weight: float = 1.15,
) -> float:
    """
    Map raw logit/entropy & verification signals into calibrated vote weight w_i.
    """
    norm_ent = signals.normalized_option_entropy
    ent_certainty = 1.0 - norm_ent
    margin_certainty = math.tanh(max(0.0, signals.logit_margin) / max(1e-6, margin_scale))
    seq_penalty = 1.0 / (1.0 + 0.12 * max(0.0, signals.seq_token_entropy - 0.6))

    base = (0.55 * ent_certainty + 0.45 * margin_certainty) * seq_penalty
    if signals.tool_verified:
        mult = tool_verified_bonus
    elif signals.has_cot_trace:
        mult = 1.00
    else:
        mult = unverified_shallow_discount

    return min(max_weight, max(0.05, base * mult))


def posterior_entropy_and_margin_from_logits(logits: Sequence[float]) -> tuple[float, float, list[float]]:
    """
    Compute exact Shannon entropy (in nats), top-2 logit margin, and normalized
    probabilities from a vector of candidate option logits.
    """
    if not logits:
        raise ValueError("logits must be non-empty")
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    z_sum = sum(exps)
    probs = [e / z_sum for e in exps]
    ent = -sum(p * math.log(max(p, 1e-15)) for p in probs)
    s_logits = sorted(logits, reverse=True)
    margin = (s_logits[0] - s_logits[1]) if len(s_logits) >= 2 else float("inf")
    return ent, margin, probs

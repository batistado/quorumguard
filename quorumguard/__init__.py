"""QuorumGuard: Claim-Genealogy-Gated Hedged Quorums for Asynchronous Multi-Agent LLM Consensus."""

from .entropy_gate import (
    EpistemicSignals,
    compute_epistemic_weight,
    posterior_entropy_and_margin_from_logits,
)
from .genealogy import ProvenanceDAG, ProvenanceNode
from .scheduler import (
    VLLM_PRIORITY_BACKGROUND_SPECULATIVE,
    VLLM_PRIORITY_DISSENT_BOOST,
    VLLM_PRIORITY_NORMAL,
    AgentVote,
    PriorityBudgetPool,
    QuorumGuardConfig,
    QuorumGuardController,
    StageCommitResult,
)

__all__ = [
    "AgentVote",
    "EpistemicSignals",
    "PriorityBudgetPool",
    "ProvenanceDAG",
    "ProvenanceNode",
    "QuorumGuardConfig",
    "QuorumGuardController",
    "StageCommitResult",
    "VLLM_PRIORITY_BACKGROUND_SPECULATIVE",
    "VLLM_PRIORITY_DISSENT_BOOST",
    "VLLM_PRIORITY_NORMAL",
    "compute_epistemic_weight",
    "posterior_entropy_and_margin_from_logits",
]
__version__ = "0.1.0"

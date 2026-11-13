"""
Claim-Genealogy Independence & Provenance Lineage Tracking.
===========================================================
Prevents correlated upstream context echoes from masquerading as independent
fast-quorum consensus votes. Given a provenance DAG G = (V, E) linking upstream
retrieval chunks, triage summaries, and agent responses, computes each agent's
minimal upstream provenance root set and evaluates genealogy-deduplicated
effective quorum support N_eff(t, c; G).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class ProvenanceNode:
    node_id: str
    kind: str = "source"  # "source", "summary", "agent"
    parents: Tuple[str, ...] = ()


class ProvenanceDAG:
    """
    Tracks information lineage from raw independent sources/verifiers to agent votes.

    If two fast agents consume the same intermediate summary `S_1` derived from
    source `R_0`, both trace back to `{R_0}` and collectively contribute at most
    one source root's epistemic capacity toward early quorum closure.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ProvenanceNode] = {}
        self._root_cache: Dict[str, FrozenSet[str]] = {}

    def add_node(self, node_id: str, kind: str = "source", parents: Sequence[str] = ()) -> None:
        self._nodes[node_id] = ProvenanceNode(
            node_id=node_id, kind=kind, parents=tuple(dict.fromkeys(parents))
        )
        self._root_cache.clear()

    def register_agent_lineage(self, agent_node_id: str, upstream_sources: Sequence[str]) -> None:
        for src in upstream_sources:
            if src not in self._nodes:
                self._nodes[src] = ProvenanceNode(node_id=src, kind="source", parents=())
        self.add_node(agent_node_id, kind="agent", parents=upstream_sources)

    def resolve_roots(self, node_id: str) -> FrozenSet[str]:
        """Return minimal ancestor nodes (indegree 0) in the provenance DAG."""
        if node_id in self._root_cache:
            return self._root_cache[node_id]
        if node_id not in self._nodes or not self._nodes[node_id].parents:
            res = frozenset([node_id])
            self._root_cache[node_id] = res
            return res

        visited: Set[str] = set()
        roots: Set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            node = self._nodes.get(cur)
            if node is None or not node.parents:
                roots.add(cur)
            else:
                for p in node.parents:
                    stack.append(p)
        res = frozenset(roots)
        self._root_cache[node_id] = res
        return res

    def compute_effective_support(
        self,
        votes: Sequence[Tuple[str, str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        """
        Compute genealogy-deduplicated effective support N_eff(c; G) and distinct
        independent root counts per candidate claim `c`.

        Args:
            votes: Sequence of `(agent_node_id, claim, calibrated_weight)` tuples.

        Returns:
            `(claim_weights, claim_distinct_roots)` where each provenance root `r`
            can contribute at most unit capacity across agents sharing `r`. When an
            agent depends on `m` roots `{r_1, ..., r_m}`, its weight `w_i` is split
            equally (`w_i / m`) across its supporting roots before per-root max
            aggregation, preserving submodularity and single-root exactness.
        """
        root_claim_cap: Dict[Tuple[str, str], float] = {}
        claim_root_sets: Dict[str, Set[str]] = {}

        for agent_id, claim, weight in votes:
            if weight <= 0.0:
                continue
            roots = self.resolve_roots(agent_id)
            share = float(weight) / max(1, len(roots))
            claim_root_sets.setdefault(claim, set())
            for r in roots:
                key = (r, claim)
                root_claim_cap[key] = max(root_claim_cap.get(key, 0.0), share)
                claim_root_sets[claim].add(r)

        claim_weights: Dict[str, float] = {}
        for (_, claim), cap in root_claim_cap.items():
            claim_weights[claim] = claim_weights.get(claim, 0.0) + cap

        claim_distinct_roots: Dict[str, int] = {
            c: len(rset) for c, rset in claim_root_sets.items()
        }
        return claim_weights, claim_distinct_roots

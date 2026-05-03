"""DualMapScheduler — dual-hash affinity + semantic-hit-rate-weighted routing (Activity A).

Each request is mapped to two candidate nodes via independent hash functions.
The node with the higher routing_score = semantic_hit_rate × (1 - load) is selected.
SLO violation and long-wait fairness both fall back to load-only routing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.engine.runner import InferenceRequest


@dataclass
class NodeState:
    node_id: str
    cache: object  # SemanticSegmentCache — avoid circular import with string type
    current_load: float = 0.0
    slo_violation: bool = False
    _wait_steps: Dict[str, int] = field(default_factory=dict)


class DualMapScheduler:
    """Routes requests to cache nodes using dual-hash + semantic-hit-rate scoring.

    Routing score formula:
        score = semantic_hit_rate(req_emb, node) × (1 - node.current_load)

    Degrades to load-only scoring when:
        - The selected node has an active SLO violation, OR
        - The request's wait_steps >= fairness_max_wait (starvation protection).

    Single-node mode (len(nodes)==1): h1 and h2 both map to the same node, which is
    explicitly allowed per Spec §10.
    """

    def __init__(
        self,
        nodes: List[NodeState],
        slo_ttft_ms: float = 200.0,
        top_k_semantic: int = 5,
        fairness_max_wait: int = 10,
        hash_seed_1: int = 2654435761,
        hash_seed_2: int = 1234567891,
    ) -> None:
        self.nodes = nodes
        self.slo_ttft_ms = slo_ttft_ms
        self.top_k_semantic = top_k_semantic
        self.fairness_max_wait = fairness_max_wait
        self.hash_seed_1 = hash_seed_1
        self.hash_seed_2 = hash_seed_2
        # Maps node_id → NodeState for O(1) lookup
        self._node_map: Dict[str, NodeState] = {n.node_id: n for n in nodes}
        # Per-request wait step counters
        self._wait_steps: Dict[str, int] = {}

    def _node_index_h1(self, request_id: str) -> int:
        """Primary hash: seed1 XOR hash(request_id) mod num_nodes."""
        raw = hash(request_id) & 0xFFFFFFFF
        return (self.hash_seed_1 ^ raw) % len(self.nodes)

    def _node_index_h2(self, request_id: str) -> int:
        """Secondary hash: guaranteed different from h1 when len(nodes) > 1."""
        raw = hash(request_id) & 0xFFFFFFFF
        idx = (self.hash_seed_2 ^ raw) % len(self.nodes)
        # Ensure two distinct candidates when multiple nodes exist
        h1 = self._node_index_h1(request_id)
        if len(self.nodes) > 1 and idx == h1:
            idx = (idx + 1) % len(self.nodes)
        return idx

    def _semantic_hit_score(
        self,
        request_embedding: torch.Tensor,
        node: NodeState,
    ) -> float:
        """Compute mean cosine similarity of request embedding vs node's cached embeddings.

        Reads _semantic_index directly (no cache.get() call) to avoid stats pollution.
        Returns 0.0 when the node's cache index is empty.
        """
        semantic_index = getattr(node.cache, "_semantic_index", [])
        if not semantic_index:
            return 0.0

        emb_matrix = torch.stack([emb for _, emb in semantic_index])
        q_norm = F.normalize(request_embedding.unsqueeze(0).float(), dim=-1)
        e_norm = F.normalize(emb_matrix.float(), dim=-1)
        sims = (q_norm @ e_norm.T).squeeze(0)

        actual_k = min(self.top_k_semantic, len(semantic_index))
        top_sims, _ = sims.topk(actual_k)
        return float(top_sims.mean().item())

    def _request_embedding(self, request: InferenceRequest) -> torch.Tensor:
        """Approximate request embedding from token IDs (no external model required).

        Uses a deterministic pseudo-random vector seeded by the mean token id so that
        semantically-similar prompts (with similar vocabularies) produce nearby embeddings.
        """
        d_head = 64
        # Determine d_head from any available node cache index
        for node in self.nodes:
            idx = getattr(node.cache, "_semantic_index", [])
            if idx:
                d_head = idx[0][1].shape[-1]
                break

        token_ids = request.token_ids
        token_mean = float(sum(token_ids)) / max(1, len(token_ids))
        g = torch.Generator()
        g.manual_seed(int(token_mean) & 0xFFFFFFFF)
        raw = torch.randn(d_head, generator=g)
        return F.normalize(raw, dim=-1)

    def route(self, request: InferenceRequest) -> str:
        """Select a target node_id for the request.

        Selection logic (in priority order):
        1. Fairness override — wait_steps >= fairness_max_wait → load-only scoring.
        2. SLO-safe path — both candidates healthy → semantic-hit × (1-load) scoring.
        3. SLO fallback — any violating candidate → load-only scoring.
        """
        wait_steps = self._wait_steps.get(request.request_id, 0)
        fairness_override = wait_steps >= self.fairness_max_wait

        idx1 = self._node_index_h1(request.request_id)
        idx2 = self._node_index_h2(request.request_id)
        candidates = [self.nodes[idx1], self.nodes[idx2]]

        any_slo_violation = any(n.slo_violation for n in candidates)

        if fairness_override or any_slo_violation:
            # Load-only: pick least-loaded candidate
            chosen = min(candidates, key=lambda n: n.current_load)
        else:
            req_emb = self._request_embedding(request)
            scored: List[Tuple[float, NodeState]] = []
            for node in candidates:
                sem_score = self._semantic_hit_score(req_emb, node)
                routing_score = sem_score * (1.0 - node.current_load)
                scored.append((routing_score, node))
            chosen = max(scored, key=lambda t: t[0])[1]

        return chosen.node_id

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Annotate each request with target_node_id and sort for cache locality.

        Requests routed to the same node are batched together (sorted by node_id)
        to maximise sequential cache reuse within each node.
        """
        for req in requests:
            target = self.route(req)
            req.target_node_id = target  # type: ignore[attr-defined]

        # Sort by target_node_id so same-node requests are contiguous
        requests.sort(key=lambda r: getattr(r, "target_node_id", ""))

        # Advance wait counters for requests not yet processed (all were given a node this round)
        processed_ids = {r.request_id for r in requests}
        for rid in list(self._wait_steps.keys()):
            if rid not in processed_ids:
                self._wait_steps[rid] += 1
        # Reset wait counter for requests that just got scheduled
        for req in requests:
            self._wait_steps.pop(req.request_id, None)

        return requests

    def update_load(self, node_id: str, load: float) -> None:
        """Update current load for a node (0.0–1.0)."""
        node = self._node_map.get(node_id)
        if node is not None:
            node.current_load = float(load)

    def update_slo_status(self, node_id: str, violated: bool) -> None:
        """Update SLO violation flag for a node."""
        node = self._node_map.get(node_id)
        if node is not None:
            node.slo_violation = violated

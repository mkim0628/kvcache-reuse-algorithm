"""TriangleInequalitySegmentIndex — LycheeCluster (arXiv 2603.08453) based O(log N) KV segment index.

Activity B: Recursive hierarchical index with triangle inequality pruning.
Wraps a backend CacheStore (e.g. StaticDynamicSegmentCache) and adds an index layer.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# Module-level counter used as tie-breaker in heap entries so that _IndexNode
# objects are never compared directly (they don't implement __lt__).
_counter = itertools.count()

from src.cache.base import CacheStore


@dataclass
class _IndexNode:
    """Single node in the hierarchical segment index."""

    center_key: str
    center_embedding: torch.Tensor  # [d_embed]
    max_radius: float
    left: Optional["_IndexNode"] = None
    right: Optional["_IndexNode"] = None
    segment_keys: List[str] = field(default_factory=list)  # leaf-only


class TriangleInequalitySegmentIndex(CacheStore):
    """
    LycheeCluster (arXiv 2603.08453) based recursive hierarchical KV segment index.

    - Triangle inequality pruning: d(q,c) - max_radius(node) > best_dist → skip subtree
    - Theoretical search complexity: O(N) → O(log N)
    - Backend: any CacheStore instance (e.g. StaticDynamicSegmentCache)
    - Full CacheStore interface: put/get/evict/hit_rate/memory_bytes/reset_stats
    """

    def __init__(
        self,
        backend_cache: CacheStore,
        embedding_dim: int = 64,
        leaf_size: int = 8,
        distance_fn: str = "cosine",
    ) -> None:
        self._backend = backend_cache
        self._embedding_dim = embedding_dim
        self._leaf_size = leaf_size
        self._distance_fn = distance_fn

        self._embeddings: Dict[str, torch.Tensor] = {}  # key → [d_embed]
        self._root: Optional[_IndexNode] = None
        self._dirty: bool = False

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store segment in backend + extract embedding + mark index dirty."""
        self._backend.put(key, value)
        self._embeddings[key] = self._extract_embedding(value)
        self._dirty = True

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Direct key lookup bypassing index (use when exact key is known)."""
        return self._backend.get(key)

    def evict(self) -> int:
        """Evict from backend + remove corresponding embedding from index."""
        freed = self._backend.evict()
        current_keys = set(self._get_backend_keys())
        evicted_keys = set(self._embeddings.keys()) - current_keys
        for k in evicted_keys:
            self._embeddings.pop(k, None)
        if evicted_keys:
            self._dirty = True
        return freed

    def hit_rate(self) -> float:
        return self._backend.hit_rate()

    def memory_bytes(self) -> int:
        embedding_bytes = sum(e.nbytes for e in self._embeddings.values())
        return self._backend.memory_bytes() + embedding_bytes

    def reset_stats(self) -> None:
        self._backend.reset_stats()

    # ------------------------------------------------------------------ #
    # Core search API                                                      #
    # ------------------------------------------------------------------ #

    def search_nearest(
        self,
        query_embedding: torch.Tensor,  # [d_embed]
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        Best-first search with triangle inequality pruning for top_k nearest segments.

        Returns: [(key, distance), ...] sorted by ascending distance.

        Algorithm:
        1. Rebuild index if dirty
        2. Push root node onto min-heap (priority = lower bound distance)
        3. Pop minimum node; if leaf, score segment keys; if internal, expand children
           Triangle inequality pruning: lower_bound = d(q,c) - max_radius > best_so_far → skip
        4. Collect top_k results
        """
        if self._dirty or self._root is None:
            self._rebuild_index()
        if self._root is None:
            return []

        # Max-heap of size top_k: store (-distance, key) so smallest distance at top
        results: List[Tuple[float, str]] = []  # (-dist, key) max-heap
        pq: List[Tuple[float, int, _IndexNode]] = []
        heapq.heappush(pq, (0.0, next(_counter), self._root))

        while pq:
            lb_dist, _, node = heapq.heappop(pq)

            best_so_far = -results[0][0] if len(results) >= top_k else float("inf")
            if lb_dist > best_so_far:
                break

            if node.left is None and node.right is None:
                # Leaf node: compute exact distances for all segment keys
                for seg_key in node.segment_keys:
                    if seg_key in self._embeddings:
                        dist = self._distance(query_embedding, self._embeddings[seg_key])
                        if dist <= max_distance:
                            if len(results) < top_k:
                                heapq.heappush(results, (-dist, seg_key))
                            elif dist < -results[0][0]:
                                heapq.heapreplace(results, (-dist, seg_key))
            else:
                # Internal node: enqueue children with triangle inequality lower bounds
                best_so_far = -results[0][0] if len(results) >= top_k else float("inf")
                for child in (node.left, node.right):
                    if child is None:
                        continue
                    child_dist = self._distance(query_embedding, child.center_embedding)
                    lower_bound = max(0.0, child_dist - child.max_radius)
                    if lower_bound <= best_so_far:
                        heapq.heappush(pq, (lower_bound, next(_counter), child))

        # Sort results by ascending distance
        final = [(-neg_d, k) for neg_d, k in results]
        final.sort(key=lambda x: x[0])
        return [(k, d) for d, k in final[:top_k]]

    def estimate_hit_probability(
        self,
        query_segments: List[torch.Tensor],  # list of KV tensors for query
        threshold_distance: float = 0.3,
    ) -> float:
        """
        O(log N) hit probability estimate for Turn 2+ requests.
        Used by PPDAppendPrefillRouter for D/P node routing.

        Returns fraction of query segments with a cached neighbor within threshold_distance.
        """
        if not query_segments or not self._embeddings:
            return 0.0
        hits = 0
        for seg_tensor in query_segments:
            query_emb = self._extract_embedding(seg_tensor)
            nearest = self.search_nearest(query_emb, top_k=1, max_distance=threshold_distance)
            if nearest and nearest[0][1] <= threshold_distance:
                hits += 1
        return hits / len(query_segments)

    # ------------------------------------------------------------------ #
    # Index reconstruction                                                 #
    # ------------------------------------------------------------------ #

    def _rebuild_index(self) -> None:
        """Rebuild recursive hierarchical index from all stored embeddings."""
        keys = list(self._embeddings.keys())
        if not keys:
            self._root = None
            self._dirty = False
            return
        self._root = self._build_node(keys)
        self._dirty = False

    def _build_node(self, keys: List[str]) -> _IndexNode:
        """Recursively build an index node over the given key set."""
        if len(keys) <= self._leaf_size:
            embeddings = torch.stack([self._embeddings[k] for k in keys])
            center_emb = embeddings.mean(dim=0)
            dists = [self._distance(center_emb, self._embeddings[k]) for k in keys]
            center_key = keys[int(torch.tensor(dists).argmin().item())]
            max_radius = max(dists) if dists else 0.0
            return _IndexNode(
                center_key=center_key,
                center_embedding=center_emb,
                max_radius=max_radius,
                segment_keys=keys,
            )

        # Select the two farthest segments as pivots to split the space
        pivot_a_key, pivot_b_key = self._select_pivots(keys)
        emb_a = self._embeddings[pivot_a_key]
        emb_b = self._embeddings[pivot_b_key]

        left_keys: List[str] = []
        right_keys: List[str] = []
        for k in keys:
            d_a = self._distance(self._embeddings[k], emb_a)
            d_b = self._distance(self._embeddings[k], emb_b)
            if d_a <= d_b:
                left_keys.append(k)
            else:
                right_keys.append(k)

        # Guarantee both halves are non-empty
        if not left_keys or not right_keys:
            half = len(keys) // 2
            left_keys, right_keys = keys[:half], keys[half:]

        all_embs = torch.stack([self._embeddings[k] for k in keys])
        center_emb = all_embs.mean(dim=0)
        dists_all = [self._distance(center_emb, self._embeddings[k]) for k in keys]
        center_key = keys[int(torch.tensor(dists_all).argmin().item())]
        max_radius = max(dists_all) if dists_all else 0.0

        node = _IndexNode(
            center_key=center_key,
            center_embedding=center_emb,
            max_radius=max_radius,
        )
        node.left = self._build_node(left_keys)
        node.right = self._build_node(right_keys)
        return node

    def _select_pivots(self, keys: List[str]) -> Tuple[str, str]:
        """Return the two farthest keys (O(N²) over up to 64 sampled keys)."""
        sample = keys[:64]
        max_dist = -1.0
        pivot_a, pivot_b = sample[0], sample[-1]
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                d = self._distance(self._embeddings[sample[i]], self._embeddings[sample[j]])
                if d > max_dist:
                    max_dist = d
                    pivot_a, pivot_b = sample[i], sample[j]
        return pivot_a, pivot_b

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Distance between two embeddings (cosine or euclidean)."""
        a_f = a.float()
        b_f = b.float()
        if self._distance_fn == "cosine":
            cos_sim = torch.nn.functional.cosine_similarity(
                a_f.unsqueeze(0), b_f.unsqueeze(0)
            ).item()
            return float(1.0 - cos_sim)
        else:
            return float(torch.norm(a_f - b_f).item())

    def _extract_embedding(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        """Extract embedding from a KV tensor: mean key vector [d_embed]."""
        if kv_tensor.dim() == 1:
            flat = kv_tensor.float()
        else:
            flat = kv_tensor.float().mean(dim=0)

        # Resize to embedding_dim if needed
        d = self._embedding_dim
        if flat.shape[0] >= d:
            return flat[:d]
        # Pad with zeros if smaller
        padded = torch.zeros(d, dtype=torch.float32)
        padded[: flat.shape[0]] = flat
        return padded

    def _get_backend_keys(self) -> List[str]:
        """Get current keys from the backend cache (StaticDynamicSegmentCache compatible)."""
        backend = self._backend
        keys: List[str] = []
        if hasattr(backend, "_static_store"):
            keys.extend(backend._static_store.keys())
        if hasattr(backend, "_dynamic_store"):
            keys.extend(backend._dynamic_store.keys())
        # Fallback for other backends with _store attribute
        if not keys and hasattr(backend, "_store"):
            keys.extend(backend._store.keys())
        return keys

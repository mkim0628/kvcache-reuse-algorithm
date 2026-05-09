"""SemanticBoundarySegmentCache — SemantiCache (arXiv 2603.14303) based semantic chunking.

Activity B: Semantic boundary detection + GSC (Greedy Seed-based Clustering) +
proportional attention weighting. Integrates with TriangleInequalitySegmentIndex.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore


class SemanticBoundarySegmentCache(CacheStore):
    """
    SemantiCache (arXiv 2603.14303) based semantic boundary chunking + GSC clustering.

    - Semantic boundary detection: delimiters (.!? + \\n\\n + code blocks ```)
    - GSC (Greedy Seed-based Clustering): top attention score seeds + greedy merge
    - Proportional attention: attention_weight_core = sum(cluster attention weights) (NOT mean)

    Full CacheStore interface implementation.
    Integrates with TriangleInequalitySegmentIndex: semantic core embeddings improve index precision.
    """

    # Semantic boundary pattern (compiled constant)
    _BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=\n\n)|(?<=```)")

    def __init__(
        self,
        capacity_bytes: int,
        min_cluster_size: int = 3,
        max_merge_ratio: float = 0.7,
        attention_threshold: float = 0.1,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.min_cluster_size = min_cluster_size
        self.max_merge_ratio = max_merge_ratio
        self.attention_threshold = attention_threshold

        self._store: Dict[str, torch.Tensor] = {}
        self._lru_order: List[str] = []
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store semantic core KV tensor (size-reduced after GSC merging)."""
        self._store[key] = value
        if key in self._lru_order:
            self._lru_order.remove(key)
        self._lru_order.append(key)
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._hit_count += 1
            self._lru_order.remove(key)
            self._lru_order.append(key)
            return self._store[key]
        self._miss_count += 1
        return None

    def evict(self) -> int:
        """Evict the LRU entry."""
        if not self._lru_order:
            return 0
        evict_key = self._lru_order.pop(0)
        kv = self._store.pop(evict_key, None)
        return kv.nbytes if kv is not None else 0

    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------ #
    # Semantic boundary + GSC API                                          #
    # ------------------------------------------------------------------ #

    def detect_semantic_boundaries(self, text: str) -> List[int]:
        """
        Return boundary positions (character index estimates) from semantic delimiters.
        Delimiters: sentence end (.!?), paragraph (\\n\\n), code block (```)
        """
        positions = [0]
        for match in self._BOUNDARY_PATTERN.finditer(text):
            positions.append(match.start())
        return sorted(set(positions))

    def apply_gsc_clustering(
        self,
        kv_tensor: torch.Tensor,          # [n_tokens, d_head]
        attention_scores: torch.Tensor,   # [n_tokens]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GSC clustering to produce semantic core KV.

        Steps:
        1. Select seed tokens: top (1 - max_merge_ratio) fraction by attention score
        2. Assign each non-seed token to the nearest seed cluster (cosine similarity)
        3. Merge cluster: core_kv = sum(attn_t * kv_t) / sum(attn_t) for t in cluster
        4. Proportional attention: attention_weight_core = sum(attn weights in cluster)

        Returns: (core_kv [n_seeds, d_head], core_attention_weights [n_seeds])
        """
        n_tokens = kv_tensor.shape[0]
        if n_tokens == 0:
            return kv_tensor.clone(), attention_scores.clone()

        d_head = kv_tensor.shape[1] if kv_tensor.dim() > 1 else kv_tensor.shape[0]
        max_seeds = max(1, int(n_tokens * (1 - self.max_merge_ratio)))

        # Seed selection: top tokens by attention score
        k_seeds = min(max_seeds, n_tokens)
        _, seed_indices = torch.topk(attention_scores, k=k_seeds)
        seed_indices = seed_indices.sort().values  # maintain order

        seed_kv = kv_tensor[seed_indices]          # [n_seeds, d_head]
        seed_attn = attention_scores[seed_indices]  # [n_seeds]

        # Initialize cluster accumulators with seed contributions
        cluster_sums = seed_kv.clone() * seed_attn.unsqueeze(-1)  # [n_seeds, d_head]
        cluster_attn_sums = seed_attn.clone()                     # [n_seeds]

        seed_set = set(seed_indices.tolist())
        for t in range(n_tokens):
            if t in seed_set:
                continue
            # Assign to nearest seed by cosine similarity
            sims = torch.nn.functional.cosine_similarity(
                kv_tensor[t].unsqueeze(0), seed_kv
            )
            nearest_seed_idx = int(sims.argmax().item())

            cluster_sums[nearest_seed_idx] += kv_tensor[t] * attention_scores[t]
            # Proportional attention: sum (not mean)
            cluster_attn_sums[nearest_seed_idx] += attention_scores[t]

        # Weighted average KV per cluster
        core_kv = cluster_sums / cluster_attn_sums.unsqueeze(-1).clamp(min=1e-9)
        core_attn_weights = cluster_attn_sums  # proportional attention sums

        return core_kv, core_attn_weights

    def put_with_gsc(
        self,
        key: str,
        kv_tensor: torch.Tensor,          # [n_tokens, d_head]
        attention_scores: torch.Tensor,   # [n_tokens]
    ) -> None:
        """Apply GSC clustering and store only the semantic core KV."""
        core_kv, _ = self.apply_gsc_clustering(kv_tensor, attention_scores)
        self.put(key, core_kv)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _maybe_evict(self) -> None:
        """Evict until memory usage is within capacity."""
        while self.memory_bytes() > self.capacity_bytes and self._lru_order:
            self.evict()

"""ManifoldKVWindowedEviction — Euclidean outlier-based sliding-window KV eviction.

Activity C: ManifoldKV (arXiv 2602.08343) inspired windowed eviction policy.
Cosine-similarity-based eviction ignores token magnitude, discarding semantically
important outlier tokens. Euclidean distance to the local windowed centroid preserves
scale information and better identifies important tokens to retain.

Drop-in replacement for LRU or cosine-based eviction via outlier_score_fn injection.
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore


class ManifoldKVWindowedEviction(CacheStore):
    """Sliding-window Euclidean outlier score KV cache eviction.

    Segments with low outlier scores (close to local centroid = less distinctive)
    are evicted first. High-outlier segments are retained as semantically important.
    """

    def __init__(
        self,
        capacity_bytes: int,
        window_size: int = 4096,
        evict_ratio: float = 0.2,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.window_size = window_size
        self.evict_ratio = evict_ratio

        self._store: Dict[str, torch.Tensor] = {}
        # Separate key vectors for score computation to support compressed value stores
        self._key_vectors: Dict[str, torch.Tensor] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV tensor. Uses value as key vector for outlier scoring."""
        self._store[key] = value
        self._key_vectors[key] = value
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._hit_count += 1
            return self._store[key]
        self._miss_count += 1
        return None

    def evict(self) -> int:
        """Evict the segment with the lowest Euclidean outlier score."""
        if not self._store:
            return 0
        scores = self._compute_outlier_scores()
        if not scores:
            return 0
        worst_key = min(scores, key=lambda k: scores[k])
        kv = self._store.pop(worst_key, None)
        self._key_vectors.pop(worst_key, None)
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
    # Euclidean outlier score computation                                  #
    # ------------------------------------------------------------------ #

    def _compute_outlier_scores(self) -> Dict[str, float]:
        """Compute per-segment Euclidean outlier scores using sliding window centroids.

        Algorithm:
        1. Concatenate all K vectors → [total_tokens, d_head].
        2. For each window, compute local centroid and Euclidean distance per token.
        3. Segment score = mean of its token distances (higher = more distinctive = keep).
        """
        scores: Dict[str, float] = {}
        keys_list = list(self._key_vectors.keys())
        if not keys_list:
            return scores

        all_kvs: List[torch.Tensor] = []
        seg_ranges: List[Tuple[str, int, int]] = []
        cursor = 0
        for key in keys_list:
            kv = self._key_vectors[key]
            if kv.dim() == 1:
                kv = kv.unsqueeze(0)
            all_kvs.append(kv.float())
            seg_ranges.append((key, cursor, cursor + kv.shape[0]))
            cursor += kv.shape[0]

        if not all_kvs:
            return scores

        all_k = torch.cat(all_kvs, dim=0)
        total_tokens = all_k.shape[0]

        token_scores = torch.zeros(total_tokens)
        for win_start in range(0, total_tokens, self.window_size):
            win_end = min(win_start + self.window_size, total_tokens)
            window_k = all_k[win_start:win_end]
            centroid = window_k.mean(dim=0, keepdim=True)
            # Euclidean distance from each token to local centroid
            dists = torch.cdist(window_k, centroid).squeeze(-1)
            token_scores[win_start:win_end] = dists

        for key, start, end in seg_ranges:
            scores[key] = token_scores[start:end].mean().item()

        return scores

    def outlier_score_fn(self, key_vectors: torch.Tensor) -> torch.Tensor:
        """Compute per-token outlier scores for external drop-in use.

        Args:
            key_vectors: [n_tokens, d_head] tensor.
        Returns:
            [n_tokens] tensor of Euclidean outlier scores (higher = more important).
        """
        kv = key_vectors.float()
        n_tokens = kv.shape[0]
        token_scores = torch.zeros(n_tokens)
        for win_start in range(0, n_tokens, self.window_size):
            win_end = min(win_start + self.window_size, n_tokens)
            window_k = kv[win_start:win_end]
            centroid = window_k.mean(dim=0, keepdim=True)
            dists = torch.cdist(window_k, centroid).squeeze(-1)
            token_scores[win_start:win_end] = dists
        return token_scores

    def _maybe_evict(self) -> None:
        while self.memory_bytes() > self.capacity_bytes and self._store:
            self.evict()

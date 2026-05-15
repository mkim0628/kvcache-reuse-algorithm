"""Activity B: RelayUShapeLayerSelectiveSegmentCache.

Non-contiguous segment cache with U-shape layer-selective reuse based on
RelayCaching (arXiv 2603.13289). Even non-identical segments can reuse
middle-layer KV (low deviation) while boundary layers are recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import yaml

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class RelayULayerConfig:
    chunk_size: int = 128
    max_entries: int = 1000
    n_layers: int = 12
    n_heads: int = 8
    d_head: int = 64
    similarity_threshold: float = 0.95
    profile_path: str = "configs/relay_ulayer_profile.yaml"
    seed: int = 42


@dataclass
class LayerReuseProfile:
    """Offline profiling result: per-layer reuse eligibility."""

    n_layers: int
    reuse_layer_indices: List[int]
    boundary_layer_indices: List[int]
    similarity_scores: List[float]

    @classmethod
    def from_yaml(cls, path: str) -> "LayerReuseProfile":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            n_layers=data["n_layers"],
            reuse_layer_indices=data["reuse_layer_indices"],
            boundary_layer_indices=data["boundary_layer_indices"],
            similarity_scores=data["similarity_scores"],
        )

    def to_bitmask(self) -> bytes:
        """n_layers bitmask: reusable layer = 1."""
        mask = 0
        for idx in self.reuse_layer_indices:
            mask |= 1 << idx
        return mask.to_bytes((self.n_layers + 7) // 8, byteorder="little")


class RelayUShapeLayerSelectiveSegmentCache(CacheStore):
    """Non-contiguous segment cache with U-shape layer-selective reuse.

    Even non-identical segments can reuse middle-layer KV (small deviation)
    while boundary layers (large deviation) are recomputed. Per-segment
    layer_reuse_mask (bitmask) is stored alongside KV to enable selective load.

    CacheStore interface fully implemented:
      put / get / evict / hit_rate / memory_bytes / reset_stats
    """

    def __init__(self, config: RelayULayerConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._base_cache = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._profile: Optional[LayerReuseProfile] = self._load_profile()
        # segment key → layer reuse bitmask
        self._layer_masks: Dict[str, bytes] = {}
        self._hits = 0
        self._misses = 0
        self._partial_reuse_hits = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store segment KV with all-layers-reusable mask (byte-identical path).

        value shape: [n_tokens, n_layers, 2, n_heads, d_head] or
                     [n_tokens, 2, n_heads, d_head] (single layer)
        """
        self._base_cache.put(key, value)
        # Byte-identical segment: all layers reusable
        all_reuse_mask = ((1 << self.config.n_layers) - 1).to_bytes(
            (self.config.n_layers + 7) // 8, byteorder="little"
        )
        self._layer_masks[key] = all_reuse_mask

    def get(self, key: str) -> Optional[torch.Tensor]:
        result = self._base_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        freed = self._base_cache.evict()
        return freed

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._base_cache.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._partial_reuse_hits = 0
        self._noncontiguous_hits = 0
        self._base_cache.reset_stats()

    # ------------------------------------------------------------------ #
    # Layer-selective reuse API                                            #
    # ------------------------------------------------------------------ #

    def put_with_layer_mask(
        self,
        key: str,
        value: torch.Tensor,
        layer_reuse_mask: bytes,
    ) -> None:
        """Store segment with explicit layer reuse bitmask."""
        self._base_cache.put(key, value)
        self._layer_masks[key] = layer_reuse_mask

    def get_with_layer_selection(
        self,
        key: str,
        target_layer_indices: Optional[List[int]] = None,
    ) -> Optional[Tuple[torch.Tensor, List[int], List[int]]]:
        """Load KV with per-layer reuse decision.

        Returns:
            (kv_tensor, reusable_layers, boundary_layers) or None on miss.
        """
        kv = self._base_cache.get(key)
        if kv is None:
            return None
        mask_bytes = self._layer_masks.get(key)
        reusable, boundary = self._decode_layer_mask(mask_bytes)
        if target_layer_indices is not None:
            target_set = set(target_layer_indices)
            reusable = [l for l in reusable if l in target_set]
            boundary = [l for l in boundary if l in target_set]
        return kv, reusable, boundary

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Store segment using content-hash key with profile-based layer mask."""
        key = self._chunk_key(token_ids, chunk_idx)
        if self._profile is not None:
            mask = self._profile.to_bitmask()
        else:
            mask = self._default_middle_layer_mask()
        self.put_with_layer_mask(key, kv, mask)

    def get_segments_layer_selective(
        self,
        token_ids: List[int],
        request_kv: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor, List[int], List[int]]], List[int]]:
        """Look up all chunks with per-layer reuse decisions.

        Returns:
            hits: [(chunk_idx, kv, reusable_layers, boundary_layers), ...]
            misses: [chunk_idx, ...]

        Non-contiguous hit tracking: if a hit follows any miss, it counts as
        non-contiguous.
        """
        n_chunks = max(
            1,
            (len(token_ids) + self.config.chunk_size - 1) // self.config.chunk_size,
        )
        hits: List[Tuple[int, torch.Tensor, List[int], List[int]]] = []
        misses: List[int] = []
        for i in range(n_chunks):
            key = self._chunk_key(token_ids, i)
            result = self.get_with_layer_selection(key)
            if result is not None:
                kv, reusable, boundary = result
                hits.append((i, kv, reusable, boundary))
                if any(m < i for m in misses):
                    self._noncontiguous_hits += 1
                    self._partial_reuse_hits += 1
            else:
                misses.append(i)
        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        total = self._hits
        if total == 0:
            return 0.0
        return self._noncontiguous_hits / total

    def partial_reuse_rate(self) -> float:
        """Fraction of hits that involved layer-selective (non-identical) reuse."""
        total = self._hits
        if total == 0:
            return 0.0
        return self._partial_reuse_hits / total

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_profile(self) -> Optional[LayerReuseProfile]:
        import os

        if os.path.exists(self.config.profile_path):
            return LayerReuseProfile.from_yaml(self.config.profile_path)
        return None

    def _decode_layer_mask(
        self,
        mask_bytes: Optional[bytes],
    ) -> Tuple[List[int], List[int]]:
        """Decode bitmask into (reusable_layers, boundary_layers)."""
        n = self.config.n_layers
        if mask_bytes is None:
            # Default: middle 70% reusable
            reusable = list(range(n // 6, n - n // 6))
            boundary = [i for i in range(n) if i not in set(reusable)]
            return reusable, boundary
        mask_int = int.from_bytes(mask_bytes, byteorder="little")
        reusable = [i for i in range(n) if (mask_int >> i) & 1]
        boundary = [i for i in range(n) if not ((mask_int >> i) & 1)]
        return reusable, boundary

    def _default_middle_layer_mask(self) -> bytes:
        """Default middle ~70% layers when no profile available."""
        n = self.config.n_layers
        start = n // 6
        end = n - n // 6
        mask = 0
        for i in range(start, end):
            mask |= 1 << i
        return mask.to_bytes((n + 7) // 8, byteorder="little")

    def _chunk_key(self, token_ids: List[int], chunk_idx: int) -> str:
        """Content-hash chunk key (delegates to SegmentedHashCache)."""
        return self._base_cache.chunk_key(token_ids, chunk_idx, layer_idx=0)

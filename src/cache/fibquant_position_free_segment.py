"""FibQuantPositionFreeSegmentCache — B+C Integration.

Cross-activity cache that combines:
  - Pre-RoPE position-decoupled storage (from RoPEReencodingNonContiguousCache)
  - Per-segment FibQuant VQ compression (from FibQuantVQSegmentCache)

KV tensors are stored before RoPE is applied (position-independent content hash),
FibQuant-compressed for memory efficiency, then decompressed and re-RoPE'd at
retrieval time for the target token positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.fibquant_vq_segment_cache import (
    FibQuantSegmentCacheConfig,
    FibQuantVQSegmentCache,
)
from src.cache.segmented import SegmentedHashCache


@dataclass
class FibQuantPositionFreeConfig:
    chunk_size: int = 64
    max_entries: int = 1000
    compression_target: float = 10.0
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 12
    bits_radial: int = 4
    bits_direction: int = 9
    rope_base: float = 10000.0
    seed: int = 42


class FibQuantPositionFreeSegmentCache(CacheStore):
    """B+C Integration: position-decoupled storage + FibQuant VQ compression.

    Stores KV in pre-RoPE form (position-independent content hash),
    then FibQuant-compresses each segment independently.
    On retrieval, decompresses and re-applies RoPE for the target positions.

    Combines logic from:
      - RoPEReencodingNonContiguousCache: pre-RoPE storage + RoPE re-application
      - FibQuantVQSegmentCache: per-segment FibQuant compression

    Implements the full CacheStore interface (put/get/evict/hit_rate/memory_bytes/reset_stats).
    Also overrides store_pre_rope() and load_with_rope() from CacheStore base.
    """

    def __init__(self, config: FibQuantPositionFreeConfig) -> None:
        self.config = config
        seg_cfg = FibQuantSegmentCacheConfig(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
            compression_target=config.compression_target,
            d_head=config.d_head,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            bits_radial=config.bits_radial,
            bits_direction=config.bits_direction,
            seed=config.seed,
        )
        self._compressed_cache = FibQuantVQSegmentCache(seg_cfg)
        # Lazy rotation-matrix cache: (position, d) -> [d//2, 2, 2]
        self._rope_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._key_helper = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=1,
        )

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store already-formed KV tensor (standard path, no position decoupling)."""
        self._compressed_cache.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve and decompress KV tensor (standard path, no RoPE adjustment)."""
        result = self._compressed_cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        """Delegate LRU eviction to the compressed inner cache."""
        return self._compressed_cache.evict()

    def hit_rate(self) -> float:
        """Cumulative cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Total bytes used by compressed code tensors."""
        return self._compressed_cache.memory_bytes()

    def reset_stats(self) -> None:
        """Reset hit/miss/noncontiguous counters for both this layer and inner cache."""
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._compressed_cache.reset_stats()

    # ------------------------------------------------------------------ #
    # RoPE-aware extension (CacheStore optional overrides)               #
    # ------------------------------------------------------------------ #

    def store_pre_rope(
        self,
        key: str,
        value: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] pre-RoPE
        layer_idx: int = 0,
    ) -> None:
        """Store pre-RoPE KV with FibQuant compression under a scoped content-hash key."""
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        self._compressed_cache.encode_segment(value, scoped_key, layer_idx)

    def load_with_rope(
        self,
        key: str,
        target_positions: torch.Tensor,  # [n_tokens] long
        layer_idx: int = 0,
        rope_dim: int = -1,
    ) -> Optional[torch.Tensor]:
        """Load FibQuant-compressed pre-RoPE KV, decompress, re-apply RoPE.

        Returns None on miss.
        """
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        pre_rope_kv = self._compressed_cache.decode_segment(scoped_key, layer_idx)
        if pre_rope_kv is None:
            self._misses += 1
            return None
        self._hits += 1
        return self._apply_rope(pre_rope_kv, target_positions, rope_dim)

    # ------------------------------------------------------------------ #
    # Segment-level API                                                    #
    # ------------------------------------------------------------------ #

    def put_segment_pre_rope(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Compress and store a chunk's pre-RoPE KV keyed by content hash."""
        key = self._key_helper.chunk_key(token_ids, chunk_idx, layer_idx)
        self.store_pre_rope(key, pre_rope_kv, layer_idx)

    def get_segments_with_rope(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Retrieve all chunks, decompress, apply RoPE for target_offset.

        Returns:
            hits:   [(chunk_idx, rope-applied kv tensor), ...]
            misses: [chunk_idx, ...]
        Non-contiguous tracking mirrors RoPEReencodingNonContiguousCache.
        """
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for chunk_idx in range(n_chunks):
            key = self._key_helper.chunk_key(token_ids, chunk_idx, layer_idx)
            start_tok = chunk_idx * chunk_size
            end_tok = min(start_tok + chunk_size, len(token_ids))
            positions = torch.arange(
                target_offset + start_tok, target_offset + end_tok, dtype=torch.long
            )
            kv = self.load_with_rope(key, positions, layer_idx)
            if kv is not None:
                hits.append((chunk_idx, kv))
                if any(m < chunk_idx for m in misses):
                    self._noncontiguous_hits += 1
            else:
                misses.append(chunk_idx)

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

    # ------------------------------------------------------------------ #
    # RoPE computation (identical logic to RoPEReencodingNonContiguousCache)
    # ------------------------------------------------------------------ #

    def _get_rope_rotation(self, position: int, d: int) -> torch.Tensor:
        """Return [d//2, 2, 2] rotation matrix for a single position."""
        cache_key = (position, d)
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]
        i = torch.arange(0, d, 2, dtype=torch.float32)
        theta = 1.0 / (self.config.rope_base ** (i / d))
        angle = position * theta
        rot = torch.stack(
            [
                torch.stack([angle.cos(), -angle.sin()], dim=-1),
                torch.stack([angle.sin(), angle.cos()], dim=-1),
            ],
            dim=-2,
        )  # [d//2, 2, 2]
        self._rope_cache[cache_key] = rot
        return rot

    def _apply_rope(
        self,
        pre_rope_kv: torch.Tensor,       # [n_tokens, 2, n_heads, d_head]
        target_positions: torch.Tensor,  # [n_tokens] long
        rope_dim: int = -1,
    ) -> torch.Tensor:
        """Apply position-specific RoPE to keys; values are unchanged."""
        n_tokens, _, n_heads, d_head = pre_rope_kv.shape
        d = d_head if rope_dim == -1 else rope_dim
        assert d % 2 == 0, f"rope_dim must be even, got {d}"

        result = pre_rope_kv.clone().float()
        key_slice = result[:, 0, :, :d]  # [n_tokens, n_heads, d]

        unique_pos = target_positions.unique().tolist()
        rot_by_pos: Dict[int, torch.Tensor] = {
            int(pos): self._get_rope_rotation(int(pos), d) for pos in unique_pos
        }

        for tok_idx in range(n_tokens):
            pos = int(target_positions[tok_idx].item())
            rot = rot_by_pos[pos]  # [d//2, 2, 2]
            k_tok = key_slice[tok_idx].reshape(n_heads, d // 2, 2)  # [n_heads, d//2, 2]
            k_tok_rot = torch.einsum("hpi,pij->hpj", k_tok, rot)
            key_slice[tok_idx] = k_tok_rot.reshape(n_heads, d)

        result[:, 0, :, :d] = key_slice
        return result.to(pre_rope_kv.dtype)

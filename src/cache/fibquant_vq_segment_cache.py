"""FibQuantVQSegmentCache — Non-contiguous segment cache with per-segment FibQuant compression.

Activity B: Each segment is encoded independently with FibQuantVQCodec, enabling
random-access restoration of any segment without decompressing the full cache.
Position-independent keying reuses SegmentedHashCache's content-hash approach.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.fibquant_vq_codec import FibQuantConfig, FibQuantVQCodec
from src.cache.segmented import SegmentedHashCache


@dataclass
class FibQuantSegmentCacheConfig:
    chunk_size: int = 64          # segment size in tokens
    max_entries: int = 1000       # maximum cached segments
    compression_target: float = 10.0   # target compression ratio (5/10/20x)
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 12
    bits_radial: int = 4
    bits_direction: int = 9
    seed: int = 42


class FibQuantVQSegmentCache(CacheStore):
    """Non-contiguous segment cache with per-segment FibQuant VQ compression.

    Each segment is encoded independently, enabling random-access restoration
    of any segment without decompressing the full cache.

    Uses SegmentedHashCache's content-hash keying (position-independent),
    so segments with identical token content at different positions share a cache entry.
    Satisfies the full CacheStore interface.
    """

    def __init__(self, config: FibQuantSegmentCacheConfig) -> None:
        self.config = config
        fibquant_cfg = FibQuantConfig(
            d_head=config.d_head,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            bits_radial=config.bits_radial,
            bits_direction=config.bits_direction,
            seed=config.seed,
        )
        self._codec = FibQuantVQCodec(fibquant_cfg)
        # key -> compressed dict (code tensors, not raw FP16)
        self._compressed_store: Dict[str, Dict] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        # Reuse SegmentedHashCache only for chunk_key computation
        self._key_helper = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=1,
        )

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store raw tensor with FibQuant compression (layer_idx=0 default)."""
        self._put_compressed(key, value, layer_idx=0)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve and FibQuant-decompress segment on-demand."""
        if key not in self._compressed_store:
            self._misses += 1
            return None
        self._lru.move_to_end(key)
        self._hits += 1
        compressed = self._compressed_store[key]
        layer_idx = compressed.get("layer_idx", 0)
        return self._codec.decode_segment(compressed, layer_idx)

    def evict(self) -> int:
        """LRU eviction of the oldest compressed segment. Returns bytes freed."""
        if not self._lru:
            return 0
        oldest_key, _ = self._lru.popitem(last=False)
        compressed = self._compressed_store.pop(oldest_key, {})
        freed = sum(
            v.nbytes for v in compressed.values()
            if isinstance(v, torch.Tensor)
        )
        return max(freed, 1)

    def hit_rate(self) -> float:
        """Cumulative cache hit rate (0.0–1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Sum of compressed code tensor bytes (not raw FP16 tensors)."""
        total = 0
        for compressed in self._compressed_store.values():
            total += sum(
                v.nbytes for v in compressed.values()
                if isinstance(v, torch.Tensor)
            )
        return total

    def reset_stats(self) -> None:
        """Reset hit/miss/noncontiguous counters."""
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # Segment-level API                                                    #
    # ------------------------------------------------------------------ #

    def encode_segment(
        self,
        segment_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        segment_id: str,
        layer_idx: int = 0,
    ) -> None:
        """FibQuant-compress and store a segment under segment_id."""
        compressed = self._codec.encode_segment(segment_kv, layer_idx, segment_id)
        if len(self._lru) >= self.config.max_entries and segment_id not in self._lru:
            self.evict()
        if segment_id in self._lru:
            self._lru.move_to_end(segment_id)
        else:
            self._lru[segment_id] = None
        self._compressed_store[segment_id] = compressed

    def decode_segment(
        self,
        segment_id: str,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Decompress and return segment KV on-demand. Returns None on miss."""
        if segment_id not in self._compressed_store:
            return None
        self._lru.move_to_end(segment_id)
        compressed = self._compressed_store[segment_id]
        actual_layer = compressed.get("layer_idx", layer_idx)
        return self._codec.decode_segment(compressed, actual_layer)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Content-hash keyed segment store with FibQuant compression."""
        key = self._key_helper.chunk_key(token_ids, chunk_idx, layer_idx)
        self.encode_segment(kv, key, layer_idx)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Look up all chunks for token_ids, decompress hits.

        Returns:
            hits:   [(chunk_idx, kv_tensor), ...]
            misses: [chunk_idx, ...]

        Non-contiguous tracking: a hit is counted as non-contiguous if any
        lower-indexed chunk was a miss.
        """
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            key = self._key_helper.chunk_key(token_ids, i, layer_idx)
            kv = self.get(key)
            if kv is not None:
                hits.append((i, kv))
                if any(m < i for m in misses):
                    self._noncontiguous_hits += 1
            else:
                misses.append(i)

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _put_compressed(
        self,
        key: str,
        value: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Compress value and store under key with LRU management."""
        if len(self._lru) >= self.config.max_entries and key not in self._lru:
            self.evict()
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = None
        compressed = self._codec.encode_segment(value, layer_idx, key)
        self._compressed_store[key] = compressed

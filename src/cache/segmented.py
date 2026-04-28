import hashlib
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch

from src.cache.base import CacheStore


class SegmentedHashCache(CacheStore):
    """Position-independent segmented hash cache (Activity B).

    Splits token sequences into fixed-size chunks and hashes each chunk
    independently of its position, enabling non-contiguous KV reuse.
    """

    def __init__(self, chunk_size: int = 128, max_entries: int = 1000) -> None:
        self.chunk_size = chunk_size
        self.max_entries = max_entries
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store a pre-keyed segment (used internally and by subclasses)."""
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_entries:
                self.evict()
            self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        _, evicted = self._store.popitem(last=False)
        return evicted.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # Segment-level API                                                    #
    # ------------------------------------------------------------------ #

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """Hash token IDs within a chunk, scoped to a layer.

        Position-independent: only token content (not position) determines the key,
        enabling reuse when the same tokens appear at different offsets.
        Layer index is included so each layer maintains independent KV entries.
        """
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        chunk = token_ids[start:end]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Cache the KV tensor for a specific chunk and layer."""
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)
        self.put(key, kv)

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Look up all chunks for a given layer; return (hits, miss_chunk_indices).

        hits: list of (chunk_idx, kv_tensor) for cache hits
        misses: list of chunk_idx values that were not cached
        """
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            key = self.chunk_key(token_ids, i, layer_idx)
            kv = self.get(key)
            if kv is not None:
                hits.append((i, kv))
            else:
                misses.append(i)

        # Track non-contiguous hits: hits not at a contiguous prefix boundary
        if hits:
            miss_set = set(misses)
            for idx, _ in hits:
                if any(m < idx for m in miss_set):
                    self._noncontiguous_hits += 1

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous (out-of-prefix)."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

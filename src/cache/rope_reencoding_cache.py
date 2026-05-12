"""RoPEReencodingNonContiguousCache — Activity B.

Stores KV tensors in a position-decoupled (pre-RoPE) form keyed by content hash.
On retrieval, re-applies RoPE for the caller-specified target token positions,
enabling non-contiguous segment reuse across requests whose tokens appear at
different positions.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class RoPEReencodingConfig:
    chunk_size: int = 128
    max_entries: int = 2000
    d_head: int = 64
    n_heads: int = 8
    rope_base: float = 10000.0
    rope_dim: int = -1   # -1 → use full d_head
    seed: int = 42


class RoPEReencodingNonContiguousCache(CacheStore):
    """Non-contiguous KV cache that stores pre-RoPE tensors and re-encodes on load.

    Implements the full CacheStore interface, plus overrides
    store_pre_rope() / load_with_rope() from the base class.

    Internally delegates storage to SegmentedHashCache so that content-hash
    keying (position-independent) is inherited for free.
    """

    def __init__(self, config: RoPEReencodingConfig) -> None:
        self.config = config
        self._store = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        # Lazy rotation-matrix cache: (position, d) → [d//2, 2, 2]
        self._rope_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store already-RoPE-encoded KV (standard path, no re-encoding on load)."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Standard get — no RoPE re-application."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._store.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._store.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._store.reset_stats()

    # ------------------------------------------------------------------ #
    # RoPE-aware extension (CacheStore optional methods overridden)        #
    # ------------------------------------------------------------------ #

    def store_pre_rope(
        self,
        key: str,
        value: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Store pre-RoPE KV under a scoped content-hash key."""
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        self._store.put(scoped_key, value)

    def load_with_rope(
        self,
        key: str,
        target_positions: torch.Tensor,  # [n_tokens] long
        layer_idx: int = 0,
        rope_dim: int = -1,
    ) -> Optional[torch.Tensor]:
        """Load pre-RoPE KV and apply rotation for target_positions.

        Returns None on miss.
        """
        scoped_key = f"pre_rope:{layer_idx}:{key}"
        # Bypass SegmentedHashCache's own hit/miss counting for the inner store
        # by calling put/get directly; we track stats here.
        pre_rope_kv = self._store.get(scoped_key)
        if pre_rope_kv is None:
            self._misses += 1
            return None
        self._hits += 1
        return self._apply_rope(pre_rope_kv, target_positions, rope_dim)

    # ------------------------------------------------------------------ #
    # RoPE computation utilities                                           #
    # ------------------------------------------------------------------ #

    def _get_rope_rotation(self, position: int, d: int) -> torch.Tensor:
        """Return [d//2, 2, 2] rotation matrix for a single position."""
        cache_key = (position, d)
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        i = torch.arange(0, d, 2, dtype=torch.float32)
        theta = 1.0 / (self.config.rope_base ** (i / d))  # [d//2]
        angle = position * theta                            # [d//2]
        cos_a = angle.cos()
        sin_a = angle.sin()
        # [[cos, -sin], [sin, cos]] — standard 2-D rotation
        rot = torch.stack(
            [
                torch.stack([cos_a, -sin_a], dim=-1),
                torch.stack([sin_a, cos_a], dim=-1),
            ],
            dim=-2,
        )  # [d//2, 2, 2]

        self._rope_cache[cache_key] = rot
        return rot

    def _apply_rope(
        self,
        pre_rope_kv: torch.Tensor,      # [n_tokens, 2, n_heads, d_head]
        target_positions: torch.Tensor, # [n_tokens] long
        rope_dim: int = -1,
    ) -> torch.Tensor:
        """Apply position-specific RoPE to the key (index 0) of each token.

        Value (index 1) is left unchanged — standard transformer practice.
        """
        n_tokens, _, n_heads, d_head = pre_rope_kv.shape
        d = d_head if rope_dim == -1 else rope_dim
        assert d % 2 == 0, "RoPE dimension must be even"

        result = pre_rope_kv.clone().float()
        key_slice = result[:, 0, :, :d]  # [n_tokens, n_heads, d]

        # Batch-build rotation matrices for all unique positions
        unique_pos = target_positions.unique().tolist()
        rot_by_pos: Dict[int, torch.Tensor] = {}
        for pos in unique_pos:
            rot_by_pos[int(pos)] = self._get_rope_rotation(int(pos), d)  # [d//2, 2, 2]

        for tok_idx in range(n_tokens):
            pos = int(target_positions[tok_idx].item())
            rot = rot_by_pos[pos]                              # [d//2, 2, 2]
            k_tok = key_slice[tok_idx].reshape(n_heads, d // 2, 2)  # [n_heads, d//2, 2]
            k_tok_rot = torch.einsum("hpi,pij->hpj", k_tok, rot)    # [n_heads, d//2, 2]
            key_slice[tok_idx] = k_tok_rot.reshape(n_heads, d)

        result[:, 0, :, :d] = key_slice
        return result.to(pre_rope_kv.dtype)

    # ------------------------------------------------------------------ #
    # Segment-level API (SegmentedHashCache-compatible)                   #
    # ------------------------------------------------------------------ #

    def put_segment_pre_rope(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int = 0,
    ) -> None:
        """Store a chunk's pre-RoPE KV keyed by content hash."""
        key = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
        self.store_pre_rope(key, pre_rope_kv, layer_idx)

    def get_segments_with_rope(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Retrieve all chunks; apply RoPE for this request's token positions.

        Returns:
            hits:   [(chunk_idx, rope-applied kv tensor), ...]
            misses: [chunk_idx, ...]
        """
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for chunk_idx in range(n_chunks):
            key = self._store.chunk_key(token_ids, chunk_idx, layer_idx)
            start_tok = chunk_idx * chunk_size
            end_tok = min(start_tok + chunk_size, len(token_ids))
            positions = torch.arange(
                target_offset + start_tok,
                target_offset + end_tok,
                dtype=torch.long,
            )
            kv = self.load_with_rope(key, positions, layer_idx)
            if kv is not None:
                hits.append((chunk_idx, kv))
                # Non-contiguous: there is a miss at a lower chunk index
                if any(m < chunk_idx for m in misses):
                    self._noncontiguous_hits += 1
            else:
                misses.append(chunk_idx)

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous (a miss exists at a lower index)."""
        total_hits = self._hits
        if total_hits == 0:
            return 0.0
        return self._noncontiguous_hits / total_hits

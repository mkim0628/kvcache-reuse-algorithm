"""AdapShotMixedDimSegmentPipeline — Activity B+C integration.

Combines RoPEReencodingNonContiguousCache (B-1) with MixedDimPerTokenBudgetCodec (C-1).

Store order contract:
    raw KV (pre-RoPE)
        → [1] MixedDimPerTokenBudgetCodec.encode()  (mixed-dim compression)
        → [2] RoPEReencodingNonContiguousCache.store_pre_rope()  (content-hash storage)

Restore order contract:
    stored compressed KV
        → [1] RoPEReencodingNonContiguousCache.load_with_rope()  (pre-RoPE load + RoPE re-apply)
        → [2] MixedDimPerTokenBudgetCodec.decode()  (masked_kv returned as-is)
        → final KV (RoPE-encoded, mixed-dim compressed, zeroed dims = 0)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.rope_reencoding_cache import (
    RoPEReencodingConfig,
    RoPEReencodingNonContiguousCache,
)
from src.cache.mixed_dim_codec import MixedDimConfig, MixedDimPerTokenBudgetCodec


@dataclass
class AdapShotPipelineConfig:
    rope: Optional[RoPEReencodingConfig] = None
    mixed_dim: Optional[MixedDimConfig] = None

    def __post_init__(self) -> None:
        if self.rope is None:
            self.rope = RoPEReencodingConfig()
        if self.mixed_dim is None:
            self.mixed_dim = MixedDimConfig()


class AdapShotMixedDimSegmentPipeline(CacheStore):
    """B+C integration pipeline: pre-RoPE storage + mixed-dim compression (Cross-2).

    Implements the full CacheStore interface by delegating to
    RoPEReencodingNonContiguousCache and MixedDimPerTokenBudgetCodec.

    Store order:  raw KV → mixed-dim compress → pre-RoPE store
    Restore order: pre-RoPE load → RoPE re-apply → mixed-dim decode (no-op)
    """

    def __init__(self, config: AdapShotPipelineConfig) -> None:
        self.config = config
        self.rope_cache = RoPEReencodingNonContiguousCache(config.rope)
        self.codec = MixedDimPerTokenBudgetCodec(config.mixed_dim)
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Compress via mixed-dim codec then store pre-RoPE (store order contract).

        value: [n_tokens, 2, n_heads, d_head] — raw, pre-RoPE KV.
        """
        compressed = self.codec.compression_hook(key, value)
        self.rope_cache.store_pre_rope(key, compressed, layer_idx=0)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Standard get — returns compressed KV without RoPE re-application.

        For position-correct retrieval use get_with_rope() or load_segment().
        """
        scoped_key = f"pre_rope:0:{key}"
        result = self.rope_cache._store.get(scoped_key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self.rope_cache.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self.rope_cache.memory_bytes()

    def reset_stats(self) -> None:
        self.rope_cache.reset_stats()
        self._hits = 0
        self._misses = 0

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CacheStore-compatible compression hook (delegates to MixedDimCodec)."""
        return self.codec.compression_hook(key, value, attn_weights)

    # ------------------------------------------------------------------ #
    # B+C pipeline API                                                     #
    # ------------------------------------------------------------------ #

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] — raw pre-RoPE
        layer_idx: int = 0,
        attn_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Store order contract: [1] mixed-dim compress → [2] pre-RoPE store.

        pre_rope_kv must be the raw (non-RoPE-encoded) KV tensor.
        """
        key = self.rope_cache._store.chunk_key(token_ids, chunk_idx, layer_idx)
        # [1] mixed-dim compression
        compressed_kv = self.codec.compression_hook(key, pre_rope_kv, attn_weights)
        # [2] store pre-RoPE
        self.rope_cache.store_pre_rope(key, compressed_kv, layer_idx)

    def load_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        target_offset: int,  # absolute position of this chunk's first token
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Restore order contract: [1] pre-RoPE load + RoPE re-apply → [2] decode (no-op).

        Returns [n_tokens, 2, n_heads, d_head] (RoPE-encoded, mixed-dim compressed).
        Returns None on cache miss.
        """
        key = self.rope_cache._store.chunk_key(token_ids, chunk_idx, layer_idx)
        chunk_size = self.rope_cache.config.chunk_size
        start_tok = chunk_idx * chunk_size
        end_tok = min(start_tok + chunk_size, len(token_ids))
        positions = torch.arange(
            target_offset + start_tok,
            target_offset + end_tok,
            dtype=torch.long,
        )
        # [1] pre-RoPE load + RoPE re-application
        rope_applied_kv = self.rope_cache.load_with_rope(key, positions, layer_idx)
        if rope_applied_kv is None:
            return None
        # [2] decode — masked_kv is already the final form (zeroed dims stay zero)
        return rope_applied_kv

    def get_segments(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Retrieve all chunks via load_segment(); classify as hits or misses.

        Tracks non-contiguous hits in rope_cache for noncontiguous_hit_rate().

        Returns:
            hits:   [(chunk_idx, rope_applied_compressed_kv), ...]
            misses: [chunk_idx, ...]
        """
        chunk_size = self.rope_cache.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor]] = []
        misses: List[int] = []

        for chunk_idx in range(n_chunks):
            kv = self.load_segment(token_ids, chunk_idx, target_offset, layer_idx)
            if kv is not None:
                hits.append((chunk_idx, kv))
                # Track non-contiguous: hit follows a miss at a lower index
                if any(m < chunk_idx for m in misses):
                    self.rope_cache._noncontiguous_hits += 1
            else:
                misses.append(chunk_idx)

        return hits, misses

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that are non-contiguous (mirrors rope_cache tracking)."""
        return self.rope_cache.noncontiguous_hit_rate()

    def save_pipeline(self, path: str) -> None:
        """Serialise pipeline state to disk."""
        torch.save(
            {
                "rope_store": self.rope_cache._store._store,
                "rope_config": self.config.rope,
                "codec_config": self.config.mixed_dim,
            },
            path,
        )

    def load_pipeline(self, path: str) -> None:
        """Restore pipeline state from disk."""
        state = torch.load(path, weights_only=False)
        self.rope_cache._store._store = state["rope_store"]
        self.config.rope = state["rope_config"]
        self.config.mixed_dim = state["codec_config"]

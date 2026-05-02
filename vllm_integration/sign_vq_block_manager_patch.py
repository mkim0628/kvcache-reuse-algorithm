"""Activity B+C — SignVQ non-contiguous KV reuse for vLLM 0.20.0.

This module ports ``src/cache/sign_vq_segment.SignVQSegmentCache`` (B+C Cross-1,
verified 2026-05-02) into vLLM's v1 KV cache management layer.

Architecture overview
---------------------
vLLM 0.20.0 v1 engine uses:
    vllm.v1.core.kv_cache_manager.KVCacheManager
        ├── BlockPool         (physical block allocation)
        └── KVCacheCoordinator(s) (per KV-cache-group prefix matching)

The standard prefix cache only reuses KV blocks when the token sequence
prefix is byte-identical.  This patch adds a **second lookup layer** that
enables *non-contiguous* reuse via 1-bit sign VQ similarity:

    3-stage lookup per chunk:
      Stage 1: exact SHA-256 block hash  → FP16 full KV  (exact_fp16)
      Stage 2: sign-code Hamming check   → approx ±1 KV  (approx_sign)
      Stage 3: full miss → normal prefill

The ``_sign_store`` dict (``OrderedDict`` for LRU) maps a position-independent
SHA-256 key (content-only, not position-based) to ``(sign_code, value_fp16)``.

Import safety
-------------
vLLM imports are wrapped in try/except so this module can be imported in
CPU-only unit-test environments where vLLM is not installed.
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
    from vllm.v1.request import Request as VllmRequest
    _VLLM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _VLLM_AVAILABLE = False
    KVCacheManager = object          # type: ignore[assignment, misc]
    KVCacheBlocks = object           # type: ignore[assignment, misc]
    VllmRequest = object             # type: ignore[assignment, misc]

from vllm_integration.leverage_compressor_patch import (
    VllmLeverageCompressor,
    _packbits_2d,
    _unpack_signs_to_pm1,
)


# --------------------------------------------------------------------------- #
# Popcount helper                                                               #
# --------------------------------------------------------------------------- #

def _popcount_uint8(x: torch.Tensor) -> int:
    """Count set bits across an entire uint8 tensor.

    Uses torch.bitwise_count (PyTorch >= 2.1) when available; falls back to a
    256-entry lookup table for older versions.
    """
    if hasattr(torch, "bitwise_count"):
        return int(torch.bitwise_count(x).sum().item())
    lut = torch.tensor(
        [bin(i).count("1") for i in range(256)], dtype=torch.int32
    )
    return int(lut[x.long()].sum().item())


# --------------------------------------------------------------------------- #
# Position-independent chunk key                                                #
# --------------------------------------------------------------------------- #

def _make_chunk_key(
    token_ids: List[int],
    chunk_idx: int,
    chunk_size: int,
    layer_idx: int,
) -> str:
    """Compute a position-independent SHA-256 hash key for a token chunk.

    The hash covers only the *content* of the chunk (the token IDs) and the
    layer index, but NOT the absolute position within the sequence.  This
    enables cache hits even when the same token subsequence appears at a
    different offset in a later request.

    Args:
        token_ids:  full token sequence.
        chunk_idx:  index of this chunk within the sequence.
        chunk_size: tokens per chunk.
        layer_idx:  transformer layer index.

    Returns:
        Hex-digest SHA-256 string (64 chars).
    """
    start = chunk_idx * chunk_size
    end = min(start + chunk_size, len(token_ids))
    chunk = token_ids[start:end]
    h = hashlib.sha256()
    h.update(struct.pack(f"<{len(chunk)}I", *chunk))
    h.update(struct.pack("<I", layer_idx))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# SignVQSegmentIndex — standalone lookup store                                  #
# --------------------------------------------------------------------------- #

class SignVQSegmentIndex:
    """In-process 3-stage KV segment lookup index (Activity B+C port).

    This is the vLLM-side equivalent of ``SignVQSegmentCache`` from
    ``src/cache/sign_vq_segment.py``.  It does NOT inherit from ``CacheStore``
    (which is a standalone-benchmark abstraction), but implements the same
    logical lookup protocol.

    The index maintains two separate LRU stores:
      * ``_fp16_store`` : key → (2, n_tokens, d_head) float16 tensor
                          (Tier-1 exact FP16 hits)
      * ``_sign_store`` : key → (sign_code uint8, value_fp16 fp16)
                          (Tier-2 approximate sign-VQ hits)

    Args:
        compressor:       ``VllmLeverageCompressor`` instance (required for
                          sign-code computation and sign-code lookup).
        chunk_size:       tokens per chunk (must match vLLM block_size or its
                          divisor; defaults to 16 to match vLLM DEFAULT_BLOCK_SIZE).
        max_entries:      LRU capacity per store.
        hamming_threshold: normalised Hamming distance threshold for approx hits.
    """

    def __init__(
        self,
        compressor: Optional[VllmLeverageCompressor] = None,
        chunk_size: int = 16,
        max_entries: int = 1000,
        hamming_threshold: float = 0.15,
    ) -> None:
        self.compressor = compressor
        self.chunk_size = chunk_size
        self.max_entries = max_entries
        self.hamming_threshold = hamming_threshold

        # Tier-1 exact FP16 store
        # key (str) → kv_fp16: torch.Tensor  shape (2, n_tokens, d_head) fp16
        self._fp16_store: OrderedDict[str, torch.Tensor] = OrderedDict()

        # Tier-2 sign VQ store
        # key (str) → (sign_code uint8, value_fp16 fp16)
        self._sign_store: OrderedDict[
            str, Tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()

        # Hit/miss counters
        self._exact_fp16_hits: int = 0
        self._approx_sign_hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------ #
    # Storage API                                                          #
    # ------------------------------------------------------------------ #

    def put(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Compress and store a KV chunk.

        Dispatches to:
          * Tier-1 FP16 store (high-leverage tokens, per compressor.encode_block).
          * Tier-2 sign store (mid-leverage tokens).

        Without a compressor, stores the full FP16 KV in the FP16 store only.

        Args:
            token_ids: full token sequence (for key computation).
            chunk_idx: chunk index within the sequence.
            keys:      (n_tokens, d_head) float tensor — Key slice.
            values:    (n_tokens, d_head) float tensor — Value slice.
            layer_idx: transformer layer index.
        """
        key = _make_chunk_key(token_ids, chunk_idx, self.chunk_size, layer_idx)

        if self.compressor is None:
            # No compressor: store raw FP16
            kv_fp16 = torch.stack([keys.half(), values.half()], dim=0)
            self._put_fp16(key, kv_fp16)
            return

        storage = self.compressor.encode_block(keys, values, layer_idx=layer_idx)

        # Tier-1: FP16 full KV
        t1 = storage["tier1_indices"]
        tier1_kv = storage["tier1_kv"]
        if isinstance(t1, torch.Tensor) and t1.numel() > 0:
            d_head: int = storage["d_head"]  # type: ignore[assignment]
            # Reconstruct (2, n1, d_head) fp16 for exact storage
            tier1_kv_t = tier1_kv  # type: ignore[assignment]
            kv_t1 = torch.stack(
                [tier1_kv_t[:, :d_head], tier1_kv_t[:, d_head:]], dim=0
            )
            self._put_fp16(key, kv_t1)

        # Tier-2: sign code + FP16 value
        t2 = storage["tier2_indices"]
        tier2_sign_k = storage["tier2_sign_k"]
        tier2_v_fp16 = storage["tier2_v_fp16"]
        if isinstance(t2, torch.Tensor) and t2.numel() > 0:
            self._put_sign(key, tier2_sign_k, tier2_v_fp16)

    def _put_fp16(self, key: str, kv_fp16: torch.Tensor) -> None:
        """Insert into FP16 store with LRU eviction."""
        if len(self._fp16_store) >= self.max_entries:
            self._fp16_store.popitem(last=False)
        self._fp16_store[key] = kv_fp16

    def _put_sign(
        self,
        key: str,
        sign_code: torch.Tensor,
        value_fp16: torch.Tensor,
    ) -> None:
        """Insert into sign store with LRU eviction."""
        if len(self._sign_store) >= self.max_entries:
            self._sign_store.popitem(last=False)
        self._sign_store[key] = (sign_code, value_fp16)

    # ------------------------------------------------------------------ #
    # Lookup API — 3-stage                                                 #
    # ------------------------------------------------------------------ #

    def get(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
        query_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Look up a single chunk via 3-stage protocol.

        Stage 1: exact SHA-256 hash → FP16 KV                (exact_fp16)
        Stage 2: sign-code Hamming  → approximate ±1 KV      (approx_sign)
        Stage 3: miss                                         (miss)

        Args:
            token_ids:  full token sequence.
            chunk_idx:  chunk index.
            layer_idx:  transformer layer index.
            query_keys: (n_tokens, d_head) current query key slice for Hamming
                        distance computation.  Required for approx hits; without
                        it only exact hits are returned.

        Returns:
            (kv_tensor_or_None, hit_type_str)
            hit_type ∈ {"exact_fp16", "approx_sign", "miss"}
        """
        key = _make_chunk_key(token_ids, chunk_idx, self.chunk_size, layer_idx)

        # Stage 1: exact FP16
        fp16_kv = self._fp16_store.get(key)
        if fp16_kv is not None:
            self._fp16_store.move_to_end(key)  # LRU refresh
            self._exact_fp16_hits += 1
            return fp16_kv.float(), "exact_fp16"

        # Stage 2: approximate sign-VQ
        if query_keys is not None and key in self._sign_store:
            sign_code, val_fp16 = self._sign_store[key]
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, len(token_ids))
            q_chunk = query_keys[chunk_start:chunk_end]

            if (
                q_chunk.shape[0] == sign_code.shape[0]
                and self._is_similar(q_chunk, sign_code)
            ):
                self._sign_store.move_to_end(key)
                self._approx_sign_hits += 1
                d_head = q_chunk.shape[-1]
                approx_k = _unpack_signs_to_pm1(sign_code, d_head)  # (n_toks, d_head)
                kv = torch.stack([approx_k, val_fp16.float()], dim=0)
                return kv, "approx_sign"

        # Stage 3: miss
        self._misses += 1
        return None, "miss"

    def get_all_chunks(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        query_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor, str]], List[int]]:
        """Look up all chunks in a sequence.

        Returns:
            hits:   list of (chunk_idx, kv_tensor, hit_type)
            misses: list of chunk_idx values not found
        """
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits: List[Tuple[int, torch.Tensor, str]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            kv, hit_type = self.get(
                token_ids, i, layer_idx=layer_idx, query_keys=query_keys
            )
            if hit_type != "miss":
                hits.append((i, kv, hit_type))
            else:
                misses.append(i)

        return hits, misses

    # ------------------------------------------------------------------ #
    # Similarity helper                                                     #
    # ------------------------------------------------------------------ #

    def _is_similar(
        self,
        query_keys: torch.Tensor,
        stored_sign_code: torch.Tensor,
    ) -> bool:
        """Return True if normalised Hamming distance ≤ hamming_threshold.

        Normalised distance = total differing bits / (n_tokens * d_head).

        Args:
            query_keys:       (n_toks, d_head) float
            stored_sign_code: (n_toks, ceil(d_head/8)) uint8
        """
        if self.compressor is None:
            return False
        d_head = query_keys.shape[-1]
        q_sign = self.compressor.to_sign_code(query_keys)
        xor = q_sign ^ stored_sign_code
        hamming_dist_norm = _popcount_uint8(xor) / (query_keys.shape[0] * d_head)
        return bool(hamming_dist_norm <= self.hamming_threshold)

    # ------------------------------------------------------------------ #
    # Eviction                                                             #
    # ------------------------------------------------------------------ #

    def evict_fp16(self) -> int:
        """Evict the LRU entry from the FP16 store. Returns freed bytes."""
        if not self._fp16_store:
            return 0
        _, kv = self._fp16_store.popitem(last=False)
        return kv.nbytes

    def evict_sign(self) -> int:
        """Evict the LRU entry from the sign store. Returns freed bytes."""
        if not self._sign_store:
            return 0
        _, (sc, vf) = self._sign_store.popitem(last=False)
        return sc.nbytes + vf.nbytes

    def evict(self) -> int:
        """Evict one entry from each store. Returns total freed bytes."""
        return self.evict_fp16() + self.evict_sign()

    # ------------------------------------------------------------------ #
    # Metrics                                                               #
    # ------------------------------------------------------------------ #

    def hit_rate(self) -> float:
        """Overall (exact + approx) hit rate."""
        total = self._exact_fp16_hits + self._approx_sign_hits + self._misses
        if total == 0:
            return 0.0
        return (self._exact_fp16_hits + self._approx_sign_hits) / total

    def tier_hit_rates(self) -> Dict[str, float]:
        """Return per-tier and overall hit rate metrics.

        Returns:
            dict with keys: exact_fp16, approx_sign, overall,
                            noncontiguous_ratio.
        """
        total = self._exact_fp16_hits + self._approx_sign_hits + self._misses
        if total == 0:
            return {
                "exact_fp16": 0.0,
                "approx_sign": 0.0,
                "overall": 0.0,
                "noncontiguous_ratio": 0.0,
            }
        total_hits = self._exact_fp16_hits + self._approx_sign_hits
        return {
            "exact_fp16": self._exact_fp16_hits / total,
            "approx_sign": self._approx_sign_hits / total,
            "overall": total_hits / total,
            "noncontiguous_ratio": (
                self._approx_sign_hits / total_hits if total_hits > 0 else 0.0
            ),
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._exact_fp16_hits = 0
        self._approx_sign_hits = 0
        self._misses = 0

    def memory_bytes(self) -> int:
        """Total memory used by both stores."""
        fp16_bytes = sum(v.nbytes for v in self._fp16_store.values())
        sign_bytes = sum(
            sc.nbytes + vf.nbytes for sc, vf in self._sign_store.values()
        )
        return fp16_bytes + sign_bytes


# --------------------------------------------------------------------------- #
# NonContiguousKVCacheManagerV2                                                 #
# (monkey-patchable subclass of vLLM KVCacheManager)                           #
# --------------------------------------------------------------------------- #

class NonContiguousKVCacheManagerV2(KVCacheManager):  # type: ignore[misc]
    """vLLM KVCacheManager subclass with SignVQ non-contiguous reuse.

    Adds a ``SignVQSegmentIndex`` (``_sign_index``) to the standard
    ``KVCacheManager`` and exposes ``store_segment`` / ``lookup_segment``
    methods that the attention backend can call after every prefill step.

    The existing ``get_computed_blocks`` logic (standard prefix cache) runs
    first; this class supplements it with a second, non-contiguous lookup for
    chunks that the prefix cache missed.

    Usage:
        # Replace the default KVCacheManager at engine construction:
        kv_manager = NonContiguousKVCacheManagerV2(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            hash_block_size=block_size,
            enable_caching=True,
            # sign-VQ params:
            sign_vq_chunk_size=16,
            sign_vq_max_entries=2000,
            sign_vq_hamming_threshold=0.15,
            sign_vq_rank=32,
            sign_vq_tier1_ratio=0.20,
            sign_vq_tier3_ratio=0.20,
        )

    Notes:
        * This class only overrides ``__init__`` and adds new public methods.
          All existing ``KVCacheManager`` methods are inherited unchanged.
        * When vLLM is not installed the class degrades gracefully (inherits
          from ``object`` via the import fallback).
    """

    def __init__(
        self,
        *args: object,
        sign_vq_chunk_size: int = 16,
        sign_vq_max_entries: int = 2000,
        sign_vq_hamming_threshold: float = 0.15,
        sign_vq_rank: int = 32,
        sign_vq_tier1_ratio: float = 0.20,
        sign_vq_tier3_ratio: float = 0.20,
        **kwargs: object,
    ) -> None:
        if _VLLM_AVAILABLE:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        # else: running in unit-test fallback; no super().__init__ needed

        compressor = VllmLeverageCompressor(
            rank=sign_vq_rank,
            tier1_ratio=sign_vq_tier1_ratio,
            tier3_ratio=sign_vq_tier3_ratio,
        )
        self._sign_index = SignVQSegmentIndex(
            compressor=compressor,
            chunk_size=sign_vq_chunk_size,
            max_entries=sign_vq_max_entries,
            hamming_threshold=sign_vq_hamming_threshold,
        )

    # ------------------------------------------------------------------ #
    # Attention-backend facing API                                          #
    # ------------------------------------------------------------------ #

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Store a KV chunk into the sign-VQ index.

        Called by the attention wrapper after every write-to-cache step.

        Args:
            token_ids: full token sequence.
            chunk_idx: chunk index within the sequence.
            keys:      (n_tokens, d_head) float tensor.
            values:    (n_tokens, d_head) float tensor.
            layer_idx: transformer layer index.
        """
        self._sign_index.put(
            token_ids, chunk_idx, keys, values, layer_idx=layer_idx
        )

    def lookup_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
        query_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Look up a single chunk from the sign-VQ index.

        Args:
            token_ids:  full token sequence.
            chunk_idx:  chunk index.
            layer_idx:  transformer layer index.
            query_keys: (n_tokens, d_head) current query keys for approx lookup.

        Returns:
            (kv_or_None, hit_type)  — hit_type ∈ {"exact_fp16", "approx_sign", "miss"}.
        """
        return self._sign_index.get(
            token_ids, chunk_idx, layer_idx=layer_idx, query_keys=query_keys
        )

    def lookup_all_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        query_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor, str]], List[int]]:
        """Look up all chunks in a token sequence.

        Returns:
            hits:   list of (chunk_idx, kv_tensor, hit_type)
            misses: list of chunk_idx values not found
        """
        return self._sign_index.get_all_chunks(
            token_ids, layer_idx=layer_idx, query_keys=query_keys
        )

    def sign_index_stats(self) -> Dict[str, float]:
        """Return tier hit rate metrics from the sign-VQ index."""
        return self._sign_index.tier_hit_rates()

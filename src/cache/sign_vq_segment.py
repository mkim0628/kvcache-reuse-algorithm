"""Activity B+C — SignVQSegmentCache: 3-tier FP16/sign-VQ segment cache.

Combines non-contiguous KV reuse (Activity B) with leverage-score-based
compression (Activity C).

Lookup order per chunk:
  1. Exact SHA-256 hash → Tier-1 FP16 entry  (exact_fp16)
  2. Exact SHA-256 key present in sign store → XOR+popcount Hamming check
     → if within threshold: approximate hit          (approx_sign)
  3. Full miss

Reference: Self-Indexing KVCache (AAAI 2026, arXiv 2603.14224).
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch

from src.cache.segmented import SegmentedHashCache
from src.cache.leverage_compressor import LeverageScoreCompressor, _unpack_signs_to_pm1


class SignVQSegmentCache(SegmentedHashCache):
    """1-bit sign VQ index + leverage-score 3-tier segment cache (Activity B+C).

    Tier-1 (FP16) segments are stored in the parent ``_store`` (OrderedDict)
    and retrieved via exact hash.  Tier-2 (sign-only Key + FP16 Value)
    segments are stored in ``_sign_store`` and retrieved via Hamming distance.
    Tier-3 tokens are evicted and not stored.

    hit_type annotations:
      "exact_fp16"  — Tier-1 exact hash hit
      "approx_sign" — Tier-2 Hamming-distance approximate hit
    """

    def __init__(
        self,
        compressor: Optional[LeverageScoreCompressor] = None,
        chunk_size: int = 128,
        max_entries: int = 1000,
        hamming_threshold: float = 0.15,
    ) -> None:
        super().__init__(chunk_size=chunk_size, max_entries=max_entries)
        self.compressor = compressor
        self.hamming_threshold = hamming_threshold
        # key → (sign_code: uint8 tensor, value_fp16: fp16 tensor)
        self._sign_store: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._exact_fp16_hits: int = 0
        self._approx_sign_hits: int = 0

    # ------------------------------------------------------------------ #
    # Storage API                                                          #
    # ------------------------------------------------------------------ #

    def put_segment_compressed(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Compress and store a KV chunk using the attached compressor.

        Without a compressor, falls back to storing raw FP16 via the parent
        ``put_segment()`` method.

        Args:
            token_ids: full token sequence (used for key computation).
            chunk_idx: index of this chunk within the sequence.
            keys:      (n_tokens_in_chunk, d_head) float tensor.
            values:    (n_tokens_in_chunk, d_head) float tensor.
            layer_idx: transformer layer index.
        """
        if self.compressor is None:
            kv = torch.cat([keys, values], dim=-1)
            self.put_segment(token_ids, chunk_idx, kv, layer_idx)
            return

        storage = self.compressor.encode(keys, values, layer_idx, tensor_id=chunk_idx)
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)

        # Tier-1: FP16 full KV → exact-hash store (parent _store)
        if storage["tier1_kv"].numel() > 0:
            self.put(key, storage["tier1_kv"])

        # Tier-2: sign code + FP16 value → approximate store
        if storage["tier2_sign_k"].numel() > 0:
            if len(self._sign_store) >= self.max_entries:
                self._evict_sign_store()
            self._sign_store[key] = (
                storage["tier2_sign_k"],
                storage["tier2_v_fp16"],
            )

    # ------------------------------------------------------------------ #
    # Lookup API                                                           #
    # ------------------------------------------------------------------ #

    def get_segments_with_approx(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
        query_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[List[Tuple[int, torch.Tensor, str]], List[int]]:
        """Look up all chunks: exact FP16 → approx sign → miss.

        Args:
            token_ids:  full token sequence.
            layer_idx:  transformer layer index.
            query_keys: (n_tokens, d_head) current query keys for Hamming
                        similarity check.  Required for approx hits; without
                        this argument only exact hits are returned.

        Returns:
            hits:   list of (chunk_idx, kv_tensor, hit_type) where
                    hit_type ∈ {"exact_fp16", "approx_sign"}.
            misses: list of chunk_idx values not found.
        """
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)
        hits: List[Tuple[int, torch.Tensor, str]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            key = self.chunk_key(token_ids, i, layer_idx)

            # Stage 1: exact FP16 hash lookup
            fp16_kv = self.get(key)
            if fp16_kv is not None:
                self._exact_fp16_hits += 1
                hits.append((i, fp16_kv.float(), "exact_fp16"))
                continue

            # Stage 2: approximate sign-VQ lookup (only when query_keys provided)
            if query_keys is not None and key in self._sign_store:
                sign_code, val_fp16 = self._sign_store[key]
                chunk_start = i * self.chunk_size
                chunk_end = min(chunk_start + self.chunk_size, len(token_ids))
                q_keys_chunk = query_keys[chunk_start:chunk_end]

                if (q_keys_chunk.shape[0] == sign_code.shape[0]
                        and self._is_similar(q_keys_chunk, sign_code)):
                    self._approx_sign_hits += 1
                    self._noncontiguous_hits += 1
                    d_head = q_keys_chunk.shape[-1]
                    approx_k = _unpack_signs_to_pm1(sign_code, d_head)
                    kv = torch.cat([approx_k, val_fp16.float()], dim=-1)
                    hits.append((i, kv, "approx_sign"))
                    # Move to end (LRU refresh)
                    self._sign_store.move_to_end(key)
                    continue

            # Stage 3: full miss
            misses.append(i)

        return hits, misses

    # ------------------------------------------------------------------ #
    # Similarity helpers                                                   #
    # ------------------------------------------------------------------ #

    def _is_similar(
        self,
        query_keys: torch.Tensor,
        stored_sign_code: torch.Tensor,
    ) -> bool:
        """Return True if normalised Hamming distance ≤ hamming_threshold.

        Normalised distance = total differing bits / (n_tokens * d_head).
        Uses XOR + popcount; O(n_tokens × ceil(d_head/64)) bitwise ops.

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

    def evict(self) -> int:
        """Evict one entry from _store (FP16) and one from _sign_store."""
        freed = super().evict()
        freed += self._evict_sign_store()
        return freed

    def _evict_sign_store(self) -> int:
        """Evict the LRU entry from _sign_store."""
        if not self._sign_store:
            return 0
        evict_key, (sign_code, val_fp16) = next(iter(self._sign_store.items()))
        del self._sign_store[evict_key]
        return sign_code.nbytes + val_fp16.nbytes

    # ------------------------------------------------------------------ #
    # CacheStore interface overrides                                       #
    # ------------------------------------------------------------------ #

    def memory_bytes(self) -> int:
        """Total bytes used by FP16 store + sign store."""
        fp16_bytes = sum(v.nbytes for v in self._store.values())
        sign_bytes = sum(
            sc.nbytes + vf.nbytes for sc, vf in self._sign_store.values()
        )
        return fp16_bytes + sign_bytes

    def reset_stats(self) -> None:
        """Reset hit/miss counters including tier-specific counters."""
        super().reset_stats()
        self._exact_fp16_hits = 0
        self._approx_sign_hits = 0

    # ------------------------------------------------------------------ #
    # Metrics                                                              #
    # ------------------------------------------------------------------ #

    def tier_hit_rates(self) -> Dict[str, float]:
        """Return per-tier and overall hit rate metrics.

        Returns:
            dict with keys: exact_fp16, approx_sign, overall,
                            noncontiguous_ratio.
        """
        total = self._hits + self._misses
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _popcount_uint8(x: torch.Tensor) -> int:
    """Count set bits (1s) across an entire uint8 tensor.

    Uses torch.bitwise_count (PyTorch ≥ 2.1) when available; falls back to
    a 256-entry lookup table for older versions.

    Args:
        x: uint8 tensor of any shape.

    Returns:
        Total number of set bits as a Python int.
    """
    if hasattr(torch, "bitwise_count"):
        return int(torch.bitwise_count(x).sum().item())
    # Lookup table fallback: precompute popcount for all 256 uint8 values
    lut = torch.tensor(
        [bin(i).count("1") for i in range(256)], dtype=torch.int32
    )
    return int(lut[x.long()].sum().item())

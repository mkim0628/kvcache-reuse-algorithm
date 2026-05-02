"""Activity C — LeverageScoreCompressor adapted for vLLM 0.20.0 paged KV blocks.

This module ports ``src/cache/leverage_compressor.LeverageScoreCompressor``
(B+C Cross-1 implementation, verified 2026-05-02) into vLLM's block-paged KV
cache layout.

vLLM KV-cache tensor shapes (v1 engine):
    kv_cache : Tensor  shape [2, num_blocks, block_size, num_kv_heads, head_size]
                 [0] = key blocks,  [1] = value blocks

``VllmLeverageCompressor`` operates on individual blocks extracted from this
paged layout.  It preserves vLLM's block boundaries and does NOT split or merge
blocks, in accordance with the porting principle "블록 경계를 벗어나는 세그먼트
분할 금지".

Usage pattern (inside a custom attention wrapper or model-execution hook):
    comp = VllmLeverageCompressor(rank=32, tier1_ratio=0.20, tier3_ratio=0.20)
    storage = comp.encode_block(key_block, value_block, layer_idx=layer_idx,
                                block_id=block_id)
    kv_reconstructed = comp.decode_block(storage)  # (2, block_size, d_head)

The encode/decode interface mirrors
``src/cache/leverage_compressor.LeverageScoreCompressor`` so that unit tests can
reuse the same test logic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Helpers (shared with src/cache/leverage_compressor via copy-free re-export)  #
# --------------------------------------------------------------------------- #

def _packbits_2d(bits: torch.Tensor) -> torch.Tensor:
    """Pack a 2-D uint8 {0,1} tensor along axis=-1 into bytes (numpy packbits).

    Args:
        bits: (n_tokens, d_head) uint8 with values in {0, 1}

    Returns:
        (n_tokens, ceil(d_head / 8)) uint8
    """
    arr = bits.cpu().numpy().astype(np.uint8)
    packed_np = np.packbits(arr, axis=-1)
    return torch.from_numpy(packed_np.copy())


def _unpackbits_2d(packed: torch.Tensor, d_head: int) -> torch.Tensor:
    """Unpack a 2-D uint8 byte tensor to individual bits, truncating to d_head.

    Args:
        packed: (n_tokens, ceil(d_head/8)) uint8
        d_head: number of bit columns to keep

    Returns:
        (n_tokens, d_head) uint8 with values in {0, 1}
    """
    arr = packed.cpu().numpy().astype(np.uint8)
    unpacked_np = np.unpackbits(arr, axis=-1)[:, :d_head]
    return torch.from_numpy(unpacked_np.copy())


def _unpack_signs_to_pm1(sign_code: torch.Tensor, d_head: int) -> torch.Tensor:
    """Unpack uint8 bit-packed sign codes to ±1.0 float32 tensor.

    Args:
        sign_code: (n_tokens, ceil(d_head/8)) uint8
        d_head:    original key dimension

    Returns:
        (n_tokens, d_head) float32 with values in {-1.0, +1.0}
    """
    unpacked = _unpackbits_2d(sign_code, d_head)  # (n_tokens, d_head) uint8
    return 2.0 * unpacked.float() - 1.0            # 0 → -1.0, 1 → +1.0


# --------------------------------------------------------------------------- #
# VllmLeverageCompressor                                                        #
# --------------------------------------------------------------------------- #

class VllmLeverageCompressor:
    """Training-free 3-tier KV compressor adapted for vLLM paged-block layout.

    Mirrors ``src/cache/leverage_compressor.LeverageScoreCompressor`` but
    accepts vLLM block tensors directly and handles the paged (blocked) memory
    layout.

    Tier classification (per block):
        Tier-1 (top ``tier1_ratio``)  → FP16 full KV (exact)
        Tier-2 (middle tier)          → 1-bit sign packed Key + FP16 Value
        Tier-3 (bottom ``tier3_ratio``) → evicted (zero bytes stored)

    The block boundary is fully respected: this compressor processes one
    ``(block_size, d_head)`` slice at a time.

    Args:
        rank:         Rank-k approximation for leverage score computation.
        reg_lambda:   Regularisation constant (λ) for (K^T K + λI)^{-1}.
        tier1_ratio:  Fraction of tokens kept as full FP16 (default 0.20).
        tier3_ratio:  Fraction of tokens evicted entirely (default 0.20).
    """

    def __init__(
        self,
        rank: int = 32,
        reg_lambda: float = 1e-3,
        tier1_ratio: float = 0.20,
        tier3_ratio: float = 0.20,
    ) -> None:
        self.rank = rank
        self.reg_lambda = reg_lambda
        self.tier1_ratio = tier1_ratio
        self.tier3_ratio = tier3_ratio

    # ------------------------------------------------------------------ #
    # Core leverage score computation                                       #
    # ------------------------------------------------------------------ #

    def compute_leverage_scores(self, keys: torch.Tensor) -> torch.Tensor:
        """Compute rank-k approximate leverage scores for each token in a block.

        Args:
            keys: (n_tokens, d_head) float tensor — Key matrix K.

        Returns:
            scores: (n_tokens,) float32 — leverage score per token.
        """
        n_tokens, d_head = keys.shape
        k = min(self.rank, n_tokens, d_head)

        kf = keys.float()
        ktk = kf.T @ kf                      # (d_head, d_head)
        eigenvalues, eigenvectors = torch.linalg.eigh(ktk)
        # eigh returns ascending; take top-k
        v_k = eigenvectors[:, -k:]           # (d_head, k)
        lambda_k = eigenvalues[-k:]          # (k,)
        proj = kf @ v_k                      # (n_tokens, k)
        inv_diag = 1.0 / (lambda_k + self.reg_lambda)
        scores = (proj ** 2 * inv_diag).sum(dim=-1)  # (n_tokens,)
        return scores

    def classify(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Classify tokens in a block into 3 tiers by leverage score.

        Args:
            keys:   (n_tokens, d_head)
            values: (n_tokens, d_head)

        Returns:
            dict with index tensors: tier1, tier2, tier3, scores.
        """
        n_tokens = keys.shape[0]
        scores = self.compute_leverage_scores(keys)
        sorted_indices = torch.argsort(scores, descending=True)

        n1 = max(1, int(n_tokens * self.tier1_ratio))
        n3 = max(0, int(n_tokens * self.tier3_ratio))

        tier1_indices = sorted_indices[:n1]
        if n_tokens - n3 > n1:
            tier3_indices = sorted_indices[n_tokens - n3:]
            tier2_indices = sorted_indices[n1: n_tokens - n3]
        else:
            tier3_indices = torch.empty(0, dtype=torch.long)
            tier2_indices = torch.empty(0, dtype=torch.long)

        return {
            "tier1": tier1_indices,
            "tier2": tier2_indices,
            "tier3": tier3_indices,
            "scores": scores,
        }

    def to_sign_code(self, keys: torch.Tensor) -> torch.Tensor:
        """Pack the sign bits of each Key dimension into uint8 bytes.

        Args:
            keys: (n_tokens, d_head) — any float dtype.

        Returns:
            packed: (n_tokens, ceil(d_head / 8)) uint8 tensor.
        """
        signs = (keys >= 0).to(torch.uint8)  # (n_tokens, d_head)
        return _packbits_2d(signs)            # (n_tokens, ceil(d_head/8))

    # ------------------------------------------------------------------ #
    # Block-level encode / decode (primary vLLM API)                       #
    # ------------------------------------------------------------------ #

    def encode_block(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        layer_idx: int,
        block_id: int = 0,
    ) -> Dict[str, object]:
        """Compress a single vLLM KV block (n_tokens == block_size).

        ``key_block`` and ``value_block`` are single-block slices of the
        paged KV cache, i.e. shape (block_size, num_kv_heads, head_size) or the
        flattened form (block_size, d_head).  Both 3-D and 2-D inputs are
        accepted; 3-D inputs are automatically reshaped to 2-D.

        Args:
            key_block:   (block_size, [num_kv_heads,] head_size) float tensor.
            value_block: (block_size, [num_kv_heads,] head_size) float tensor.
            layer_idx:   transformer layer index (metadata only).
            block_id:    vLLM physical block identifier (metadata only).

        Returns:
            storage dict compatible with ``decode_block``.
        """
        original_shape = key_block.shape
        # Flatten to 2-D for compression
        keys_2d = key_block.reshape(key_block.shape[0], -1).float()
        values_2d = value_block.reshape(value_block.shape[0], -1).float()
        n_tokens, d_head = keys_2d.shape

        classification = self.classify(keys_2d, values_2d)
        t1 = classification["tier1"]
        t2 = classification["tier2"]
        t3 = classification["tier3"]

        # Tier-1: full FP16 KV concatenated
        if t1.numel() > 0:
            tier1_kv = torch.cat(
                [keys_2d[t1].half(), values_2d[t1].half()], dim=-1
            )  # (n1, 2*d_head)
        else:
            tier1_kv = torch.empty(0, 2 * d_head, dtype=torch.float16)

        # Tier-2: sign-packed Key + FP16 Value
        if t2.numel() > 0:
            tier2_sign_k = self.to_sign_code(keys_2d[t2])
            tier2_v_fp16 = values_2d[t2].half()
        else:
            n_packed = math.ceil(d_head / 8)
            tier2_sign_k = torch.empty(0, n_packed, dtype=torch.uint8)
            tier2_v_fp16 = torch.empty(0, d_head, dtype=torch.float16)

        return {
            "tier1_kv": tier1_kv,
            "tier2_sign_k": tier2_sign_k,
            "tier2_v_fp16": tier2_v_fp16,
            "tier1_indices": t1,
            "tier2_indices": t2,
            "tier3_indices": t3,
            "n_tokens": n_tokens,
            "d_head": d_head,
            "original_shape": original_shape,
            "layer_idx": layer_idx,
            "block_id": block_id,
        }

    def decode_block(
        self,
        storage: Dict[str, object],
    ) -> torch.Tensor:
        """Reconstruct a KV block tensor from tier storage.

        Returns:
            (2, n_tokens, d_head) float32 — [0]=keys, [1]=values.
            Tier-3 tokens are zero-filled.
        """
        n_tokens: int = storage["n_tokens"]  # type: ignore[assignment]
        d_head: int = storage["d_head"]      # type: ignore[assignment]
        t1: torch.Tensor = storage["tier1_indices"]   # type: ignore[assignment]
        t2: torch.Tensor = storage["tier2_indices"]   # type: ignore[assignment]

        keys_out = torch.zeros(n_tokens, d_head, dtype=torch.float32)
        vals_out = torch.zeros(n_tokens, d_head, dtype=torch.float32)

        # Tier-1: exact FP16 round-trip
        tier1_kv: torch.Tensor = storage["tier1_kv"]   # type: ignore[assignment]
        if t1.numel() > 0 and tier1_kv.numel() > 0:
            keys_out[t1] = tier1_kv[:, :d_head].float()
            vals_out[t1] = tier1_kv[:, d_head:].float()

        # Tier-2: approximate Key (±1 sign) + exact FP16 Value
        tier2_sign_k: torch.Tensor = storage["tier2_sign_k"]  # type: ignore[assignment]
        tier2_v_fp16: torch.Tensor = storage["tier2_v_fp16"]  # type: ignore[assignment]
        if t2.numel() > 0 and tier2_sign_k.numel() > 0:
            approx_k = _unpack_signs_to_pm1(tier2_sign_k, d_head)
            keys_out[t2] = approx_k
            vals_out[t2] = tier2_v_fp16.float()

        # Tier-3 tokens remain zero
        return torch.stack([keys_out, vals_out], dim=0)  # (2, n_tokens, d_head)

    # ------------------------------------------------------------------ #
    # Multi-head block encode / decode (vLLM 3-D shape support)            #
    # ------------------------------------------------------------------ #

    def encode_block_multihead(
        self,
        key_block: torch.Tensor,
        value_block: torch.Tensor,
        layer_idx: int,
        block_id: int = 0,
    ) -> List[Dict[str, object]]:
        """Encode a 3-D block (block_size, num_kv_heads, head_size) per-head.

        vLLM paged KV cache shape:
            [2, num_blocks, block_size, num_kv_heads, head_size]

        For a single extracted block:
            key_block   : (block_size, num_kv_heads, head_size)
            value_block : (block_size, num_kv_heads, head_size)

        This method compresses each head independently and returns a list of
        per-head storage dicts (length == num_kv_heads).

        Args:
            key_block:   (block_size, num_kv_heads, head_size)
            value_block: (block_size, num_kv_heads, head_size)
            layer_idx:   transformer layer index.
            block_id:    physical block id.

        Returns:
            List of per-head storage dicts, each as returned by encode_block.
        """
        block_size, num_kv_heads, head_size = key_block.shape
        per_head: List[Dict[str, object]] = []
        for h in range(num_kv_heads):
            k_h = key_block[:, h, :]    # (block_size, head_size)
            v_h = value_block[:, h, :]  # (block_size, head_size)
            storage = self.encode_block(k_h, v_h, layer_idx=layer_idx, block_id=block_id)
            storage["head_idx"] = h
            per_head.append(storage)
        return per_head

    def decode_block_multihead(
        self,
        per_head_storages: List[Dict[str, object]],
    ) -> torch.Tensor:
        """Decode a list of per-head storage dicts back to 3-D block tensors.

        Args:
            per_head_storages: list of dicts as returned by encode_block_multihead.

        Returns:
            (2, block_size, num_kv_heads, head_size) float32 — [0]=keys, [1]=values.
        """
        decoded = [self.decode_block(s) for s in per_head_storages]
        # each decoded: (2, block_size, d_head)  where d_head == head_size
        # stack along new head dimension → (2, block_size, num_kv_heads, head_size)
        keys = torch.stack([d[0] for d in decoded], dim=1)  # (block_size, H, head_size)
        vals = torch.stack([d[1] for d in decoded], dim=1)
        return torch.stack([keys, vals], dim=0)

    # ------------------------------------------------------------------ #
    # Memory estimation                                                     #
    # ------------------------------------------------------------------ #

    def memory_bytes_estimate(
        self,
        n_tokens: int,
        d_head: int,
    ) -> Dict[str, Union[int, float]]:
        """Estimate compressed memory usage for n_tokens tokens with d_head dims.

        Provides the same interface as
        ``LeverageScoreCompressor.memory_bytes_estimate``.

        Returns:
            dict: tier1_bytes, tier2_bytes, tier3_bytes, total_bytes,
                  baseline_bytes (FP32), reduction_ratio.
        """
        n1 = max(1, int(n_tokens * self.tier1_ratio))
        n3 = max(0, int(n_tokens * self.tier3_ratio))
        n2 = max(0, n_tokens - n1 - n3)
        n_packed = math.ceil(d_head / 8)

        tier1_bytes = n1 * 2 * d_head * 2              # FP16 key + value
        tier2_sign_bytes = n2 * n_packed                # 1-bit packed key
        tier2_val_bytes = n2 * d_head * 2               # FP16 value
        tier2_bytes = tier2_sign_bytes + tier2_val_bytes
        tier3_bytes = 0

        total_bytes = tier1_bytes + tier2_bytes
        baseline_bytes = n_tokens * 2 * d_head * 4     # FP32

        reduction_ratio = (
            1.0 - total_bytes / baseline_bytes if baseline_bytes > 0 else 0.0
        )
        return {
            "tier1_bytes": tier1_bytes,
            "tier2_bytes": tier2_bytes,
            "tier3_bytes": tier3_bytes,
            "total_bytes": total_bytes,
            "baseline_bytes": baseline_bytes,
            "reduction_ratio": reduction_ratio,
        }

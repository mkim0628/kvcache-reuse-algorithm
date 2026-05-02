"""Activity C — LeverageScoreCompressor: training-free 3-tier KV compression.

Classifies tokens by approximate leverage score (rank-k approximation of
the statistical leverage score) into:
  Tier-1 (top 20%)    → FP16 full KV (high information tokens)
  Tier-2 (middle 60%) → 1-bit sign VQ Key + FP16 Value
  Tier-3 (bottom 20%) → evicted (near-zero information)

Reference: CapKV (arXiv 2604.25975) — Information Bottleneck + leverage scores.
"""

import math
from typing import Dict, Union

import numpy as np
import torch


class LeverageScoreCompressor:
    """Training-free 3-tier KV compressor based on approximate leverage scores.

    Leverage score for token i:
        s_i = K_i^T (K^T K + λI)^{-1} K_i

    Computed via rank-k eigendecomposition of K^T K for O(N·k·d) complexity.
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
        # middle tier is implicitly 1 - tier1_ratio - tier3_ratio

    def compute_leverage_scores(
        self,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rank-k approximate leverage scores for each token.

        Args:
            keys: (n_tokens, d_head) float tensor — Key matrix K.

        Returns:
            scores: (n_tokens,) float32 — leverage score per token.
        """
        n_tokens, d_head = keys.shape
        k = min(self.rank, n_tokens, d_head)

        kf = keys.float()
        ktk = kf.T @ kf  # (d_head, d_head)

        eigenvalues, eigenvectors = torch.linalg.eigh(ktk)
        # eigh returns ascending order; take top-k
        v_k = eigenvectors[:, -k:]       # (d_head, k)
        lambda_k = eigenvalues[-k:]      # (k,)

        proj = kf @ v_k                  # (n_tokens, k)
        inv_diag = 1.0 / (lambda_k + self.reg_lambda)  # (k,)
        scores = (proj ** 2 * inv_diag).sum(dim=-1)    # (n_tokens,)
        return scores

    def classify(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Classify tokens into 3 tiers by leverage score.

        Args:
            keys:   (n_tokens, d_head)
            values: (n_tokens, d_head) — values are not used for scoring,
                    only passed through to keep the API symmetric.

        Returns:
            dict with keys: tier1, tier2, tier3 (index tensors), scores.
        """
        n_tokens = keys.shape[0]
        scores = self.compute_leverage_scores(keys)

        sorted_indices = torch.argsort(scores, descending=True)

        n1 = max(1, int(n_tokens * self.tier1_ratio))
        n3 = max(0, int(n_tokens * self.tier3_ratio))

        tier1_indices = sorted_indices[:n1]
        # ensure bottom tier does not overlap with top tier for tiny n_tokens
        if n_tokens - n3 > n1:
            tier3_indices = sorted_indices[n_tokens - n3:]
            tier2_indices = sorted_indices[n1: n_tokens - n3]
        else:
            # edge case: n_tokens too small; keep tier1, no tier2/tier3
            tier3_indices = torch.empty(0, dtype=torch.long)
            tier2_indices = torch.empty(0, dtype=torch.long)

        return {
            "tier1": tier1_indices,
            "tier2": tier2_indices,
            "tier3": tier3_indices,
            "scores": scores,
        }

    def to_sign_code(
        self,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        """Pack the sign bits of each Key dimension into uint8 bytes.

        Args:
            keys: (n_tokens, d_head) — any float dtype.

        Returns:
            packed: (n_tokens, ceil(d_head / 8)) uint8 tensor.
        """
        signs = (keys >= 0).to(torch.uint8)  # (n_tokens, d_head)
        packed = _packbits_2d(signs)          # (n_tokens, ceil(d_head/8))
        return packed

    def encode(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Classify and compress KV tensors into tier-specific representations.

        Args:
            keys:      (n_tokens, d_head) float32
            values:    (n_tokens, d_head) float32
            layer_idx: transformer layer index (stored as metadata)
            tensor_id: optional tensor identifier (stored as metadata)

        Returns:
            storage dict with:
              tier1_kv        — FP16, (n1, 2*d_head)
              tier2_sign_k    — uint8 packed, (n2, ceil(d_head/8))
              tier2_v_fp16    — FP16, (n2, d_head)
              tier1_indices, tier2_indices, tier3_indices — int64 index tensors
              n_tokens, d_head, layer_idx, tensor_id      — scalar metadata
        """
        n_tokens, d_head = keys.shape
        classification = self.classify(keys, values)

        t1 = classification["tier1"]
        t2 = classification["tier2"]
        t3 = classification["tier3"]

        # Tier-1: full FP16 KV concatenated
        if t1.numel() > 0:
            tier1_kv = torch.cat(
                [keys[t1].half(), values[t1].half()], dim=-1
            )  # (n1, 2*d_head)
        else:
            tier1_kv = torch.empty(0, 2 * d_head, dtype=torch.float16)

        # Tier-2: sign-packed Key + FP16 Value
        if t2.numel() > 0:
            tier2_sign_k = self.to_sign_code(keys[t2])  # (n2, ceil(d_head/8))
            tier2_v_fp16 = values[t2].half()            # (n2, d_head)
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
            "layer_idx": layer_idx,
            "tensor_id": tensor_id,
        }

    def decode(
        self,
        storage: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct a (n_tokens, 2*d_head) float32 tensor from tier storage.

        Tier-1: FP16 → float32 placed at tier1_indices.
        Tier-2 Key: sign code → ±1 approximation; Value: FP16 → float32.
        Tier-3: zeros (evicted).
        """
        n_tokens = storage["n_tokens"]
        d_head = storage["d_head"]
        t1 = storage["tier1_indices"]
        t2 = storage["tier2_indices"]

        out = torch.zeros(n_tokens, 2 * d_head, dtype=torch.float32)

        # Tier-1: exact FP16 round-trip
        if t1.numel() > 0:
            out[t1] = storage["tier1_kv"].float()

        # Tier-2: approximate Key (±1 sign) + exact FP16 Value
        if t2.numel() > 0:
            sign_code = storage["tier2_sign_k"]
            approx_k = _unpack_signs_to_pm1(sign_code, d_head)  # (n2, d_head)
            approx_v = storage["tier2_v_fp16"].float()          # (n2, d_head)
            out[t2, :d_head] = approx_k
            out[t2, d_head:] = approx_v

        # Tier-3 tokens remain zero (already initialised)
        return out

    def memory_bytes_estimate(
        self,
        n_tokens: int,
        d_head: int,
    ) -> Dict[str, Union[int, float]]:
        """Estimate memory usage for n_tokens tokens with given d_head.

        Returns a dict with per-tier bytes and the overall reduction ratio
        versus a FP32 baseline (n_tokens * 2 * d_head * 4 bytes).
        """
        n1 = max(1, int(n_tokens * self.tier1_ratio))
        n3 = max(0, int(n_tokens * self.tier3_ratio))
        n2 = max(0, n_tokens - n1 - n3)

        n_packed = math.ceil(d_head / 8)

        # Tier-1: FP16 key + value
        tier1_bytes = n1 * 2 * d_head * 2

        # Tier-2: 1-bit packed key + FP16 value
        tier2_sign_bytes = n2 * n_packed
        tier2_val_bytes = n2 * d_head * 2
        tier2_bytes = tier2_sign_bytes + tier2_val_bytes

        # Tier-3: evicted, 0 bytes
        tier3_bytes = 0

        total_bytes = tier1_bytes + tier2_bytes + tier3_bytes

        # FP32 baseline: both key and value in float32
        baseline_bytes = n_tokens * 2 * d_head * 4

        reduction_ratio = 1.0 - total_bytes / baseline_bytes if baseline_bytes > 0 else 0.0

        return {
            "tier1_bytes": tier1_bytes,
            "tier2_bytes": tier2_bytes,
            "tier3_bytes": tier3_bytes,
            "total_bytes": total_bytes,
            "baseline_bytes": baseline_bytes,
            "reduction_ratio": reduction_ratio,
        }


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _packbits_2d(bits: torch.Tensor) -> torch.Tensor:
    """Pack a 2-D uint8 tensor of 0/1 values along axis=-1 into bytes.

    Equivalent to ``numpy.packbits(x, axis=-1)`` but implemented via numpy
    for broad PyTorch version compatibility.

    Args:
        bits: (n_tokens, d_head) uint8 tensor with values in {0, 1}.

    Returns:
        (n_tokens, ceil(d_head / 8)) uint8 tensor.
    """
    arr = bits.cpu().numpy().astype(np.uint8)
    packed_np = np.packbits(arr, axis=-1)        # (n_tokens, ceil(d_head/8))
    return torch.from_numpy(packed_np.copy())


def _unpackbits_2d(packed: torch.Tensor, d_head: int) -> torch.Tensor:
    """Unpack a 2-D uint8 byte tensor to individual bits, truncating to d_head.

    Args:
        packed: (n_tokens, ceil(d_head/8)) uint8 tensor.
        d_head: number of bit columns to keep.

    Returns:
        (n_tokens, d_head) uint8 tensor with values in {0, 1}.
    """
    arr = packed.cpu().numpy().astype(np.uint8)
    unpacked_np = np.unpackbits(arr, axis=-1)[:, :d_head]  # (n_tokens, d_head)
    return torch.from_numpy(unpacked_np.copy())


def _unpack_signs_to_pm1(
    sign_code: torch.Tensor,
    d_head: int,
) -> torch.Tensor:
    """Unpack uint8 bit-packed sign codes to ±1.0 float tensor.

    Args:
        sign_code: (n_tokens, ceil(d_head/8)) uint8
        d_head:    original key dimension

    Returns:
        (n_tokens, d_head) float32 with values in {-1.0, +1.0}
    """
    unpacked = _unpackbits_2d(sign_code, d_head)  # (n_tokens, d_head) uint8
    return 2.0 * unpacked.float() - 1.0           # 0 → -1.0, 1 → +1.0

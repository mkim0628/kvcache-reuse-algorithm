"""RSimVQCodec — Residual Simple VQ codec operating in pre-RoPE space.

Based on VQKV (arXiv 2603.16435). Training-free: codebooks are learned once
via k-means on calibration data and then reused for all subsequent encoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class VQCodebookConfig:
    codebook_size: int = 256      # M: number of codewords
    n_residuals: int = 4          # residual VQ stages
    d_head: int = 128             # head dimension
    n_layers: int = 32
    n_heads: int = 8
    max_iter_kmeans: int = 100
    rope_base: int = 10000
    seed: int = 42
    recent_window: int = 64       # FP16 tokens kept uncompressed


class VQCodec:
    """Training-free Residual Simple VQ codec (VQKV arXiv 2603.16435).

    Codebooks are learned once via fit() and reused for all encodes.
    key_codebooks[layer_idx][residual_idx]: Tensor [M, d_head]
    val_codebooks[layer_idx][residual_idx]: Tensor [M, d_head]
    """

    def __init__(self, config: VQCodebookConfig) -> None:
        self.config = config
        self.key_codebooks: Dict[int, List[torch.Tensor]] = {}
        self.val_codebooks: Dict[int, List[torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # RoPE helpers                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_cos_sin(
        positions: torch.Tensor,  # [n_tokens]
        d_head: int,
        base: int = 10000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables for RoPE. Returns [n_tokens, d_head/2] each."""
        half = d_head // 2
        freq_idx = torch.arange(half, dtype=torch.float32, device=positions.device)
        # theta_i = 1 / base^(2i/d_head)
        inv_freq = 1.0 / (base ** (freq_idx / half))
        # [n_tokens, half]
        angles = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def apply_rope(
        k_pre: torch.Tensor,      # [n_tokens, n_heads, d_head]
        positions: torch.Tensor,  # [n_tokens]
        base: int = 10000,
    ) -> torch.Tensor:
        """Apply RoPE: interleaved rotation on consecutive (d0, d1) pairs."""
        n_tokens, n_heads, d_head = k_pre.shape
        cos, sin = VQCodec._make_cos_sin(positions, d_head, base)
        # cos/sin: [n_tokens, d_head/2]

        # Split into even/odd
        x = k_pre.float()
        x0 = x[..., 0::2]  # [n_tokens, n_heads, d_head/2]
        x1 = x[..., 1::2]

        cos = cos.unsqueeze(1)  # [n_tokens, 1, d_head/2]
        sin = sin.unsqueeze(1)

        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos

        # Interleave back
        out = torch.stack([out0, out1], dim=-1).reshape(n_tokens, n_heads, d_head)
        return out.to(k_pre.dtype)

    @staticmethod
    def inverse_rope(
        k_post: torch.Tensor,     # [n_tokens, n_heads, d_head]
        positions: torch.Tensor,  # [n_tokens]
        base: int = 10000,
    ) -> torch.Tensor:
        """Inverse RoPE: recover pre-RoPE representation from post-RoPE tensor.

        Since RoPE is an orthogonal rotation, the inverse is the transpose:
          k_pre = R(-theta) * k_post
        which means applying cos(theta) with negated sin.
        """
        n_tokens, n_heads, d_head = k_post.shape
        cos, sin = VQCodec._make_cos_sin(positions, d_head, base)

        x = k_post.float()
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Inverse rotation: multiply by transpose of rotation matrix
        out0 = x0 * cos + x1 * sin
        out1 = -x0 * sin + x1 * cos

        out = torch.stack([out0, out1], dim=-1).reshape(n_tokens, n_heads, d_head)
        return out.to(k_post.dtype)

    # ------------------------------------------------------------------ #
    # K-means helpers                                                      #
    # ------------------------------------------------------------------ #

    def _kmeans(
        self,
        data: torch.Tensor,  # [N, d_head]
        codebook_size: int,
        max_iter: int,
    ) -> torch.Tensor:
        """Simple k-means returning codebook [M, d_head]."""
        N, D = data.shape
        M = min(codebook_size, N)
        # Random initialization (seed already set by caller)
        idx = torch.randperm(N, device=data.device)[:M]
        centroids = data[idx].float().clone()

        for _ in range(max_iter):
            # Assign each point to nearest centroid
            dists = torch.cdist(data.float(), centroids)  # [N, M]
            assignments = dists.argmin(dim=1)             # [N]

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(M, device=data.device, dtype=torch.float32)
            new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand_as(data.float()), data.float())
            counts.scatter_add_(0, assignments, torch.ones(N, device=data.device, dtype=torch.float32))

            # Avoid division by zero (keep old centroid for empty clusters)
            mask = counts > 0
            new_centroids[mask] = new_centroids[mask] / counts[mask].unsqueeze(1)
            new_centroids[~mask] = centroids[~mask]

            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids.to(data.dtype)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        calibration_keys: torch.Tensor,  # [n_tokens, d_head] pre-RoPE keys
        calibration_vals: torch.Tensor,  # [n_tokens, d_head] values
        layer_idx: int,
        head_idx: int = 0,
    ) -> None:
        """Learn codebooks via k-means. Call per layer (aggregates all heads)."""
        torch.manual_seed(self.config.seed)
        M = self.config.codebook_size
        n_r = self.config.n_residuals
        max_iter = self.config.max_iter_kmeans

        key_books: List[torch.Tensor] = []
        val_books: List[torch.Tensor] = []

        # Residual VQ for keys
        residual_k = calibration_keys.float().clone()
        for r in range(n_r):
            cb = self._kmeans(residual_k, M, max_iter)
            key_books.append(cb)
            codes = torch.cdist(residual_k, cb).argmin(dim=1)
            residual_k = residual_k - cb[codes]

        # Residual VQ for values
        residual_v = calibration_vals.float().clone()
        for r in range(n_r):
            cb = self._kmeans(residual_v, M, max_iter)
            val_books.append(cb)
            codes = torch.cdist(residual_v, cb).argmin(dim=1)
            residual_v = residual_v - cb[codes]

        self.key_codebooks[layer_idx] = key_books
        self.val_codebooks[layer_idx] = val_books

    def _encode_residual(
        self,
        data: torch.Tensor,          # [N, d_head]
        codebooks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Encode via residual VQ. Returns codes [N, n_residuals] int16."""
        N = data.shape[0]
        n_r = len(codebooks)
        codes = torch.zeros(N, n_r, dtype=torch.int16)
        residual = data.float().clone()
        for r, cb in enumerate(codebooks):
            dists = torch.cdist(residual, cb.float())
            assigned = dists.argmin(dim=1)  # [N]
            codes[:, r] = assigned.to(torch.int16)
            residual = residual - cb.float()[assigned]
        return codes

    def _decode_residual(
        self,
        codes: torch.Tensor,         # [N, n_residuals] int16
        codebooks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Decode by summing residual VQ stages. Returns [N, d_head]."""
        N = codes.shape[0]
        d = codebooks[0].shape[1]
        recon = torch.zeros(N, d, dtype=torch.float32)
        for r, cb in enumerate(codebooks):
            idx = codes[:, r].long()
            recon = recon + cb.float()[idx]
        return recon

    def encode(
        self,
        kv: torch.Tensor,          # [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
        positions: torch.Tensor,   # [n_tokens] int64
    ) -> dict:
        """Encode KV tensor with residual VQ. Keys are inverse-RoPEd first."""
        n_tokens, _, n_heads, d_head = kv.shape

        if layer_idx not in self.key_codebooks:
            # Fallback: auto-fit on the provided data
            k_flat = kv[:, 0].reshape(n_tokens * n_heads, d_head)
            v_flat = kv[:, 1].reshape(n_tokens * n_heads, d_head)
            k_pre = VQCodec.inverse_rope(kv[:, 0], positions, self.config.rope_base)
            k_pre_flat = k_pre.reshape(n_tokens * n_heads, d_head)
            self.fit(k_pre_flat, v_flat, layer_idx)

        base = self.config.rope_base
        k_post = kv[:, 0]  # [n_tokens, n_heads, d_head]
        v = kv[:, 1]       # [n_tokens, n_heads, d_head]

        k_pre = VQCodec.inverse_rope(k_post, positions, base)

        # Encode each head independently using same codebook (shared across heads)
        key_codes_list = []
        val_codes_list = []
        for h in range(n_heads):
            k_h = k_pre[:, h, :]  # [n_tokens, d_head]
            v_h = v[:, h, :]

            k_codes_h = self._encode_residual(k_h, self.key_codebooks[layer_idx])
            v_codes_h = self._encode_residual(v_h, self.val_codebooks[layer_idx])
            key_codes_list.append(k_codes_h)
            val_codes_list.append(v_codes_h)

        # Stack: [n_tokens, n_heads, n_residuals]
        key_codes = torch.stack(key_codes_list, dim=1)
        val_codes = torch.stack(val_codes_list, dim=1)

        return {
            "key_codes": key_codes,
            "val_codes": val_codes,
            "layer_idx": layer_idx,
            "n_tokens": n_tokens,
            "positions": positions,
        }

    def decode(
        self,
        codes: dict,
        layer_idx: int,
    ) -> torch.Tensor:
        """Decode VQ codes back to [n_tokens, 2, n_heads, d_head] float16."""
        key_codes = codes["key_codes"]   # [n_tokens, n_heads, n_residuals]
        val_codes = codes["val_codes"]
        positions = codes["positions"]
        n_tokens = codes["n_tokens"]

        if layer_idx not in self.key_codebooks:
            raise ValueError(f"Codebooks for layer {layer_idx} not fitted.")

        n_heads = key_codes.shape[1]
        d_head = self.key_codebooks[layer_idx][0].shape[1]

        k_pre_heads = []
        v_heads = []
        for h in range(n_heads):
            k_codes_h = key_codes[:, h, :]  # [n_tokens, n_residuals]
            v_codes_h = val_codes[:, h, :]

            k_pre_h = self._decode_residual(k_codes_h, self.key_codebooks[layer_idx])
            v_h = self._decode_residual(v_codes_h, self.val_codebooks[layer_idx])
            k_pre_heads.append(k_pre_h)
            v_heads.append(v_h)

        k_pre = torch.stack(k_pre_heads, dim=1).to(torch.float16)  # [n_t, n_h, d_h]
        v_tensor = torch.stack(v_heads, dim=1).to(torch.float16)

        # Apply RoPE to get post-RoPE keys
        k_post = VQCodec.apply_rope(k_pre, positions, self.config.rope_base)

        # Stack into [n_tokens, 2, n_heads, d_head]
        return torch.stack([k_post, v_tensor], dim=1)

    def compression_ratio(self) -> float:
        """Effective compression ratio for the VQ+recent_window scheme.

        VQ operates on vectors of d_head dimensions: each residual stage encodes
        an entire d_head-dimensional vector into one code index.  So the VQ storage
        cost for one token-head entry is:
            n_residuals × ceil(log2(M)) bits       [for the codes]
        vs the FP16 baseline:
            d_head × 16 bits                        [for the raw tensor]

        With recent_window tokens kept as FP16, the effective ratio across a
        typical block of `typical_total` tokens is:
            vq_fraction  = (total - recent_window) / total
            avg_bits_per_dhead_fp16 = d_head × 16
            avg_bits_vq  = n_r × bits_per_code
            ratio = 1 - (vq_fraction × avg_bits_vq + fp16_fraction × avg_bits_fp16)
                                     / avg_bits_fp16
        """
        M = self.config.codebook_size
        n_r = self.config.n_residuals
        d_head = self.config.d_head
        bits_per_code = math.ceil(math.log2(max(M, 2)))

        # Bits to represent one token-head vector
        fp16_bits_per_vec = d_head * 16.0          # FP16 baseline
        vq_bits_per_vec = n_r * bits_per_code       # VQ codes (one code per residual stage)

        # Fraction of tokens that are VQ-compressed vs kept FP16
        typical_total = 512
        recent_w = self.config.recent_window
        vq_fraction = max(0.0, (typical_total - recent_w) / typical_total)
        fp16_fraction = 1.0 - vq_fraction

        avg_bits = vq_fraction * vq_bits_per_vec + fp16_fraction * fp16_bits_per_vec
        ratio = 1.0 - avg_bits / fp16_bits_per_vec
        return max(0.0, ratio)

    def save(self, path: str) -> None:
        torch.save(
            {
                "key_codebooks": self.key_codebooks,
                "val_codebooks": self.val_codebooks,
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        # weights_only=False needed because VQCodebookConfig dataclass is not a plain tensor
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.key_codebooks = data["key_codebooks"]
        self.val_codebooks = data["val_codebooks"]
        self.config = data["config"]

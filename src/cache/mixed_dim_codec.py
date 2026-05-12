"""MixedDimPerTokenBudgetCodec — Activity C (training-free KV compression).

Each token's "loss score" is a product of attention importance, value magnitude,
and per-dimension compressibility (PCA variance ratio).  A bisection search finds
the global threshold λ* that satisfies the caller-specified memory budget_ratio.
Dimensions whose loss score is below λ* are zeroed (dropped); the rest are kept.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class MixedDimConfig:
    n_heads: int = 8
    d_head: int = 64
    budget_ratio: float = 0.50      # fraction of dims to retain (0‒1)
    bisection_iters: int = 64
    min_retain_ratio: float = 0.10  # per-token minimum retention guard
    attn_importance_weight: float = 1.0
    value_magnitude_weight: float = 1.0
    compressibility_weight: float = 1.0
    seed: int = 42


class MixedDimPerTokenBudgetCodec:
    """Training-free, per-token mixed-dim budget KV codec (Activity C).

    encode() zeroes out low-importance dimensions determined by bisection search.
    decode() is a no-op (returns masked_kv as-is) since zeroed dims carry no info.
    """

    def __init__(self, config: MixedDimConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Core computation                                                     #
    # ------------------------------------------------------------------ #

    def compute_loss_scores(
        self,
        kv: torch.Tensor,                           # [n_tokens, 2, n_heads, d_head]
        attn_weights: Optional[torch.Tensor] = None, # [n_tokens] — external importance
    ) -> torch.Tensor:
        """Return per-(token, head, dim) loss scores [n_tokens, n_heads, d_head].

        Higher score → more important → retained.
        """
        k = kv[:, 0].float()  # [n_tokens, n_heads, d_head]
        v = kv[:, 1].float()  # [n_tokens, n_heads, d_head]

        # 1. Attention importance — [n_tokens, n_heads, 1]
        if attn_weights is not None:
            # Broadcast per-token scalar to per-(token,head,1)
            attn_imp = attn_weights.float().view(-1, 1, 1).expand(
                kv.shape[0], kv.shape[2], 1
            )
        else:
            attn_imp = v.norm(dim=-1, keepdim=True)  # value magnitude as proxy

        # 2. Value magnitude — [n_tokens, n_heads, 1]
        val_mag = v.norm(dim=-1, keepdim=True)

        # 3. Per-dimension compressibility = normalised per-dim variance
        #    (high variance → informative → (1 - compress_score) is low → less to drop)
        k_var = k.var(dim=0)                                      # [n_heads, d_head]
        v_var = v.var(dim=0)
        kv_var = (k_var + v_var) / 2                              # [n_heads, d_head]
        total_var = kv_var.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        compress_score = kv_var / total_var                        # [n_heads, d_head]
        compress_score = compress_score.unsqueeze(0)               # [1, n_heads, d_head]

        loss = (
            self.config.attn_importance_weight * attn_imp
            * self.config.value_magnitude_weight * val_mag
            * self.config.compressibility_weight * compress_score
        )  # [n_tokens, n_heads, d_head]

        return loss

    def find_threshold(
        self,
        loss_scores: torch.Tensor,          # [n_tokens, n_heads, d_head]
        budget_ratio: Optional[float] = None,
    ) -> float:
        """Bisection search for λ* such that retained_fraction ≈ budget_ratio."""
        ratio = budget_ratio if budget_ratio is not None else self.config.budget_ratio
        lo = 0.0
        hi = float(loss_scores.max().item())
        if hi == 0.0:
            return 0.0
        for _ in range(self.config.bisection_iters):
            mid = (lo + hi) / 2
            retained = (loss_scores >= mid).float().mean().item()
            if retained > ratio:
                lo = mid  # threshold too low — raise it to drop more dims
            else:
                hi = mid
        return (lo + hi) / 2

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def encode(
        self,
        kv: torch.Tensor,                            # [n_tokens, 2, n_heads, d_head]
        attn_weights: Optional[torch.Tensor] = None, # [n_tokens]
        budget_ratio: Optional[float] = None,
    ) -> dict:
        """Compress KV by zeroing low-loss-score dimensions.

        Returns a dict with keys:
          masked_kv, retain_mask, lambda_star, budget_ratio, n_tokens, n_heads, d_head
        """
        n_tokens, _, n_heads, d_head = kv.shape
        loss_scores = self.compute_loss_scores(kv, attn_weights)  # [n_tokens, n_heads, d_head]
        lam = self.find_threshold(loss_scores, budget_ratio)
        retain_mask = loss_scores >= lam  # [n_tokens, n_heads, d_head] bool

        # Enforce min_retain_ratio per (token, head) pair
        # Use ceil so that e.g. 0.10 * 32 = 3.2 → 4 dims (not 3), ensuring >= ratio
        import math
        topk_dim = max(1, math.ceil(d_head * self.config.min_retain_ratio))
        per_token_ratio = retain_mask.float().mean(dim=-1)   # [n_tokens, n_heads]
        under_min = per_token_ratio < self.config.min_retain_ratio  # [n_tokens, n_heads]
        if under_min.any():
            # For tokens/heads below minimum, force-keep the top-k dims by loss score
            scores_flat = loss_scores[under_min]                      # [M, d_head]
            topk_idx = scores_flat.topk(topk_dim, dim=-1).indices     # [M, topk_dim]
            mask_flat = retain_mask[under_min]                        # [M, d_head]
            mask_flat.scatter_(-1, topk_idx, True)
            retain_mask[under_min] = mask_flat

        masked_kv = kv.clone()
        # Apply mask to both key (dim=1,idx=0) and value (dim=1,idx=1)
        mask_4d = retain_mask.unsqueeze(1).expand_as(masked_kv)
        masked_kv = masked_kv * mask_4d.to(masked_kv.dtype)

        actual_ratio = retain_mask.float().mean().item()
        return {
            "masked_kv": masked_kv,
            "retain_mask": retain_mask,
            "lambda_star": lam,
            "budget_ratio": actual_ratio,
            "n_tokens": n_tokens,
            "n_heads": n_heads,
            "d_head": d_head,
        }

    def decode(self, encoded: dict) -> torch.Tensor:
        """Reconstruct KV — zeroed dims remain zero (lossy, but near-lossless for low-importance dims)."""
        return encoded["masked_kv"]

    def memory_reduction_ratio(self, encoded: dict) -> float:
        """Fraction of memory saved = 1 − retained_ratio."""
        return 1.0 - encoded["budget_ratio"]

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,                         # [n_tokens, 2, n_heads, d_head]
        attn_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CacheStore-compatible hook: encode then decode → masked_kv."""
        encoded = self.encode(value, attn_weights)
        return self.decode(encoded)

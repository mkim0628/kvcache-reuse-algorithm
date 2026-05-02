"""Activity C — CacheConfig extension for SignVQ + LeverageScore compression.

This module extends vLLM 0.20.0's ``CacheConfig`` with the compression
parameters required for the 2026-05-02 B+C cycle:

    enable_sign_vq      : bool   — enable SignVQ non-contiguous reuse
    tier1_ratio         : float  — fraction of tokens stored as FP16 (Tier-1)
    tier2_ratio         : float  — fraction stored as sign+FP16 Value (Tier-2)
    hamming_threshold   : float  — Hamming distance threshold for approx hits

Because vLLM's ``CacheConfig`` is a frozen Pydantic ``@config`` dataclass, we
do NOT modify it directly.  Instead we provide:

1. ``SignVQCacheParams`` — a standalone dataclass that carries the new fields.
   This can be constructed independently and passed alongside a standard
   ``CacheConfig``.

2. ``SignVQCacheConfigMixin`` — a mixin class for use when subclassing the
   vLLM engine config (``VllmConfig``).  The mixin exposes an additional
   ``sign_vq`` attribute of type ``SignVQCacheParams``.

3. ``build_sign_vq_compressor`` — convenience factory that reads a
   ``SignVQCacheParams`` and returns a ready-to-use ``VllmLeverageCompressor``.

4. ``build_sign_vq_index`` — convenience factory that reads a
   ``SignVQCacheParams`` and returns a ready-to-use ``SignVQSegmentIndex``.

This approach follows the porting principle "기존 CacheConfig 필드를 건드리지
않는다. 새 필드는 서브클래스에서 추가한다."

Usage:
    from vllm_integration.cache_config_extension import (
        SignVQCacheParams,
        build_sign_vq_compressor,
        build_sign_vq_index,
    )

    params = SignVQCacheParams(
        enable_sign_vq=True,
        tier1_ratio=0.20,
        tier2_ratio=0.60,
        hamming_threshold=0.15,
    )
    compressor = build_sign_vq_compressor(params)
    index      = build_sign_vq_index(params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex


# --------------------------------------------------------------------------- #
# SignVQCacheParams                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class SignVQCacheParams:
    """Additional KV-cache parameters for the SignVQ + LeverageScore pipeline.

    These fields extend vLLM's CacheConfig without modifying it.

    Attributes:
        enable_sign_vq:      Master toggle.  When False all other fields have
                             no effect and vLLM behaves as a stock deployment.
        tier1_ratio:         Fraction of tokens kept as full FP16 (Tier-1).
                             Default 0.20 (top 20% by leverage score).
        tier2_ratio:         Fraction stored as 1-bit sign Key + FP16 Value
                             (Tier-2).  Tier-3 is 1 − tier1_ratio − tier2_ratio
                             and is evicted entirely.
                             Default 0.60 (middle 60%).
        hamming_threshold:   Normalised Hamming distance upper bound for an
                             approximate sign-VQ hit.
                             Normalised distance = differing bits /
                                                   (n_tokens × d_head).
                             Default 0.15.
        rank:                Rank-k approximation for leverage score computation.
                             Default 32.
        reg_lambda:          Regularisation constant λ for (K^T K + λI)^{-1}.
                             Default 1e-3.
        chunk_size:          Tokens per KV chunk / segment.  Should match (or
                             divide) vLLM's block_size.
                             Default 16 (= vLLM CacheConfig.DEFAULT_BLOCK_SIZE).
        max_entries:         LRU capacity per store (FP16 store and sign store
                             are each capped at this value).
                             Default 2000.
    """

    enable_sign_vq: bool = False
    tier1_ratio: float = 0.20
    tier2_ratio: float = 0.60
    hamming_threshold: float = 0.15
    rank: int = 32
    reg_lambda: float = 1e-3
    chunk_size: int = 16
    max_entries: int = 2000

    def __post_init__(self) -> None:
        """Validate parameter constraints."""
        if not (0.0 < self.tier1_ratio <= 1.0):
            raise ValueError(
                f"tier1_ratio must be in (0, 1], got {self.tier1_ratio}"
            )
        if not (0.0 <= self.tier2_ratio < 1.0):
            raise ValueError(
                f"tier2_ratio must be in [0, 1), got {self.tier2_ratio}"
            )
        tier3_ratio = 1.0 - self.tier1_ratio - self.tier2_ratio
        if tier3_ratio < 0.0:
            raise ValueError(
                f"tier1_ratio ({self.tier1_ratio}) + tier2_ratio "
                f"({self.tier2_ratio}) must be ≤ 1.0"
            )
        if not (0.0 <= self.hamming_threshold <= 1.0):
            raise ValueError(
                f"hamming_threshold must be in [0, 1], got {self.hamming_threshold}"
            )
        if self.rank < 1:
            raise ValueError(f"rank must be ≥ 1, got {self.rank}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {self.chunk_size}")
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be ≥ 1, got {self.max_entries}")

    @property
    def tier3_ratio(self) -> float:
        """Derived Tier-3 (eviction) ratio = 1 − tier1 − tier2."""
        return max(0.0, 1.0 - self.tier1_ratio - self.tier2_ratio)


# --------------------------------------------------------------------------- #
# SignVQCacheConfigMixin                                                        #
# --------------------------------------------------------------------------- #

class SignVQCacheConfigMixin:
    """Mixin for injecting ``sign_vq`` params alongside a vLLM engine config.

    Typical usage — subclass or composition:

        class MyVllmConfig(VllmConfig, SignVQCacheConfigMixin):
            def __init__(self, *args, sign_vq_params=None, **kwargs):
                super().__init__(*args, **kwargs)
                SignVQCacheConfigMixin.__init__(self, sign_vq_params)

    The mixin intentionally does not call ``super().__init__`` to avoid MRO
    complications with Pydantic models.
    """

    def __init__(
        self,
        sign_vq_params: Optional[SignVQCacheParams] = None,
    ) -> None:
        self.sign_vq: SignVQCacheParams = sign_vq_params or SignVQCacheParams()


# --------------------------------------------------------------------------- #
# Convenience factories                                                         #
# --------------------------------------------------------------------------- #

def build_sign_vq_compressor(
    params: SignVQCacheParams,
) -> VllmLeverageCompressor:
    """Build a ``VllmLeverageCompressor`` from ``SignVQCacheParams``.

    Returns a compressor configured with the tier ratios and rank from
    ``params``.  The Tier-3 ratio is derived as ``1 − tier1 − tier2``.

    Args:
        params: ``SignVQCacheParams`` instance.

    Returns:
        ``VllmLeverageCompressor`` ready for use.
    """
    return VllmLeverageCompressor(
        rank=params.rank,
        reg_lambda=params.reg_lambda,
        tier1_ratio=params.tier1_ratio,
        tier3_ratio=params.tier3_ratio,
    )


def build_sign_vq_index(
    params: SignVQCacheParams,
    compressor: Optional[VllmLeverageCompressor] = None,
) -> SignVQSegmentIndex:
    """Build a ``SignVQSegmentIndex`` from ``SignVQCacheParams``.

    Args:
        params:      ``SignVQCacheParams`` instance.
        compressor:  Optional pre-built ``VllmLeverageCompressor``.
                     If None, one is constructed from ``params``.

    Returns:
        ``SignVQSegmentIndex`` ready for use.
    """
    if compressor is None:
        compressor = build_sign_vq_compressor(params)
    return SignVQSegmentIndex(
        compressor=compressor,
        chunk_size=params.chunk_size,
        max_entries=params.max_entries,
        hamming_threshold=params.hamming_threshold,
    )

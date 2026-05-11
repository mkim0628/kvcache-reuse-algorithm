"""Activity C mandatory accuracy preservation tests for RateQuantReverseWaterfillingCodec.

These tests verify the ±1% perplexity proxy requirement from evaluation_criteria.md §4.
All accuracy checks use synthetic KV tensors — no real model is required.
"""

import pytest
import torch

from src.cache.ratequant_codec import RateQuantConfig, RateQuantReverseWaterfillingCodec
from src.metrics.perplexity import (
    attention_kl_divergence,
    attention_output_relative_error,
    cosine_similarity_output,
)


@pytest.fixture
def calibrated_codec() -> RateQuantReverseWaterfillingCodec:
    """Codec pre-calibrated with 20 synthetic samples (unit-test scale-down of 512)."""
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=42)
    codec = RateQuantReverseWaterfillingCodec(config)
    torch.manual_seed(42)
    cal_kvs = [torch.randn(64, 2, 4, 32) for _ in range(20)]
    codec.calibrate(cal_kvs)
    return codec


# -------------------------------------------------------------------------
# Bit allocation correctness
# -------------------------------------------------------------------------

def test_reverse_waterfilling_total_budget(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """Sum of allocated bits must be within 1 bit of the target budget."""
    n_heads = calibrated_codec.config.n_heads
    total_budget = calibrated_codec.config.total_bit_budget * n_heads
    alloc = calibrated_codec.bit_allocation[0]
    assert abs(sum(alloc) - total_budget) <= n_heads, (
        f"Bit sum {sum(alloc)} deviates too far from budget {total_budget}"
    )


def test_encode_decode_shape(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """encode → decode must preserve tensor shape [n_tokens, 2, n_heads, d_head]."""
    torch.manual_seed(10)
    kv = torch.randn(32, 2, 4, 32)
    encoded = calibrated_codec.encode(kv, layer_idx=0)
    decoded = calibrated_codec.decode(encoded, layer_idx=0)
    assert decoded.shape == kv.shape, f"Shape mismatch: {decoded.shape} vs {kv.shape}"


# -------------------------------------------------------------------------
# PRIMARY ±1% accuracy preservation (mandatory evaluation_criteria.md §4)
# -------------------------------------------------------------------------

def test_accuracy_relative_error_within_1pct(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """PRIMARY ±1% test: relative attention output error < 0.01.

    Uses calibration-independent validation data (seed=99, not seed=42).
    """
    torch.manual_seed(99)  # independent of calibration seed=42
    kv = torch.randn(64, 2, 4, 32)
    q = torch.randn(16, 32)

    encoded = calibrated_codec.encode(kv, layer_idx=0)
    kv_dec = calibrated_codec.decode(encoded, layer_idx=0)

    error = attention_output_relative_error(
        q,
        kv[:, 0, 0, :].float(),   # head 0 key
        kv[:, 1, 0, :].float(),   # head 0 value
        kv_dec[:, 0, 0, :].float(),
        kv_dec[:, 1, 0, :].float(),
    )
    assert error < 0.01, f"Relative error {error:.4f} exceeds 1% limit"


def test_kl_divergence_within_threshold(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """KL divergence < 0.015 (±1% perplexity proxy)."""
    torch.manual_seed(55)
    kv = torch.randn(64, 2, 4, 32)
    q = torch.randn(16, 32)

    encoded = calibrated_codec.encode(kv, layer_idx=0)
    kv_dec = calibrated_codec.decode(encoded, layer_idx=0)

    kl = attention_kl_divergence(
        q.float(),
        kv[:, 0, 0, :].float(),
        kv_dec[:, 0, 0, :].float(),
    )
    assert kl < 0.015, f"KL divergence {kl:.5f} exceeds threshold 0.015"


def test_cosine_similarity_above_threshold(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """Attention output cosine similarity >= 0.99."""
    torch.manual_seed(77)
    kv = torch.randn(64, 2, 4, 32)
    q = torch.randn(16, 32)

    encoded = calibrated_codec.encode(kv, layer_idx=0)
    kv_dec = calibrated_codec.decode(encoded, layer_idx=0)

    sim = cosine_similarity_output(
        q.float(),
        kv[:, 0, 0, :].float(),
        kv[:, 1, 0, :].float(),
        kv_dec[:, 0, 0, :].float(),
        kv_dec[:, 1, 0, :].float(),
    )
    assert sim >= 0.99, f"Cosine similarity {sim:.4f} below threshold 0.99"


def test_compression_ratio_meets_target(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """avg_bits=4.0 should yield >= 70% compression relative to FP16."""
    ratio = calibrated_codec.compression_ratio(layer_idx=0)
    assert ratio >= 0.70, f"Compression ratio {ratio:.3f} below 0.70"


def test_low_variance_head_gets_fewer_bits(calibrated_codec: RateQuantReverseWaterfillingCodec) -> None:
    """Core reverse water-filling property: lower-variance heads get fewer or equal bits."""
    # Build a codec with clearly different variances per head
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=0)
    codec = RateQuantReverseWaterfillingCodec(config)
    torch.manual_seed(0)

    # Head 0: very low variance; Head 3: very high variance
    base = torch.randn(64, 2, 4, 32) * 0.01
    base[:, :, 3, :] = torch.randn(64, 2, 32) * 5.0   # high variance for head 3
    cal_kvs = [base.clone() + torch.randn_like(base) * 0.001 for _ in range(20)]
    codec.calibrate(cal_kvs)

    alloc = codec.bit_allocation[0]
    # Head 3 should receive >= bits than head 0 (high variance → more bits)
    assert alloc[3] >= alloc[0], (
        f"Expected head 3 (high var) bits >= head 0 (low var), got {alloc[3]} < {alloc[0]}"
    )


# -------------------------------------------------------------------------
# Calibration-independent accuracy (key evidence for evaluation_criteria.md §4)
# -------------------------------------------------------------------------

def test_calibration_independent_accuracy() -> None:
    """Accuracy on data completely independent of calibration.

    This is the primary evidence for evaluation_criteria.md §4 mandatory requirement.
    Calibration: seed=0. Validation: seed=999.
    """
    torch.manual_seed(0)
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=0)
    codec = RateQuantReverseWaterfillingCodec(config)

    # Calibration data (seed 0)
    cal_kvs = [torch.randn(32, 2, 4, 32) for _ in range(10)]
    codec.calibrate(cal_kvs)

    # Validation data — completely independent (seed 999)
    torch.manual_seed(999)
    test_kv = torch.randn(32, 2, 4, 32)
    q = torch.randn(8, 32)

    encoded = codec.encode(test_kv, layer_idx=0)
    kv_dec = codec.decode(encoded, layer_idx=0)

    error = attention_output_relative_error(
        q.float(),
        test_kv[:, 0, 0, :].float(),
        test_kv[:, 1, 0, :].float(),
        kv_dec[:, 0, 0, :].float(),
        kv_dec[:, 1, 0, :].float(),
    )
    assert error < 0.01, f"Independent test relative error {error:.4f} exceeds ±1%"

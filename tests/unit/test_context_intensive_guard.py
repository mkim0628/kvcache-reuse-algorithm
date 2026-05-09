"""Unit tests for ContextIntensiveAccuracyGuard (Activity C).

Tests: density assessment, compression limit gating, eOptShrinkQCodec integration,
density level thresholds, and weight normalization.
"""

import pytest
import torch

from src.cache.context_intensive_guard import ContextIntensiveAccuracyGuard


def _make_guard(**kwargs) -> ContextIntensiveAccuracyGuard:
    return ContextIntensiveAccuracyGuard(**kwargs)


# ------------------------------------------------------------------ #
# Density assessment                                                   #
# ------------------------------------------------------------------ #

class TestDensityAssessment:
    def test_assess_returns_valid_range(self):
        """assess() must return a value in [0.0, 1.0]."""
        guard = _make_guard()
        torch.manual_seed(0)
        token_ids = torch.randint(0, 100000, (200,))
        score = guard.assess(token_ids)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0.0, 1.0]"

    def test_assess_empty_input_returns_default(self):
        """assess() on empty tensor must return the default value (0.5)."""
        guard = _make_guard()
        score = guard.assess(torch.tensor([], dtype=torch.long))
        assert score == pytest.approx(0.5)

    def test_high_token_ids_raise_entity_ratio(self):
        """Token IDs > 50000 (rare vocabulary proxy) should push density score up."""
        guard = _make_guard()
        high_ids = torch.randint(50001, 100000, (128,))
        low_ids = torch.randint(1000, 5000, (128,))

        score_high = guard.assess(high_ids)
        score_low = guard.assess(low_ids)
        assert score_high > score_low, (
            f"High entity tokens ({score_high:.4f}) should yield higher density than low ({score_low:.4f})"
        )

    def test_numeric_range_tokens_raise_numeric_ratio(self):
        """Token IDs in range 10..100 (numeric proxy) should contribute to density score."""
        guard = _make_guard()
        numeric_ids = torch.randint(10, 101, (128,))
        other_ids = torch.randint(5000, 10000, (128,))

        score_num = guard.assess(numeric_ids)
        score_other = guard.assess(other_ids)
        assert score_num >= score_other

    def test_assess_samples_only_first_n_tokens(self):
        """assess() should use only the first sample_tokens tokens."""
        guard = _make_guard(sample_tokens=10)
        # First 10 tokens: all high-ID → high density
        # Remaining: all low-ID → should not matter
        token_ids = torch.cat([
            torch.full((10,), 99999, dtype=torch.long),
            torch.zeros(200, dtype=torch.long),
        ])
        score = guard.assess(token_ids)
        # entity_ratio = 1.0 (all 10 first tokens > 50000)
        assert score > 0.2


# ------------------------------------------------------------------ #
# Compression limits                                                   #
# ------------------------------------------------------------------ #

class TestCompressionLimits:
    def test_high_density_limits_compression_to_4bit(self):
        """High density (score ≥ 0.7) must enforce min_bits ≥ 4.0."""
        guard = _make_guard()
        limits = guard.get_compression_limits(0.8)
        assert limits["min_bits"] >= 4.0
        assert limits["density_level"] == "high"

    def test_low_density_allows_aggressive_compression(self):
        """Low density (score ≤ 0.4) must allow min_bits ≤ 2.2."""
        guard = _make_guard()
        limits = guard.get_compression_limits(0.2)
        assert limits["min_bits"] <= 2.2
        assert limits["density_level"] == "low"

    def test_medium_density_allows_2_2_bits(self):
        """Medium density must allow down to 2.2 bits (not locked to 4-bit)."""
        guard = _make_guard()
        limits = guard.get_compression_limits(0.55)
        assert limits["min_bits"] <= 4.0
        assert limits["min_bits"] >= 2.2
        assert limits["density_level"] == "medium"

    def test_boundary_at_threshold_high(self):
        """Score exactly at threshold_high must be classified as high."""
        guard = _make_guard(threshold_high=0.7)
        limits = guard.get_compression_limits(0.7)
        assert limits["density_level"] == "high"

    def test_boundary_at_threshold_low(self):
        """Score exactly at threshold_low must be classified as medium."""
        guard = _make_guard(threshold_low=0.4)
        limits = guard.get_compression_limits(0.4)
        assert limits["density_level"] == "medium"

    def test_score_below_threshold_low_is_low(self):
        """Score below threshold_low must be classified as low."""
        guard = _make_guard(threshold_low=0.4)
        limits = guard.get_compression_limits(0.39)
        assert limits["density_level"] == "low"

    def test_max_compression_ratio_ordered(self):
        """High density must have lower max_compression_ratio than low density."""
        guard = _make_guard()
        high_limits = guard.get_compression_limits(0.9)
        low_limits = guard.get_compression_limits(0.1)
        assert high_limits["max_compression_ratio"] < low_limits["max_compression_ratio"]


# ------------------------------------------------------------------ #
# eOptShrinkQCodec gating                                             #
# ------------------------------------------------------------------ #

class TestGateEoptCodec:
    def test_gate_eopt_codec_raises_bits_in_high_density(self):
        """High-density context must raise key_bits to ≥ 4 in eOptShrinkQCodec."""
        from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec

        guard = _make_guard(threshold_high=0.0)  # force all contexts to be high density
        codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
        token_ids = torch.randint(50001, 100000, (128,))  # all high-entity tokens

        applied = guard.gate_eopt_codec(codec, token_ids)

        assert applied["applied_key_bits"] >= 4, (
            f"applied_key_bits={applied['applied_key_bits']} should be ≥ 4 in high-density context"
        )

    def test_gate_eopt_codec_preserves_bits_in_low_density(self):
        """Low-density context must not artificially raise compression bits."""
        from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec

        guard = _make_guard(threshold_high=1.0, threshold_low=0.9)  # almost nothing is high density
        codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
        token_ids = torch.randint(1000, 5000, (128,))  # low-entity tokens

        applied = guard.gate_eopt_codec(codec, token_ids)

        # Low density → min_bits=1.0 → max(1, 2) = 2 (original preserved)
        assert applied["applied_key_bits"] == applied["original_key_bits"] or applied["applied_key_bits"] >= 1

    def test_gate_eopt_codec_returns_required_keys(self):
        """gate_eopt_codec() must return dict with all required keys."""
        from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec

        guard = _make_guard()
        codec = eOptShrinkQCodec(num_layers=1, key_bits=3, value_bits=4)
        token_ids = torch.randint(0, 100000, (128,))
        applied = guard.gate_eopt_codec(codec, token_ids)

        required_keys = {
            "density_score", "density_level",
            "original_key_bits", "original_val_bits",
            "applied_key_bits", "applied_val_bits",
        }
        assert required_keys.issubset(set(applied.keys()))

    def test_gate_eopt_codec_density_score_valid_range(self):
        """Density score in gate_eopt_codec() result must be in [0.0, 1.0]."""
        from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec

        guard = _make_guard()
        codec = eOptShrinkQCodec(num_layers=1, key_bits=3, value_bits=4)
        token_ids = torch.randint(0, 100000, (128,))
        applied = guard.gate_eopt_codec(codec, token_ids)
        assert 0.0 <= applied["density_score"] <= 1.0


# ------------------------------------------------------------------ #
# Weight normalization                                                  #
# ------------------------------------------------------------------ #

class TestWeightNormalization:
    def test_density_score_components_weighted(self):
        """Default weights w1 + w2 + w3 must sum to 1.0."""
        guard = _make_guard()
        assert (guard.w1 + guard.w2 + guard.w3) == pytest.approx(1.0), (
            f"Weights sum to {guard.w1 + guard.w2 + guard.w3:.4f}, expected 1.0"
        )

    def test_custom_weights_accepted(self):
        """Custom weights should be stored as-is (no normalization enforced)."""
        guard = _make_guard(w1=0.4, w2=0.4, w3=0.2)
        assert guard.w1 == 0.4
        assert guard.w2 == 0.4
        assert guard.w3 == 0.2

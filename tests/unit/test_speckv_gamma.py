"""Unit tests for SpecKVCompressionGammaController (Activity C).

Tests: compression level → γ selection, valid range, online adaptation,
perplexity proxy, eOptShrinkQCodec integration, and MLP training guard.
"""

import pytest
import torch

from src.cache.speckv_gamma_controller import SpecKVCompressionGammaController


# ------------------------------------------------------------------ #
# Basic γ selection                                                    #
# ------------------------------------------------------------------ #

class TestGammaSelection:
    def test_gamma_in_valid_range(self):
        """γ must always be in {1, 2, 3, 4, 5, 6} for all inputs."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        for comp_lvl in [0, 1, 2]:
            for min_conf in [0.1, 0.5, 0.9]:
                gamma = ctrl.select_gamma(comp_lvl, min_conf, 1.0 - min_conf)
                assert 1 <= gamma <= 6, f"γ={gamma} out of range for comp={comp_lvl}, conf={min_conf}"

    def test_select_gamma_returns_int(self):
        """select_gamma() must return a Python int."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        result = ctrl.select_gamma(0, 0.8, 0.2)
        assert isinstance(result, int)

    def test_high_compression_selects_lower_gamma(self):
        """
        NF4 (highest compression) should select γ ≤ FP16 γ with the same draft signals.
        This verifies the accuracy-preserving property.
        """
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        gamma_fp16 = ctrl.select_gamma(
            SpecKVCompressionGammaController.COMPRESSION_FP16, 0.9, 0.1
        )
        gamma_nf4 = ctrl.select_gamma(
            SpecKVCompressionGammaController.COMPRESSION_NF4, 0.9, 0.1
        )
        assert gamma_nf4 <= gamma_fp16, (
            f"NF4 γ({gamma_nf4}) > FP16 γ({gamma_fp16}); expected NF4 to be more conservative"
        )

    def test_compression_constants_distinct(self):
        """Compression level constants must be distinct integers."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        levels = {ctrl.COMPRESSION_FP16, ctrl.COMPRESSION_INT8, ctrl.COMPRESSION_NF4}
        assert len(levels) == 3


# ------------------------------------------------------------------ #
# Online adaptation                                                    #
# ------------------------------------------------------------------ #

class TestOnlineAdaptation:
    def test_online_adaptation_lowers_gamma_on_low_pass_rate(self):
        """
        With 0% verification pass rate (all rejected), γ_bias must go negative.
        """
        ctrl = SpecKVCompressionGammaController(base_seed=42, ema_alpha=0.2)
        for _ in range(20):
            ctrl.record_verification(was_accepted=False)
        assert ctrl._gamma_bias < 0.0, (
            f"γ_bias={ctrl._gamma_bias:.4f}; expected negative after 0% pass rate"
        )

    def test_online_adaptation_raises_gamma_on_high_pass_rate(self):
        """
        With 100% verification pass rate (all accepted), γ_bias must go positive.
        """
        ctrl = SpecKVCompressionGammaController(base_seed=42, ema_alpha=0.2)
        for _ in range(20):
            ctrl.record_verification(was_accepted=True)
        assert ctrl._gamma_bias > 0.0, (
            f"γ_bias={ctrl._gamma_bias:.4f}; expected positive after 100% pass rate"
        )

    def test_gamma_bias_bounded(self):
        """γ_bias must stay within [-2.0, 2.0] regardless of feedback stream."""
        ctrl = SpecKVCompressionGammaController(base_seed=42, ema_alpha=0.5)
        for _ in range(200):
            ctrl.record_verification(was_accepted=False)
        assert ctrl._gamma_bias >= -2.0

        ctrl2 = SpecKVCompressionGammaController(base_seed=42, ema_alpha=0.5)
        for _ in range(200):
            ctrl2.record_verification(was_accepted=True)
        assert ctrl2._gamma_bias <= 2.0

    def test_no_adaptation_below_10_samples(self):
        """Adaptation must not trigger until at least 10 verification records."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        bias_before = ctrl._gamma_bias
        for _ in range(9):
            ctrl.record_verification(was_accepted=False)
        assert ctrl._gamma_bias == bias_before, "Adaptation should not occur with < 10 samples"


# ------------------------------------------------------------------ #
# eOptShrinkQCodec integration                                         #
# ------------------------------------------------------------------ #

class TestEOptIntegration:
    def test_integrate_with_eopt_selects_nf4_for_2bit(self):
        """key_bits=2 in eOptShrinkQCodec must map to NF4 compression level."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)

        class FakeCodec:
            key_bits = 2

        gamma = ctrl.integrate_with_eopt(FakeCodec(), 0.8, 0.2)
        # Should use COMPRESSION_NF4 (most conservative γ)
        gamma_direct = ctrl.select_gamma(ctrl.COMPRESSION_NF4, 0.8, 0.2)
        assert gamma == gamma_direct

    def test_integrate_with_eopt_selects_fp16_for_4bit(self):
        """key_bits=4 in eOptShrinkQCodec must map to FP16 compression level."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)

        class FakeCodec:
            key_bits = 4

        gamma = ctrl.integrate_with_eopt(FakeCodec(), 0.8, 0.2)
        gamma_direct = ctrl.select_gamma(ctrl.COMPRESSION_FP16, 0.8, 0.2)
        assert gamma == gamma_direct

    def test_integrate_with_eopt_selects_int8_for_3bit(self):
        """key_bits=3 in eOptShrinkQCodec must map to INT8 compression level."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)

        class FakeCodec:
            key_bits = 3

        gamma = ctrl.integrate_with_eopt(FakeCodec(), 0.8, 0.2)
        gamma_direct = ctrl.select_gamma(ctrl.COMPRESSION_INT8, 0.8, 0.2)
        assert gamma == gamma_direct


# ------------------------------------------------------------------ #
# MLP training guard                                                   #
# ------------------------------------------------------------------ #

class TestMLPTraining:
    def test_train_mlp_requires_512_records(self):
        """train_mlp_from_profile() must return inf when < 512 records collected."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        for _ in range(100):
            ctrl.collect_profile_record(0, 0.8, 0.2, 4)
        loss = ctrl.train_mlp_from_profile()
        assert loss == float("inf"), f"Expected inf with < 512 records, got {loss}"

    def test_collect_profile_record_buffers(self):
        """collect_profile_record() must append to buffer."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        ctrl.collect_profile_record(1, 0.5, 0.5, 3)
        assert len(ctrl._profile_buffer) == 1

    def test_train_mlp_with_enough_records(self):
        """train_mlp_from_profile() must return finite loss with ≥ 512 records."""
        ctrl = SpecKVCompressionGammaController(base_seed=42)
        torch.manual_seed(42)
        for i in range(512):
            comp = i % 3
            ctrl.collect_profile_record(comp, 0.5, 0.5, 3)
        loss = ctrl.train_mlp_from_profile(epochs=2)
        assert loss < float("inf")
        assert loss >= 0.0


# ------------------------------------------------------------------ #
# Perplexity proxy via eOptShrinkQCodec MSE                           #
# ------------------------------------------------------------------ #

class TestPerplexityProxy:
    def test_perplexity_proxy_within_tolerance(self):
        """
        After eOptShrinkQCodec compression + reconstruction, relative MSE < 5%.
        This acts as a proxy for perplexity change ±1%.
        """
        from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec

        codec = eOptShrinkQCodec(num_layers=1, key_bits=3, value_bits=4)
        torch.manual_seed(42)
        codec.calibrate([torch.randn(64, 32) for _ in range(20)])

        kv_key = torch.randn(256, 32)
        kv_val = torch.randn(256, 32)

        compressed = codec.encode(kv_key, kv_val, layer_idx=0)
        key_approx, val_approx = codec.decode(compressed)

        mse_key = ((kv_key - key_approx) ** 2).mean() / (kv_key ** 2).mean()
        mse_val = ((kv_val - val_approx) ** 2).mean() / (kv_val ** 2).mean()

        assert mse_key.item() < 0.05, f"Key relative MSE {mse_key.item():.4f} >= 0.05"
        assert mse_val.item() < 0.05, f"Val relative MSE {mse_val.item():.4f} >= 0.05"

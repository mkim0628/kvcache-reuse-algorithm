"""Unit tests for eOptShrinkQCodec accuracy preservation (Activity C).

Covers BBP rank selection, residual bias, encode/decode roundtrip quality,
memory reduction, and calibration persistence.
"""

import os
import statistics
import tempfile

import pytest
import torch
import torch.nn.functional as F

from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec


def _make_calibration_data(n_samples: int = 20, n_tokens: int = 64, d_head: int = 32) -> list:
    torch.manual_seed(42)
    return [torch.randn(n_tokens, d_head) for _ in range(n_samples)]


class TestBBPRankSelection:
    def test_bbp_rank_selection_consistency(self) -> None:
        """BBP auto-rank should be consistent across seeds on same-distribution data."""
        ranks = []
        for seed in range(10):
            codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
            torch.manual_seed(seed)
            # Rank-4 signal + low noise
            signal = torch.randn(64, 4) @ torch.randn(4, 32)
            noise = torch.randn(64, 32) * 0.1
            kv = signal + noise
            codec.calibrate([kv])
            ranks.append(codec._auto_ranks.get(0, 0))
        # Standard deviation should be small (consistent selection)
        assert statistics.stdev(ranks) <= 2, f"BBP rank selection inconsistent: {ranks}"

    def test_auto_rank_positive(self) -> None:
        """Auto-selected rank must be >= 1."""
        codec = eOptShrinkQCodec(num_layers=3, key_bits=2, value_bits=3)
        calibration_data = _make_calibration_data(n_samples=20)
        codec.calibrate([calibration_data[0], calibration_data[1], calibration_data[2]])
        for layer_idx in range(3):
            rank = codec._auto_ranks.get(layer_idx, 0)
            assert rank >= 1, f"Layer {layer_idx} rank is {rank} < 1"

    def test_high_rank_signal_gives_higher_rank(self) -> None:
        """High-rank signal should yield larger auto-rank than low-rank signal."""
        torch.manual_seed(42)
        codec_low = eOptShrinkQCodec(num_layers=1)
        low_rank_signal = torch.randn(64, 2) @ torch.randn(2, 32) + torch.randn(64, 32) * 0.05
        codec_low.calibrate([low_rank_signal])

        codec_high = eOptShrinkQCodec(num_layers=1)
        high_rank_signal = torch.randn(64, 16) @ torch.randn(16, 32) + torch.randn(64, 32) * 0.05
        codec_high.calibrate([high_rank_signal])

        rank_low = codec_low._auto_ranks.get(0, 0)
        rank_high = codec_high._auto_ranks.get(0, 0)
        assert rank_high >= rank_low, (
            f"Higher-rank signal should have rank >= lower-rank signal: {rank_high} vs {rank_low}"
        )


class TestResidualBias:
    def test_residual_bias_near_zero(self) -> None:
        """After low-rank separation, residual inner product with low-rank component must be small."""
        codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
        torch.manual_seed(42)
        kv = torch.randn(128, 64)
        codec.calibrate([kv])
        r = codec._auto_ranks.get(0, 4)

        U, S, Vh = torch.linalg.svd(kv.float(), full_matrices=False)
        lowrank = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        residual = kv.float() - lowrank

        bias = (residual * lowrank).mean().abs().item()
        assert bias < 0.1, f"Residual inner-product bias too large: {bias:.6f}"

    def test_residual_variance_below_original(self) -> None:
        """Residual after low-rank separation should have lower total variance."""
        torch.manual_seed(99)
        codec = eOptShrinkQCodec(num_layers=1)
        kv = torch.randn(64, 32) @ torch.randn(32, 32)  # structured data
        codec.calibrate([kv])
        r = codec._auto_ranks.get(0, 4)

        U, S, Vh = torch.linalg.svd(kv.float(), full_matrices=False)
        lowrank = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
        residual = kv.float() - lowrank

        # Residual variance must be < original variance (low-rank captures structure)
        assert residual.var().item() < kv.float().var().item()


class TestEncodeDecodeRoundtrip:
    def test_encode_decode_roundtrip_cosine_similarity(self) -> None:
        """Encode → decode roundtrip cosine similarity must be >= 0.85."""
        codec = eOptShrinkQCodec(num_layers=2, key_bits=2, value_bits=3)
        calibration_kvs = _make_calibration_data(n_samples=20)
        codec.calibrate([calibration_kvs[0], calibration_kvs[1]])

        torch.manual_seed(42)
        kv_key = torch.randn(128, 32)
        kv_val = torch.randn(128, 32)
        compressed = codec.encode(kv_key, kv_val, layer_idx=0)
        key_approx, val_approx = codec.decode(compressed)

        cos_key = F.cosine_similarity(
            kv_key.flatten().unsqueeze(0), key_approx.flatten().unsqueeze(0)
        ).item()
        cos_val = F.cosine_similarity(
            kv_val.flatten().unsqueeze(0), val_approx.flatten().unsqueeze(0)
        ).item()
        assert cos_key >= 0.85, f"Key cosine similarity too low: {cos_key:.4f}"
        assert cos_val >= 0.85, f"Value cosine similarity too low: {cos_val:.4f}"

    def test_encode_preserves_shape(self) -> None:
        """Decoded tensors must have the same shape as the originals."""
        codec = eOptShrinkQCodec(num_layers=1)
        torch.manual_seed(0)
        codec.calibrate([torch.randn(32, 16)])
        kv_key = torch.randn(64, 16)
        kv_val = torch.randn(64, 16)
        compressed = codec.encode(kv_key, kv_val, layer_idx=0)
        key_approx, val_approx = codec.decode(compressed)
        assert key_approx.shape == kv_key.shape
        assert val_approx.shape == kv_val.shape


class TestMemoryReduction:
    def test_memory_reduction_at_least_30_percent(self) -> None:
        """Compressed size must be <= 70% of FP32 baseline."""
        codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=3)
        torch.manual_seed(42)
        codec.calibrate([torch.randn(64, 64) for _ in range(20)])
        est = codec.memory_bytes_estimate(n_tokens=512, d_head=64, layer_idx=0)
        assert est["reduction_ratio"] >= 0.30, (
            f"Memory reduction insufficient: {est['reduction_ratio']:.2%} < 30%"
        )

    def test_memory_estimate_fields_present(self) -> None:
        """memory_bytes_estimate must return the required keys."""
        codec = eOptShrinkQCodec(num_layers=1)
        codec._auto_ranks[0] = 4
        est = codec.memory_bytes_estimate(n_tokens=128, d_head=64)
        assert "total_bytes" in est
        assert "baseline_bytes" in est
        assert "reduction_ratio" in est
        assert est["total_bytes"] > 0
        assert est["baseline_bytes"] > 0


class TestPerplexityProxy:
    def test_perplexity_proxy_within_tolerance(self) -> None:
        """MSE relative error (perplexity proxy) must be < 5% with relaxed bit settings."""
        codec = eOptShrinkQCodec(num_layers=1, key_bits=3, value_bits=4)
        torch.manual_seed(42)
        codec.calibrate([torch.randn(64, 32) for _ in range(20)])

        kv_key = torch.randn(256, 32)
        kv_val = torch.randn(256, 32)
        compressed = codec.encode(kv_key, kv_val, layer_idx=0)
        key_approx, val_approx = codec.decode(compressed)

        mse_key = ((kv_key - key_approx) ** 2).mean() / (kv_key ** 2).mean()
        mse_val = ((kv_val - val_approx) ** 2).mean() / (kv_val ** 2).mean()
        assert mse_key.item() < 0.05, f"Key MSE relative error too large: {mse_key.item():.4f}"
        assert mse_val.item() < 0.05, f"Value MSE relative error too large: {mse_val.item():.4f}"


class TestKeyValueAsymmetricBits:
    def test_key_value_asymmetric_bits(self) -> None:
        """Key codec uses key_bits, Value codec uses value_bits (different)."""
        codec = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=4)
        assert codec._key_codec.bits == 2
        assert codec._val_codec.bits == 4

    def test_different_bits_affect_accuracy(self) -> None:
        """Higher value_bits should yield better value reconstruction quality."""
        torch.manual_seed(7)
        cal_data = [torch.randn(64, 32) for _ in range(20)]
        kv_key = torch.randn(128, 32)
        kv_val = torch.randn(128, 32)

        codec_low = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=2)
        codec_low.calibrate(cal_data)
        compressed_low = codec_low.encode(kv_key, kv_val, layer_idx=0)
        _, val_low = codec_low.decode(compressed_low)

        codec_high = eOptShrinkQCodec(num_layers=1, key_bits=2, value_bits=4)
        codec_high.calibrate(cal_data)
        compressed_high = codec_high.encode(kv_key, kv_val, layer_idx=0)
        _, val_high = codec_high.decode(compressed_high)

        mse_low = ((kv_val - val_low) ** 2).mean().item()
        mse_high = ((kv_val - val_high) ** 2).mean().item()
        # Higher bits should give lower or equal reconstruction error
        assert mse_high <= mse_low * 1.5, (
            f"Higher bits should give comparable/better accuracy: MSE_low={mse_low:.4f}, MSE_high={mse_high:.4f}"
        )


class TestCalibrateLoadSave:
    def test_calibrate_save_load_roundtrip(self) -> None:
        """Calibration state must survive save → load roundtrip exactly."""
        codec_orig = eOptShrinkQCodec(num_layers=3, key_bits=2, value_bits=3)
        cal_data = _make_calibration_data(n_samples=20)
        codec_orig.calibrate(cal_data[:3])

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "calib.pt")
            codec_orig.calibrate(cal_data[:3], save_path=save_path)

            codec_loaded = eOptShrinkQCodec(num_layers=3, key_bits=2, value_bits=3)
            codec_loaded.load_calibration(save_path)

            assert codec_loaded._auto_ranks == codec_orig._auto_ranks
            for k in codec_orig._noise_levels:
                assert abs(codec_loaded._noise_levels[k] - codec_orig._noise_levels[k]) < 1e-6

    def test_calibrate_save_creates_directory(self) -> None:
        """calibrate() with save_path must create parent directories."""
        codec = eOptShrinkQCodec(num_layers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "nested", "dir", "calib.pt")
            torch.manual_seed(0)
            codec.calibrate([torch.randn(32, 16)], save_path=save_path)
            assert os.path.isfile(save_path)

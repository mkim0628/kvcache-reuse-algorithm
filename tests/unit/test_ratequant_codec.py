"""Unit tests for RateQuantReverseWaterfillingCodec (Activity C).

Covers: reverse water-filling bit allocation, encode/decode shape and
accuracy, compression ratio, serialisation.
"""

import math
import os
import tempfile

import pytest
import torch

from src.cache.ratequant_codec import RateQuantConfig, RateQuantReverseWaterfillingCodec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def codec_4bit() -> RateQuantReverseWaterfillingCodec:
    """Calibrated codec with total_bit_budget=4.0 (equal-variance data)."""
    torch.manual_seed(42)
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=42)
    codec = RateQuantReverseWaterfillingCodec(config)
    cal_kvs = [torch.randn(32, 2, 4, 32) for _ in range(16)]
    codec.calibrate(cal_kvs)
    return codec


@pytest.fixture
def codec_varied_variance() -> RateQuantReverseWaterfillingCodec:
    """Calibrated codec where heads have clearly different variances."""
    torch.manual_seed(0)
    config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=0)
    codec = RateQuantReverseWaterfillingCodec(config)
    torch.manual_seed(0)
    base = torch.randn(32, 2, 4, 32)
    base[:, :, 0, :] *= 0.01   # very low variance — head 0
    base[:, :, 3, :] *= 5.0    # very high variance — head 3
    cal_kvs = [base.clone() + torch.randn_like(base) * 0.001 for _ in range(16)]
    codec.calibrate(cal_kvs)
    return codec


# ---------------------------------------------------------------------------
# Reverse water-filling algorithm
# ---------------------------------------------------------------------------

class TestReverseWaterfilling:
    def test_bit_sum_near_total_budget(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        n_heads = codec_4bit.config.n_heads
        total_budget = codec_4bit.config.total_bit_budget * n_heads  # 16
        alloc = codec_4bit.bit_allocation[0]
        assert abs(sum(alloc) - total_budget) <= n_heads, (
            f"Bit sum {sum(alloc)} deviates more than {n_heads} from budget {total_budget}"
        )

    def test_all_bits_in_valid_range(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        alloc = codec_4bit.bit_allocation[0]
        cfg = codec_4bit.config
        for b in alloc:
            assert cfg.min_bits <= b <= cfg.max_bits, (
                f"Bit width {b} outside [{cfg.min_bits}, {cfg.max_bits}]"
            )

    def test_high_variance_head_gets_more_bits(
        self, codec_varied_variance: RateQuantReverseWaterfillingCodec
    ) -> None:
        """Head 3 (high variance) must get >= bits than head 0 (low variance)."""
        alloc = codec_varied_variance.bit_allocation[0]
        assert alloc[3] >= alloc[0], (
            f"Expected head 3 bits >= head 0 bits, got {alloc[3]} vs {alloc[0]}"
        )

    def test_static_reverse_waterfilling(self) -> None:
        """Direct unit test of the static helper."""
        variances = torch.tensor([1.0, 2.0, 0.5, 4.0])
        total_budget = 4.0 * 4  # 16 bits for 4 heads
        bits = RateQuantReverseWaterfillingCodec._reverse_waterfilling(
            variances, total_budget, min_bits=2, max_bits=8
        )
        assert len(bits) == 4
        assert all(2 <= b <= 8 for b in bits)
        # Higher variance heads should get >= bits than lower variance heads
        assert bits[3] >= bits[2]   # var 4.0 >= var 0.5


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_sets_calibrated_flag(self) -> None:
        config = RateQuantConfig(n_heads=2, d_head=16)
        codec = RateQuantReverseWaterfillingCodec(config)
        assert not codec._calibrated
        codec.calibrate([torch.randn(8, 2, 2, 16)])
        assert codec._calibrated

    def test_calibrate_layer_independent(self) -> None:
        config = RateQuantConfig(n_heads=2, d_head=16, n_layers=3)
        codec = RateQuantReverseWaterfillingCodec(config)
        kvs = [torch.randn(8, 2, 2, 16) for _ in range(4)]
        codec.calibrate_layer(kvs, layer_idx=0)
        codec.calibrate_layer(kvs, layer_idx=2)
        assert 0 in codec.bit_allocation
        assert 2 in codec.bit_allocation

    def test_calibrate_empty_raises(self) -> None:
        config = RateQuantConfig(n_heads=2, d_head=16)
        codec = RateQuantReverseWaterfillingCodec(config)
        with pytest.raises(ValueError):
            codec.calibrate_layer([], layer_idx=0)


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------

class TestEncodeDecode:
    def test_encode_requires_calibration(self) -> None:
        config = RateQuantConfig(n_heads=2, d_head=16)
        codec = RateQuantReverseWaterfillingCodec(config)
        with pytest.raises(RuntimeError):
            codec.encode(torch.randn(8, 2, 2, 16), layer_idx=0)

    def test_encode_decode_shape_preserved(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv = torch.randn(32, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        decoded = codec_4bit.decode(encoded, layer_idx=0)
        assert decoded.shape == kv.shape

    def test_encode_decode_dtype_is_half(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv = torch.randn(16, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        decoded = codec_4bit.decode(encoded, layer_idx=0)
        assert decoded.dtype == torch.float16

    def test_encode_dict_keys(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv = torch.randn(8, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        for key in ("quantized", "scales", "zero_pts", "bit_widths", "layer_idx", "n_tokens", "n_heads"):
            assert key in encoded, f"Missing key '{key}' in encoded dict"

    def test_encode_head_count(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv = torch.randn(16, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        assert len(encoded["quantized"]) == 4
        assert len(encoded["scales"]) == 4

    def test_bit_widths_list_length(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv = torch.randn(16, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        assert len(encoded["bit_widths"]) == 4

    def test_decode_is_close_to_original(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        """Decode output should be numerically close to the original tensor."""
        torch.manual_seed(7)
        kv = torch.randn(32, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        decoded = codec_4bit.decode(encoded, layer_idx=0)
        rel_error = (kv - decoded.float()).norm() / kv.norm()
        assert rel_error < 0.20, f"Relative reconstruction error {rel_error:.4f} exceeds 20%"


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def test_compression_ratio_formula(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        """compression_ratio = 1 - avg_bits / 16."""
        ratio = codec_4bit.compression_ratio(layer_idx=0)
        alloc = codec_4bit.bit_allocation[0]
        expected = 1.0 - (sum(alloc) / len(alloc)) / 16.0
        assert ratio == pytest.approx(expected, abs=1e-6)

    def test_compression_ratio_meets_70pct(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        """4-bit average should give >= 70% theoretical compression vs FP16."""
        assert codec_4bit.compression_ratio(0) >= 0.70

    def test_compression_ratio_increases_with_fewer_bits(self) -> None:
        """Lower bit budget → higher compression ratio."""
        torch.manual_seed(0)
        kvs = [torch.randn(8, 2, 2, 16) for _ in range(4)]

        c_high = RateQuantReverseWaterfillingCodec(
            RateQuantConfig(n_heads=2, d_head=16, total_bit_budget=8.0)
        )
        c_high.calibrate(kvs)

        c_low = RateQuantReverseWaterfillingCodec(
            RateQuantConfig(n_heads=2, d_head=16, total_bit_budget=2.0)
        )
        c_low.calibrate(kvs)

        assert c_low.compression_ratio(0) >= c_high.compression_ratio(0)


# ---------------------------------------------------------------------------
# memory_bytes
# ---------------------------------------------------------------------------

class TestMemoryBytes:
    def test_memory_bytes_positive(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv = torch.randn(16, 2, 4, 32)
        encoded = codec_4bit.encode(kv, layer_idx=0)
        assert codec_4bit.memory_bytes(encoded) > 0

    def test_larger_tensor_uses_more_memory(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        kv_small = torch.randn(8, 2, 4, 32)
        kv_large = torch.randn(32, 2, 4, 32)
        enc_small = codec_4bit.encode(kv_small, layer_idx=0)
        enc_large = codec_4bit.encode(kv_large, layer_idx=0)
        assert codec_4bit.memory_bytes(enc_large) > codec_4bit.memory_bytes(enc_small)


# ---------------------------------------------------------------------------
# Serialisation (save / load calibration)
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_save_load_calibration(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            codec_4bit.save_calibration(path)
            config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=42)
            codec2 = RateQuantReverseWaterfillingCodec(config)
            codec2.load_calibration(path)
            assert codec2._calibrated
            assert codec2.bit_allocation == codec_4bit.bit_allocation
        finally:
            os.unlink(path)

    def test_loaded_codec_encodes_correctly(self, codec_4bit: RateQuantReverseWaterfillingCodec) -> None:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            codec_4bit.save_calibration(path)
            config = RateQuantConfig(n_heads=4, d_head=32, total_bit_budget=4.0, seed=42)
            codec2 = RateQuantReverseWaterfillingCodec(config)
            codec2.load_calibration(path)
            kv = torch.randn(16, 2, 4, 32)
            enc = codec2.encode(kv, layer_idx=0)
            dec = codec2.decode(enc, layer_idx=0)
            assert dec.shape == kv.shape
        finally:
            os.unlink(path)

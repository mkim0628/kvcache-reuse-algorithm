import pytest
import torch
from src.cache.compression import CompressionCodec


@pytest.fixture
def codec() -> CompressionCodec:
    return CompressionCodec(num_layers=12, cutoff_ratio=1 / 3)


def test_fp16_early_layers(codec: CompressionCodec) -> None:
    kv = torch.randn(16, 64)
    compressed = codec.encode(kv, layer_idx=0)
    assert compressed.dtype == torch.float16


def test_int8_late_layers(codec: CompressionCodec) -> None:
    kv = torch.randn(16, 64)
    compressed = codec.encode(kv, layer_idx=6)
    assert compressed.dtype == torch.int8


def test_fp16_roundtrip_accuracy(codec: CompressionCodec) -> None:
    kv = torch.randn(16, 64)
    compressed = codec.encode(kv, layer_idx=0)
    restored = codec.decode(compressed, layer_idx=0)
    l2_error = (kv.float() - restored).norm() / kv.float().norm()
    assert l2_error < 0.001, f"FP16 roundtrip error too large: {l2_error:.6f}"


def test_int8_roundtrip_accuracy(codec: CompressionCodec) -> None:
    kv = torch.randn(16, 64)
    layer_idx = 6
    compressed = codec.encode(kv, layer_idx=layer_idx, tensor_id=0)
    restored = codec.decode(compressed, layer_idx=layer_idx, tensor_id=0)
    l2_error = (kv.float() - restored).norm() / kv.float().norm()
    assert l2_error < 0.01, f"INT8 roundtrip error too large: {l2_error:.6f}"


def test_compression_ratio_early(codec: CompressionCodec) -> None:
    ratio = codec.compression_ratio(layer_idx=0)
    assert abs(ratio - 0.5) < 1e-6, "FP16 should give 50% savings"


def test_compression_ratio_late(codec: CompressionCodec) -> None:
    ratio = codec.compression_ratio(layer_idx=6)
    assert abs(ratio - 0.75) < 1e-6, "INT8 should give 75% savings"


def test_average_compression_ratio(codec: CompressionCodec) -> None:
    avg = codec.average_compression_ratio()
    # cutoff=4/12, early 4 layers FP16 (0.5), late 8 layers INT8 (0.75)
    expected = (4 * 0.5 + 8 * 0.75) / 12
    assert abs(avg - expected) < 1e-6


def test_multiple_tensors_independent_scales(codec: CompressionCodec) -> None:
    kv_a = torch.randn(8, 32) * 10
    kv_b = torch.randn(8, 32) * 0.1
    layer_idx = 8
    comp_a = codec.encode(kv_a, layer_idx, tensor_id=1)
    comp_b = codec.encode(kv_b, layer_idx, tensor_id=2)
    rest_a = codec.decode(comp_a, layer_idx, tensor_id=1)
    rest_b = codec.decode(comp_b, layer_idx, tensor_id=2)
    err_a = (kv_a - rest_a).norm() / kv_a.norm()
    err_b = (kv_b - rest_b).norm() / kv_b.norm()
    assert err_a < 0.01
    assert err_b < 0.01

"""Unit tests for TurboQuantCodec (Activity C)."""

import torch
import pytest

from src.cache.turbo_quant import TurboQuantCodec
from src.cache.base import CacheStore


@pytest.fixture
def codec() -> TurboQuantCodec:
    torch.manual_seed(42)
    return TurboQuantCodec(num_layers=12, bits=3, qjl_bits=1, base_seed=42, sensitive_layers_ratio=0.25)


def test_encode_returns_dict_with_required_keys(codec: TurboQuantCodec) -> None:
    torch.manual_seed(42)
    kv = torch.randn(10, 64)
    result = codec.encode(kv, layer_idx=6)
    for key in ("quantized", "scale", "qjl_packed", "layer_idx"):
        assert key in result, f"Missing key: {key}"


def test_decode_shape_matches_input(codec: TurboQuantCodec) -> None:
    torch.manual_seed(42)
    kv = torch.randn(20, 128)
    compressed = codec.encode(kv, layer_idx=6)
    decoded = codec.decode(compressed, layer_idx=6)
    assert decoded.shape == kv.shape


def test_compression_ratio_3bit(codec: TurboQuantCodec) -> None:
    ratio = codec.compression_ratio(layer_idx=6)
    assert ratio >= 0.60, f"Expected ≥ 0.60, got {ratio:.4f}"


def test_compression_ratio_4bit_sensitive(codec: TurboQuantCodec) -> None:
    # layer_idx=0 is a sensitive layer (4-bit), still expect ≥ 50% reduction
    ratio = codec.compression_ratio(layer_idx=0)
    assert ratio >= 0.50, f"Expected ≥ 0.50, got {ratio:.4f}"


def test_layer_specific_rotation(codec: TurboQuantCodec) -> None:
    d_head = 64
    R0 = codec._get_rotation_matrix(0, d_head)
    R6 = codec._get_rotation_matrix(6, d_head)
    assert not torch.allclose(R0, R6), "Different layers must produce different rotation matrices"


def test_edge_case_n_tokens_1(codec: TurboQuantCodec) -> None:
    torch.manual_seed(42)
    kv = torch.randn(1, 64)
    compressed = codec.encode(kv, layer_idx=6)
    decoded = codec.decode(compressed, layer_idx=6)
    assert decoded.shape == (1, 64)


def test_memory_bytes_estimate_format(codec: TurboQuantCodec) -> None:
    result = codec.memory_bytes_estimate(n_tokens=100, d_head=128, layer_idx=6)
    for key in ("total_bytes", "baseline_bytes", "reduction_ratio"):
        assert key in result, f"Missing key: {key}"
    assert result["baseline_bytes"] == 100 * 128 * 4
    assert result["reduction_ratio"] > 0.0


def test_cachestore_not_inherited(codec: TurboQuantCodec) -> None:
    assert not isinstance(codec, CacheStore), "TurboQuantCodec must NOT inherit CacheStore"

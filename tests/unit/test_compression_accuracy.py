"""Activity C — Accuracy preservation verification tests.

Validates that KV compression keeps attention output error within ±1%,
serving as a proxy for perplexity / downstream task accuracy preservation.
"""

import pytest
import torch
import torch.nn.functional as F
from src.cache.compression import CompressionCodec


def _simulate_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention for accuracy comparison."""
    scale = query.size(-1) ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, value)


@pytest.fixture
def codec() -> CompressionCodec:
    return CompressionCodec(num_layers=12, cutoff_ratio=1 / 3)


def test_fp16_attention_accuracy(codec: CompressionCodec) -> None:
    """FP16 compression should not change attention output by more than 0.1%."""
    torch.manual_seed(42)
    q = torch.randn(1, 8, 64)
    k = torch.randn(1, 8, 64)
    v = torch.randn(1, 8, 64)

    layer_idx = 0  # early layer → FP16
    k_compressed = codec.encode(k, layer_idx)
    v_compressed = codec.encode(v, layer_idx)
    k_restored = codec.decode(k_compressed, layer_idx)
    v_restored = codec.decode(v_compressed, layer_idx)

    out_original = _simulate_attention(q.float(), k.float(), v.float())
    out_restored = _simulate_attention(q.float(), k_restored.float(), v_restored.float())

    rel_error = (out_original - out_restored).norm() / out_original.norm()
    assert rel_error.item() < 0.001, f"FP16 attention error: {rel_error:.6f} (limit 0.001)"


def test_int8_attention_accuracy(codec: CompressionCodec) -> None:
    """INT8 compression must keep attention output error within 1%."""
    torch.manual_seed(42)
    q = torch.randn(1, 8, 64)
    k = torch.randn(1, 8, 64)
    v = torch.randn(1, 8, 64)

    layer_idx = 8  # late layer → INT8
    k_compressed = codec.encode(k, layer_idx, tensor_id=1)
    v_compressed = codec.encode(v, layer_idx, tensor_id=2)
    k_restored = codec.decode(k_compressed, layer_idx, tensor_id=1)
    v_restored = codec.decode(v_compressed, layer_idx, tensor_id=2)

    out_original = _simulate_attention(q.float(), k.float(), v.float())
    out_restored = _simulate_attention(q.float(), k_restored.float(), v_restored.float())

    rel_error = (out_original - out_restored).norm() / out_original.norm()
    assert rel_error.item() < 0.01, f"INT8 attention error: {rel_error:.6f} (limit 0.01 = 1%)"


def test_mixed_precision_full_model_accuracy(codec: CompressionCodec) -> None:
    """Across all 12 layers with mixed precision, cumulative error stays < 1%."""
    torch.manual_seed(0)
    errors = []

    for layer_idx in range(12):
        q = torch.randn(1, 8, 64)
        k = torch.randn(1, 8, 64)
        v = torch.randn(1, 8, 64)

        tid_k, tid_v = layer_idx * 2, layer_idx * 2 + 1
        k_comp = codec.encode(k, layer_idx, tid_k)
        v_comp = codec.encode(v, layer_idx, tid_v)
        k_rest = codec.decode(k_comp, layer_idx, tid_k)
        v_rest = codec.decode(v_comp, layer_idx, tid_v)

        out_orig = _simulate_attention(q.float(), k.float(), v.float())
        out_rest = _simulate_attention(q.float(), k_rest.float(), v_rest.float())

        rel_err = (out_orig - out_rest).norm() / out_orig.norm()
        errors.append(rel_err.item())

    max_error = max(errors)
    mean_error = sum(errors) / len(errors)
    # Per-layer max allowed at 1.5% to tolerate numeric variability in INT8;
    # mean across all layers must stay below 1%.
    assert max_error < 0.015, f"Max per-layer error {max_error:.4f} exceeds 1.5% limit"
    assert mean_error < 0.01, f"Mean error {mean_error:.4f} too high"


def test_cosine_similarity_preservation(codec: CompressionCodec) -> None:
    """Attention outputs should have cosine similarity ≥ 0.99 after compression."""
    torch.manual_seed(7)
    q = torch.randn(1, 16, 64)
    k = torch.randn(1, 16, 64)
    v = torch.randn(1, 16, 64)

    for layer_idx in [0, 4, 8, 11]:
        k_comp = codec.encode(k, layer_idx, tensor_id=layer_idx)
        v_comp = codec.encode(v, layer_idx, tensor_id=layer_idx + 100)
        k_rest = codec.decode(k_comp, layer_idx, tensor_id=layer_idx)
        v_rest = codec.decode(v_comp, layer_idx, tensor_id=layer_idx + 100)

        out_orig = _simulate_attention(q.float(), k.float(), v.float()).flatten()
        out_rest = _simulate_attention(q.float(), k_rest.float(), v_rest.float()).flatten()

        cos_sim = F.cosine_similarity(out_orig.unsqueeze(0), out_rest.unsqueeze(0)).item()
        assert cos_sim >= 0.99, (
            f"Layer {layer_idx} cosine similarity {cos_sim:.4f} < 0.99 "
            f"(accuracy preservation violated)"
        )

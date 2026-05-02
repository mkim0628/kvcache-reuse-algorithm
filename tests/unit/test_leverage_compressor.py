"""Unit tests for LeverageScoreCompressor (Activity C).

Verifies leverage-score computation, 3-tier classification, sign-code
packing, encode/decode round-trips, memory estimates, and edge cases.
All tests run on CPU tensors with torch.manual_seed(42).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.cache.leverage_compressor import LeverageScoreCompressor, _unpack_signs_to_pm1


SEED = 42
N_TOKENS = 100
D_HEAD = 64


@pytest.fixture
def compressor() -> LeverageScoreCompressor:
    return LeverageScoreCompressor(
        rank=32,
        reg_lambda=1e-3,
        tier1_ratio=0.20,
        tier3_ratio=0.20,
    )


@pytest.fixture
def kv_pair():
    torch.manual_seed(SEED)
    keys = torch.randn(N_TOKENS, D_HEAD)
    values = torch.randn(N_TOKENS, D_HEAD)
    return keys, values


# ---------------------------------------------------------------------------
# Leverage score computation
# ---------------------------------------------------------------------------

def test_leverage_scores_shape(compressor: LeverageScoreCompressor, kv_pair) -> None:
    """compute_leverage_scores() returns (n_tokens,) non-negative float tensor."""
    keys, _ = kv_pair
    scores = compressor.compute_leverage_scores(keys)
    assert scores.shape == (N_TOKENS,), f"Expected ({N_TOKENS},), got {scores.shape}"
    assert (scores >= 0).all(), "All leverage scores must be non-negative"


def test_leverage_scores_relative_order(compressor: LeverageScoreCompressor) -> None:
    """Tokens that span a wider range should have larger leverage scores."""
    torch.manual_seed(SEED)
    # Construct two obvious outliers
    keys = torch.zeros(20, D_HEAD)
    keys[0] = torch.ones(D_HEAD) * 10.0   # large magnitude → high leverage
    keys[1] = torch.ones(D_HEAD) * 0.01   # near zero → low leverage
    scores = compressor.compute_leverage_scores(keys)
    assert scores[0] > scores[1], "High-magnitude token should have higher leverage score"


# ---------------------------------------------------------------------------
# 3-tier classification
# ---------------------------------------------------------------------------

def test_classify_tier_sizes(compressor: LeverageScoreCompressor, kv_pair) -> None:
    """classify() must produce roughly 20 / 60 / 20 split for 100 tokens."""
    keys, values = kv_pair
    result = compressor.classify(keys, values)

    n1, n2, n3 = result["tier1"].numel(), result["tier2"].numel(), result["tier3"].numel()
    total = n1 + n2 + n3
    assert total == N_TOKENS, f"Tier indices do not cover all tokens: {total} != {N_TOKENS}"

    # Allow ±1 due to integer rounding
    assert abs(n1 - 20) <= 1, f"Tier-1 size {n1} not ≈ 20"
    assert abs(n3 - 20) <= 1, f"Tier-3 size {n3} not ≈ 20"
    assert abs(n2 - 60) <= 2, f"Tier-2 size {n2} not ≈ 60"


def test_classify_no_index_overlap(compressor: LeverageScoreCompressor, kv_pair) -> None:
    """Tier indices must be disjoint and cover every token exactly once."""
    keys, values = kv_pair
    result = compressor.classify(keys, values)

    t1 = result["tier1"].tolist()
    t2 = result["tier2"].tolist()
    t3 = result["tier3"].tolist()

    combined = set(t1) | set(t2) | set(t3)
    assert len(combined) == N_TOKENS, "Tiers do not cover all tokens"
    assert len(t1) + len(t2) + len(t3) == N_TOKENS, "Tiers overlap"


# ---------------------------------------------------------------------------
# Sign code packing
# ---------------------------------------------------------------------------

def test_to_sign_code_shape(compressor: LeverageScoreCompressor, kv_pair) -> None:
    """to_sign_code() returns (n_tokens, ceil(d_head/8)) uint8 tensor."""
    keys, _ = kv_pair
    code = compressor.to_sign_code(keys)
    expected_cols = math.ceil(D_HEAD / 8)
    assert code.shape == (N_TOKENS, expected_cols), (
        f"Expected ({N_TOKENS}, {expected_cols}), got {code.shape}"
    )
    assert code.dtype == torch.uint8, f"Expected uint8, got {code.dtype}"


def test_to_sign_code_hamming_distance_self_zero(
    compressor: LeverageScoreCompressor, kv_pair
) -> None:
    """XOR of a sign code with itself must be all zeros (zero Hamming distance)."""
    keys, _ = kv_pair
    code = compressor.to_sign_code(keys)
    xor = code ^ code
    assert xor.sum().item() == 0, "Self-XOR must be zero"


def test_to_sign_code_sign_consistency(
    compressor: LeverageScoreCompressor,
) -> None:
    """Unpacked bits must match the original key signs element-wise."""
    from src.cache.leverage_compressor import _unpackbits_2d
    torch.manual_seed(SEED)
    keys = torch.randn(10, D_HEAD)
    code = compressor.to_sign_code(keys)
    unpacked = _unpackbits_2d(code, D_HEAD)  # (10, d_head) uint8
    expected = (keys >= 0).to(torch.uint8)
    assert (unpacked == expected).all(), "Sign code does not match key signs"


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def test_encode_decode_tier1_precision(
    compressor: LeverageScoreCompressor, kv_pair
) -> None:
    """Tier-1 tokens decoded from FP16 must have cosine similarity ≥ 0.99."""
    keys, values = kv_pair
    storage = compressor.encode(keys, values, layer_idx=0)

    t1 = storage["tier1_indices"]
    tier1_kv_decoded = storage["tier1_kv"].float()  # (n1, 2*d_head)

    original_kv = torch.cat([keys[t1], values[t1]], dim=-1)  # (n1, 2*d_head)

    cos_sim = F.cosine_similarity(
        original_kv.flatten().unsqueeze(0),
        tier1_kv_decoded.flatten().unsqueeze(0),
    ).item()

    assert cos_sim >= 0.99, (
        f"Tier-1 FP16 cosine similarity {cos_sim:.6f} < 0.99"
    )


def test_encode_keys_count(compressor: LeverageScoreCompressor, kv_pair) -> None:
    """encode() must return roughly 20% of tokens in Tier-1."""
    keys, values = kv_pair
    storage = compressor.encode(keys, values, layer_idx=3)

    n1 = storage["tier1_kv"].shape[0]
    assert abs(n1 - int(N_TOKENS * 0.20)) <= 1, (
        f"Tier-1 count {n1} not ≈ {int(N_TOKENS * 0.20)}"
    )


def test_encode_decode_full_roundtrip_shapes(
    compressor: LeverageScoreCompressor, kv_pair
) -> None:
    """decode() output must have shape (n_tokens, 2*d_head)."""
    keys, values = kv_pair
    storage = compressor.encode(keys, values, layer_idx=0)
    reconstructed = compressor.decode(storage)

    assert reconstructed.shape == (N_TOKENS, 2 * D_HEAD), (
        f"Expected ({N_TOKENS}, {2 * D_HEAD}), got {reconstructed.shape}"
    )


def test_encode_decode_tier2_value_preserved(
    compressor: LeverageScoreCompressor, kv_pair
) -> None:
    """Tier-2 values decoded from FP16 must have cosine similarity ≥ 0.99."""
    keys, values = kv_pair
    storage = compressor.encode(keys, values, layer_idx=0)

    t2 = storage["tier2_indices"]
    if t2.numel() == 0:
        pytest.skip("No Tier-2 tokens (edge case)")

    val_original = values[t2].float().flatten()
    val_decoded = storage["tier2_v_fp16"].float().flatten()

    cos_sim = F.cosine_similarity(
        val_original.unsqueeze(0),
        val_decoded.unsqueeze(0),
    ).item()
    assert cos_sim >= 0.99, (
        f"Tier-2 value FP16 cosine similarity {cos_sim:.6f} < 0.99"
    )


# ---------------------------------------------------------------------------
# Memory estimate
# ---------------------------------------------------------------------------

def test_memory_estimate_reduction(compressor: LeverageScoreCompressor) -> None:
    """memory_bytes_estimate(1000, 64) must show ≥70% reduction vs FP32 baseline."""
    est = compressor.memory_bytes_estimate(1000, D_HEAD)

    assert est["reduction_ratio"] >= 0.70, (
        f"Memory reduction {est['reduction_ratio']:.4f} < 70%"
    )
    assert est["total_bytes"] < est["baseline_bytes"], (
        "Compressed size must be smaller than FP32 baseline"
    )


def test_memory_estimate_structure(compressor: LeverageScoreCompressor) -> None:
    """memory_bytes_estimate() must return all required keys."""
    est = compressor.memory_bytes_estimate(100, D_HEAD)
    for key in ("tier1_bytes", "tier2_bytes", "tier3_bytes",
                "total_bytes", "baseline_bytes", "reduction_ratio"):
        assert key in est, f"Missing key '{key}' in memory estimate"
    assert est["tier3_bytes"] == 0, "Tier-3 (evicted) must have 0 bytes"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_edge_case_single_token(compressor: LeverageScoreCompressor) -> None:
    """n_tokens=1: Tier-1 gets 1 token, Tier-2 and Tier-3 are empty."""
    torch.manual_seed(SEED)
    keys = torch.randn(1, D_HEAD)
    values = torch.randn(1, D_HEAD)

    storage = compressor.encode(keys, values, layer_idx=0)

    assert storage["tier1_kv"].shape[0] == 1, "Single token must go to Tier-1"
    assert storage["tier2_sign_k"].shape[0] == 0, "Tier-2 must be empty for single token"
    assert storage["tier3_indices"].numel() == 0, "Tier-3 must be empty for single token"


def test_edge_case_two_tokens(compressor: LeverageScoreCompressor) -> None:
    """n_tokens=2: Tier-1=1, Tier-2=1, Tier-3=0 (floor(2*0.2)=0 so n3=0)."""
    torch.manual_seed(SEED)
    keys = torch.randn(2, D_HEAD)
    values = torch.randn(2, D_HEAD)

    result = compressor.classify(keys, values)

    # With int(2 * 0.20) = 0 for tier3, we clamp n3=0 so tier3 is empty
    # n1 = max(1, 0) = 1
    # Tier1=1, Tier3=0, Tier2=1
    assert result["tier1"].numel() == 1, f"Expected 1 in Tier-1, got {result['tier1'].numel()}"
    total = (result["tier1"].numel()
             + result["tier2"].numel()
             + result["tier3"].numel())
    assert total == 2, f"All tokens must be accounted for, got {total}"


def test_edge_case_small_rank_clamp(compressor: LeverageScoreCompressor) -> None:
    """When n_tokens < rank, rank is clamped to n_tokens without error."""
    torch.manual_seed(SEED)
    # rank=32 but only 5 tokens
    keys = torch.randn(5, D_HEAD)
    scores = compressor.compute_leverage_scores(keys)
    assert scores.shape == (5,), "Must handle n_tokens < rank"
    assert (scores >= 0).all(), "Scores must be non-negative even with clamped rank"

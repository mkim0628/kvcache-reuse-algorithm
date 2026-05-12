"""Unit tests for MixedDimPerTokenBudgetCodec (Activity C).

Tests:
  - compute_loss_scores shape and non-negativity
  - find_threshold bisection accuracy
  - encode/decode shape preservation
  - budget_ratio approximately satisfied
  - min_retain_ratio enforcement
  - memory_reduction_ratio
  - compression_hook interface
  - Serialisation compatibility
"""

import pytest
import torch

from src.cache.mixed_dim_codec import MixedDimConfig, MixedDimPerTokenBudgetCodec


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg():
    return MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.50, seed=42)


@pytest.fixture
def codec(cfg):
    return MixedDimPerTokenBudgetCodec(cfg)


def make_kv(n_tokens: int, n_heads: int = 4, d_head: int = 32, seed: int = 42):
    torch.manual_seed(seed)
    return torch.randn(n_tokens, 2, n_heads, d_head)


# --------------------------------------------------------------------------- #
# compute_loss_scores                                                           #
# --------------------------------------------------------------------------- #


class TestComputeLossScores:
    def test_output_shape(self, codec):
        kv = make_kv(16)
        scores = codec.compute_loss_scores(kv)
        assert scores.shape == (16, 4, 32)

    def test_non_negative(self, codec):
        kv = make_kv(16)
        scores = codec.compute_loss_scores(kv)
        assert (scores >= 0).all()

    def test_with_attn_weights_shape(self, codec):
        kv = make_kv(16)
        attn_w = torch.rand(16)
        scores = codec.compute_loss_scores(kv, attn_weights=attn_w)
        assert scores.shape == (16, 4, 32)

    def test_high_attn_weight_gives_higher_scores(self, codec):
        """Higher attention weight → higher loss scores (more important → retain)."""
        kv = make_kv(8)
        low_w = torch.zeros(8)
        high_w = torch.ones(8) * 10.0
        scores_low = codec.compute_loss_scores(kv, attn_weights=low_w)
        scores_high = codec.compute_loss_scores(kv, attn_weights=high_w)
        assert scores_high.mean() > scores_low.mean()


# --------------------------------------------------------------------------- #
# find_threshold                                                                #
# --------------------------------------------------------------------------- #


class TestFindThreshold:
    def test_returns_float(self, codec):
        kv = make_kv(32)
        scores = codec.compute_loss_scores(kv)
        lam = codec.find_threshold(scores)
        assert isinstance(lam, float)

    def test_threshold_satisfies_budget(self, codec):
        """Retained fraction after applying λ* should be close to budget_ratio."""
        kv = make_kv(64)
        scores = codec.compute_loss_scores(kv)
        lam = codec.find_threshold(scores, budget_ratio=0.50)
        retained = (scores >= lam).float().mean().item()
        assert abs(retained - 0.50) < 0.05

    def test_higher_budget_gives_lower_threshold(self, codec):
        """A higher budget_ratio (keep more) should require a lower threshold."""
        kv = make_kv(32)
        scores = codec.compute_loss_scores(kv)
        lam_30 = codec.find_threshold(scores, budget_ratio=0.30)
        lam_70 = codec.find_threshold(scores, budget_ratio=0.70)
        assert lam_30 >= lam_70

    def test_zero_tensor_threshold(self, codec):
        """All-zero loss scores → threshold 0.0 (retain everything or nothing)."""
        scores = torch.zeros(8, 4, 32)
        lam = codec.find_threshold(scores)
        assert isinstance(lam, float)


# --------------------------------------------------------------------------- #
# encode / decode                                                               #
# --------------------------------------------------------------------------- #


class TestEncodeDecode:
    def test_encode_returns_dict_with_expected_keys(self, codec):
        kv = make_kv(16)
        encoded = codec.encode(kv)
        required = {"masked_kv", "retain_mask", "lambda_star", "budget_ratio", "n_tokens", "n_heads", "d_head"}
        assert required.issubset(encoded.keys())

    def test_encode_decode_shape_preserved(self, codec):
        kv = make_kv(32)
        encoded = codec.encode(kv)
        recovered = codec.decode(encoded)
        assert recovered.shape == kv.shape

    def test_decode_returns_masked_kv(self, codec):
        kv = make_kv(16)
        encoded = codec.encode(kv)
        recovered = codec.decode(encoded)
        assert torch.allclose(recovered, encoded["masked_kv"])

    def test_budget_ratio_approximately_met(self, codec):
        kv = make_kv(64)
        encoded = codec.encode(kv)
        assert abs(encoded["budget_ratio"] - 0.50) < 0.05

    def test_masked_kv_dtype_matches_input(self, codec):
        kv = make_kv(16).half()
        encoded = codec.encode(kv)
        assert encoded["masked_kv"].dtype == kv.dtype

    def test_zeroed_dims_are_zero(self, codec):
        kv = make_kv(16)
        encoded = codec.encode(kv)
        mask = encoded["retain_mask"]  # [n_tokens, n_heads, d_head]
        masked_kv = encoded["masked_kv"]
        # Dimensions not in retain_mask should be exactly 0
        mask_4d = mask.unsqueeze(1).expand_as(masked_kv)
        dropped = masked_kv[~mask_4d]
        assert torch.allclose(dropped, torch.zeros_like(dropped))

    def test_encode_with_custom_budget_ratio(self, codec):
        kv = make_kv(64)
        encoded = codec.encode(kv, budget_ratio=0.30)
        # min_retain_ratio enforcement may push actual retention above the target;
        # allow looser bound when budget is near min_retain_ratio
        assert encoded["budget_ratio"] >= 0.30 - 0.01  # never below target
        assert encoded["budget_ratio"] <= 0.30 + 0.10  # ceiling from min_retain guard

    def test_encode_n_tokens_n_heads_d_head_correct(self, codec):
        kv = make_kv(12)
        encoded = codec.encode(kv)
        assert encoded["n_tokens"] == 12
        assert encoded["n_heads"] == 4
        assert encoded["d_head"] == 32


# --------------------------------------------------------------------------- #
# min_retain_ratio enforcement                                                  #
# --------------------------------------------------------------------------- #


class TestMinRetainRatio:
    def test_min_retain_ratio_respected(self):
        """Each (token, head) pair should retain at least min_retain_ratio dims."""
        cfg = MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.10, min_retain_ratio=0.10)
        codec = MixedDimPerTokenBudgetCodec(cfg)
        kv = make_kv(8, n_heads=4, d_head=32)
        encoded = codec.encode(kv)
        mask = encoded["retain_mask"]  # [n_tokens, n_heads, d_head]
        per_pair_ratio = mask.float().mean(dim=-1)  # [n_tokens, n_heads]
        assert (per_pair_ratio >= cfg.min_retain_ratio - 1e-6).all()

    def test_very_low_budget_still_keeps_minimum(self):
        """Even budget_ratio=0.01 must keep min_retain_ratio per pair."""
        cfg = MixedDimConfig(n_heads=2, d_head=16, budget_ratio=0.01, min_retain_ratio=0.25)
        codec = MixedDimPerTokenBudgetCodec(cfg)
        kv = make_kv(4, n_heads=2, d_head=16)
        encoded = codec.encode(kv)
        mask = encoded["retain_mask"]
        per_pair = mask.float().mean(dim=-1)
        assert (per_pair >= cfg.min_retain_ratio - 1e-6).all()


# --------------------------------------------------------------------------- #
# memory_reduction_ratio                                                        #
# --------------------------------------------------------------------------- #


class TestMemoryReductionRatio:
    def test_reduction_ratio_for_50pct_budget(self, codec):
        kv = make_kv(64)
        encoded = codec.encode(kv)
        ratio = codec.memory_reduction_ratio(encoded)
        # 50% budget → ~50% memory reduction
        assert 0.30 <= ratio <= 0.70

    def test_reduction_ratio_increases_with_lower_budget(self):
        kv = make_kv(64)
        cfg_high = MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.70)
        cfg_low = MixedDimConfig(n_heads=4, d_head=32, budget_ratio=0.30)
        codec_high = MixedDimPerTokenBudgetCodec(cfg_high)
        codec_low = MixedDimPerTokenBudgetCodec(cfg_low)
        enc_high = codec_high.encode(kv)
        enc_low = codec_low.encode(kv)
        assert codec_low.memory_reduction_ratio(enc_low) > codec_high.memory_reduction_ratio(enc_high)

    def test_memory_reduction_meets_30pct_target(self, codec):
        """budget_ratio=0.50 → memory reduction >= 30%."""
        kv = make_kv(64)
        encoded = codec.encode(kv)
        assert codec.memory_reduction_ratio(encoded) >= 0.30


# --------------------------------------------------------------------------- #
# compression_hook                                                              #
# --------------------------------------------------------------------------- #


class TestCompressionHook:
    def test_hook_returns_tensor_with_correct_shape(self, codec):
        kv = make_kv(16)
        out = codec.compression_hook("key", kv)
        assert out.shape == kv.shape
        assert isinstance(out, torch.Tensor)

    def test_hook_with_attn_weights(self, codec):
        kv = make_kv(16)
        w = torch.rand(16)
        out = codec.compression_hook("key", kv, attn_weights=w)
        assert out.shape == kv.shape


# --------------------------------------------------------------------------- #
# Budget sweep                                                                  #
# --------------------------------------------------------------------------- #


class TestBudgetSweep:
    def test_all_budget_ratios_produce_correct_shapes(self):
        cfg = MixedDimConfig(n_heads=4, d_head=32, seed=42)
        kv = make_kv(64)
        for ratio in [0.30, 0.40, 0.50, 0.60, 0.70]:
            cfg.budget_ratio = ratio
            codec = MixedDimPerTokenBudgetCodec(cfg)
            encoded = codec.encode(kv)
            recovered = codec.decode(encoded)
            assert recovered.shape == kv.shape

    def test_higher_budget_ratio_retains_more(self):
        cfg = MixedDimConfig(n_heads=4, d_head=32, seed=42)
        kv = make_kv(64)
        ratios = [0.30, 0.40, 0.50, 0.60, 0.70]
        actual_retentions = []
        for ratio in ratios:
            cfg.budget_ratio = ratio
            codec = MixedDimPerTokenBudgetCodec(cfg)
            encoded = codec.encode(kv)
            actual_retentions.append(encoded["budget_ratio"])
        # Should be monotonically non-decreasing
        for i in range(len(actual_retentions) - 1):
            assert actual_retentions[i] <= actual_retentions[i + 1] + 0.05

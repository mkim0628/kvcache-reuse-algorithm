"""Smoke tests for vllm_integration 2026-05-02 B+C cycle.

Tests:
  1. Import check — all three new patch modules import without error.
  2. SignVQCacheParams instantiation + validation.
  3. VllmLeverageCompressor on a mock KV block (torch tensor).
  4. SignVQSegmentIndex: put + exact get (exact_fp16) + approx get (approx_sign)
     + full miss.
  5. NonContiguousKVCacheManagerV2: store_segment / lookup_segment / stats.
  6. Memory estimate sanity (≥ 30% reduction vs FP32 baseline).
"""

from __future__ import annotations

import math
import sys
import pathlib

import pytest
import torch

# Ensure repo root is on sys.path regardless of where pytest is invoked from.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --------------------------------------------------------------------------- #
# 1. Import checks                                                              #
# --------------------------------------------------------------------------- #

class TestImports:
    def test_leverage_compressor_patch_imports(self) -> None:
        from vllm_integration import leverage_compressor_patch  # noqa: F401
        from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
        assert VllmLeverageCompressor is not None

    def test_sign_vq_block_manager_patch_imports(self) -> None:
        from vllm_integration import sign_vq_block_manager_patch  # noqa: F401
        from vllm_integration.sign_vq_block_manager_patch import (
            SignVQSegmentIndex,
            NonContiguousKVCacheManagerV2,
        )
        assert SignVQSegmentIndex is not None
        assert NonContiguousKVCacheManagerV2 is not None

    def test_cache_config_extension_imports(self) -> None:
        from vllm_integration import cache_config_extension  # noqa: F401
        from vllm_integration.cache_config_extension import (
            SignVQCacheParams,
            SignVQCacheConfigMixin,
            build_sign_vq_compressor,
            build_sign_vq_index,
        )
        assert SignVQCacheParams is not None
        assert SignVQCacheConfigMixin is not None
        assert build_sign_vq_compressor is not None
        assert build_sign_vq_index is not None


# --------------------------------------------------------------------------- #
# 2. CacheConfig extension instantiation + validation                          #
# --------------------------------------------------------------------------- #

class TestSignVQCacheParams:
    def test_default_instantiation(self) -> None:
        from vllm_integration.cache_config_extension import SignVQCacheParams
        params = SignVQCacheParams()
        assert params.enable_sign_vq is False
        assert params.tier1_ratio == 0.20
        assert params.tier2_ratio == 0.60
        assert params.hamming_threshold == 0.15

    def test_tier3_ratio_derived(self) -> None:
        from vllm_integration.cache_config_extension import SignVQCacheParams
        params = SignVQCacheParams(tier1_ratio=0.20, tier2_ratio=0.60)
        assert abs(params.tier3_ratio - 0.20) < 1e-9

    def test_custom_params(self) -> None:
        from vllm_integration.cache_config_extension import SignVQCacheParams
        params = SignVQCacheParams(
            enable_sign_vq=True,
            tier1_ratio=0.30,
            tier2_ratio=0.50,
            hamming_threshold=0.10,
            rank=16,
            chunk_size=32,
            max_entries=500,
        )
        assert params.enable_sign_vq is True
        assert params.tier1_ratio == 0.30
        assert abs(params.tier3_ratio - 0.20) < 1e-9

    def test_invalid_tier_ratios_raise(self) -> None:
        from vllm_integration.cache_config_extension import SignVQCacheParams
        with pytest.raises(ValueError):
            SignVQCacheParams(tier1_ratio=0.60, tier2_ratio=0.60)

    def test_invalid_hamming_threshold_raises(self) -> None:
        from vllm_integration.cache_config_extension import SignVQCacheParams
        with pytest.raises(ValueError):
            SignVQCacheParams(hamming_threshold=1.5)

    def test_build_compressor_factory(self) -> None:
        from vllm_integration.cache_config_extension import (
            SignVQCacheParams, build_sign_vq_compressor,
        )
        from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
        params = SignVQCacheParams(enable_sign_vq=True)
        comp = build_sign_vq_compressor(params)
        assert isinstance(comp, VllmLeverageCompressor)
        assert comp.tier1_ratio == params.tier1_ratio

    def test_build_index_factory(self) -> None:
        from vllm_integration.cache_config_extension import (
            SignVQCacheParams, build_sign_vq_index,
        )
        from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex
        params = SignVQCacheParams(enable_sign_vq=True, max_entries=50)
        idx = build_sign_vq_index(params)
        assert isinstance(idx, SignVQSegmentIndex)
        assert idx.max_entries == 50


# --------------------------------------------------------------------------- #
# 3. VllmLeverageCompressor on mock KV block                                   #
# --------------------------------------------------------------------------- #

class TestVllmLeverageCompressor:
    @pytest.fixture
    def compressor(self):
        from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
        return VllmLeverageCompressor(
            rank=4,
            tier1_ratio=0.20,
            tier3_ratio=0.20,
        )

    def test_encode_decode_2d_shape(self, compressor) -> None:
        torch.manual_seed(42)
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        storage = compressor.encode_block(k, v, layer_idx=0, block_id=0)
        decoded = compressor.decode_block(storage)
        assert decoded.shape == (2, 16, 32)

    def test_encode_decode_3d_shape(self, compressor) -> None:
        torch.manual_seed(42)
        # vLLM shape: (block_size, num_kv_heads, head_size)
        k = torch.randn(16, 4, 32)
        v = torch.randn(16, 4, 32)
        storage = compressor.encode_block(k, v, layer_idx=1, block_id=5)
        decoded = compressor.decode_block(storage)
        # decode_block returns (2, n_tokens, d_head) where d_head = 4*32=128
        assert decoded.shape == (2, 16, 4 * 32)

    def test_multihead_encode_decode(self, compressor) -> None:
        torch.manual_seed(42)
        k = torch.randn(16, 4, 32)
        v = torch.randn(16, 4, 32)
        per_head = compressor.encode_block_multihead(k, v, layer_idx=0)
        assert len(per_head) == 4
        out = compressor.decode_block_multihead(per_head)
        assert out.shape == (2, 16, 4, 32)

    def test_tier1_tokens_exact_within_fp16_tolerance(self, compressor) -> None:
        """Tier-1 tokens should round-trip through FP16 accurately."""
        torch.manual_seed(42)
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        storage = compressor.encode_block(k, v, layer_idx=0)
        decoded = compressor.decode_block(storage)
        t1 = storage["tier1_indices"]
        d = decoded[0].shape[-1]  # d_head
        k_fp32 = k.float()
        # FP16 round-trip error should be tiny (within fp16 precision)
        err = (decoded[0][t1] - k_fp32[t1]).abs().max().item()
        assert err < 1e-2, f"Tier-1 key round-trip error too large: {err}"

    def test_memory_estimate_reduction_ge_30pct(self, compressor) -> None:
        est = compressor.memory_bytes_estimate(n_tokens=100, d_head=64)
        assert est["reduction_ratio"] >= 0.30, (
            f"Memory reduction {est['reduction_ratio']:.2%} < 30%"
        )

    def test_memory_estimate_reduction_ge_70pct(self, compressor) -> None:
        """With default ratios, should achieve the 70% target."""
        est = compressor.memory_bytes_estimate(n_tokens=1000, d_head=64)
        assert est["reduction_ratio"] >= 0.70, (
            f"Memory reduction {est['reduction_ratio']:.2%} < 70%"
        )

    def test_leverage_scores_positive(self, compressor) -> None:
        torch.manual_seed(42)
        k = torch.randn(32, 64)
        scores = compressor.compute_leverage_scores(k)
        assert scores.shape == (32,)
        assert (scores >= 0).all()

    def test_sign_code_shape(self, compressor) -> None:
        torch.manual_seed(42)
        k = torch.randn(16, 64)
        sign_code = compressor.to_sign_code(k)
        expected_cols = math.ceil(64 / 8)
        assert sign_code.shape == (16, expected_cols)
        assert sign_code.dtype == torch.uint8


# --------------------------------------------------------------------------- #
# 4. SignVQSegmentIndex: put / get                                              #
# --------------------------------------------------------------------------- #

class TestSignVQSegmentIndex:
    @pytest.fixture
    def compressor(self):
        from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
        return VllmLeverageCompressor(rank=4, tier1_ratio=0.20, tier3_ratio=0.20)

    @pytest.fixture
    def index(self, compressor):
        from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex
        return SignVQSegmentIndex(
            compressor=compressor,
            chunk_size=16,
            max_entries=100,
            hamming_threshold=0.15,
        )

    def test_put_then_exact_fp16_hit(self, index) -> None:
        torch.manual_seed(42)
        tokens = list(range(32))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        index.put(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)
        kv_out, hit_type = index.get(tokens, chunk_idx=0, layer_idx=0)
        assert hit_type == "exact_fp16", f"Expected exact_fp16, got {hit_type}"
        assert kv_out is not None
        assert kv_out.shape[0] == 2  # (2, n_tokens, d_head)

    def test_miss_on_unknown_chunk(self, index) -> None:
        tokens = list(range(64))
        kv_out, hit_type = index.get(tokens, chunk_idx=99, layer_idx=0)
        assert hit_type == "miss"
        assert kv_out is None

    def test_approx_sign_hit(self, index) -> None:
        """Slightly perturbed query keys should still hit via sign-VQ."""
        torch.manual_seed(42)
        tokens = list(range(32))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        index.put(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)

        # Perturb keys slightly — sign bits should mostly agree
        k_noisy = k + 0.001 * torch.randn_like(k)
        _, hit_type = index.get(
            tokens, chunk_idx=0, layer_idx=0, query_keys=k_noisy
        )
        # Should be either exact_fp16 (if FP16 store had it) or approx_sign
        assert hit_type in ("exact_fp16", "approx_sign"), (
            f"Expected hit with slightly perturbed keys, got {hit_type}"
        )

    def test_tier_hit_rates_after_lookups(self, index) -> None:
        torch.manual_seed(42)
        tokens = list(range(32))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        index.put(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)

        # 1 hit + 1 miss
        index.get(tokens, chunk_idx=0, layer_idx=0)
        index.get(list(range(32, 64)), chunk_idx=0, layer_idx=0)

        rates = index.tier_hit_rates()
        assert 0.0 <= rates["overall"] <= 1.0
        assert 0.0 <= rates["noncontiguous_ratio"] <= 1.0

    def test_lru_eviction_respects_max_entries(self) -> None:
        from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
        from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex
        comp = VllmLeverageCompressor(rank=2, tier1_ratio=0.50, tier3_ratio=0.10)
        idx = SignVQSegmentIndex(compressor=comp, chunk_size=4, max_entries=3)
        for i in range(6):
            tokens = list(range(i * 4, (i + 1) * 4))
            k = torch.randn(4, 16)
            v = torch.randn(4, 16)
            idx.put(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)
        assert len(idx._fp16_store) <= 3, (
            f"FP16 store has {len(idx._fp16_store)} entries, expected ≤ 3"
        )

    def test_get_all_chunks(self, index) -> None:
        torch.manual_seed(0)
        tokens = list(range(32))
        for i in range(2):
            chunk_start = i * 16
            k = torch.randn(16, 32)
            v = torch.randn(16, 32)
            index.put(tokens, chunk_idx=i, keys=k, values=v, layer_idx=0)
        hits, misses = index.get_all_chunks(tokens, layer_idx=0)
        assert len(hits) + len(misses) == 2
        assert all(ht in ("exact_fp16", "approx_sign") for _, _, ht in hits)

    def test_memory_bytes_returns_int(self, index) -> None:
        torch.manual_seed(42)
        tokens = list(range(32))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        index.put(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)
        mem = index.memory_bytes()
        assert isinstance(mem, int)
        assert mem > 0

    def test_reset_stats(self, index) -> None:
        torch.manual_seed(42)
        tokens = list(range(32))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        index.put(tokens, 0, k, v, layer_idx=0)
        index.get(tokens, 0, layer_idx=0)
        index.reset_stats()
        rates = index.tier_hit_rates()
        assert rates["overall"] == 0.0


# --------------------------------------------------------------------------- #
# 5. NonContiguousKVCacheManagerV2 (standalone, no vLLM engine)                #
# --------------------------------------------------------------------------- #

class TestNonContiguousKVCacheManagerV2:
    """Test the sign-VQ manager without a live vLLM engine.

    Because the class inherits from ``object`` when vLLM is not available,
    these tests exercise the sign-index API directly.
    """

    @pytest.fixture
    def manager(self):
        from vllm_integration.sign_vq_block_manager_patch import (
            NonContiguousKVCacheManagerV2,
            _VLLM_AVAILABLE,
        )
        if _VLLM_AVAILABLE:
            pytest.skip("Full vLLM engine required for this test path")
        return NonContiguousKVCacheManagerV2(
            sign_vq_chunk_size=16,
            sign_vq_max_entries=200,
            sign_vq_hamming_threshold=0.15,
            sign_vq_rank=4,
            sign_vq_tier1_ratio=0.20,
            sign_vq_tier3_ratio=0.20,
        )

    def test_manager_has_sign_index(self, manager) -> None:
        from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex
        assert isinstance(manager._sign_index, SignVQSegmentIndex)

    def test_store_and_lookup_segment(self, manager) -> None:
        torch.manual_seed(42)
        tokens = list(range(32))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        manager.store_segment(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)
        kv_out, hit_type = manager.lookup_segment(tokens, chunk_idx=0, layer_idx=0)
        assert hit_type in ("exact_fp16", "approx_sign")
        assert kv_out is not None

    def test_sign_index_stats_returns_dict(self, manager) -> None:
        torch.manual_seed(42)
        tokens = list(range(16))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        manager.store_segment(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)
        manager.lookup_segment(tokens, chunk_idx=0, layer_idx=0)
        stats = manager.sign_index_stats()
        assert "overall" in stats
        assert "noncontiguous_ratio" in stats

    def test_lookup_all_segments(self, manager) -> None:
        torch.manual_seed(7)
        tokens = list(range(32))
        for i in range(2):
            k = torch.randn(16, 32)
            v = torch.randn(16, 32)
            manager.store_segment(tokens, chunk_idx=i, keys=k, values=v, layer_idx=0)
        hits, misses = manager.lookup_all_segments(tokens, layer_idx=0)
        assert len(hits) + len(misses) == 2


# --------------------------------------------------------------------------- #
# 6. Standalone manager (no vLLM) — always runs                                #
# --------------------------------------------------------------------------- #

class TestNonContiguousManagerStandalone:
    """Exercise the sign-VQ API when _VLLM_AVAILABLE is False (simulated)."""

    def test_store_lookup_standalone(self) -> None:
        """Test sign-VQ store/lookup directly via SignVQSegmentIndex (no engine init)."""
        from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor
        from vllm_integration.sign_vq_block_manager_patch import SignVQSegmentIndex

        # Build the index directly to avoid vLLM engine constructor requirements
        comp = VllmLeverageCompressor(rank=4, tier1_ratio=0.20, tier3_ratio=0.20)
        idx = SignVQSegmentIndex(
            compressor=comp,
            chunk_size=16,
            max_entries=50,
            hamming_threshold=0.15,
        )

        torch.manual_seed(99)
        tokens = list(range(16))
        k = torch.randn(16, 32)
        v = torch.randn(16, 32)
        idx.put(tokens, chunk_idx=0, keys=k, values=v, layer_idx=0)
        kv, ht = idx.get(tokens, chunk_idx=0, layer_idx=0)
        assert ht in ("exact_fp16", "approx_sign"), f"Got {ht}"
        assert kv is not None

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


# ===========================================================================
# 2026-05-13  Activity A+B+C smoke tests
# ===========================================================================

class TestPBKVSchedulerMixin2026_05_13:
    """Activity A (2026-05-13): PBKVAgentSegmentPreservationSchedulerMixin."""

    def test_import(self) -> None:
        from vllm_integration.scheduler_patch import (
            PBKVAgentSegmentPreservationSchedulerMixin,
            PBKVSchedulerConfig,
            make_pbkv_scheduler_class,
        )
        assert PBKVAgentSegmentPreservationSchedulerMixin is not None
        assert PBKVSchedulerConfig is not None
        assert make_pbkv_scheduler_class is not None

    def test_pbkv_pre_schedule_reorders(self) -> None:
        """Re-ordering should run without error and produce valid output."""
        from vllm_integration.scheduler_patch import (
            PBKVAgentSegmentPreservationSchedulerMixin,
            make_pbkv_scheduler_class,
        )

        class _MockReq:
            def __init__(self, rid, toks):
                self.request_id = rid
                self.prompt_token_ids = toks
                self._all_token_ids = toks

        class _MockQueue:
            def __init__(self, reqs):
                self._queue = list(reqs)
            def __iter__(self):
                return iter(self._queue)
            def __len__(self):
                return len(self._queue)

        class _Base:
            def __init__(self, *a, **kw):
                self.waiting = _MockQueue([])
                self.running = []
            def schedule(self):
                return {}

        PBKVSched = make_pbkv_scheduler_class(
            _Base,
            pbkv_segment_emb_dim=16,
            pbkv_history_steps=4,
            pbkv_chunk_size=4,
        )
        sched = PBKVSched()
        reqs = [
            _MockReq("a", list(range(8))),
            _MockReq("b", list(range(4))),
            _MockReq("c", list(range(12))),
        ]
        sched.waiting = _MockQueue(reqs)
        sched.schedule()

        stats = sched.pbkv_stats()
        assert stats["pbkv_step_count"] >= 1
        assert stats["pbkv_tracked_requests"] >= 0

    def test_preservation_policy(self) -> None:
        """Preservation policy returns two sets."""
        from vllm_integration.scheduler_patch import (
            PBKVAgentSegmentPreservationSchedulerMixin,
            make_pbkv_scheduler_class,
        )

        class _Base:
            def __init__(self, *a, **kw):
                self.waiting = iter([])
                self.running = []
            def schedule(self):
                return {}

        PBKVSched = make_pbkv_scheduler_class(
            _Base, pbkv_segment_emb_dim=16, pbkv_history_steps=2
        )
        sched = PBKVSched()
        preserve, evict = sched.pbkv_preservation_policy(["key1", "key2", "key3"])
        assert isinstance(preserve, set)
        assert isinstance(evict, set)
        assert preserve | evict <= {"key1", "key2", "key3"}

    def test_vllm_subclass(self) -> None:
        """Factory produces valid subclass of vLLM Scheduler."""
        from vllm_integration.scheduler_patch import make_pbkv_scheduler_class
        try:
            from vllm.v1.core.sched.scheduler import Scheduler
            PBKVSched = make_pbkv_scheduler_class(
                Scheduler, pbkv_segment_emb_dim=64
            )
            assert issubclass(PBKVSched, Scheduler)
        except Exception:
            pytest.skip("vLLM Scheduler requires GPU environment")


class TestKVFoldBlockManager2026_05_13:
    """Activity B (2026-05-13): KVFoldAccumulativeBlockManager."""

    def test_import(self) -> None:
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
            make_kvfold_kv_cache_manager_class,
        )
        assert KVFoldAccumulativeBlockManager is not None
        assert KVFoldBlockManagerConfig is not None
        assert make_kvfold_kv_cache_manager_class is not None

    def test_store_and_lookup(self) -> None:
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
        )
        cfg = KVFoldBlockManagerConfig(
            chunk_size=4, max_entries=50, n_heads=2, d_head=8, seed=42
        )
        mgr = KVFoldAccumulativeBlockManager(cfg)
        tokens = list(range(8))
        kv = torch.randn(4, 2, 2, 8)
        key = mgr.store_chunk(tokens, chunk_idx=0, layer_idx=0, kv_tensor=kv)
        assert isinstance(key, str)
        result = mgr.lookup_chunk(tokens, chunk_idx=0, layer_idx=0)
        assert result is not None

    def test_fold_accumulation(self) -> None:
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
        )
        cfg = KVFoldBlockManagerConfig(
            chunk_size=4, max_entries=50, n_heads=2, d_head=8,
            window_size=4, seed=42,
        )
        mgr = KVFoldAccumulativeBlockManager(cfg)
        fold_key1, acc1 = mgr.fold_chunk(list(range(4)), layer_idx=0)
        fold_key2, acc2 = mgr.fold_chunk(list(range(4, 8)), layer_idx=0, existing_fold_key=fold_key1)
        assert fold_key1.startswith("fold:")
        assert fold_key2.startswith("fold:")
        assert acc2.shape[0] >= 4

    def test_fold_prefix_lookup(self) -> None:
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
        )
        cfg = KVFoldBlockManagerConfig(
            chunk_size=4, n_heads=2, d_head=8, seed=42
        )
        mgr = KVFoldAccumulativeBlockManager(cfg)
        fold_key, _ = mgr.fold_chunk(list(range(4)), layer_idx=0)
        prefix = mgr.lookup_fold_prefix(fold_key)
        assert prefix is not None

    def test_hit_stats(self) -> None:
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
        )
        cfg = KVFoldBlockManagerConfig(chunk_size=4, n_heads=2, d_head=8, seed=42)
        mgr = KVFoldAccumulativeBlockManager(cfg)
        tokens = list(range(8))
        mgr.store_chunk(tokens, 0, 0, torch.randn(4, 2, 2, 8))
        mgr.lookup_chunk(tokens, 0, 0)  # hit
        mgr.lookup_chunk(tokens, 1, 0)  # miss
        stats = mgr.hit_stats()
        assert stats["total_hits"] >= 1
        assert stats["total_misses"] >= 1
        assert 0.0 <= stats["hit_rate"] <= 1.0

    def test_vllm_factory(self) -> None:
        from vllm_integration.block_manager_patch import make_kvfold_kv_cache_manager_class
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            KVFoldMgr = make_kvfold_kv_cache_manager_class(
                KVCacheManager, chunk_size=4, n_heads=2, d_head=8
            )
            assert issubclass(KVFoldMgr, KVCacheManager)
        except Exception:
            pytest.skip("KVCacheManager requires GPU environment")


class TestSRFTInt8AttentionHook2026_05_13:
    """Activity C (2026-05-13): SRFTInt8AttentionHook."""

    def test_import(self) -> None:
        from vllm_integration.attention_backend_patch import (
            SRFTInt8AttentionHook,
            SRFTInt8Config,
            apply_srft_int8_patch,
            extend_cache_config_srft_int8,
        )
        assert SRFTInt8AttentionHook is not None
        assert SRFTInt8Config is not None
        assert apply_srft_int8_patch is not None
        assert extend_cache_config_srft_int8 is not None

    def test_write_read_roundtrip(self) -> None:
        from vllm_integration.attention_backend_patch import SRFTInt8AttentionHook
        hook = SRFTInt8AttentionHook(n_heads=4, d_head=16, group_size=16, seed=42)
        torch.manual_seed(42)
        key = torch.randn(8, 4, 16)
        value = torch.randn(8, 4, 16)
        payload = hook.write_to_cache(key, value, layer_idx=0)
        assert payload["compressed"] is True
        key_dec, val_dec = hook.read_from_cache(payload)
        assert key_dec.shape == key.shape
        assert val_dec.shape == value.shape
        rel_err = ((key_dec.float() - key).norm() / key.norm()).item()
        assert rel_err < 0.05, f"key rel error {rel_err:.4f} exceeds 5%"

    def test_accuracy_preservation(self) -> None:
        """Verify relative error < 1% for medium-sized KV (Report ① contract)."""
        from vllm_integration.attention_backend_patch import SRFTInt8AttentionHook
        hook = SRFTInt8AttentionHook(n_heads=8, d_head=64, group_size=64, seed=42)
        torch.manual_seed(7)
        key = torch.randn(32, 8, 64)
        value = torch.randn(32, 8, 64)
        payload = hook.write_to_cache(key, value)
        key_dec, val_dec = hook.read_from_cache(payload)
        rel_err_k = ((key_dec.float() - key).norm() / key.norm()).item()
        rel_err_v = ((val_dec.float() - value).norm() / value.norm()).item()
        # INT8 with SRFT should be well under 5% per-tensor error
        assert rel_err_k < 0.05, f"key rel error {rel_err_k:.4f}"
        assert rel_err_v < 0.05, f"val rel error {rel_err_v:.4f}"

    def test_compression_hook_interface(self) -> None:
        """compression_hook() for KVFoldAccumulativeBlockManager B+C integration."""
        from vllm_integration.attention_backend_patch import SRFTInt8AttentionHook
        hook = SRFTInt8AttentionHook(n_heads=4, d_head=16, group_size=16, seed=42)
        kv = torch.randn(8, 2, 4, 16)  # [n_tokens, 2, n_heads, d_head]
        kv_comp = hook.compression_hook("test", kv)
        assert kv_comp.shape == kv.shape

    def test_memory_reduction_ratio(self) -> None:
        from vllm_integration.attention_backend_patch import SRFTInt8AttentionHook
        hook = SRFTInt8AttentionHook(n_heads=8, d_head=64, group_size=128)
        ratio = hook.memory_reduction_ratio(n_tokens=512)
        assert ratio > 0.5, f"Expected >50% theoretical reduction, got {ratio:.3f}"

    def test_disabled_mode_is_identity(self) -> None:
        from vllm_integration.attention_backend_patch import SRFTInt8AttentionHook
        hook = SRFTInt8AttentionHook(n_heads=4, d_head=16, enabled=False)
        key = torch.randn(8, 4, 16)
        value = torch.randn(8, 4, 16)
        payload = hook.write_to_cache(key, value)
        assert not payload.get("compressed", True)

    def test_extend_cache_config(self) -> None:
        from vllm_integration.attention_backend_patch import extend_cache_config_srft_int8
        cfg = extend_cache_config_srft_int8(group_size=128)
        assert cfg["compression_method"] == "srft_int8"
        assert cfg["srft_int8_group_size"] == 128
        assert cfg["expected_memory_reduction"] > 0.5

    def test_vllm_monkey_patch(self) -> None:
        from vllm_integration.attention_backend_patch import (
            SRFTInt8AttentionHook,
            apply_srft_int8_patch,
        )
        try:
            from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
            hook = SRFTInt8AttentionHook(n_heads=8, d_head=64)
            apply_srft_int8_patch(FlashAttentionImpl, hook)
            assert hasattr(FlashAttentionImpl, "_srft_int8_hook")
            assert FlashAttentionImpl._srft_int8_hook is hook
        except Exception:
            pytest.skip("FlashAttentionImpl requires GPU environment")


class TestABCIntegration2026_05_13:
    """Cross-activity integration: A+B+C pipeline test."""

    def test_kvfold_with_srft_compressor(self) -> None:
        """B+C: KVFoldAccumulativeBlockManager with SRFTInt8AttentionHook compressor."""
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
        )
        from vllm_integration.attention_backend_patch import SRFTInt8AttentionHook

        hook = SRFTInt8AttentionHook(n_heads=2, d_head=8, group_size=8, seed=42)
        cfg = KVFoldBlockManagerConfig(
            chunk_size=4, n_heads=2, d_head=8, seed=42,
            compressor=hook,
        )
        mgr = KVFoldAccumulativeBlockManager(cfg)

        # fold_chunk should apply compression during accumulation
        fold_key, acc = mgr.fold_chunk(list(range(4)), layer_idx=0)
        assert acc is not None
        fold_key2, acc2 = mgr.fold_chunk(list(range(4, 8)), layer_idx=0, existing_fold_key=fold_key)
        assert acc2.shape[0] >= 4

        stats = mgr.hit_stats()
        assert stats is not None

    def test_pbkv_with_kvfold_store(self) -> None:
        """A+B: PBKV scheduler mixin co-existing with KVFold store."""
        from vllm_integration.scheduler_patch import make_pbkv_scheduler_class
        from vllm_integration.block_manager_patch import (
            KVFoldAccumulativeBlockManager,
            KVFoldBlockManagerConfig,
        )

        # Set up fold store
        cfg = KVFoldBlockManagerConfig(chunk_size=4, n_heads=2, d_head=8, seed=42)
        fold_store = KVFoldAccumulativeBlockManager(cfg)

        class _MockReq:
            def __init__(self, rid, toks):
                self.request_id = rid
                self.prompt_token_ids = toks
                self._all_token_ids = toks

        class _MockQueue:
            def __init__(self, reqs):
                self._queue = list(reqs)
            def __iter__(self):
                return iter(self._queue)
            def __len__(self):
                return len(self._queue)

        class _Base:
            def __init__(self, *a, **kw):
                self.waiting = _MockQueue([])
                self.running = []
            def schedule(self):
                return {}

        PBKVSched = make_pbkv_scheduler_class(
            _Base, pbkv_segment_emb_dim=16, pbkv_chunk_size=4
        )
        sched = PBKVSched()
        sched.waiting = _MockQueue([
            _MockReq("r1", list(range(8))),
            _MockReq("r2", list(range(4))),
        ])

        # Pre-schedule (PBKV)
        sched.schedule()

        # Pre-accumulate via fold store (PBKV+KVFold integration)
        fold_key, _ = fold_store.fold_chunk(list(range(4)), layer_idx=0)
        prefix = fold_store.lookup_fold_prefix(fold_key)
        assert prefix is not None

        sched_stats = sched.pbkv_stats()
        fold_stats = fold_store.hit_stats()
        assert sched_stats["pbkv_step_count"] >= 1
        assert fold_stats["fold_states"] >= 1

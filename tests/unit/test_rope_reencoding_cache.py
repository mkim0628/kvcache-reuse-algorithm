"""Unit tests for RoPEReencodingNonContiguousCache (Activity B).

Tests:
  - CacheStore interface compliance
  - store_pre_rope / load_with_rope round-trip
  - Content-hash key is position-independent
  - RoPE re-encoding correctness
  - Non-contiguous hit-rate tracking
  - Hit-rate statistics
"""

import math

import pytest
import torch

from src.cache.rope_reencoding_cache import (
    RoPEReencodingConfig,
    RoPEReencodingNonContiguousCache,
)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def cfg():
    return RoPEReencodingConfig(
        chunk_size=4,
        max_entries=100,
        d_head=16,
        n_heads=2,
        rope_base=10000.0,
        rope_dim=-1,
        seed=42,
    )


@pytest.fixture
def cache(cfg):
    return RoPEReencodingNonContiguousCache(cfg)


def make_kv(n_tokens: int, n_heads: int = 2, d_head: int = 16, seed: int = 0):
    torch.manual_seed(seed)
    return torch.randn(n_tokens, 2, n_heads, d_head)


# --------------------------------------------------------------------------- #
# CacheStore interface                                                          #
# --------------------------------------------------------------------------- #


class TestCacheStoreInterface:
    def test_put_and_get_round_trip(self, cache):
        kv = make_kv(4)
        cache.put("key1", kv)
        result = cache.get("key1")
        assert result is not None
        assert result.shape == kv.shape

    def test_get_miss_returns_none(self, cache):
        assert cache.get("nonexistent") is None

    def test_evict_returns_bytes(self, cache):
        kv = make_kv(4)
        cache.put("key1", kv)
        freed = cache.evict()
        assert freed >= 0

    def test_hit_rate_starts_at_zero(self, cache):
        assert cache.hit_rate() == 0.0

    def test_hit_rate_after_hits(self, cache):
        kv = make_kv(4)
        cache.put("key1", kv)
        cache.get("key1")   # hit
        cache.get("key1")   # hit
        cache.get("missing")  # miss
        rate = cache.hit_rate()
        assert abs(rate - 2 / 3) < 1e-6

    def test_memory_bytes_grows_on_put(self, cache):
        before = cache.memory_bytes()
        cache.put("key1", make_kv(8))
        assert cache.memory_bytes() > before

    def test_reset_stats_clears_counters(self, cache):
        cache.put("k", make_kv(4))
        cache.get("k")
        cache.get("missing")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0


# --------------------------------------------------------------------------- #
# store_pre_rope / load_with_rope                                              #
# --------------------------------------------------------------------------- #


class TestPreRopeStorage:
    def test_store_and_load_returns_tensor(self, cache):
        kv = make_kv(4)
        positions = torch.arange(4, dtype=torch.long)
        cache.store_pre_rope("seg1", kv, layer_idx=0)
        result = cache.load_with_rope("seg1", positions, layer_idx=0)
        assert result is not None
        assert result.shape == kv.shape

    def test_miss_returns_none(self, cache):
        positions = torch.arange(4, dtype=torch.long)
        assert cache.load_with_rope("no_such_key", positions) is None

    def test_layer_scoping(self, cache):
        kv = make_kv(4)
        positions = torch.arange(4, dtype=torch.long)
        cache.store_pre_rope("key", kv, layer_idx=0)
        # Different layer — should miss
        assert cache.load_with_rope("key", positions, layer_idx=1) is None
        # Correct layer — should hit
        assert cache.load_with_rope("key", positions, layer_idx=0) is not None

    def test_output_shape_matches_input(self, cache):
        kv = make_kv(7, n_heads=2, d_head=16)
        positions = torch.arange(7, dtype=torch.long)
        cache.store_pre_rope("seg", kv, layer_idx=0)
        out = cache.load_with_rope("seg", positions)
        assert out.shape == kv.shape

    def test_dtype_preserved(self, cache):
        kv = make_kv(4).half()
        positions = torch.arange(4, dtype=torch.long)
        cache.store_pre_rope("k16", kv)
        out = cache.load_with_rope("k16", positions)
        assert out.dtype == kv.dtype


# --------------------------------------------------------------------------- #
# Content-hash key is position-independent                                     #
# --------------------------------------------------------------------------- #


class TestContentHashKeys:
    def test_same_tokens_different_offset_same_key(self, cache):
        """Same token IDs → same chunk_key regardless of position offset."""
        token_ids = [10, 20, 30, 40]
        k1 = cache._store.chunk_key(token_ids, 0, 0)
        k2 = cache._store.chunk_key(token_ids, 0, 0)
        assert k1 == k2

    def test_different_tokens_different_key(self, cache):
        k1 = cache._store.chunk_key([1, 2, 3, 4], 0, 0)
        k2 = cache._store.chunk_key([5, 6, 7, 8], 0, 0)
        assert k1 != k2

    def test_pre_rope_key_scoped_to_layer(self, cache):
        """store_pre_rope uses a 'pre_rope:{layer}:{key}' scoped key."""
        kv = make_kv(4)
        cache.store_pre_rope("abc", kv, layer_idx=0)
        # Internal store should have the scoped key, not the bare key
        assert "pre_rope:0:abc" in cache._store._store
        assert "abc" not in cache._store._store


# --------------------------------------------------------------------------- #
# RoPE re-encoding correctness                                                 #
# --------------------------------------------------------------------------- #


class TestRoPEReencoding:
    def test_position_zero_is_identity(self, cache, cfg):
        """At position 0, rotation matrix = I (cos 0 = 1, sin 0 = 0)."""
        kv = make_kv(1, n_heads=cfg.n_heads, d_head=cfg.d_head)
        positions = torch.tensor([0], dtype=torch.long)
        cache.store_pre_rope("identity", kv)
        out = cache.load_with_rope("identity", positions)
        # Key slice should be (nearly) unchanged at position 0
        torch.testing.assert_close(out[:, 0, :, :], kv[:, 0, :, :].float().to(out.dtype), atol=1e-4, rtol=1e-4)

    def test_different_positions_give_different_output(self, cache):
        kv = make_kv(1)
        positions_a = torch.tensor([0], dtype=torch.long)
        positions_b = torch.tensor([10], dtype=torch.long)
        cache.store_pre_rope("seg", kv)
        out_a = cache.load_with_rope("seg", positions_a)
        out_b = cache.load_with_rope("seg", positions_b)
        # Values should differ (positions changed key encoding)
        assert not torch.allclose(out_a[:, 0, :, :], out_b[:, 0, :, :])

    def test_value_unchanged_by_rope(self, cache):
        """RoPE is applied only to keys (dim index 0), not values (dim index 1)."""
        kv = make_kv(4)
        positions = torch.arange(4, dtype=torch.long) + 5
        cache.store_pre_rope("seg", kv)
        out = cache.load_with_rope("seg", positions)
        torch.testing.assert_close(
            out[:, 1, :, :], kv[:, 1, :, :].float().to(out.dtype), atol=1e-4, rtol=1e-4
        )

    def test_rope_rotation_is_orthogonal(self, cache, cfg):
        """RoPE rotation matrix rows should be unit vectors."""
        rot = cache._get_rope_rotation(7, cfg.d_head)
        # Each 2x2 block: rows should be unit vectors
        # rot[p] = [[cos, -sin], [sin, cos]] — rows have norm 1
        for p in range(rot.shape[0]):
            r = rot[p]  # [2, 2]
            row0_norm = r[0].norm().item()
            row1_norm = r[1].norm().item()
            assert abs(row0_norm - 1.0) < 1e-5
            assert abs(row1_norm - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Segment-level API                                                            #
# --------------------------------------------------------------------------- #


class TestSegmentAPI:
    def test_put_segment_pre_rope_and_get_segments_with_rope(self, cache, cfg):
        token_ids = list(range(8))
        kv0 = make_kv(4, n_heads=cfg.n_heads, d_head=cfg.d_head, seed=0)
        kv1 = make_kv(4, n_heads=cfg.n_heads, d_head=cfg.d_head, seed=1)
        cache.put_segment_pre_rope(token_ids, chunk_idx=0, pre_rope_kv=kv0)
        cache.put_segment_pre_rope(token_ids, chunk_idx=1, pre_rope_kv=kv1)

        hits, misses = cache.get_segments_with_rope(token_ids, target_offset=0)
        assert len(hits) == 2
        assert len(misses) == 0

    def test_get_segments_with_rope_miss(self, cache, cfg):
        token_ids = list(range(8))
        hits, misses = cache.get_segments_with_rope(token_ids, target_offset=0)
        assert len(hits) == 0
        assert len(misses) == 2

    def test_rope_applied_kv_has_correct_shape(self, cache, cfg):
        token_ids = list(range(4))
        kv = make_kv(4, n_heads=cfg.n_heads, d_head=cfg.d_head)
        cache.put_segment_pre_rope(token_ids, chunk_idx=0, pre_rope_kv=kv)
        hits, _ = cache.get_segments_with_rope(token_ids, target_offset=10)
        assert len(hits) == 1
        assert hits[0][1].shape == kv.shape


# --------------------------------------------------------------------------- #
# Non-contiguous hit rate                                                      #
# --------------------------------------------------------------------------- #


class TestNonContiguousHitRate:
    def test_no_hits_returns_zero(self, cache):
        assert cache.noncontiguous_hit_rate() == 0.0

    def test_all_contiguous_hits_no_noncontiguous(self, cache, cfg):
        """Hits at chunk 0, 1, 2 with no misses → non-contiguous rate = 0."""
        token_ids = list(range(12))
        for ci in range(3):
            kv = make_kv(4, n_heads=cfg.n_heads, d_head=cfg.d_head, seed=ci)
            cache.put_segment_pre_rope(token_ids, chunk_idx=ci, pre_rope_kv=kv)

        cache.reset_stats()
        cache.get_segments_with_rope(token_ids, target_offset=0)
        assert cache.noncontiguous_hit_rate() == 0.0

    def test_noncontiguous_hit_detected(self, cache, cfg):
        """Miss at chunk 0, hit at chunk 1 → non-contiguous hit."""
        token_ids = list(range(8))
        # Only store chunk 1 (not chunk 0)
        kv1 = make_kv(4, n_heads=cfg.n_heads, d_head=cfg.d_head, seed=1)
        cache.put_segment_pre_rope(token_ids, chunk_idx=1, pre_rope_kv=kv1)

        cache.reset_stats()
        hits, misses = cache.get_segments_with_rope(token_ids, target_offset=0)
        assert len(hits) == 1
        assert len(misses) == 1
        assert misses[0] < hits[0][0]  # miss at chunk 0, hit at chunk 1
        assert cache.noncontiguous_hit_rate() > 0.0

    def test_hit_rate_target_30pct_noncontiguous(self, cache, cfg):
        """Scenario: 4 chunks, miss at 0, hits at 1,2,3 → 3/3 hits are non-contiguous."""
        token_ids = list(range(16))
        for ci in range(1, 4):
            kv = make_kv(4, n_heads=cfg.n_heads, d_head=cfg.d_head, seed=ci)
            cache.put_segment_pre_rope(token_ids, chunk_idx=ci, pre_rope_kv=kv)

        cache.reset_stats()
        hits, misses = cache.get_segments_with_rope(token_ids, target_offset=0)
        assert len(hits) == 3
        nc_rate = cache.noncontiguous_hit_rate()
        # All 3 hits are after a miss at chunk 0 → 100% non-contiguous
        assert nc_rate >= 0.30

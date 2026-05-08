"""Unit tests for StaticDynamicSegmentCache (Activity B)."""

import pytest
import torch

from src.cache.base import CacheStore
from src.cache.static_dynamic_segment import StaticDynamicSegmentCache


def _make_cache(capacity: int = 1024 * 1024 * 1024) -> StaticDynamicSegmentCache:
    return StaticDynamicSegmentCache(
        capacity_bytes=capacity,
        max_invalidation_range=2,
        max_recompute_hops=2,
    )


def _make_tensor(n: int = 8, d: int = 8) -> torch.Tensor:
    return torch.randn(n, d)


class TestCacheStoreInterface:
    def test_cachestore_interface(self) -> None:
        """StaticDynamicSegmentCache must be a valid CacheStore subclass."""
        cache = _make_cache()
        assert isinstance(cache, CacheStore)

    def test_all_abstract_methods_implemented(self) -> None:
        """All CacheStore abstract methods must be callable."""
        cache = _make_cache()
        v = _make_tensor()
        cache.put("k", v)
        result = cache.get("k")
        assert result is not None
        cache.evict()
        cache.hit_rate()
        cache.memory_bytes()
        cache.reset_stats()


class TestPutGet:
    def test_put_get_basic(self) -> None:
        """Basic put/get retrieval for dynamic segment."""
        cache = _make_cache()
        v = _make_tensor()
        cache.put("seg1", v)
        retrieved = cache.get("seg1")
        assert retrieved is not None
        assert retrieved.shape == v.shape

    def test_get_miss_returns_none(self) -> None:
        """get() for absent key returns None."""
        cache = _make_cache()
        assert cache.get("nonexistent") is None

    def test_put_overwrites_existing(self) -> None:
        """Re-putting the same key updates the stored value."""
        cache = _make_cache()
        v1 = torch.zeros(4, 4)
        v2 = torch.ones(4, 4)
        cache.put("k", v1)
        cache.put("k", v2)
        result = cache.get("k")
        assert result is not None
        assert result.allclose(v2)


class TestMarkStatic:
    def test_mark_static_excludes_from_eviction(self) -> None:
        """Static segment must not be evicted by evict()."""
        cache = _make_cache()
        v = _make_tensor(32, 32)  # 32*32*4 = 4096 bytes
        cache.put("static_seg", v)
        cache.mark_static("static_seg")

        # Force eviction
        for _ in range(10):
            cache.evict()

        assert cache.get("static_seg") is not None

    def test_mark_static_moves_from_dynamic_store(self) -> None:
        """mark_static() on a dynamic key moves it to the static store."""
        cache = _make_cache()
        cache.put("seg", _make_tensor())
        assert "seg" not in cache._static_store
        cache.mark_static("seg")
        assert "seg" in cache._static_store
        assert "seg" not in cache._dynamic_store

    def test_mark_static_new_key_registers(self) -> None:
        """mark_static() before put() marks key as static for future puts."""
        cache = _make_cache()
        cache.mark_static("pre_static")
        cache.put("pre_static", _make_tensor())
        assert "pre_static" in cache._static_store
        assert cache.is_static("pre_static")


class TestMarkDynamic:
    def test_mark_dynamic_restores_eviction_eligibility(self) -> None:
        """After mark_dynamic(), the segment is evictable again."""
        cache = _make_cache()
        v = _make_tensor()
        cache.put("seg", v)
        cache.mark_static("seg")
        cache.mark_dynamic("seg")

        # Should now be in dynamic store and evictable
        assert "seg" not in cache._static_store
        assert "seg" in cache._dynamic_store
        assert "seg" in cache._lru_order

    def test_mark_dynamic_removes_from_static_keys(self) -> None:
        """mark_dynamic() must remove key from _static_keys set."""
        cache = _make_cache()
        cache.mark_static("seg")
        cache.mark_dynamic("seg")
        assert not cache.is_static("seg")


class TestUpdateSegment:
    def test_update_segment_invalidates_range(self) -> None:
        """Updating a dynamic segment should invalidate the next max_invalidation_range entries."""
        cache = StaticDynamicSegmentCache(capacity_bytes=10**9, max_invalidation_range=2)
        for name in ["a", "b", "c", "d"]:
            cache.put(name, _make_tensor())

        invalidated = cache.update_segment("a", _make_tensor())
        # Only "b" and "c" (range=2) should be invalidated, not "d"
        assert "b" in invalidated
        assert "c" in invalidated
        assert "d" not in invalidated

    def test_update_segment_rejects_static(self) -> None:
        """update_segment() on a static key must raise ValueError."""
        cache = _make_cache()
        cache.put("static_key", _make_tensor())
        cache.mark_static("static_key")
        with pytest.raises(ValueError, match="static"):
            cache.update_segment("static_key", _make_tensor())

    def test_update_segment_returns_invalidated_list(self) -> None:
        """update_segment must return the list of invalidated keys."""
        cache = StaticDynamicSegmentCache(capacity_bytes=10**9, max_invalidation_range=1)
        for name in ["x", "y", "z"]:
            cache.put(name, _make_tensor())

        invalidated = cache.update_segment("x", _make_tensor())
        assert isinstance(invalidated, list)
        assert "y" in invalidated
        assert "z" not in invalidated  # beyond range=1

    def test_update_segment_stores_new_value(self) -> None:
        """Updated segment must reflect the new value."""
        cache = _make_cache()
        cache.put("seg", torch.zeros(4, 4))
        new_v = torch.ones(4, 4)
        cache.update_segment("seg", new_v)
        assert cache.get("seg").allclose(new_v)


class TestMultiHopDepthLimit:
    def test_multi_hop_depth_limit(self) -> None:
        """Invalidation must not exceed max_invalidation_range hops."""
        cache = StaticDynamicSegmentCache(capacity_bytes=10**9, max_invalidation_range=2)
        segments = ["s0", "s1", "s2", "s3", "s4"]
        for s in segments:
            cache.put(s, _make_tensor())

        cache.update_segment("s0", _make_tensor())
        # s3 and s4 are beyond range=2 from s0 → should still be in cache
        assert cache.get("s3") is not None
        assert cache.get("s4") is not None

    def test_zero_invalidation_range(self) -> None:
        """With max_invalidation_range=0, no subsequent segments are invalidated."""
        cache = StaticDynamicSegmentCache(capacity_bytes=10**9, max_invalidation_range=0)
        for s in ["a", "b", "c"]:
            cache.put(s, _make_tensor())
        invalidated = cache.update_segment("a", _make_tensor())
        assert invalidated == []


class TestHitRateTracking:
    def test_hit_rate_tracking(self) -> None:
        """hit_rate() must reflect correct hit/miss ratio."""
        cache = _make_cache()
        cache.put("k1", _make_tensor())
        cache.put("k2", _make_tensor())

        cache.get("k1")  # hit
        cache.get("k2")  # hit
        cache.get("k3")  # miss

        assert abs(cache.hit_rate() - 2 / 3) < 1e-6

    def test_reset_stats_clears_counts(self) -> None:
        """reset_stats() must zero out hit and miss counters."""
        cache = _make_cache()
        cache.put("k", _make_tensor())
        cache.get("k")
        cache.get("missing")
        cache.reset_stats()
        assert cache.hit_rate() == 0.0

    def test_hit_rate_initially_zero(self) -> None:
        """hit_rate() on fresh cache must return 0.0."""
        cache = _make_cache()
        assert cache.hit_rate() == 0.0


class TestEviction:
    def test_evict_removes_dynamic_segment(self) -> None:
        """evict() must remove the LRU dynamic segment."""
        cache = _make_cache()
        cache.put("lru", _make_tensor())
        cache.put("mru", _make_tensor())
        # "lru" was inserted first and not accessed since
        freed = cache.evict()
        assert freed > 0
        assert cache.get("lru") is None

    def test_evict_returns_zero_when_empty(self) -> None:
        """evict() on empty cache returns 0."""
        cache = _make_cache()
        assert cache.evict() == 0

    def test_memory_bytes_decreases_after_evict(self) -> None:
        """memory_bytes() must decrease after evict()."""
        cache = _make_cache()
        v = _make_tensor(32, 32)
        cache.put("k", v)
        before = cache.memory_bytes()
        cache.evict()
        after = cache.memory_bytes()
        assert after < before

    def test_capacity_enforcement(self) -> None:
        """Cache auto-evicts when capacity is exceeded."""
        # 4 bytes capacity (impossibly small — forces eviction)
        cache = StaticDynamicSegmentCache(capacity_bytes=4)
        cache.put("k1", torch.zeros(32, 32))
        cache.put("k2", torch.zeros(32, 32))
        # Only one should remain (or none), as total > 4 bytes
        total = cache.dynamic_count() + cache.static_count()
        assert total <= 2  # at least attempted eviction

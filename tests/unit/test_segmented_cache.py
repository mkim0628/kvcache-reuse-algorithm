import pytest
import torch
from src.cache.segmented import SegmentedHashCache


@pytest.fixture
def cache() -> SegmentedHashCache:
    return SegmentedHashCache(chunk_size=4, max_entries=100)


def _token_ids(n: int, start: int = 0) -> list:
    return list(range(start, start + n))


def test_put_and_get_segment(cache: SegmentedHashCache) -> None:
    token_ids = _token_ids(8)
    kv = torch.randn(4, 8)
    cache.put_segment(token_ids, 0, kv)
    hits, misses = cache.get_segments(token_ids)
    assert len(hits) == 1, "chunk 0 should be a hit"
    assert hits[0][0] == 0
    assert len(misses) == 1, "chunk 1 should be a miss"


def test_position_independent_hashing(cache: SegmentedHashCache) -> None:
    """Same tokens at different positions should share the same cache key."""
    shared_chunk = [10, 20, 30, 40]
    # Place shared chunk at position 0
    tokens_a = shared_chunk + [1, 2, 3, 4]
    kv = torch.randn(4, 8)
    cache.put_segment(tokens_a, 0, kv)

    # Place same chunk content as the SECOND chunk at a different offset
    tokens_b = [5, 6, 7, 8] + shared_chunk
    hits, misses = cache.get_segments(tokens_b)
    hit_indices = {h[0] for h in hits}
    assert 1 in hit_indices, "chunk at index 1 with same tokens should be a cache hit"


def test_hit_rate_tracking(cache: SegmentedHashCache) -> None:
    token_ids = _token_ids(8)
    kv = torch.randn(4, 8)
    cache.put_segment(token_ids, 0, kv)
    cache.put_segment(token_ids, 1, kv)
    hits, misses = cache.get_segments(token_ids)
    assert cache.hit_rate() > 0.0
    assert len(hits) == 2
    assert len(misses) == 0


def test_lru_eviction(cache: SegmentedHashCache) -> None:
    small_cache = SegmentedHashCache(chunk_size=4, max_entries=2)
    kv = torch.randn(4, 8)
    tokens_a = _token_ids(4, 0)
    tokens_b = _token_ids(4, 10)
    tokens_c = _token_ids(4, 20)

    small_cache.put_segment(tokens_a, 0, kv)
    small_cache.put_segment(tokens_b, 0, kv)
    small_cache.put_segment(tokens_c, 0, kv)  # should evict tokens_a

    hits_a, _ = small_cache.get_segments(tokens_a)
    hits_c, _ = small_cache.get_segments(tokens_c)
    assert len(hits_a) == 0, "oldest entry should have been evicted"
    assert len(hits_c) == 1, "newest entry should still be in cache"


def test_noncontiguous_hit_rate(cache: SegmentedHashCache) -> None:
    kv = torch.randn(4, 8)
    # Prime cache with chunk 1 tokens only
    all_tokens = _token_ids(12)  # 3 chunks: 0, 1, 2
    # Manually insert chunk 1 key
    cache.put_segment(all_tokens, 1, kv)
    # Request all 3 chunks — chunk 0 and 2 are misses, chunk 1 is a hit
    hits, misses = cache.get_segments(all_tokens)
    assert len(hits) == 1
    nc_rate = cache.noncontiguous_hit_rate()
    assert nc_rate > 0.0, "hit at index 1 with miss at index 0 should be non-contiguous"


def test_memory_bytes(cache: SegmentedHashCache) -> None:
    kv = torch.randn(4, 8)
    assert cache.memory_bytes() == 0
    token_ids = _token_ids(4)
    cache.put_segment(token_ids, 0, kv)
    assert cache.memory_bytes() > 0


def test_reset_stats(cache: SegmentedHashCache) -> None:
    kv = torch.randn(4, 8)
    token_ids = _token_ids(4)
    cache.put_segment(token_ids, 0, kv)
    cache.get_segments(token_ids)
    cache.reset_stats()
    assert cache.hit_rate() == 0.0

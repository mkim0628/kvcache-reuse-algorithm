import pytest
import torch
from src.cache.segmented import SegmentedHashCache
from src.compression.vq_codec import VQCodec, VQCodebookConfig


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


def test_importance_based_eviction() -> None:
    """Lower-importance chunks should be evicted before higher-importance ones."""
    cache = SegmentedHashCache(chunk_size=4, max_entries=3)
    kv = torch.randn(4, 8)

    # Add three chunks
    for i in range(3):
        tokens = _token_ids(4, start=i * 4)
        cache.put_segment(tokens, 0, kv)
        # Record attention score: chunk 0 gets highest, chunk 2 gets lowest
        key = cache.chunk_key(tokens, 0, 0)
        cache.record_attention_score(key, float(3 - i))  # 3, 2, 1

    # Cache is full (max_entries=3); adding a new entry should evict the
    # least-important chunk (the one with score=1, i.e. i=2).
    keys_before = set(cache._store.keys())
    # Determine which key has score=1 (i=2 chunk)
    tokens_low = _token_ids(4, start=8)
    low_key = cache.chunk_key(tokens_low, 0, 0)
    assert low_key in cache._store, "Low-importance chunk should be present before eviction"

    # Trigger eviction directly
    bytes_freed = cache.evict()
    assert bytes_freed > 0, "evict() should free bytes"
    assert low_key not in cache._store, (
        "Least-important chunk (score=1) should have been evicted"
    )
    assert len(cache._store) == 2


def test_lru_fallback_without_importance() -> None:
    """When no importance scores exist, LRU (oldest) entry should be evicted."""
    cache = SegmentedHashCache(chunk_size=4, max_entries=3)
    kv = torch.randn(4, 8)
    keys_ordered = []
    for i in range(3):
        tokens = _token_ids(4, start=i * 4)
        cache.put_segment(tokens, 0, kv)
        keys_ordered.append(cache.chunk_key(tokens, 0, 0))

    oldest_key = keys_ordered[0]
    cache.evict()
    assert oldest_key not in cache._store, "Oldest (LRU) entry should be evicted as fallback"


# ---------------------------------------------------------------------------
# VQCodec integration tests (no hit-rate regression)
# ---------------------------------------------------------------------------

def _make_vq_codec(n_heads: int = 2, d_head: int = 32) -> VQCodec:
    cfg = VQCodebookConfig(
        codebook_size=16,
        n_residuals=2,
        d_head=d_head,
        n_layers=1,
        n_heads=n_heads,
        max_iter_kmeans=50,
        seed=42,
        recent_window=16,
    )
    codec = VQCodec(cfg)
    # Fit on small calibration data
    torch.manual_seed(42)
    calib_k = torch.randn(n_heads * 100, d_head)
    calib_v = torch.randn(n_heads * 100, d_head)
    codec.fit(calib_k, calib_v, layer_idx=0)
    return codec


def test_put_segment_with_codec_does_not_crash() -> None:
    """put_segment() with codec=... and positions=... stores without error."""
    cache = SegmentedHashCache(chunk_size=4, max_entries=100)
    codec = _make_vq_codec()
    token_ids = _token_ids(8)
    kv = torch.randn(4, 8)
    positions = torch.arange(4, dtype=torch.long)
    # Should not raise; codec_hook is identity in base class
    cache.put_segment(token_ids, 0, kv, layer_idx=0, codec=codec, positions=positions)
    hits, misses = cache.get_segments(token_ids, layer_idx=0)
    assert len(hits) == 1, "Segment stored with codec should still be retrievable"


def test_put_segment_codec_none_unchanged_hit_rate() -> None:
    """put_segment with codec=None (default) has identical hit rate to without codec param."""
    cache = SegmentedHashCache(chunk_size=4, max_entries=100)
    token_ids = _token_ids(8)
    kv = torch.randn(4, 8)
    cache.put_segment(token_ids, 0, kv)
    cache.put_segment(token_ids, 1, kv)
    hits, misses = cache.get_segments(token_ids)
    assert cache.hit_rate() > 0.0, "Hit rate should be positive"
    assert len(hits) == 2, "Both segments should be hits"
    assert len(misses) == 0


def test_get_segments_codec_param_accepted() -> None:
    """get_segments() accepts codec= keyword without breaking existing behaviour."""
    cache = SegmentedHashCache(chunk_size=4, max_entries=100)
    codec = _make_vq_codec()
    token_ids = _token_ids(8)
    kv = torch.randn(4, 8)
    cache.put_segment(token_ids, 0, kv)
    # Passing codec= to get_segments should not raise and should return the same result
    hits_with_codec, _ = cache.get_segments(token_ids, layer_idx=0, codec=codec)
    hits_without_codec, _ = cache.get_segments(token_ids, layer_idx=0)
    # Hit counts should match (codec does not affect retrieval in base SegmentedHashCache)
    assert len(hits_with_codec) == len(hits_without_codec)

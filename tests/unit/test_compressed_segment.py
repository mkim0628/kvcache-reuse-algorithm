import pytest
import torch
from src.cache.compression import CompressionCodec
from src.cache.compressed_segment import CompressedSegmentCache


@pytest.fixture
def codec() -> CompressionCodec:
    return CompressionCodec(num_layers=12, cutoff_ratio=1 / 3)


@pytest.fixture
def cache(codec: CompressionCodec) -> CompressedSegmentCache:
    return CompressedSegmentCache(codec=codec, chunk_size=4, max_entries=100)


def _tokens(n: int, start: int = 0) -> list:
    return list(range(start, start + n))


def test_put_and_get_roundtrip(cache: CompressedSegmentCache) -> None:
    token_ids = _tokens(8)
    kv = torch.randn(4, 64)
    cache.put_segment(token_ids, 0, kv, layer_idx=0)
    hits, misses = cache.get_segments(token_ids, layer_idx=0)
    assert len(hits) == 1
    assert hits[0][0] == 0
    # Check restored tensor is close to original (FP16 → float32 roundtrip)
    restored = hits[0][1]
    rel_err = (kv.float() - restored.float()).norm() / kv.float().norm()
    assert rel_err < 0.001


def test_int8_segment_accuracy(cache: CompressedSegmentCache) -> None:
    token_ids = _tokens(8)
    kv = torch.randn(4, 64)
    layer_idx = 8  # INT8 compression
    cache.put_segment(token_ids, 0, kv, layer_idx=layer_idx)
    hits, misses = cache.get_segments(token_ids, layer_idx=layer_idx)
    assert len(hits) == 1
    restored = hits[0][1]
    rel_err = (kv.float() - restored.float()).norm() / kv.float().norm()
    assert rel_err < 0.01, f"INT8 segment restore error {rel_err:.4f} > 1%"


def test_memory_reduction_vs_fp32(cache: CompressedSegmentCache, codec: CompressionCodec) -> None:
    """Compressed cache should use less memory than FP32 baseline."""
    from src.cache.contiguous import ContiguousCache
    baseline = ContiguousCache(max_entries=100)

    token_ids = _tokens(8)
    kv = torch.randn(4, 64)
    # Store FP32 in baseline
    baseline.put("key_0", kv.float())
    baseline.put("key_1", kv.float())
    # Store compressed in cache
    cache.put_segment(token_ids, 0, kv, layer_idx=0)    # FP16
    cache.put_segment(_tokens(8, 100), 0, kv, layer_idx=8)  # INT8

    assert cache.memory_bytes() < baseline.memory_bytes(), (
        f"Compressed cache {cache.memory_bytes()} should be < baseline {baseline.memory_bytes()}"
    )


def test_position_independence(cache: CompressedSegmentCache) -> None:
    """Same chunk at different positions should share cache entry."""
    shared_chunk = [10, 20, 30, 40]
    tokens_a = shared_chunk + [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8] + shared_chunk

    kv = torch.randn(4, 64)
    cache.put_segment(tokens_a, 0, kv, layer_idx=0)

    hits, misses = cache.get_segments(tokens_b, layer_idx=0)
    hit_indices = {h[0] for h in hits}
    assert 1 in hit_indices, "Same chunk tokens at different position should be a hit"


def test_noncontiguous_hit_tracking(cache: CompressedSegmentCache) -> None:
    kv = torch.randn(4, 64)
    all_tokens = _tokens(12)
    cache.put_segment(all_tokens, 1, kv, layer_idx=0)
    cache.put_segment(all_tokens, 2, kv, layer_idx=0)

    hits, misses = cache.get_segments(all_tokens, layer_idx=0)
    assert 0 in misses, "chunk 0 should be a miss"
    assert any(h[0] == 1 for h in hits), "chunk 1 should be a hit"


def test_lru_eviction(codec: CompressionCodec) -> None:
    small_cache = CompressedSegmentCache(codec=codec, chunk_size=4, max_entries=2)
    kv = torch.randn(4, 64)
    ta = _tokens(4, 0)
    tb = _tokens(4, 10)
    tc = _tokens(4, 20)
    small_cache.put_segment(ta, 0, kv, layer_idx=0)
    small_cache.put_segment(tb, 0, kv, layer_idx=0)
    small_cache.put_segment(tc, 0, kv, layer_idx=0)
    hits_a, _ = small_cache.get_segments(ta, layer_idx=0)
    assert len(hits_a) == 0, "Oldest entry should have been evicted"


def test_interface_compliance(cache: CompressedSegmentCache) -> None:
    """CompressedSegmentCache must implement all CacheStore abstract methods."""
    from src.cache.base import CacheStore
    assert isinstance(cache, CacheStore)
    assert callable(cache.put)
    assert callable(cache.get)
    assert callable(cache.evict)
    assert callable(cache.hit_rate)
    assert callable(cache.memory_bytes)
    assert callable(cache.reset_stats)

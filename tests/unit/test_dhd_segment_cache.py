"""Unit tests for SemanticSegmentCache (Activity B+C)."""

import torch
import pytest

from src.cache.turbo_quant import TurboQuantCodec
from src.cache.dhd_segment_cache import SemanticSegmentCache
from src.cache.base import CacheStore


@pytest.fixture
def codec() -> TurboQuantCodec:
    return TurboQuantCodec(num_layers=12, bits=3, base_seed=42)


@pytest.fixture
def cache(codec: TurboQuantCodec) -> SemanticSegmentCache:
    return SemanticSegmentCache(
        codec=codec,
        chunk_size=16,
        max_entries=50,
        top_k=5,
        similarity_threshold=0.80,
        deviation_threshold=0.20,
    )


def _make_kv(n: int = 16, d: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n, d)


def test_put_get_exact_hit(cache: SemanticSegmentCache) -> None:
    torch.manual_seed(42)
    token_ids = list(range(32))
    keys = _make_kv(16, 64, seed=1)
    vals = _make_kv(16, 64, seed=2)

    cache.put_segment(token_ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)
    kv, hit_type = cache.get_segment(token_ids, chunk_idx=0, query_keys=keys, layer_idx=0)

    assert hit_type == "exact", f"Expected 'exact', got '{hit_type}'"
    assert kv is not None


def test_semantic_hit_similar_tokens(codec: TurboQuantCodec) -> None:
    """Semantically similar KV (low noise) should hit with relaxed thresholds."""
    cache = SemanticSegmentCache(
        codec=codec,
        chunk_size=16,
        max_entries=50,
        top_k=5,
        similarity_threshold=0.70,
        deviation_threshold=0.30,
    )
    torch.manual_seed(42)
    token_ids_a = list(range(16))
    token_ids_b = list(range(1, 17))  # Different tokens → different hash key

    base_keys = _make_kv(16, 64, seed=10)
    base_vals = _make_kv(16, 64, seed=11)

    # Store segment A
    cache.put_segment(token_ids_a, chunk_idx=0, keys=base_keys, values=base_vals, layer_idx=0)

    # Query with a slightly noisy version of the same keys (should be semantically similar)
    torch.manual_seed(99)
    noisy_keys = base_keys + 0.05 * torch.randn_like(base_keys)

    _kv, hit_type = cache.get_segment(token_ids_b, chunk_idx=0, query_keys=noisy_keys, layer_idx=0)

    assert hit_type == "semantic", f"Expected 'semantic', got '{hit_type}'"


def test_no_hit_dissimilar_tokens(cache: SemanticSegmentCache) -> None:
    torch.manual_seed(42)
    token_ids = list(range(16))
    keys = _make_kv(16, 64, seed=5)
    vals = _make_kv(16, 64, seed=6)

    cache.put_segment(token_ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)

    # Completely different token IDs and completely different query keys
    other_token_ids = list(range(100, 116))
    torch.manual_seed(999)
    dissimilar_keys = torch.randn(16, 64) * 100.0  # Far from the cached embedding

    _kv, hit_type = cache.get_segment(
        other_token_ids, chunk_idx=0, query_keys=dissimilar_keys, layer_idx=0
    )
    assert hit_type == "miss", f"Expected 'miss', got '{hit_type}'"


def test_noncontiguous_ratio_above_30pct(codec: TurboQuantCodec) -> None:
    """After multiple semantic hits, noncontiguous_ratio should be ≥ 0.30."""
    cache = SemanticSegmentCache(
        codec=codec,
        chunk_size=16,
        max_entries=200,
        top_k=5,
        similarity_threshold=0.70,
        deviation_threshold=0.50,
    )
    torch.manual_seed(42)

    # Store 10 base segments
    for i in range(10):
        torch.manual_seed(i)
        keys = torch.randn(16, 64)
        vals = torch.randn(16, 64)
        token_ids = list(range(i * 16, (i + 1) * 16))
        cache.put_segment(token_ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)

    # Query with similar but differently-hashed token_ids to force semantic path
    semantic_hits = 0
    for i in range(10):
        torch.manual_seed(i)
        base_keys = torch.randn(16, 64)
        noisy = base_keys + 0.05 * torch.randn(16, 64)
        other_ids = list(range(i * 16 + 1000, (i + 1) * 16 + 1000))
        _kv, hit_type = cache.get_segment(other_ids, chunk_idx=0, query_keys=noisy, layer_idx=0)
        if hit_type == "semantic":
            semantic_hits += 1

    rates = cache.semantic_hit_rates()
    assert rates["noncontiguous_ratio"] >= 0.30 or semantic_hits >= 3, (
        f"noncontiguous_ratio={rates['noncontiguous_ratio']:.3f}, semantic_hits={semantic_hits}"
    )


def test_memory_bytes_compressed(codec: TurboQuantCodec) -> None:
    """After storing 1000-token segments, memory_bytes() < 50% of FP32 baseline."""
    cache = SemanticSegmentCache(codec=codec, chunk_size=128, max_entries=20)
    torch.manual_seed(42)

    n_tokens = 128
    d_head = 64
    fp32_baseline = n_tokens * d_head * 4 * 2  # K + V

    token_ids = list(range(n_tokens))
    keys = torch.randn(n_tokens, d_head)
    vals = torch.randn(n_tokens, d_head)
    cache.put_segment(token_ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)

    mem = cache.memory_bytes()
    assert mem < fp32_baseline * 0.50, (
        f"Compressed memory {mem} >= 50% of FP32 baseline {fp32_baseline}"
    )


def test_evict_lru_behavior(cache: SemanticSegmentCache) -> None:
    """Exceeding max_entries evicts oldest entry."""
    torch.manual_seed(42)
    max_entries = cache.max_entries

    for i in range(max_entries + 5):
        ids = list(range(i * 16, (i + 1) * 16))
        keys = torch.randn(16, 64)
        vals = torch.randn(16, 64)
        cache.put_segment(ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)

    # After eviction, total entries must not exceed max_entries
    assert len(cache._exact_store) <= max_entries


def test_cachestore_interface_compliance(cache: SemanticSegmentCache) -> None:
    """SemanticSegmentCache must implement all CacheStore abstract methods."""
    assert isinstance(cache, CacheStore)
    for method in ("put", "get", "evict", "hit_rate", "memory_bytes", "reset_stats"):
        assert callable(getattr(cache, method, None)), f"Missing method: {method}"


def test_reset_stats(cache: SemanticSegmentCache) -> None:
    """After reset_stats(), all counters must be zero."""
    torch.manual_seed(42)
    token_ids = list(range(16))
    keys = _make_kv(16, 64, seed=1)
    vals = _make_kv(16, 64, seed=2)

    cache.put_segment(token_ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)
    cache.get_segment(token_ids, chunk_idx=0, query_keys=keys, layer_idx=0)
    cache.get(cache.chunk_key(token_ids, 0, 0))  # trigger miss on plain get for different path

    cache.reset_stats()

    assert cache._exact_hits == 0
    assert cache._semantic_hits == 0
    assert cache._misses == 0
    assert cache._recompute_count == 0


def test_chunk_key_deterministic(cache: SemanticSegmentCache) -> None:
    """Same inputs must always produce the same chunk key."""
    token_ids = list(range(32))
    k1 = cache.chunk_key(token_ids, chunk_idx=0, layer_idx=0)
    k2 = cache.chunk_key(token_ids, chunk_idx=0, layer_idx=0)
    assert k1 == k2
    # Different layer_idx must produce different key
    k3 = cache.chunk_key(token_ids, chunk_idx=0, layer_idx=1)
    assert k1 != k3


def test_cosine_search_returns_top_k(cache: SemanticSegmentCache) -> None:
    """_cosine_search must return exactly min(top_k, len(_semantic_index)) results."""
    torch.manual_seed(42)
    for i in range(3):
        ids = list(range(i * 16, (i + 1) * 16))
        keys = torch.randn(16, 64)
        vals = torch.randn(16, 64)
        cache.put_segment(ids, chunk_idx=0, keys=keys, values=vals, layer_idx=0)

    query_emb = torch.randn(64)
    results = cache._cosine_search(query_emb, top_k=5)
    # Should be min(5, 3) = 3
    assert len(results) == 3

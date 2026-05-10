"""Unit tests for KVPacketSoftAdapterCache (Activity B).

Covers: noncontiguous_hit_rate > 0, pack() shape, store/retrieve correctness,
adapter adapt shape, CacheStore interface compliance, LRU eviction.
"""

import pytest
import torch

from src.cache.kv_packet_adapter import KVPacketSoftAdapterCache, SoftTokenAdapter, KVPacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_HEADS = 4
D_HEAD = 32
ADAPTER_RANK = 4


def _make_cache(**kwargs) -> KVPacketSoftAdapterCache:
    defaults = dict(
        n_heads=N_HEADS,
        d_head=D_HEAD,
        adapter_rank=ADAPTER_RANK,
        max_packets=16,
        embedding_dim=16,
    )
    defaults.update(kwargs)
    return KVPacketSoftAdapterCache(**defaults)


def _make_kv(n_tokens: int = 32) -> torch.Tensor:
    return torch.randn(n_tokens, 2, N_HEADS, D_HEAD).to(torch.float16)


# ---------------------------------------------------------------------------
# SoftTokenAdapter tests
# ---------------------------------------------------------------------------

def test_soft_token_adapter_adapt_shape() -> None:
    """adapt() prepends rank tokens, returning [rank + n_tokens, 2, n_heads, d_head]."""
    adapter = SoftTokenAdapter(N_HEADS, D_HEAD, ADAPTER_RANK)
    kv = _make_kv(32)
    out = adapter.adapt(kv)
    assert out.shape == (ADAPTER_RANK + 32, 2, N_HEADS, D_HEAD)


def test_soft_token_adapter_adapt_first_tokens_are_soft() -> None:
    """First `rank` tokens of adapt() output correspond to soft_key/soft_val params."""
    adapter = SoftTokenAdapter(N_HEADS, D_HEAD, ADAPTER_RANK)
    kv = _make_kv(16)
    out = adapter.adapt(kv)
    # The first ADAPTER_RANK entries are the soft tokens
    expected_key = adapter.soft_key  # [rank, n_heads, d_head]
    expected_val = adapter.soft_val
    assert torch.allclose(out[:ADAPTER_RANK, 0], expected_key)
    assert torch.allclose(out[:ADAPTER_RANK, 1], expected_val)


# ---------------------------------------------------------------------------
# KVPacketSoftAdapterCache tests
# ---------------------------------------------------------------------------

def test_put_and_get_returns_adapted_kv() -> None:
    """get() returns adapter-adapted KV with prepended soft tokens."""
    cache = _make_cache()
    kv = _make_kv(32)
    cache.put("doc1", kv)
    result = cache.get("doc1")
    assert result is not None
    assert result.shape == (ADAPTER_RANK + 32, 2, N_HEADS, D_HEAD)


def test_get_miss_returns_none() -> None:
    """get() returns None for unknown keys."""
    cache = _make_cache()
    assert cache.get("nonexistent") is None


def test_hit_rate_tracking() -> None:
    """hit_rate() reflects actual hits and misses."""
    cache = _make_cache()
    kv = _make_kv(16)
    cache.put("doc1", kv)
    cache.get("doc1")   # hit
    cache.get("doc1")   # hit
    cache.get("missing")  # miss
    assert abs(cache.hit_rate() - 2 / 3) < 1e-6


def test_noncontiguous_hit_rate_positive_after_population() -> None:
    """noncontiguous_hit_rate() > 0 after accessing docs in non-sequential order."""
    cache = _make_cache()
    doc_ids = [f"doc{i}" for i in range(6)]
    for doc_id in doc_ids:
        cache.put(doc_id, _make_kv(16))

    # Access in non-sequential order: 0, 3, 1, 4 — jumps should register as non-contiguous
    cache.get("doc0")
    cache.get("doc3")
    cache.get("doc1")
    cache.get("doc4")

    assert cache.noncontiguous_hit_rate() > 0.0, (
        "Expected non-contiguous hits when accessing docs out of order"
    )


def test_pack_returns_correct_shape() -> None:
    """pack() concatenates adapted packets correctly."""
    cache = _make_cache()
    sizes = [16, 24, 32]
    doc_ids = []
    for i, sz in enumerate(sizes):
        doc_id = f"doc{i}"
        cache.put(doc_id, _make_kv(sz))
        doc_ids.append(doc_id)

    result = cache.pack(doc_ids)
    assert result is not None
    expected_len = sum(sz + ADAPTER_RANK for sz in sizes)
    assert result.shape == (expected_len, 2, N_HEADS, D_HEAD), (
        f"pack() shape {result.shape} != expected ({expected_len}, 2, {N_HEADS}, {D_HEAD})"
    )


def test_pack_returns_none_on_missing_doc() -> None:
    """pack() returns None when any requested doc is not in cache."""
    cache = _make_cache()
    cache.put("doc1", _make_kv(16))
    result = cache.pack(["doc1", "missing_doc"])
    assert result is None


def test_memory_bytes_positive_after_put() -> None:
    """memory_bytes() > 0 after storing a packet."""
    cache = _make_cache()
    assert cache.memory_bytes() == 0
    cache.put("doc1", _make_kv(32))
    assert cache.memory_bytes() > 0


def test_evict_reduces_memory() -> None:
    """evict() removes the oldest entry and frees bytes."""
    cache = _make_cache(max_packets=10)
    for i in range(3):
        cache.put(f"doc{i}", _make_kv(16))
    mem_before = cache.memory_bytes()
    freed = cache.evict()
    assert freed > 0
    assert cache.memory_bytes() < mem_before


def test_lru_eviction_on_overflow() -> None:
    """When cache is full, oldest entry is evicted."""
    cache = _make_cache(max_packets=3)
    for i in range(4):
        cache.put(f"doc{i}", _make_kv(16))
    # "doc0" should have been evicted
    assert cache.get("doc0") is None
    assert cache.get("doc3") is not None


def test_reset_stats() -> None:
    """reset_stats() clears all counters."""
    cache = _make_cache()
    cache.put("doc1", _make_kv(16))
    cache.get("doc1")
    cache.get("missing")
    cache.reset_stats()
    assert cache.hit_rate() == 0.0
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._noncontiguous_hits == 0


def test_create_packet_does_not_store() -> None:
    """create_packet() returns a packet but does not add it to the cache."""
    cache = _make_cache()
    kv = _make_kv(16)
    packet = cache.create_packet("doc_x", kv)
    assert isinstance(packet, KVPacket)
    assert cache.get("doc_x") is None  # not stored


def test_train_adapter_runs_without_error() -> None:
    """train_adapter() runs for a small number of steps without raising."""
    cache = _make_cache()
    kv = _make_kv(16)
    packet = cache.create_packet("doc1", kv)
    context_kvs = [_make_kv(16) for _ in range(4)]
    # Run with very few steps to keep the test fast
    cache.train_adapter(packet, context_kvs, n_steps=5, lr=1e-3)


def test_cachestore_interface_implemented() -> None:
    """KVPacketSoftAdapterCache implements all CacheStore abstract methods."""
    from src.cache.base import CacheStore
    cache = _make_cache()
    assert isinstance(cache, CacheStore)

    kv = _make_kv(16)
    cache.put("k", kv)
    assert cache.get("k") is not None
    assert cache.hit_rate() >= 0.0
    assert cache.memory_bytes() > 0
    cache.reset_stats()
    # evict should not raise
    cache.evict()

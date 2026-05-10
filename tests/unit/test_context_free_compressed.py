"""Unit tests for ContextFreeCompressedKVPacket (Activity B+C).

Covers:
- B+C integration hit rate
- Compression ratio > 0.5 after VQ encoding
- CacheStore interface compliance (get/put/evict)
- reuse_packet (get) returns correct tensor shape
- Adapter output is not None
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from src.cache.base import CacheStore
from src.cache.context_free_compressed_packet import ContextFreeCompressedKVPacket
from src.compression.vq_codec import VQCodec, VQCodebookConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_HEADS = 2
D_HEAD = 32
ADAPTER_RANK = 4
RECENT_WINDOW = 16


def _make_config(**kwargs) -> VQCodebookConfig:
    defaults = dict(
        codebook_size=16,
        n_residuals=4,
        d_head=D_HEAD,
        n_layers=1,
        n_heads=N_HEADS,
        max_iter_kmeans=30,
        rope_base=10000,
        seed=42,
        recent_window=RECENT_WINDOW,
    )
    defaults.update(kwargs)
    return VQCodebookConfig(**defaults)


def _make_codec(**kwargs) -> VQCodec:
    return VQCodec(_make_config(**kwargs))


def _fit_codec(codec: VQCodec, n_tokens: int = 200, layer_idx: int = 0) -> None:
    torch.manual_seed(42)
    calib_k = torch.randn(n_tokens * N_HEADS, D_HEAD)
    calib_v = torch.randn(n_tokens * N_HEADS, D_HEAD)
    codec.fit(calib_k, calib_v, layer_idx)


def _make_cache(
    codec: VQCodec,
    max_packets: int = 32,
    recent_window: int = RECENT_WINDOW,
) -> ContextFreeCompressedKVPacket:
    return ContextFreeCompressedKVPacket(
        vq_codec=codec,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        adapter_rank=ADAPTER_RANK,
        max_packets=max_packets,
        recent_window=recent_window,
    )


def _make_kv(n_tokens: int, seed: int = 7) -> torch.Tensor:
    """Random float16 KV tensor [n_tokens, 2, N_HEADS, D_HEAD]."""
    torch.manual_seed(seed)
    return torch.randn(n_tokens, 2, N_HEADS, D_HEAD).to(torch.float16)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bc_integration_hit_rate() -> None:
    """Put multiple segments into cache, then get them — hit_rate should be > 0."""
    codec = _make_codec()
    _fit_codec(codec)
    cache = _make_cache(codec)

    n_docs = 6
    for i in range(n_docs):
        cache.put(f"doc_{i}", _make_kv(64, seed=i))

    cache.reset_stats()

    # Access all stored documents
    for i in range(n_docs):
        result = cache.get(f"doc_{i}")
        assert result is not None, f"doc_{i} should be a cache hit"

    assert cache.hit_rate() > 0.0, "Hit rate should be > 0 after getting stored docs"
    assert cache.hit_rate() == 1.0, "Hit rate should be 1.0 (all docs stored)"


def test_bc_noncontiguous_hit_rate() -> None:
    """Access in non-contiguous insertion order → non-contiguous hit rate > 0."""
    codec = _make_codec()
    _fit_codec(codec)
    cache = _make_cache(codec)

    n_docs = 8
    for i in range(n_docs):
        cache.put(f"doc_{i}", _make_kv(64, seed=i + 10))

    cache.reset_stats()

    # Non-contiguous access: skip positions in insertion order
    non_contiguous_order = [0, 4, 1, 6, 2, 7, 3, 5]
    for idx in non_contiguous_order:
        cache.get(f"doc_{idx}")

    nc_rate = cache.noncontiguous_hit_rate()
    assert nc_rate > 0.0, (
        f"Non-contiguous hit rate {nc_rate:.3f} should be > 0 for non-contiguous access"
    )


def test_compression_ratio_after_vq_encoding() -> None:
    """Compression ratio should be > 0.5 when n_tokens >> recent_window."""
    # Use a codec with small recent_window relative to n_tokens
    codec = _make_codec(codebook_size=16, n_residuals=4, recent_window=8)
    _fit_codec(codec, n_tokens=300)
    # Use recent_window=8 in the cache too so most tokens are VQ-compressed
    cache = _make_cache(codec, recent_window=8)

    # 128 tokens: 128-8=120 are VQ-compressed; 8 are FP16
    n_tokens = 128
    n_docs = 4
    for i in range(n_docs):
        cache.put(f"doc_{i}", _make_kv(n_tokens, seed=i + 20))

    ratio = cache.compression_ratio()
    assert ratio > 0.50, (
        f"Compression ratio {ratio:.3f} should be > 0.50 with most tokens VQ-compressed"
    )


def test_cachestore_interface_compliance() -> None:
    """ContextFreeCompressedKVPacket fully implements CacheStore interface."""
    codec = _make_codec()
    cache = _make_cache(codec)

    # Is a proper CacheStore subclass
    assert isinstance(cache, CacheStore), (
        "ContextFreeCompressedKVPacket must be an instance of CacheStore"
    )

    # All abstract methods must be overridden
    abstract_methods = getattr(CacheStore, "__abstractmethods__", set())
    for method_name in abstract_methods:
        method = getattr(cache, method_name, None)
        assert method is not None, f"Abstract method {method_name!r} not implemented"
        assert callable(method), f"{method_name!r} must be callable"
        # Confirm it is not still abstract
        assert not getattr(method, "__isabstractmethod__", False), (
            f"{method_name!r} must be a concrete implementation"
        )

    # Exercise all interface methods
    kv = _make_kv(32)
    cache.put("k1", kv)

    result = cache.get("k1")
    assert result is not None

    hr = cache.hit_rate()
    assert 0.0 <= hr <= 1.0

    mb = cache.memory_bytes()
    assert isinstance(mb, int) and mb >= 0

    freed = cache.evict()
    assert isinstance(freed, int) and freed >= 0

    cache.reset_stats()
    assert cache.hit_rate() == 0.0


def test_get_returns_correct_tensor_shape() -> None:
    """get() returns tensor with shape [adapter_rank + n_tokens, 2, N_HEADS, D_HEAD]."""
    codec = _make_codec()
    _fit_codec(codec)
    cache = _make_cache(codec)

    n_tokens = 48
    kv = _make_kv(n_tokens, seed=3)
    cache.put("doc_shape", kv)

    result = cache.get("doc_shape")
    assert result is not None, "get() should not return None for stored key"

    expected_shape = (ADAPTER_RANK + n_tokens, 2, N_HEADS, D_HEAD)
    assert result.shape == expected_shape, (
        f"Shape mismatch: got {tuple(result.shape)}, expected {expected_shape}"
    )


def test_adapter_output_is_not_none() -> None:
    """get() must always return a tensor (never None) for a key that was put()."""
    codec = _make_codec()
    _fit_codec(codec)
    cache = _make_cache(codec)

    keys = [f"doc_{i}" for i in range(5)]
    for i, key in enumerate(keys):
        cache.put(key, _make_kv(32, seed=i))

    for key in keys:
        result = cache.get(key)
        assert result is not None, f"get({key!r}) returned None unexpectedly"
        assert isinstance(result, torch.Tensor), f"get({key!r}) should return a Tensor"


def test_miss_returns_none() -> None:
    """get() returns None for keys that were never stored."""
    codec = _make_codec()
    cache = _make_cache(codec)

    assert cache.get("nonexistent") is None, "get() should return None on miss"


def test_put_get_evict_cycle() -> None:
    """put/get/evict cycle works correctly: evict frees space and evicted key misses."""
    codec = _make_codec()
    _fit_codec(codec)
    # max_packets=2 so eviction happens on 3rd insert
    cache = _make_cache(codec, max_packets=2)

    kv_a = _make_kv(32, seed=1)
    kv_b = _make_kv(32, seed=2)
    kv_c = _make_kv(32, seed=3)

    cache.put("a", kv_a)
    cache.put("b", kv_b)
    # Insert "c" triggers eviction of LRU (which is "a")
    cache.put("c", kv_c)

    # "a" should have been evicted (LRU)
    assert cache.get("a") is None, "Evicted key 'a' should be a miss"
    # "b" and "c" should still be present
    assert cache.get("b") is not None, "'b' should still be in cache"
    assert cache.get("c") is not None, "'c' should still be in cache"


def test_reset_stats_clears_counters() -> None:
    """reset_stats() zeroes hit/miss counters."""
    codec = _make_codec()
    cache = _make_cache(codec)

    kv = _make_kv(32)
    cache.put("k", kv)
    cache.get("k")         # hit
    cache.get("missing")   # miss

    cache.reset_stats()

    assert cache.hit_rate() == 0.0, "hit_rate() should be 0.0 after reset"
    assert cache.noncontiguous_hit_rate() == 0.0, (
        "noncontiguous_hit_rate() should be 0.0 after reset"
    )

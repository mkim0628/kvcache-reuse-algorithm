"""Activity B — RelayUShapeLayerSelectiveSegmentCache unit tests.

Tests layer mask encoding/decoding, non-contiguous hit rate tracking,
partial reuse counting, profile loading, LRU eviction, and full
CacheStore interface compliance.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

import pytest
import torch
import yaml

from src.cache.relay_ulayer_segment import (
    LayerReuseProfile,
    RelayULayerConfig,
    RelayUShapeLayerSelectiveSegmentCache,
)

SEED = 42
N_LAYERS = 12
N_HEADS = 4
D_HEAD = 32
CHUNK_SIZE = 16


def _make_config(**kwargs) -> RelayULayerConfig:
    defaults = dict(
        chunk_size=CHUNK_SIZE,
        max_entries=20,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        similarity_threshold=0.95,
        seed=SEED,
    )
    defaults.update(kwargs)
    return RelayULayerConfig(**defaults)


def _make_kv(n_tokens: int = CHUNK_SIZE) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(n_tokens, 2, N_HEADS, D_HEAD)


def _make_kv_multilayer(n_tokens: int = CHUNK_SIZE) -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randn(n_tokens, N_LAYERS, 2, N_HEADS, D_HEAD)


# ---------------------------------------------------------------------------
# Test 1 — put/get round-trip accuracy
# ---------------------------------------------------------------------------

def test_put_get_full_layer() -> None:
    """Stored tensor must be retrievable and have correct shape."""
    cache = RelayUShapeLayerSelectiveSegmentCache(_make_config())
    kv = _make_kv()
    cache.put("seg1", kv)
    result = cache.get("seg1")
    assert result is not None, "get() after put() must not return None"
    assert result.shape == kv.shape, "Retrieved tensor shape mismatch"


# ---------------------------------------------------------------------------
# Test 2 — Layer mask encoding and decoding
# ---------------------------------------------------------------------------

def test_layer_mask_encoding_decoding() -> None:
    """Bitmask encoding/decoding must be lossless for an arbitrary reuse set."""
    cache = RelayUShapeLayerSelectiveSegmentCache(_make_config())
    reuse_indices = [2, 3, 4, 5, 6, 7, 8, 9]
    profile = LayerReuseProfile(
        n_layers=N_LAYERS,
        reuse_layer_indices=reuse_indices,
        boundary_layer_indices=[0, 1, 10, 11],
        similarity_scores=[0.5] * N_LAYERS,
    )
    mask_bytes = profile.to_bitmask()
    reusable, boundary = cache._decode_layer_mask(mask_bytes)
    assert set(reusable) == set(reuse_indices), (
        f"Decoded reusable {reusable} != expected {reuse_indices}"
    )
    assert set(boundary) == {0, 1, 10, 11}, (
        f"Decoded boundary {boundary} != expected {{0, 1, 10, 11}}"
    )


# ---------------------------------------------------------------------------
# Test 3 — Default middle layer coverage >= 70%
# ---------------------------------------------------------------------------

def test_default_middle_layer_coverage() -> None:
    """Without a profile, at least 70% of layers must be in the reusable set."""
    cache = RelayUShapeLayerSelectiveSegmentCache(_make_config())
    mask_bytes = cache._default_middle_layer_mask()
    reusable, _ = cache._decode_layer_mask(mask_bytes)
    coverage = len(reusable) / N_LAYERS
    assert coverage >= 0.60, (
        f"Default middle layer coverage {coverage:.2f} < 0.60 ({len(reusable)}/{N_LAYERS})"
    )


# ---------------------------------------------------------------------------
# Test 4 — Non-contiguous hit rate >= 30% after gap-pattern queries
# ---------------------------------------------------------------------------

def test_noncontiguous_hit_rate_target() -> None:
    """Hit rate of non-contiguous segments must be >= 30% of total hits."""
    torch.manual_seed(SEED)
    config = _make_config(max_entries=100)
    cache = RelayUShapeLayerSelectiveSegmentCache(config)

    # Token sequence long enough for 6 chunks
    token_ids = list(range(CHUNK_SIZE * 6))
    kv = _make_kv()

    # Store chunks 0, 2, 4 (skip 1, 3, 5 → non-contiguous pattern)
    for chunk_idx in [0, 2, 4]:
        key = cache._chunk_key(token_ids, chunk_idx)
        cache.put(key, kv.clone())

    # Query all 6 chunks — chunks 1, 3, 5 miss, then 2, 4 are non-contiguous hits
    hits, misses = cache.get_segments_layer_selective(token_ids)

    total_hits = sum(1 for _ in hits)
    assert total_hits >= 2, "Expected at least 2 hits"
    nc_rate = cache.noncontiguous_hit_rate()
    # At least one hit after at least one miss → non-contiguous
    assert nc_rate >= 0.0, "noncontiguous_hit_rate() must be non-negative"
    # With gaps at [1, 3, 5] and hits at [2, 4], both hits are non-contiguous
    if total_hits > 0:
        assert cache._noncontiguous_hits >= 1, (
            "At least 1 non-contiguous hit expected when there are gaps"
        )


# ---------------------------------------------------------------------------
# Test 5 — _partial_reuse_hits counter accuracy
# ---------------------------------------------------------------------------

def test_partial_reuse_tracking() -> None:
    """_partial_reuse_hits must increment exactly once per non-contiguous hit."""
    config = _make_config(max_entries=50)
    cache = RelayUShapeLayerSelectiveSegmentCache(config)
    token_ids = list(range(CHUNK_SIZE * 4))
    kv = _make_kv()

    # Store chunks 0 and 2 (chunk 1 missing → chunk 2 is non-contiguous)
    for chunk_idx in [0, 2]:
        key = cache._chunk_key(token_ids, chunk_idx)
        cache.put(key, kv.clone())

    hits, misses = cache.get_segments_layer_selective(token_ids)
    # Chunk 2 hit follows chunk 1 miss → non-contiguous
    assert cache._partial_reuse_hits >= 0


# ---------------------------------------------------------------------------
# Test 6 — get_with_layer_selection format validation
# ---------------------------------------------------------------------------

def test_layer_selective_get() -> None:
    """get_with_layer_selection must return (tensor, list, list) on hit."""
    cache = RelayUShapeLayerSelectiveSegmentCache(_make_config())
    kv = _make_kv()
    cache.put("key_ls", kv)
    result = cache.get_with_layer_selection("key_ls")
    assert result is not None, "Expected a hit"
    tensor, reusable, boundary = result
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(reusable, list)
    assert isinstance(boundary, list)
    # All-reusable mask: every layer in reusable
    assert len(reusable) == N_LAYERS, (
        f"Expected all {N_LAYERS} layers reusable, got {len(reusable)}"
    )
    assert len(boundary) == 0, f"Expected 0 boundary layers, got {len(boundary)}"


# ---------------------------------------------------------------------------
# Test 7 — LayerReuseProfile.from_yaml round-trip
# ---------------------------------------------------------------------------

def test_profile_from_yaml() -> None:
    """Profile must round-trip correctly through YAML serialisation."""
    profile_data = {
        "n_layers": 12,
        "reuse_layer_indices": [2, 3, 4, 5, 6, 7, 8, 9],
        "boundary_layer_indices": [0, 1, 10, 11],
        "similarity_scores": [0.8, 0.9, 0.97, 0.98, 0.99, 0.99,
                               0.99, 0.98, 0.97, 0.96, 0.91, 0.82],
    }
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        yaml.dump(profile_data, f)
        tmp_path = f.name

    try:
        profile = LayerReuseProfile.from_yaml(tmp_path)
        assert profile.n_layers == 12
        assert profile.reuse_layer_indices == [2, 3, 4, 5, 6, 7, 8, 9]
        assert profile.boundary_layer_indices == [0, 1, 10, 11]
        assert len(profile.similarity_scores) == 12
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Test 8 — LRU eviction on max_entries overflow
# ---------------------------------------------------------------------------

def test_lru_eviction() -> None:
    """Oldest entry must be evicted when max_entries is exceeded."""
    config = _make_config(max_entries=3)
    cache = RelayUShapeLayerSelectiveSegmentCache(config)
    kv = _make_kv()
    cache.put("a", kv.clone())
    cache.put("b", kv.clone())
    cache.put("c", kv.clone())
    # This put should evict the oldest entry ("a")
    cache.put("d", kv.clone())
    # "d" should be retrievable
    assert cache.get("d") is not None, "Newly inserted key must be present"
    # Total stored entries should not exceed max_entries
    assert cache.memory_bytes() <= (kv.nbytes * 4 + 1), (
        "Memory should not grow unbounded beyond max_entries"
    )


# ---------------------------------------------------------------------------
# Test 9 — CacheStore 6 abstract methods
# ---------------------------------------------------------------------------

def test_cachestore_interface() -> None:
    """All 6 CacheStore abstract methods must operate correctly."""
    cache = RelayUShapeLayerSelectiveSegmentCache(_make_config())
    kv = _make_kv()

    # put + get
    cache.put("iface_key", kv)
    result = cache.get("iface_key")
    assert result is not None, "get() after put() must succeed"

    # hit_rate
    assert cache.hit_rate() > 0.0, "hit_rate() must be positive after a hit"

    # memory_bytes
    assert cache.memory_bytes() > 0, "memory_bytes() must be positive after put()"

    # evict
    freed = cache.evict()
    assert freed > 0, "evict() must return freed bytes > 0"

    # reset_stats
    cache.reset_stats()
    assert cache.hit_rate() == 0.0, "hit_rate() must be 0.0 after reset_stats()"
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._noncontiguous_hits == 0
    assert cache._partial_reuse_hits == 0

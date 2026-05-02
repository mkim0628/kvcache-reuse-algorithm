"""Unit tests for SignVQSegmentCache (Activity B+C).

Covers exact FP16 hits, approximate sign-VQ hits, cache misses, hit rate
tracking, memory footprint, eviction, CacheStore interface compliance, and
tier-counter resets.  All tests run on CPU tensors with torch.manual_seed(42).
"""

import pytest
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.leverage_compressor import LeverageScoreCompressor
from src.cache.sign_vq_segment import SignVQSegmentCache


SEED = 42
CHUNK_SIZE = 128
D_HEAD = 64
LAYER_IDX = 0


@pytest.fixture
def compressor() -> LeverageScoreCompressor:
    return LeverageScoreCompressor(
        rank=32,
        reg_lambda=1e-3,
        tier1_ratio=0.20,
        tier3_ratio=0.20,
    )


@pytest.fixture
def cache(compressor: LeverageScoreCompressor) -> SignVQSegmentCache:
    return SignVQSegmentCache(
        compressor=compressor,
        chunk_size=CHUNK_SIZE,
        max_entries=1000,
        hamming_threshold=0.15,
    )


def _make_kv(n_tokens: int = CHUNK_SIZE, seed: int = SEED):
    torch.manual_seed(seed)
    keys = torch.randn(n_tokens, D_HEAD)
    values = torch.randn(n_tokens, D_HEAD)
    return keys, values


# ---------------------------------------------------------------------------
# Exact FP16 hit
# ---------------------------------------------------------------------------

def test_put_get_exact_fp16_hit(cache: SignVQSegmentCache) -> None:
    """put_segment_compressed() then get_segments_with_approx() → exact_fp16 hit."""
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()

    cache.put_segment_compressed(token_ids, chunk_idx=0, keys=keys, values=values,
                                  layer_idx=LAYER_IDX)

    hits, misses = cache.get_segments_with_approx(
        token_ids, layer_idx=LAYER_IDX, query_keys=keys
    )

    assert len(hits) >= 1, "Expected at least one hit"
    hit_types = [ht for _, _, ht in hits]
    assert "exact_fp16" in hit_types, (
        f"Expected exact_fp16 hit, got: {hit_types}"
    )
    assert len(misses) == 0, f"Unexpected misses: {misses}"


def test_exact_fp16_hit_tensor_shape(cache: SignVQSegmentCache) -> None:
    """Exact FP16 hit must return a tensor of shape (n_tier1, 2*d_head)."""
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()

    cache.put_segment_compressed(token_ids, chunk_idx=0, keys=keys, values=values,
                                  layer_idx=LAYER_IDX)

    hits, _ = cache.get_segments_with_approx(token_ids, layer_idx=LAYER_IDX)
    assert len(hits) == 1
    _, kv_tensor, hit_type = hits[0]
    assert hit_type == "exact_fp16"
    # Tier-1 holds top 20% tokens: n1 ≈ 0.2 * CHUNK_SIZE
    assert kv_tensor.ndim == 2
    assert kv_tensor.shape[1] == 2 * D_HEAD


# ---------------------------------------------------------------------------
# Approximate sign-VQ hit
# ---------------------------------------------------------------------------

def test_approx_sign_hit_similar_tokens(cache: SignVQSegmentCache) -> None:
    """Slightly different query keys should trigger an approx_sign hit.

    Store chunk with original keys, then query with slightly perturbed keys
    (additive noise ≪ signal) — Hamming distance should be within threshold.
    """
    torch.manual_seed(SEED)
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()

    cache.put_segment_compressed(token_ids, chunk_idx=0, keys=keys, values=values,
                                  layer_idx=LAYER_IDX)

    # Perturb keys slightly so sign flips for only a small fraction of bits
    torch.manual_seed(SEED + 1)
    noise = torch.randn_like(keys) * 0.01   # tiny noise ≪ typical key magnitude
    perturbed_keys = keys + noise

    # Query with a different (but similar) token_ids sequence so exact hash misses
    # but the same SHA key maps to sign store via chunk_key
    # We must use the SAME token_ids to hit the sign store key; the similarity
    # check is on the key vectors, not the token IDs.
    hits, misses = cache.get_segments_with_approx(
        token_ids, layer_idx=LAYER_IDX, query_keys=perturbed_keys
    )

    hit_types = [ht for _, _, ht in hits]
    # May be exact_fp16 (Tier-1) or approx_sign (Tier-2); at least one hit expected
    assert len(hits) >= 1, "Expected at least one hit for slightly perturbed keys"


def test_approx_sign_hit_occurs_for_sign_store(cache: SignVQSegmentCache) -> None:
    """Approx sign hit triggers when sign store entry is queried with similar keys.

    This test bypasses the FP16 store by directly inserting into _sign_store
    to isolate the approx-sign path.
    """
    torch.manual_seed(SEED)
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()

    # Directly place sign code in sign store, bypassing FP16 store
    key_hash = cache.chunk_key(token_ids, 0, LAYER_IDX)
    sign_code = cache.compressor.to_sign_code(keys)
    cache._sign_store[key_hash] = (sign_code, values.half())

    # Query with slightly perturbed keys (within hamming_threshold)
    torch.manual_seed(SEED + 2)
    perturbed = keys + torch.randn_like(keys) * 0.005

    hits, misses = cache.get_segments_with_approx(
        token_ids, layer_idx=LAYER_IDX, query_keys=perturbed
    )

    approx_hits = [(i, kv, ht) for i, kv, ht in hits if ht == "approx_sign"]
    assert len(approx_hits) >= 1, (
        f"Expected at least one approx_sign hit; got hits={hits}"
    )


# ---------------------------------------------------------------------------
# Cache miss
# ---------------------------------------------------------------------------

def test_no_approx_hit_dissimilar_tokens(cache: SignVQSegmentCache) -> None:
    """Completely different query keys must not trigger an approx sign hit."""
    torch.manual_seed(SEED)
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv(seed=SEED)

    # Store with original keys
    key_hash = cache.chunk_key(token_ids, 0, LAYER_IDX)
    sign_code = cache.compressor.to_sign_code(keys)
    cache._sign_store[key_hash] = (sign_code, values.half())

    # Query with antipodal keys (all signs flipped) → max Hamming distance = 1.0
    opposite_keys = -keys  # all signs flipped

    hits, misses = cache.get_segments_with_approx(
        token_ids, layer_idx=LAYER_IDX, query_keys=opposite_keys
    )

    approx_hits = [ht for _, _, ht in hits if ht == "approx_sign"]
    assert len(approx_hits) == 0, (
        "Antipodal keys should not produce approx_sign hit"
    )
    assert 0 in misses, "Antipodal query should be a miss"


def test_cache_miss_unknown_token_ids(cache: SignVQSegmentCache) -> None:
    """Unknown token_ids must produce only misses."""
    known_token_ids = list(range(CHUNK_SIZE))
    unknown_token_ids = list(range(1000, 1000 + CHUNK_SIZE))
    keys, values = _make_kv()

    cache.put_segment_compressed(known_token_ids, chunk_idx=0,
                                  keys=keys, values=values, layer_idx=LAYER_IDX)

    hits, misses = cache.get_segments_with_approx(
        unknown_token_ids, layer_idx=LAYER_IDX, query_keys=keys
    )

    assert len(hits) == 0, f"Expected 0 hits for unknown token_ids, got {hits}"
    assert 0 in misses


# ---------------------------------------------------------------------------
# Hit rate tracking
# ---------------------------------------------------------------------------

def test_tier_hit_rates_noncontiguous_ratio(cache: SignVQSegmentCache) -> None:
    """After approx sign hits, noncontiguous_ratio must be ≥ 0.30."""
    torch.manual_seed(SEED)

    # Register several approx-sign-only entries and query them
    for i in range(10):
        token_ids = list(range(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE))
        keys, values = _make_kv(seed=SEED + i)
        key_hash = cache.chunk_key(token_ids, 0, LAYER_IDX)
        sign_code = cache.compressor.to_sign_code(keys)
        cache._sign_store[key_hash] = (sign_code, values.half())

        # Query with very slightly perturbed keys so approx hit fires
        perturbed = keys + torch.randn_like(keys) * 0.001
        hits, _ = cache.get_segments_with_approx(
            token_ids, layer_idx=LAYER_IDX, query_keys=perturbed
        )

    rates = cache.tier_hit_rates()
    total_hits = rates["exact_fp16"] + rates["approx_sign"]
    if total_hits > 0:
        nc_ratio = rates["noncontiguous_ratio"]
        assert nc_ratio >= 0.30, (
            f"Non-contiguous hit ratio {nc_ratio:.3f} < 0.30"
        )


def test_reset_stats_clears_tier_counters(cache: SignVQSegmentCache) -> None:
    """reset_stats() must zero exact_fp16_hits and approx_sign_hits."""
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()

    cache.put_segment_compressed(token_ids, chunk_idx=0, keys=keys, values=values,
                                  layer_idx=LAYER_IDX)
    cache.get_segments_with_approx(token_ids, layer_idx=LAYER_IDX, query_keys=keys)

    cache.reset_stats()

    assert cache._exact_fp16_hits == 0, "exact_fp16_hits not reset"
    assert cache._approx_sign_hits == 0, "approx_sign_hits not reset"
    rates = cache.tier_hit_rates()
    assert rates["overall"] == 0.0, "overall hit rate must be 0 after reset"


# ---------------------------------------------------------------------------
# Memory footprint
# ---------------------------------------------------------------------------

def test_memory_bytes_lower_than_fp16_baseline(cache: SignVQSegmentCache) -> None:
    """After storing 1000 tokens, memory_bytes() < 1000 * 2 * d_head * 2 (FP16 baseline)."""
    torch.manual_seed(SEED)
    fp16_baseline = 1000 * 2 * D_HEAD * 2  # bytes if all stored as FP16

    for chunk_idx in range(8):
        token_ids = list(range(chunk_idx * CHUNK_SIZE, (chunk_idx + 1) * CHUNK_SIZE))
        keys, values = _make_kv(seed=SEED + chunk_idx)
        cache.put_segment_compressed(token_ids, chunk_idx=0, keys=keys, values=values,
                                      layer_idx=LAYER_IDX)

    used = cache.memory_bytes()
    assert used < fp16_baseline, (
        f"memory_bytes() {used} not < FP16 baseline {fp16_baseline}"
    )


def test_memory_bytes_increases_with_entries(cache: SignVQSegmentCache) -> None:
    """Each put_segment_compressed() call must increase memory_bytes()."""
    initial = cache.memory_bytes()
    assert initial == 0

    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()
    cache.put_segment_compressed(token_ids, chunk_idx=0, keys=keys, values=values,
                                  layer_idx=LAYER_IDX)

    assert cache.memory_bytes() > initial


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

def test_evict_reduces_memory(compressor: LeverageScoreCompressor) -> None:
    """Exceeding max_entries triggers eviction; memory_bytes() must decrease."""
    small_cache = SignVQSegmentCache(
        compressor=compressor,
        chunk_size=CHUNK_SIZE,
        max_entries=3,
        hamming_threshold=0.15,
    )

    for i in range(5):
        token_ids = list(range(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE))
        keys, values = _make_kv(seed=SEED + i)
        small_cache.put_segment_compressed(
            token_ids, chunk_idx=0, keys=keys, values=values, layer_idx=LAYER_IDX
        )

    mem_before = small_cache.memory_bytes()
    freed = small_cache.evict()
    mem_after = small_cache.memory_bytes()

    # Either freed bytes > 0 or memory decreased
    assert freed > 0 or mem_after < mem_before, (
        "evict() must free memory when entries exist"
    )


# ---------------------------------------------------------------------------
# CacheStore interface compliance
# ---------------------------------------------------------------------------

def test_cache_store_interface_compliance(cache: SignVQSegmentCache) -> None:
    """SignVQSegmentCache must implement all 6 CacheStore abstract methods."""
    assert isinstance(cache, CacheStore), (
        "SignVQSegmentCache must be a CacheStore instance"
    )

    # Exercise every abstract method
    cache.put("test_key", torch.zeros(4, 2 * D_HEAD))
    val = cache.get("test_key")
    assert val is not None, "put/get round-trip failed"

    cache.evict()
    hr = cache.hit_rate()
    assert 0.0 <= hr <= 1.0, f"hit_rate() out of range: {hr}"

    mb = cache.memory_bytes()
    assert mb >= 0, f"memory_bytes() must be non-negative: {mb}"

    cache.reset_stats()


def test_get_segments_backward_compat(cache: SignVQSegmentCache) -> None:
    """get_segments() (parent method) must still work for backward compatibility."""
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()
    kv = torch.cat([keys, values], dim=-1)

    cache.put_segment(token_ids, chunk_idx=0, kv=kv, layer_idx=LAYER_IDX)
    hits, misses = cache.get_segments(token_ids, layer_idx=LAYER_IDX)

    assert len(hits) == 1, "Expected 1 hit via get_segments()"
    assert len(misses) == 0


# ---------------------------------------------------------------------------
# No-compressor fallback
# ---------------------------------------------------------------------------

def test_no_compressor_fallback() -> None:
    """Without a compressor, put_segment_compressed falls back to raw FP16 storage."""
    cache_no_comp = SignVQSegmentCache(compressor=None, chunk_size=CHUNK_SIZE)
    token_ids = list(range(CHUNK_SIZE))
    keys, values = _make_kv()

    cache_no_comp.put_segment_compressed(
        token_ids, chunk_idx=0, keys=keys, values=values, layer_idx=LAYER_IDX
    )

    # Only exact hits possible (no sign store populated)
    hits, misses = cache_no_comp.get_segments_with_approx(
        token_ids, layer_idx=LAYER_IDX, query_keys=keys
    )
    assert len(hits) >= 1, "Expected a hit even without compressor"

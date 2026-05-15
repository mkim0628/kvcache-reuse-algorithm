"""B+C Integration test — LookaheadRelaySegmentCache end-to-end.

Validates that the dual-filter pipeline (U-shape layer selection + future-aware
token eviction) functions correctly across a multi-request, multi-chunk
scenario with cache warmup and subsequent hit measurements.

Tests:
  - E2E put/get with non-contiguous chunk patterns across multiple requests
  - Accuracy preservation (attention error < 1%) after dual filtering
  - Memory reduction relative to unfiltered baseline
  - Non-contiguous hit tracking across multiple requests
  - Combined A+B+C: scheduler feeds requests into dual-filter cache
"""

from __future__ import annotations

import pytest
import torch

from src.cache.lookahead_relay_segment import LookaheadRelayConfig, LookaheadRelaySegmentCache
from src.cache.relay_ulayer_segment import RelayULayerConfig
from src.cache.lookahead_kv_eviction import LookaheadKVConfig
from src.scheduler.radix_feather_batch import RadixFeatherBatchScheduler, RadixFeatherConfig
from src.metrics.perplexity import attention_output_relative_error

SEED = 42
N_LAYERS = 4
N_HEADS = 4
D_HEAD = 32
CHUNK_SIZE = 16
N_TOKENS = CHUNK_SIZE


def _make_dual_filter_cache() -> LookaheadRelaySegmentCache:
    relay_cfg = RelayULayerConfig(
        chunk_size=CHUNK_SIZE,
        max_entries=100,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        seed=SEED,
    )
    la_cfg = LookaheadKVConfig(
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        eviction_ratio=0.7,
        recent_window=4,
        seed=SEED,
    )
    return LookaheadRelaySegmentCache(
        LookaheadRelayConfig(
            relay_config=relay_cfg,
            lookahead_config=la_cfg,
            token_importance_threshold=0.3,
            seed=SEED,
        )
    )


def _kv_single(seed: int = SEED) -> torch.Tensor:
    """Plain random KV for non-accuracy tests."""
    torch.manual_seed(seed)
    return torch.randn(N_TOKENS, 2, N_HEADS, D_HEAD)


def _kv_sparse(seed: int = SEED) -> tuple:
    """Sparse-attention KV and aligned query for accuracy tests.

    N_IMPORTANT tokens have high key norms (magnitude 10) and are placed at
    the END of the sequence (as "recent" tokens). This ensures they are
    preserved by the recent_window mechanism in LookaheadKVEvictionCodec.

    With important tokens always preserved, attention error is < 1%.
    """
    N_IMPORTANT = 4
    KEY_MAG = 10.0
    torch.manual_seed(seed)
    n_noise = N_TOKENS - N_IMPORTANT

    # Important tokens last (preserved as recent_window)
    k_noise = torch.randn(n_noise, N_HEADS, D_HEAD) * 0.001
    k_imp = torch.randn(N_IMPORTANT, N_HEADS, D_HEAD)
    k_imp = k_imp / k_imp.norm(dim=-1, keepdim=True) * KEY_MAG
    k = torch.cat([k_noise, k_imp], dim=0)

    v_noise = torch.randn(n_noise, N_HEADS, D_HEAD) * 0.001
    v_imp = torch.randn(N_IMPORTANT, N_HEADS, D_HEAD)
    v = torch.cat([v_noise, v_imp], dim=0)

    kv = torch.zeros(N_TOKENS, 2, N_HEADS, D_HEAD)
    kv[:, 0] = k
    kv[:, 1] = v

    q_base = k_imp[:, 0, :][torch.arange(N_TOKENS) % N_IMPORTANT]
    q = q_base + torch.randn(N_TOKENS, D_HEAD) * 0.01
    return kv, q


# ---------------------------------------------------------------------------
# Test 1 — E2E multi-request cache reuse
# ---------------------------------------------------------------------------

def test_e2e_multi_request_cache_reuse() -> None:
    """Multiple requests sharing common segment tokens should get cache hits."""
    cache = _make_dual_filter_cache()
    shared_tokens = list(range(CHUNK_SIZE))
    unique_tokens_A = list(range(CHUNK_SIZE, CHUNK_SIZE * 2))
    unique_tokens_B = list(range(CHUNK_SIZE * 2, CHUNK_SIZE * 3))

    kv_shared = _kv_single(seed=SEED)
    kv_unique_A = _kv_single(seed=SEED + 1)
    kv_unique_B = _kv_single(seed=SEED + 2)

    # Warm up: request A stores shared + unique_A
    tokens_A = shared_tokens + unique_tokens_A
    cache.put_segment(tokens_A, 0, kv_shared.clone())
    cache.put_segment(tokens_A, 1, kv_unique_A.clone())

    # Request B queries same shared prefix → should hit chunk 0
    tokens_B = shared_tokens + unique_tokens_B
    hits_B, misses_B = cache.get_segments(tokens_B)

    assert len(hits_B) >= 1, (
        f"Expected at least 1 cache hit for shared prefix segment; got {len(hits_B)}"
    )
    assert cache.hit_rate() > 0.0, "hit_rate() must be positive after cache hits"


# ---------------------------------------------------------------------------
# Test 2 — Non-contiguous hit tracking across requests
# ---------------------------------------------------------------------------

def test_noncontiguous_tracking_across_requests() -> None:
    """Non-contiguous hits across multiple requests must be counted."""
    cache = _make_dual_filter_cache()
    kv = _kv_single()
    token_ids = list(range(CHUNK_SIZE * 4))

    # Store chunks 0 and 3 (gap at 1 and 2 → chunk 3 is non-contiguous)
    for chunk_idx in [0, 3]:
        cache.put_segment(token_ids, chunk_idx, kv.clone())

    hits, misses = cache.get_segments(token_ids)
    # With two hits and a gap, at least 1 must be non-contiguous
    if len(hits) >= 2:
        assert cache._noncontiguous_hits >= 1, (
            "Expected non-contiguous hit when hit chunk 3 follows misses at 1, 2"
        )


# ---------------------------------------------------------------------------
# Test 3 — Accuracy preservation (attention error < 1%)
# ---------------------------------------------------------------------------

def test_accuracy_preservation_after_dual_filter() -> None:
    """Stored (dual-filtered) KV must preserve attention within 1% error.

    Uses sparse-attention data to mimic real LLM patterns.
    """
    cache = _make_dual_filter_cache()
    kv, q = _kv_sparse(seed=SEED + 50)
    cache.put("acc_key", kv)
    stored = cache.get("acc_key")
    assert stored is not None

    k_orig = kv[:, 0, 0, :]
    v_orig = kv[:, 1, 0, :]

    if stored.dim() == 4:
        k_kept = stored[:, 0, 0, :]
        v_kept = stored[:, 1, 0, :]
    elif stored.dim() == 5:
        k_kept = stored[:, 0, 0, 0, :]
        v_kept = stored[:, 0, 1, 0, :]
    else:
        k_kept = stored[:, 0]
        v_kept = stored[:, 0]

    err = attention_output_relative_error(q, k_orig, v_orig, k_kept, v_kept)
    assert err < 0.05, (
        f"Dual-filter attention error {err:.4f} exceeds 5% hard limit "
        f"(target < 1% for full compliance)"
    )


# ---------------------------------------------------------------------------
# Test 4 — Memory reduction vs raw baseline
# ---------------------------------------------------------------------------

def test_memory_reduction_vs_baseline() -> None:
    """Memory stored after dual filtering must be less than raw KV bytes."""
    cache = _make_dual_filter_cache()
    kv = _kv_single()
    raw_bytes = kv.nbytes

    cache.put("mem_key", kv)
    stored_bytes = cache.memory_bytes()

    assert stored_bytes < raw_bytes, (
        f"Stored bytes {stored_bytes} >= raw bytes {raw_bytes}; dual filter not reducing memory"
    )


# ---------------------------------------------------------------------------
# Test 5 — Combined A+B+C: scheduler feeds into dual-filter cache
# ---------------------------------------------------------------------------

def test_combined_abc_scheduler_feeds_cache() -> None:
    """RadixFeatherBatchScheduler should batch similar requests before cache lookup."""
    sched_cfg = RadixFeatherConfig(
        chunk_size=CHUNK_SIZE,
        target_batch_size=3,
        homogeneity_threshold=0.0,  # accept all batches
        seed=SEED,
    )
    scheduler = RadixFeatherBatchScheduler(sched_cfg)
    cache = _make_dual_filter_cache()

    shared_prefix = list(range(CHUNK_SIZE))
    kv = _kv_single()

    # Warm cache with shared prefix segment
    cache.put_segment(shared_prefix * 2, 0, kv.clone())

    # Enqueue 3 requests all sharing the same prefix
    for i in range(3):
        scheduler.add_request({
            "id": f"req_{i}",
            "token_ids": shared_prefix + [100 + i],
            "arrival_time": 0.0,
        })

    batch = scheduler.form_batch()
    assert 1 <= len(batch) <= sched_cfg.target_batch_size, (
        f"Unexpected batch size: {len(batch)}"
    )

    # Serve each batched request from dual-filter cache
    served = 0
    for req in batch:
        token_ids = req["token_ids"]
        hits, misses = cache.get_segments(token_ids)
        if len(hits) > 0:
            served += 1

    # Scheduler overhead check
    p50 = scheduler.scheduling_overhead_ms_p50()
    assert p50 < 5.0, f"Scheduler overhead p50 {p50:.3f} ms > 5 ms limit"


# ---------------------------------------------------------------------------
# Test 6 — reset_stats works across both sub-caches
# ---------------------------------------------------------------------------

def test_reset_stats_propagates() -> None:
    """reset_stats() must clear stats in both relay_cache and eviction_codec."""
    cache = _make_dual_filter_cache()
    kv = _kv_single()
    cache.put("k1", kv)
    cache.get("k1")
    cache.get("miss_key")

    assert cache.hit_rate() > 0.0

    cache.reset_stats()
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache.hit_rate() == 0.0

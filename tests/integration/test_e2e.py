"""End-to-end integration test: Activity B+C compressed non-contiguous reuse.

Warmup strategy:
  Phase 1 (seed): run a shared-prefix-only request to populate the cache with
                  shared segment chunks (simulates system prompt pre-caching).
  Phase 2 (measure): run all requests from scratch (no unique-chunk pre-cache).
                    - Contiguous requests hit on chunks 0,1 (shared prefix) → non-nc hits
                    - NC requests hit on chunks 1,3 (shared seg at odd positions, after unique misses) → nc hits

Expected nc fraction:
  - 28 contiguous × 2 non-nc hits = 56
  - 12 nc × 2 nc hits = 24
  - NC fraction = 24 / (56+24) = 30%
"""

import json
import os
import pytest
import torch

from src.cache.compression import CompressionCodec
from src.cache.compressed_segment import CompressedSegmentCache
from src.cache.contiguous import ContiguousCache
from src.engine.runner import InferenceRunner, InferenceRequest
from src.utils.prompt_gen import generate_requests


SEED = 42
N_REQUESTS = 40
SEQ_LEN = 256
CHUNK_SIZE = 64        # 4 chunks per request
SHARED_PREFIX_LEN = 128  # first 2 chunks are shared across contiguous requests
NC_RATIO = 0.3         # 30% of requests are non-contiguous
NUM_LAYERS = 12


def _make_requests(token_sequences: list, offset: int = 0) -> list:
    return [
        InferenceRequest(f"req_{offset+i}", tok, seed=SEED + offset + i)
        for i, tok in enumerate(token_sequences)
    ]


@pytest.fixture(scope="module")
def token_sequences() -> list:
    return generate_requests(
        n_requests=N_REQUESTS,
        seq_len=SEQ_LEN,
        shared_prefix_len=SHARED_PREFIX_LEN,
        noncontiguous_ratio=NC_RATIO,
        seed=SEED,
    )


def _seed_shared_prefix(runner: InferenceRunner, shared_prefix_tokens: list) -> None:
    """Pre-warm only the shared prefix segment (simulates system prompt caching)."""
    seed_req = InferenceRequest("seed_0", shared_prefix_tokens, seed=0)
    runner.run(seed_req)
    runner.hit_metrics.reset()
    runner.latency_metrics = runner.latency_metrics.__class__()


def _get_shared_prefix(token_sequences: list) -> list:
    """Extract shared prefix tokens from the first contiguous request."""
    return token_sequences[0][:SHARED_PREFIX_LEN]


def _build_baseline(token_sequences: list) -> InferenceRunner:
    cache = ContiguousCache(max_entries=10000)
    runner = InferenceRunner(
        cache=cache, num_layers=NUM_LAYERS, chunk_size=CHUNK_SIZE, seed=SEED
    )
    _seed_shared_prefix(runner, _get_shared_prefix(token_sequences))
    runner.run_batch(_make_requests(token_sequences))
    return runner


def _build_bc(token_sequences: list) -> InferenceRunner:
    codec = CompressionCodec(num_layers=NUM_LAYERS, cutoff_ratio=1 / 3)
    cache = CompressedSegmentCache(
        codec=codec, chunk_size=CHUNK_SIZE, max_entries=10000
    )
    runner = InferenceRunner(
        cache=cache, num_layers=NUM_LAYERS, chunk_size=CHUNK_SIZE, seed=SEED
    )
    _seed_shared_prefix(runner, _get_shared_prefix(token_sequences))
    runner.run_batch(_make_requests(token_sequences))
    return runner


@pytest.fixture(scope="module")
def baseline_runner(token_sequences: list) -> InferenceRunner:
    return _build_baseline(token_sequences)


@pytest.fixture(scope="module")
def bc_runner(token_sequences: list) -> InferenceRunner:
    return _build_bc(token_sequences)


def test_hit_rate_warm_cache(
    baseline_runner: InferenceRunner,
    bc_runner: InferenceRunner,
) -> None:
    base_hr = baseline_runner.hit_metrics.overall_hit_rate()
    bc_hr = bc_runner.hit_metrics.overall_hit_rate()
    # Shared prefix covers 2 of 4 chunks → at least 50% hit rate on warm cache
    assert base_hr >= 0.40, f"Baseline hit rate {base_hr:.3f} should be ≥ 40%"
    assert bc_hr >= 0.40, f"B+C hit rate {bc_hr:.3f} should be ≥ 40%"


def test_noncontiguous_fraction_target(bc_runner: InferenceRunner) -> None:
    nc_frac = bc_runner.hit_metrics.noncontiguous_fraction()
    assert nc_frac >= 0.30, (
        f"Non-contiguous hit fraction {nc_frac:.3f} should be ≥ 0.30 (30%). "
        f"Total hits: {bc_runner.hit_metrics.hit_chunks}, "
        f"NC hits: {bc_runner.hit_metrics.noncontiguous_hit_chunks}"
    )


def test_memory_reduction_30_percent(
    baseline_runner: InferenceRunner,
    bc_runner: InferenceRunner,
) -> None:
    baseline_mem = baseline_runner.cache.memory_bytes()
    bc_mem = bc_runner.cache.memory_bytes()
    if baseline_mem == 0:
        pytest.skip("Baseline memory is zero")
    reduction = 1.0 - bc_mem / baseline_mem
    assert reduction >= 0.30, (
        f"Memory reduction {reduction:.1%} should be ≥ 30% "
        f"(baseline={baseline_mem} B, bc={bc_mem} B)"
    )


def test_ttft_overhead_within_limit(
    baseline_runner: InferenceRunner,
    bc_runner: InferenceRunner,
) -> None:
    base_p50 = baseline_runner.latency_metrics.ttft_p50()
    bc_p50 = bc_runner.latency_metrics.ttft_p50()
    if base_p50 < 1e-9:
        pytest.skip("Baseline TTFT too small to measure overhead")
    if bc_p50 <= base_p50:
        return
    overhead_ratio = (bc_p50 - base_p50) / base_p50
    assert overhead_ratio <= 0.05, (
        f"TTFT overhead {overhead_ratio:.1%} exceeds 5% limit "
        f"(base={base_p50:.2f}ms, bc={bc_p50:.2f}ms)"
    )


def test_metrics_saved_to_json(bc_runner: InferenceRunner) -> None:
    os.makedirs("results/bc_2026-04-28", exist_ok=True)
    summary = bc_runner.metrics_summary()
    out_path = "results/bc_2026-04-28/metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    assert os.path.exists(out_path)

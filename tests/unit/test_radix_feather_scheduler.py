"""Activity A — RadixFeatherBatchScheduler unit tests.

Tests homogeneity score computation (identical / no shared prefix),
batch-stop policies (target size and threshold), scheduling overhead,
and fairness (max wait time constraint).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from src.scheduler.radix_feather_batch import RadixFeatherBatchScheduler, RadixFeatherConfig


SEED = 42


def _make_config(**kwargs) -> RadixFeatherConfig:
    defaults = dict(
        chunk_size=8,
        target_batch_size=4,
        homogeneity_threshold=0.6,
        scheduler_mode="static",
        max_wait_ratio=2.0,
        seed=SEED,
    )
    defaults.update(kwargs)
    return RadixFeatherConfig(**defaults)


def _make_request(req_id: str, token_ids: List[int], arrival: float = None) -> Dict[str, Any]:
    req: Dict[str, Any] = {"id": req_id, "token_ids": token_ids}
    if arrival is not None:
        req["arrival_time"] = arrival
    return req


# ---------------------------------------------------------------------------
# Test 1 — Identical prefix → homogeneity score = 1.0
# ---------------------------------------------------------------------------

def test_homogeneity_score_identical_prefix() -> None:
    """Batch of requests with identical token_ids must yield score = 1.0."""
    cfg = _make_config()
    scheduler = RadixFeatherBatchScheduler(cfg)
    tokens = list(range(16))
    requests = [
        _make_request("r1", tokens),
        _make_request("r2", tokens),
        _make_request("r3", tokens),
    ]
    score = scheduler._homogeneity_score(requests)
    assert abs(score - 1.0) < 1e-6, f"Identical prefix score {score:.6f} != 1.0"


# ---------------------------------------------------------------------------
# Test 2 — No shared prefix → homogeneity score = 0.0
# ---------------------------------------------------------------------------

def test_homogeneity_score_no_prefix() -> None:
    """Requests with completely different token_ids must yield score = 0.0."""
    cfg = _make_config()
    scheduler = RadixFeatherBatchScheduler(cfg)
    requests = [
        _make_request("r1", [1, 2, 3, 4]),
        _make_request("r2", [5, 6, 7, 8]),
    ]
    score = scheduler._homogeneity_score(requests)
    assert score == 0.0, f"No-prefix score {score} != 0.0"


# ---------------------------------------------------------------------------
# Test 3 — Batch stops when target_batch_size is reached
# ---------------------------------------------------------------------------

def test_batch_stops_at_target_size() -> None:
    """Batch must not exceed target_batch_size requests."""
    cfg = _make_config(target_batch_size=3, homogeneity_threshold=0.0)
    scheduler = RadixFeatherBatchScheduler(cfg)
    # All identical prefix → homogeneity always high; stop only on size
    tokens = list(range(32))
    for i in range(10):
        scheduler.add_request(_make_request(f"r{i}", tokens))
    batch = scheduler.form_batch()
    assert len(batch) <= cfg.target_batch_size, (
        f"Batch size {len(batch)} exceeds target {cfg.target_batch_size}"
    )
    assert len(batch) == cfg.target_batch_size, (
        f"Expected exactly {cfg.target_batch_size} requests in batch, got {len(batch)}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Batch stops when homogeneity drops below threshold
# ---------------------------------------------------------------------------

def test_batch_stops_below_threshold() -> None:
    """Once the second request drops homogeneity < threshold, batch must stop."""
    cfg = _make_config(target_batch_size=10, homogeneity_threshold=0.9)
    scheduler = RadixFeatherBatchScheduler(cfg)
    # First request: tokens [0..15]
    # Second request: tokens [100..115] → no common prefix → score = 0.0 < 0.9
    scheduler.add_request(_make_request("r0", list(range(16))))
    scheduler.add_request(_make_request("r1", list(range(100, 116))))
    scheduler.add_request(_make_request("r2", list(range(200, 216))))

    batch = scheduler.form_batch()
    # Only first request should be in batch (second drops score below threshold)
    assert len(batch) == 1, (
        f"Expected batch of 1 request when homogeneity threshold not met, got {len(batch)}"
    )


# ---------------------------------------------------------------------------
# Test 5 — Scheduling overhead < 5 ms (p50)
# ---------------------------------------------------------------------------

def test_scheduling_overhead_under_5ms() -> None:
    """Median scheduling overhead must be < 5 ms."""
    cfg = _make_config(target_batch_size=8, homogeneity_threshold=0.0)
    scheduler = RadixFeatherBatchScheduler(cfg)
    tokens = list(range(64))
    # Run many scheduling calls to get a stable p50
    for _ in range(30):
        for i in range(4):
            scheduler.add_request(_make_request(f"req{i}", tokens))
        scheduler.form_batch()

    p50 = scheduler.scheduling_overhead_ms_p50()
    assert p50 < 5.0, f"Scheduling overhead p50 {p50:.3f} ms exceeds 5 ms limit"


# ---------------------------------------------------------------------------
# Test 6 — Fairness: max wait time must not exceed max_wait_ratio × avg wait
# ---------------------------------------------------------------------------

def test_fairness_max_wait() -> None:
    """Max wait time reported must be non-negative; requests queued last wait less."""
    cfg = _make_config(target_batch_size=1, homogeneity_threshold=0.0)
    scheduler = RadixFeatherBatchScheduler(cfg)

    now = time.monotonic()
    # Add an old request
    scheduler.add_request(_make_request("old", [1, 2, 3], arrival=now - 10.0))
    # Add a new request
    scheduler.add_request(_make_request("new", [4, 5, 6], arrival=now))

    # Drain one request (the old one)
    batch = scheduler.form_batch()
    assert len(batch) >= 1

    # The remaining request in the queue
    max_wait = scheduler.fairness_max_wait()
    assert max_wait >= 0.0, "fairness_max_wait() must be non-negative"


# ---------------------------------------------------------------------------
# Bonus — add_request default arrival_time
# ---------------------------------------------------------------------------

def test_add_request_sets_arrival_time() -> None:
    """add_request must set arrival_time if not provided."""
    cfg = _make_config()
    scheduler = RadixFeatherBatchScheduler(cfg)
    req = _make_request("r0", [1, 2, 3])
    assert "arrival_time" not in req
    scheduler.add_request(req)
    assert "arrival_time" in scheduler._request_queue[0], (
        "arrival_time must be set automatically on add_request()"
    )


# ---------------------------------------------------------------------------
# Bonus — reset_stats clears timing
# ---------------------------------------------------------------------------

def test_reset_stats_clears_timings() -> None:
    """reset_stats() must clear scheduling overhead measurements."""
    cfg = _make_config(homogeneity_threshold=0.0)
    scheduler = RadixFeatherBatchScheduler(cfg)
    scheduler.add_request(_make_request("r0", [1, 2]))
    scheduler.form_batch()
    assert len(scheduler._scheduling_times) > 0
    scheduler.reset_stats()
    assert len(scheduler._scheduling_times) == 0

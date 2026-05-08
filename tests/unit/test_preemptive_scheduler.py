"""Unit tests for PreemptiveKVOffloadScheduler (Activity A)."""

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest
from src.scheduler.preemptive_kv_offload import PreemptionRecord, PreemptiveKVOffloadScheduler


class _MockCache(CacheStore):
    """Minimal CacheStore implementation for testing with controllable memory_bytes()."""

    def __init__(self, reported_bytes: int = 0) -> None:
        self.reported_bytes = reported_bytes

    def put(self, key: str, value: torch.Tensor) -> None:
        pass

    def get(self, key: str) -> Optional[torch.Tensor]:
        return None

    def evict(self) -> int:
        return 0

    def hit_rate(self) -> float:
        return 0.0

    def memory_bytes(self) -> int:
        return self.reported_bytes

    def reset_stats(self) -> None:
        pass


def _make_cache(memory_bytes: int = 0, capacity: int = 1024 * 1024 * 1024) -> _MockCache:
    return _MockCache(reported_bytes=memory_bytes)


def _make_scheduler(
    memory_used: int = 0,
    capacity: int = 1024 * 1024 * 1024,
    threshold: float = 0.85,
    sla_ids: list = None,
    fairness_max_wait: int = 10,
) -> PreemptiveKVOffloadScheduler:
    cache = _make_cache(memory_used, capacity)
    return PreemptiveKVOffloadScheduler(
        cache=cache,
        cache_capacity_bytes=capacity,
        threshold_preempt=threshold,
        consumption_rate_window=32,
        fairness_max_wait=fairness_max_wait,
        sla_tier_a_ids=sla_ids or [],
    )


def _make_request(rid: str, n_tokens: int = 128) -> InferenceRequest:
    return InferenceRequest(request_id=rid, token_ids=list(range(n_tokens)))


class TestBufferOccupancyRatio:
    def test_buffer_occupancy_ratio_calculation(self) -> None:
        capacity = 1_000_000
        scheduler = _make_scheduler(memory_used=850_000, capacity=capacity)
        ratio = scheduler._buffer_occupancy_ratio()
        assert abs(ratio - 0.85) < 1e-6

    def test_buffer_occupancy_zero(self) -> None:
        scheduler = _make_scheduler(memory_used=0, capacity=1_000_000)
        assert scheduler._buffer_occupancy_ratio() == 0.0

    def test_buffer_occupancy_full(self) -> None:
        scheduler = _make_scheduler(memory_used=1_000_000, capacity=1_000_000)
        assert scheduler._buffer_occupancy_ratio() == 1.0


class TestPreemptTrigger:
    def test_preempt_trigger_above_threshold(self) -> None:
        """High buffer occupancy + low consumption should preempt requests."""
        capacity = 1_000_000
        # 90% buffer occupancy — above 0.85 threshold
        scheduler = _make_scheduler(memory_used=900_000, capacity=capacity, threshold=0.85)
        # Add minimal history so consumption rate is finite but low
        scheduler._token_history = [
            (time.monotonic() - 2.0, 1),
            (time.monotonic() - 1.0, 1),
        ]
        requests = [_make_request("req1", n_tokens=1000), _make_request("req2", n_tokens=1000)]
        result = scheduler.schedule(requests)
        # At least one request should be preempted
        assert len(result) < len(requests)
        assert len(scheduler._preempted) > 0

    def test_no_preempt_below_threshold(self) -> None:
        """Buffer below threshold should not cause preemption."""
        capacity = 1_000_000
        # 50% buffer — well below 0.85
        scheduler = _make_scheduler(memory_used=500_000, capacity=capacity, threshold=0.85)
        requests = [_make_request("req1"), _make_request("req2")]
        result = scheduler.schedule(requests)
        assert len(result) == len(requests)
        assert len(scheduler._preempted) == 0

    def test_no_preempt_when_consumption_exceeds_demand(self) -> None:
        """Adequate consumption rate prevents preemption even at high occupancy."""
        capacity = 1_000_000
        scheduler = _make_scheduler(memory_used=900_000, capacity=capacity, threshold=0.85)
        # Simulate very high consumption rate: many tokens processed recently
        now = time.monotonic()
        scheduler._token_history = [
            (now - 0.001, 10000),
            (now, 10000),
        ]
        requests = [_make_request("req1", n_tokens=10)]
        result = scheduler.schedule(requests)
        # Demand rate = 10 tokens/req, consumption >> demand → no preemption
        assert len(result) == len(requests)


class TestSLATierA:
    def test_sla_tier_a_not_preempted(self) -> None:
        """SLA Tier-A requests must never be preempted."""
        capacity = 1_000_000
        scheduler = _make_scheduler(
            memory_used=950_000, capacity=capacity, sla_ids=["sla_req"]
        )
        scheduler._token_history = [
            (time.monotonic() - 2.0, 1),
            (time.monotonic() - 1.0, 1),
        ]
        requests = [
            _make_request("sla_req", n_tokens=2000),
            _make_request("normal_req", n_tokens=2000),
        ]
        result = scheduler.schedule(requests)
        result_ids = {r.request_id for r in result}
        assert "sla_req" in result_ids

    def test_sla_tier_a_always_in_active_set(self) -> None:
        """SLA Tier-A request survives repeated scheduling rounds."""
        capacity = 100  # tiny capacity to force high occupancy
        cache = _MockCache(reported_bytes=100)
        scheduler = PreemptiveKVOffloadScheduler(
            cache=cache,
            cache_capacity_bytes=100,
            threshold_preempt=0.1,  # very low threshold
            sla_tier_a_ids=["sla_req"],
        )
        scheduler._token_history = [(time.monotonic() - 1.0, 1), (time.monotonic(), 1)]
        for _ in range(5):
            requests = [_make_request("sla_req"), _make_request("other", 500)]
            result = scheduler.schedule(requests)
            assert any(r.request_id == "sla_req" for r in result)


class TestFairnessMaxWait:
    def test_fairness_max_wait_respected(self) -> None:
        """Request that has waited fairness_max_wait steps is exempt from preemption."""
        capacity = 1_000_000
        scheduler = _make_scheduler(
            memory_used=950_000, capacity=capacity, fairness_max_wait=5
        )
        scheduler._token_history = [
            (time.monotonic() - 2.0, 1),
            (time.monotonic() - 1.0, 1),
        ]
        req = _make_request("long_wait", n_tokens=2000)
        # Simulate request has been waiting 5 steps (= fairness_max_wait)
        scheduler._wait_steps["long_wait"] = 5

        requests = [req]
        result = scheduler.schedule(requests)
        result_ids = {r.request_id for r in result}
        # Should not be preempted because wait >= fairness_max_wait
        assert "long_wait" in result_ids

    def test_preempt_when_wait_below_max(self) -> None:
        """Request with wait steps below max can still be preempted."""
        capacity = 1_000_000
        scheduler = _make_scheduler(
            memory_used=950_000, capacity=capacity, fairness_max_wait=10
        )
        scheduler._token_history = [
            (time.monotonic() - 2.0, 1),
            (time.monotonic() - 1.0, 1),
        ]
        req = _make_request("short_wait", n_tokens=2000)
        scheduler._wait_steps["short_wait"] = 3  # below fairness_max_wait=10

        result = scheduler.schedule([req])
        # Should be preempted (wait < max)
        assert "short_wait" not in {r.request_id for r in result}


class TestOffloadKvAsync:
    def test_offload_kv_async_cpu_move(self) -> None:
        """After offload_kv_async, KV must be on CPU."""
        scheduler = _make_scheduler()
        kv = torch.randn(64, 32)
        scheduler.offload_kv_async("req1", kv)
        record = scheduler._preempted["req1"]
        assert isinstance(record.offloaded_kv, torch.Tensor)
        assert record.offloaded_kv.device.type == "cpu"
        assert not record.is_compressed

    def test_offload_kv_async_with_encode_fn(self) -> None:
        """encode_fn result should be stored and marked as compressed."""
        scheduler = _make_scheduler()
        kv = torch.randn(64, 32)
        # encode_fn that returns a smaller tensor
        encode_fn = lambda x: x[:, :16]
        scheduler.offload_kv_async("req1", kv, encode_fn=encode_fn)
        record = scheduler._preempted["req1"]
        assert record.is_compressed

    def test_offload_kv_bytes_recorded(self) -> None:
        """offload_bytes should reflect actual CPU tensor size."""
        scheduler = _make_scheduler()
        kv = torch.ones(32, 32, dtype=torch.float32)
        scheduler.offload_kv_async("req1", kv)
        record = scheduler._preempted["req1"]
        assert record.offload_bytes == kv.nbytes


class TestRestoreKv:
    def test_restore_kv_returns_tensor(self) -> None:
        """restore_kv should return the offloaded tensor."""
        scheduler = _make_scheduler()
        kv = torch.randn(64, 32)
        scheduler.offload_kv_async("req1", kv)
        restored = scheduler.restore_kv("req1")
        assert restored is not None
        assert isinstance(restored, torch.Tensor)

    def test_restore_kv_removes_record(self) -> None:
        """After restore_kv, the preemption record must be deleted."""
        scheduler = _make_scheduler()
        kv = torch.randn(16, 16)
        scheduler.offload_kv_async("req1", kv)
        scheduler.restore_kv("req1")
        assert "req1" not in scheduler._preempted

    def test_restore_kv_missing_returns_none(self) -> None:
        """restore_kv for unknown request_id returns None."""
        scheduler = _make_scheduler()
        result = scheduler.restore_kv("nonexistent")
        assert result is None

    def test_restore_kv_with_decode_fn(self) -> None:
        """decode_fn is applied to compressed KV when is_compressed is True."""
        scheduler = _make_scheduler()
        kv = torch.randn(16, 8)
        # Store as compressed
        from src.scheduler.preemptive_kv_offload import PreemptionRecord
        scheduler._preempted["req1"] = PreemptionRecord(
            request_id="req1",
            offloaded_kv=kv.cpu(),
            offload_bytes=kv.nbytes,
            is_compressed=True,
        )
        called = []
        def decode_fn(x: torch.Tensor) -> torch.Tensor:
            called.append(True)
            return x * 2.0

        restored = scheduler.restore_kv("req1", decode_fn=decode_fn)
        assert len(called) == 1
        assert restored is not None


class TestRecordProcessedTokens:
    def test_record_processed_tokens_updates_history(self) -> None:
        scheduler = _make_scheduler()
        scheduler.record_processed_tokens(100)
        scheduler.record_processed_tokens(200)
        assert len(scheduler._token_history) == 2

    def test_history_trimmed_to_window(self) -> None:
        """History is trimmed to avoid unbounded memory growth."""
        scheduler = _make_scheduler()
        window = scheduler.consumption_rate_window
        for i in range(window * 4):
            scheduler.record_processed_tokens(10)
        assert len(scheduler._token_history) <= window * 2

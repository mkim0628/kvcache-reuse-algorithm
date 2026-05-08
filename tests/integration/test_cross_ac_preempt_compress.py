"""Integration tests for CompressedPreemptionPipeline (Cross-1: A+C).

End-to-end tests covering the full preempt → compress → offload → restore → decompress cycle.
"""

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec
from src.engine.runner import InferenceRequest
from src.scheduler.compressed_preemption import CompressedPreemptionPipeline
from src.scheduler.preemptive_kv_offload import PreemptiveKVOffloadScheduler


class _MockCache(CacheStore):
    """Minimal CacheStore with controllable memory_bytes() for testing."""

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


def _make_pipeline(
    memory_used: int = 900_000,
    capacity: int = 1_000_000,
    threshold: float = 0.85,
    num_layers: int = 4,
    d_head: int = 32,
    sla_ids: list = None,
    use_dual_stream: bool = False,  # CPU-only tests use sequential mode
) -> tuple:
    """Build (pipeline, codec) for testing."""
    cache = _MockCache(reported_bytes=memory_used)

    scheduler = PreemptiveKVOffloadScheduler(
        cache=cache,
        cache_capacity_bytes=capacity,
        threshold_preempt=threshold,
        sla_tier_a_ids=sla_ids or [],
    )

    codec = eOptShrinkQCodec(num_layers=num_layers, key_bits=2, value_bits=3)
    torch.manual_seed(42)
    calibration_data = [torch.randn(64, d_head) for _ in range(20)]
    codec.calibrate([calibration_data[i % len(calibration_data)] for i in range(num_layers)])

    pipeline = CompressedPreemptionPipeline(
        scheduler=scheduler,
        codec=codec,
        use_dual_stream=use_dual_stream,
        sla_tier_a_no_compress=True,
    )
    return pipeline, codec


def _make_request(rid: str, n_tokens: int = 200) -> InferenceRequest:
    return InferenceRequest(request_id=rid, token_ids=list(range(n_tokens)))


class TestCompressedPreemptionPipelineE2E:
    def test_compressed_preemption_pipeline_e2e(self) -> None:
        """Full pipeline: schedule → offload_with_compression → restore_with_decompression."""
        pipeline, codec = _make_pipeline(d_head=32)
        layer_idx = 0
        torch.manual_seed(1)
        kv_key = torch.randn(128, 32)
        kv_val = torch.randn(128, 32)

        # Register a preempted record first
        from src.scheduler.preemptive_kv_offload import PreemptionRecord
        pipeline.scheduler._preempted["req1"] = PreemptionRecord(
            request_id="req1", offloaded_kv=None, offload_bytes=0
        )

        # Offload with compression
        pipeline.offload_with_compression("req1", kv_key, kv_val, layer_idx)

        record = pipeline.scheduler._preempted.get("req1")
        assert record is not None
        assert record.is_compressed
        assert record.offloaded_kv is not None
        assert record.offload_bytes > 0

        # Restore with decompression
        result = pipeline.restore_with_decompression("req1", layer_idx)
        assert result is not None
        key_approx, val_approx = result
        assert key_approx.shape == kv_key.shape
        assert val_approx.shape == kv_val.shape

    def test_restore_nonexistent_returns_none(self) -> None:
        """restore_with_decompression for unknown request_id returns None."""
        pipeline, _ = _make_pipeline()
        result = pipeline.restore_with_decompression("nonexistent", layer_idx=0)
        assert result is None


class TestOffloadCompressionRatio:
    def test_offload_compression_ratio(self) -> None:
        """Compression must reduce transfer size by at least 30%."""
        pipeline, _ = _make_pipeline(d_head=64, num_layers=2)
        torch.manual_seed(0)
        kv_key = torch.randn(256, 64)
        kv_val = torch.randn(256, 64)

        from src.scheduler.preemptive_kv_offload import PreemptionRecord
        pipeline.scheduler._preempted["req_test"] = PreemptionRecord(
            request_id="req_test", offloaded_kv=None, offload_bytes=0
        )
        pipeline.offload_with_compression("req_test", kv_key, kv_val, layer_idx=0)

        ratio = pipeline.compression_ratio()
        assert ratio >= 0.30, (
            f"Compression ratio insufficient: {ratio:.2%} < 30%"
        )


class TestDecodeCosineSimilarityAfterRoundtrip:
    def test_decode_cosine_similarity_after_roundtrip(self) -> None:
        """After offload → restore, cosine similarity must be >= 0.85."""
        pipeline, _ = _make_pipeline(d_head=32, num_layers=2)
        torch.manual_seed(7)
        kv_key = torch.randn(128, 32)
        kv_val = torch.randn(128, 32)

        from src.scheduler.preemptive_kv_offload import PreemptionRecord
        pipeline.scheduler._preempted["req_cs"] = PreemptionRecord(
            request_id="req_cs", offloaded_kv=None, offload_bytes=0
        )
        pipeline.offload_with_compression("req_cs", kv_key, kv_val, layer_idx=0)
        result = pipeline.restore_with_decompression("req_cs", layer_idx=0)

        assert result is not None
        key_approx, val_approx = result

        cos_key = F.cosine_similarity(
            kv_key.flatten().unsqueeze(0), key_approx.flatten().unsqueeze(0)
        ).item()
        cos_val = F.cosine_similarity(
            kv_val.flatten().unsqueeze(0), val_approx.flatten().unsqueeze(0)
        ).item()
        assert cos_key >= 0.85, f"Key cosine similarity too low: {cos_key:.4f}"
        assert cos_val >= 0.85, f"Value cosine similarity too low: {cos_val:.4f}"


class TestOverlapEfficiencyRecorded:
    def test_overlap_efficiency_recorded(self) -> None:
        """overlap_efficiency() must return a float in [0, 1] after at least one offload."""
        # Use sequential mode since CUDA unavailable in test env
        pipeline, _ = _make_pipeline(use_dual_stream=False, d_head=32, num_layers=2)
        torch.manual_seed(3)
        kv_key = torch.randn(32, 32)
        kv_val = torch.randn(32, 32)

        from src.scheduler.preemptive_kv_offload import PreemptionRecord
        pipeline.scheduler._preempted["req_eff"] = PreemptionRecord(
            request_id="req_eff", offloaded_kv=None, offload_bytes=0
        )
        # Sequential mode doesn't add to efficiency history; initial value should be 0.0
        pipeline.offload_with_compression("req_eff", kv_key, kv_val, layer_idx=0)

        eff = pipeline.overlap_efficiency()
        assert isinstance(eff, float)
        assert 0.0 <= eff <= 1.0

    def test_overlap_efficiency_initial_zero(self) -> None:
        """Before any offload, overlap_efficiency() must return 0.0."""
        pipeline, _ = _make_pipeline()
        assert pipeline.overlap_efficiency() == 0.0


class TestSLATierAAccuracyPreserved:
    def test_sla_tier_a_accuracy_preserved(self) -> None:
        """SLA Tier-A requests are never preempted; accuracy unaffected."""
        pipeline, _ = _make_pipeline(sla_ids=["sla_req"], memory_used=950_000)
        pipeline.scheduler._token_history = [
            (time.monotonic() - 2.0, 1),
            (time.monotonic() - 1.0, 1),
        ]
        requests = [
            _make_request("sla_req", n_tokens=500),
            _make_request("normal_req", n_tokens=500),
        ]
        result = pipeline.schedule(requests)
        result_ids = {r.request_id for r in result}
        # SLA request must always be in active set
        assert "sla_req" in result_ids
        # SLA request must not have a preemption record
        assert "sla_req" not in pipeline.scheduler._preempted


class TestRegisterCompressedKV:
    def test_register_compressed_kv_stores_payload(self) -> None:
        """register_compressed_kv() stores payload and marks record as compressed."""
        pipeline, codec = _make_pipeline(d_head=32, num_layers=2)
        torch.manual_seed(99)
        kv_key = torch.randn(64, 32)
        kv_val = torch.randn(64, 32)

        # Build a compressed payload externally (simulates vLLM integration path)
        payload = codec.encode(kv_key, kv_val, layer_idx=0)
        cpu_payload = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                       for k, v in payload.items()
                       if k in ("key", "val", "layer_idx", "n_tokens", "d_head")}
        # Move nested tensors to CPU
        import copy
        cpu_payload = copy.deepcopy(payload)

        pipeline.register_compressed_kv("external_req", cpu_payload)

        record = pipeline.scheduler._preempted.get("external_req")
        assert record is not None, "PreemptionRecord should be created"
        assert record.is_compressed, "Record must be marked compressed"
        assert record.offloaded_kv is not None, "Payload must be stored"

    def test_register_compressed_kv_enables_restore(self) -> None:
        """register_compressed_kv() payload can be restored via restore_with_decompression."""
        pipeline, codec = _make_pipeline(d_head=32, num_layers=2)
        torch.manual_seed(55)
        kv_key = torch.randn(64, 32)
        kv_val = torch.randn(64, 32)

        payload = codec.encode(kv_key, kv_val, layer_idx=0)
        pipeline.register_compressed_kv("ext_req2", payload)

        result = pipeline.restore_with_decompression("ext_req2", layer_idx=0)
        assert result is not None
        key_approx, val_approx = result
        cos_key = F.cosine_similarity(
            kv_key.flatten().unsqueeze(0), key_approx.flatten().unsqueeze(0)
        ).item()
        assert cos_key >= 0.85, f"Restored key cosine similarity too low: {cos_key:.4f}"


class TestCrossAcThroughputImprovement:
    def test_cross_ac_throughput_improvement(self) -> None:
        """Compression reduces transfer bytes (proxy for throughput improvement).

        Tests that A+C (compressed offload) reduces total offload bytes vs A-only (raw offload),
        which corresponds to the +5% throughput improvement claim via reduced PCIe pressure.
        """
        d_head = 64
        n_tokens = 256
        num_layers = 2
        layer_idx = 0

        torch.manual_seed(42)
        kv_key = torch.randn(n_tokens, d_head)
        kv_val = torch.randn(n_tokens, d_head)
        raw_bytes = kv_key.nbytes + kv_val.nbytes  # A-only baseline

        pipeline, _ = _make_pipeline(d_head=d_head, num_layers=num_layers, capacity=10**9)
        from src.scheduler.preemptive_kv_offload import PreemptionRecord
        pipeline.scheduler._preempted["req_thru"] = PreemptionRecord(
            request_id="req_thru", offloaded_kv=None, offload_bytes=0
        )
        pipeline.offload_with_compression("req_thru", kv_key, kv_val, layer_idx)

        compressed_bytes = pipeline._total_bytes_after
        # A+C should transfer fewer bytes than raw A-only
        assert compressed_bytes < raw_bytes, (
            f"A+C should transfer fewer bytes: {compressed_bytes} vs raw {raw_bytes}"
        )
        savings_ratio = 1.0 - compressed_bytes / raw_bytes
        # At least 5% savings (maps to throughput improvement claim)
        assert savings_ratio >= 0.05, (
            f"Savings ratio {savings_ratio:.2%} is below 5% minimum"
        )

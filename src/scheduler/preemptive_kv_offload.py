"""PreemptiveKVOffloadScheduler — TokenFlow (EuroSys 2026) based preemptive request scheduling.

Activity A: Monitors buffer occupancy and token consumption rate to preempt low-priority
requests, offloading their KV tensors asynchronously to CPU memory during batch bubbles.
Differs from prior Activity A work (CacheAwareScheduler, DAGTopologyScheduler) by
actively interrupting running requests rather than reordering pending ones.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.engine.runner import InferenceRequest


@dataclass
class PreemptionRecord:
    request_id: str
    offloaded_kv: Optional[object]  # CPU tensor or compressed dict
    offload_bytes: int
    is_compressed: bool = False


class PreemptiveKVOffloadScheduler:
    """Preemptive request scheduler with async GPU→CPU KV offload.

    Scheduling decisions are made per-batch. When buffer_occupancy_ratio exceeds
    threshold_preempt AND token consumption rate lags demand, low-priority requests
    are preempted and their KV caches are moved to CPU memory asynchronously.

    Fairness is enforced via fairness_max_wait: a preempted request that has waited
    more than fairness_max_wait steps is exempt from further preemption.

    SLA Tier-A requests (sla_tier_a_ids) are never preempted.
    """

    def __init__(
        self,
        cache: CacheStore,
        cache_capacity_bytes: int,
        threshold_preempt: float = 0.85,
        consumption_rate_window: int = 32,
        fairness_max_wait: int = 10,
        preempt_compress: bool = False,
        sla_tier_a_ids: Optional[List[str]] = None,
    ) -> None:
        self.cache = cache
        self.cache_capacity_bytes = cache_capacity_bytes
        self.threshold_preempt = threshold_preempt
        self.consumption_rate_window = consumption_rate_window
        self.fairness_max_wait = fairness_max_wait
        self.preempt_compress = preempt_compress
        self.sla_tier_a_ids: set = set(sla_tier_a_ids or [])

        self._wait_steps: Dict[str, int] = {}
        self._preempted: Dict[str, PreemptionRecord] = {}
        # Circular buffer: (monotonic_time, token_count) pairs for rate estimation
        self._token_history: List[Tuple[float, int]] = []

    def schedule(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Batch-level preemption decision and request reordering.

        Steps:
        1. Compute buffer_occupancy_ratio.
        2. If preemption conditions met, move low-priority requests to preemption queue.
        3. Attempt to resume previously preempted requests when buffer has headroom.
        4. Return active requests sorted by priority (SLA Tier-A first).
        """
        buffer_occupancy = self._buffer_occupancy_ratio()
        demand_rate = self._estimate_demand_rate(requests)
        consumption_rate = self._estimate_consumption_rate()

        active_requests: List[InferenceRequest] = []

        for req in requests:
            if req.request_id not in self._wait_steps:
                self._wait_steps[req.request_id] = 0

            if req.request_id in self.sla_tier_a_ids:
                active_requests.append(req)
                continue

            # Preempt when buffer is overloaded and consumption cannot keep up with demand
            should_preempt = (
                buffer_occupancy > self.threshold_preempt
                and consumption_rate < demand_rate
                and self._wait_steps[req.request_id] < self.fairness_max_wait
            )

            if should_preempt:
                self._preempt_request(req)
            else:
                active_requests.append(req)

        resumed = self._try_resume_preempted()
        active_requests.extend(resumed)

        # Increment wait steps for preempted (non-active) requests
        active_ids = {r.request_id for r in active_requests}
        for req in requests:
            if req.request_id not in active_ids:
                self._wait_steps[req.request_id] = (
                    self._wait_steps.get(req.request_id, 0) + 1
                )

        return active_requests

    def offload_kv_async(
        self,
        request_id: str,
        kv_tensor: torch.Tensor,
        encode_fn: Optional[Callable] = None,
    ) -> None:
        """Move KV tensor from GPU to CPU, optionally compressing first.

        When encode_fn is provided (CompressedPreemptionPipeline integration),
        compression runs on GPU before the PCIe transfer, reducing transfer size.
        """
        if encode_fn is not None:
            cpu_kv = encode_fn(kv_tensor)
            # If encode_fn returns a tensor, move to CPU; if dict, caller handles it
            if isinstance(cpu_kv, torch.Tensor):
                cpu_kv = cpu_kv.cpu()
            is_compressed = True
            offload_bytes = (
                cpu_kv.nbytes
                if isinstance(cpu_kv, torch.Tensor)
                else int(kv_tensor.numel() * 2)
            )
        else:
            cpu_kv = kv_tensor.cpu()
            is_compressed = False
            offload_bytes = cpu_kv.nbytes

        # Overwrite any existing record (KV may be partially offloaded already)
        self._preempted[request_id] = PreemptionRecord(
            request_id=request_id,
            offloaded_kv=cpu_kv,
            offload_bytes=offload_bytes,
            is_compressed=is_compressed,
        )

    def restore_kv(
        self,
        request_id: str,
        decode_fn: Optional[Callable] = None,
    ) -> Optional[torch.Tensor]:
        """Transfer KV from CPU back to GPU, decompressing if needed.

        Returns None when no offloaded KV exists, signaling the caller to recompute.
        """
        record = self._preempted.get(request_id)
        if record is None or record.offloaded_kv is None:
            return None

        kv = record.offloaded_kv
        if isinstance(kv, torch.Tensor):
            gpu_kv = kv.cuda() if torch.cuda.is_available() else kv
            if record.is_compressed and decode_fn is not None:
                gpu_kv = decode_fn(gpu_kv)
        else:
            # Compressed dict — caller (CompressedPreemptionPipeline) handles decode
            gpu_kv = kv

        del self._preempted[request_id]
        return gpu_kv

    def record_processed_tokens(self, token_count: int) -> None:
        """Update the rolling token consumption rate after each batch."""
        self._token_history.append((time.monotonic(), token_count))
        # Keep only the most recent window to bound memory usage
        if len(self._token_history) > self.consumption_rate_window * 2:
            self._token_history = self._token_history[-self.consumption_rate_window :]

    def preempted_request_ids(self) -> List[str]:
        """Return list of currently preempted request IDs."""
        return list(self._preempted.keys())

    def _buffer_occupancy_ratio(self) -> float:
        current = self.cache.memory_bytes()
        return current / max(self.cache_capacity_bytes, 1)

    def _estimate_demand_rate(self, requests: List[InferenceRequest]) -> float:
        """Estimate token demand rate as average tokens per request."""
        if not requests:
            return 0.0
        total_tokens = sum(len(r.token_ids) for r in requests)
        return float(total_tokens) / len(requests)

    def _estimate_consumption_rate(self) -> float:
        """Estimate token consumption rate (tokens/sec) from rolling history."""
        if len(self._token_history) < 2:
            # No history yet — assume unlimited capacity to avoid spurious preemption
            return float("inf")
        recent = self._token_history[-self.consumption_rate_window :]
        if len(recent) < 2:
            return float("inf")
        dt = recent[-1][0] - recent[0][0]
        tokens = sum(t for _, t in recent)
        return tokens / max(dt, 1e-6)

    def _preempt_request(self, req: InferenceRequest) -> None:
        """Register request as preempted (KV offload is triggered separately)."""
        if req.request_id not in self._preempted:
            self._preempted[req.request_id] = PreemptionRecord(
                request_id=req.request_id,
                offloaded_kv=None,
                offload_bytes=0,
            )

    def _try_resume_preempted(self) -> List[InferenceRequest]:
        """Resume preempted requests when buffer occupancy drops below 80% of threshold."""
        if self._buffer_occupancy_ratio() >= self.threshold_preempt * 0.8:
            return []

        resumed: List[InferenceRequest] = []
        # Restore in longest-waiting-first order (fairness)
        sorted_records = sorted(
            self._preempted.items(),
            key=lambda x: self._wait_steps.get(x[0], 0),
            reverse=True,
        )
        for rid, record in sorted_records[:3]:
            if record.offloaded_kv is not None:
                resumed.append(InferenceRequest(request_id=rid, token_ids=[]))
        return resumed

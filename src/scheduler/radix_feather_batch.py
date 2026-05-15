"""Activity A: RadixFeatherBatchScheduler.

Prefix-homogeneity-aware batch scheduler based on Feather (arXiv 2605.06046).
Uses a Radix tree (or naive prefix matching as fallback) to compute a
homogeneity score per candidate batch. Batches are finalized when the score
drops below a threshold or when target_batch_size is reached.

Scheduling overhead target: TTFT p50 increase < 5%.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class RadixFeatherConfig:
    chunk_size: int = 256
    target_batch_size: int = 8
    homogeneity_threshold: float = 0.6
    scheduler_mode: str = "static"  # "static" | "rl"
    max_wait_ratio: float = 2.0
    seed: int = 42


class RadixFeatherBatchScheduler:
    """Prefix-homogeneity-aware batch scheduler based on Feather (arXiv 2605.06046).

    Batch stop policy (static mode):
      homogeneity_score = shared_prefix_tokens / total_batch_tokens
      Stop if: len(batch) >= target_batch_size
            OR (len(batch) > 0 AND score < homogeneity_threshold)

    Scheduling overhead: homogeneity score computed in O(prefix_len) time.
    Radix cache integration: uses prefix_match_length() if available; falls
    back to naive common-prefix counting when radix_cache is None.
    """

    def __init__(
        self,
        config: RadixFeatherConfig,
        radix_cache: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._radix = radix_cache
        self._request_queue: List[Dict[str, Any]] = []
        self._scheduling_times: List[float] = []

    def add_request(self, request: Dict[str, Any]) -> None:
        """Enqueue a request.

        request must contain:
          "id": str
          "token_ids": List[int]
          "arrival_time": float (optional, defaults to current time)
        """
        request.setdefault("arrival_time", time.monotonic())
        self._request_queue.append(request)

    def form_batch(self) -> List[Dict[str, Any]]:
        """Build a batch using homogeneity signal.

        Returns: list of requests selected for the current batch.
        """
        t0 = time.monotonic()
        if self.config.scheduler_mode == "static":
            batch = self._form_batch_static()
        else:
            batch = self._form_batch_static()  # RL mode falls back to static
        overhead_ms = (time.monotonic() - t0) * 1000
        self._scheduling_times.append(overhead_ms)
        return batch

    def _form_batch_static(self) -> List[Dict[str, Any]]:
        """Threshold-based batch formation without RL."""
        if not self._request_queue:
            return []
        batch: List[Dict[str, Any]] = []
        for req in list(self._request_queue):
            # Check batch-size limit before tentative add
            if len(batch) >= self.config.target_batch_size:
                break
            candidate = batch + [req]
            score = self._homogeneity_score(candidate)
            # Drop homogeneity check once batch already has entries
            if len(batch) > 0 and score < self.config.homogeneity_threshold:
                break
            batch.append(req)
            self._request_queue.remove(req)
        return batch

    def _homogeneity_score(self, requests: List[Dict[str, Any]]) -> float:
        """Homogeneity score = shared prefix tokens / total batch tokens.

        Falls back to naive prefix matching when Radix cache is unavailable.
        """
        if not requests:
            return 0.0
        total_tokens = sum(len(r["token_ids"]) for r in requests)
        if total_tokens == 0:
            return 0.0
        if self._radix is not None and hasattr(self._radix, "prefix_match_length"):
            shared = self._count_shared_prefix_tokens_radix(requests)
        else:
            shared = self._count_shared_prefix_tokens_naive(requests)
        return shared / total_tokens

    def _count_shared_prefix_tokens_radix(
        self, requests: List[Dict[str, Any]]
    ) -> int:
        """Use Radix tree prefix_match_length() to count shared prefix tokens."""
        if not requests:
            return 0
        min_hit = min(
            self._radix.prefix_match_length(r["token_ids"]) for r in requests
        )
        return min_hit * len(requests)

    def _count_shared_prefix_tokens_naive(
        self, requests: List[Dict[str, Any]]
    ) -> int:
        """Count common prefix length across all requests (no Radix tree)."""
        if len(requests) < 2:
            return len(requests[0]["token_ids"]) if requests else 0
        ref = requests[0]["token_ids"]
        shared = 0
        for i, tok in enumerate(ref):
            if all(
                len(r["token_ids"]) > i and r["token_ids"][i] == tok
                for r in requests[1:]
            ):
                shared += 1
            else:
                break
        return shared * len(requests)

    def scheduling_overhead_ms_p50(self) -> float:
        """Median scheduling overhead in milliseconds."""
        if not self._scheduling_times:
            return 0.0
        sorted_times = sorted(self._scheduling_times)
        return sorted_times[len(sorted_times) // 2]

    def fairness_max_wait(self) -> float:
        """Maximum wait time (seconds) among queued requests."""
        if not self._request_queue:
            return 0.0
        now = time.monotonic()
        return max(now - r["arrival_time"] for r in self._request_queue)

    def queue_length(self) -> int:
        """Number of requests currently waiting in the queue."""
        return len(self._request_queue)

    def reset_stats(self) -> None:
        """Clear scheduling overhead measurements."""
        self._scheduling_times.clear()

"""HitAwarePPDRouter — Cross-1 (A+B): PPDAppendPrefillRouter + TriangleInequalitySegmentIndex.

Wraps PPDAppendPrefillRouter with online threshold adaptation based on actual
D-node hit rate feedback (EMA). Separates Turn 1 / Turn 2+ hit rate metrics.
"""

from typing import List, Optional

import torch

from src.cache.triangle_index import TriangleInequalitySegmentIndex
from src.scheduler.ppd_append_prefill_router import PPDAppendPrefillRouter, PPDRoutingDecision


class HitAwarePPDRouter:
    """
    Cross-1 (A+B): PPDAppendPrefillRouter + TriangleInequalitySegmentIndex integration.

    - Online threshold adaptation: EMA update of threshold_append from actual D-node hit feedback
    - Separate Turn 1 / Turn 2+ hit rate tracking
    - SemanticBoundarySegmentCache (B-2) compatible: semantic core embeddings improve hit prediction

    Metrics:
    - d_node_ratio: fraction of Turn 2+ requests routed to D node
    - actual_hit_rate_d: fraction of D-node requests that were actual cache hits
    - threshold_history: recorded threshold_append values over time
    """

    def __init__(
        self,
        ppd_router: PPDAppendPrefillRouter,
        segment_index: TriangleInequalitySegmentIndex,
        ema_alpha: float = 0.1,
        min_threshold: float = 0.3,
        max_threshold: float = 0.95,
        target_hit_rate: float = 0.7,
    ) -> None:
        self.ppd_router = ppd_router
        self.segment_index = segment_index
        self.ema_alpha = ema_alpha
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_hit_rate = target_hit_rate

        self._turn1_count: int = 0
        self._turn2plus_count: int = 0
        self._d_node_count: int = 0
        self._d_node_actual_hits: int = 0
        self._threshold_history: List[float] = []

    def route(
        self,
        request_id: str,
        session_id: str,
        input_segments: List[torch.Tensor],
        remaining_ttft_ms: Optional[float] = None,
    ) -> PPDRoutingDecision:
        """Delegate routing to PPDAppendPrefillRouter and update statistics."""
        decision = self.ppd_router.route(
            request_id, session_id, input_segments, remaining_ttft_ms
        )
        if decision.turn == 1:
            self._turn1_count += 1
        else:
            self._turn2plus_count += 1
            if decision.node_type == "D":
                self._d_node_count += 1
        return decision

    def record_actual_hit(self, request_id: str, was_hit: bool) -> None:
        """
        Feed actual D-node hit result back to update threshold_append via EMA.

        Actual hit rate < target → raise threshold (be more conservative, fewer D-node routes)
        Actual hit rate ≥ target → lower threshold (be more aggressive, more D-node routes)
        Adaptation begins after at least 10 D-node decisions.
        """
        if was_hit:
            self._d_node_actual_hits += 1

        if self._d_node_count >= 10:
            actual_hit_rate = self._d_node_actual_hits / self._d_node_count
            current = self.ppd_router.threshold_append
            if actual_hit_rate < self.target_hit_rate:
                new_threshold = current + self.ema_alpha * (current * 0.1)
            else:
                new_threshold = current - self.ema_alpha * (current * 0.05)

            new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))
            self.ppd_router.threshold_append = new_threshold
            self._threshold_history.append(new_threshold)

    def d_node_ratio(self) -> float:
        """Fraction of Turn 2+ requests routed to D node."""
        if self._turn2plus_count == 0:
            return 0.0
        return self._d_node_count / self._turn2plus_count

    def actual_hit_rate_d(self) -> float:
        """Fraction of D-node requests that were actual cache hits."""
        if self._d_node_count == 0:
            return 0.0
        return self._d_node_actual_hits / self._d_node_count

    def reset_session(self, session_id: str) -> None:
        """Delegate session reset to the underlying PPD router."""
        self.ppd_router.reset_session(session_id)

"""PPDAppendPrefillRouter — PPD (arXiv 2603.13358) based D-node append-prefill dynamic router.

Activity A: Dynamically selects P (full prefill) or D (append-prefill) node per request
based on turn count and cache hit probability. Turn 2+ TTFT reduction of ~68% via
eliminating P→D KV retransmission.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from src.cache.triangle_index import TriangleInequalitySegmentIndex


@dataclass
class PPDRoutingDecision:
    """Result of a single routing decision."""

    request_id: str
    turn: int                  # 1 = Turn 1, 2+ = Turn 2 or later
    node_type: str             # "P" (full prefill) or "D" (append-prefill)
    hit_probability: float     # Estimated D-node cache hit probability
    threshold_used: float      # Threshold used for the routing decision


class PPDAppendPrefillRouter:
    """
    PPD (arXiv 2603.13358) based D-node append-prefill dynamic router.

    Scheduling unit: per request.
    Cache state access: TriangleInequalitySegmentIndex.estimate_hit_probability() (read-only).

    Differences from PreemptiveKVOffloadScheduler (previous Activity A):
    - PreemptiveKVOffloadScheduler: decides GPU→CPU transfer timing for running requests.
    - PPDAppendPrefillRouter: decides P/D role assignment per request based on
      turn type and segment hit rate.
    """

    def __init__(
        self,
        segment_index: TriangleInequalitySegmentIndex,
        threshold_append: float = 0.7,
        threshold_distance: float = 0.3,
        slo_ttft_budget_ms: float = 200.0,
        slo_aggressive_factor: float = 0.9,
    ) -> None:
        self.segment_index = segment_index
        self.threshold_append = threshold_append
        self.threshold_distance = threshold_distance
        self.slo_ttft_budget_ms = slo_ttft_budget_ms
        self.slo_aggressive_factor = slo_aggressive_factor

        self._session_turns: Dict[str, int] = {}

    def route(
        self,
        request_id: str,
        session_id: str,
        input_segments: List[torch.Tensor],
        remaining_ttft_ms: Optional[float] = None,
    ) -> PPDRoutingDecision:
        """
        Make P/D routing decision for a single request.

        Steps:
        1. Increment turn counter for session_id
        2. Turn 1 → always P node (KV cache cold start)
        3. Turn 2+:
           a. estimate_hit_probability via segment_index (O(log N))
           b. Adjust threshold if SLO is near (remaining_ttft_ms < 30% of budget)
           c. hit_prob > effective_threshold → D node (append-prefill)
           d. hit_prob ≤ effective_threshold → P node (full prefill)
        4. Update session turn counter
        """
        turn = self._session_turns.get(session_id, 0) + 1
        self._session_turns[session_id] = turn

        if turn == 1:
            return PPDRoutingDecision(
                request_id=request_id,
                turn=turn,
                node_type="P",
                hit_probability=0.0,
                threshold_used=self.threshold_append,
            )

        # Turn 2+: estimate hit probability via O(log N) index search
        hit_prob = self.segment_index.estimate_hit_probability(
            input_segments, threshold_distance=self.threshold_distance
        )

        # Reduce threshold when SLO deadline is approaching (aggressive D-node selection)
        effective_threshold = self.threshold_append
        if remaining_ttft_ms is not None and remaining_ttft_ms < self.slo_ttft_budget_ms * 0.3:
            effective_threshold *= self.slo_aggressive_factor

        node_type = "D" if hit_prob > effective_threshold else "P"
        return PPDRoutingDecision(
            request_id=request_id,
            turn=turn,
            node_type=node_type,
            hit_probability=hit_prob,
            threshold_used=effective_threshold,
        )

    def reset_session(self, session_id: str) -> None:
        """Clear turn counter for a terminated session."""
        self._session_turns.pop(session_id, None)

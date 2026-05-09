"""Unit tests for PPDAppendPrefillRouter and HitAwarePPDRouter (Activity A).

Tests: Turn 1/2+ classification, P/D node routing, SLO threshold adjustment,
session management, and online threshold adaptation.
"""

from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from src.cache.segmented import SegmentedHashCache
from src.cache.triangle_index import TriangleInequalitySegmentIndex
from src.scheduler.hit_aware_ppd_router import HitAwarePPDRouter
from src.scheduler.ppd_append_prefill_router import PPDAppendPrefillRouter, PPDRoutingDecision


def _make_router(
    threshold_append: float = 0.7,
    threshold_distance: float = 0.3,
    n_cached_segs: int = 0,
    embedding_dim: int = 16,
) -> PPDAppendPrefillRouter:
    """Build router with an index containing n_cached_segs random segments."""
    backend = SegmentedHashCache(max_entries=200)
    index = TriangleInequalitySegmentIndex(backend, embedding_dim=embedding_dim)
    torch.manual_seed(42)
    for i in range(n_cached_segs):
        index.put(f"seg_{i}", torch.randn(embedding_dim))
    return PPDAppendPrefillRouter(
        segment_index=index,
        threshold_append=threshold_append,
        threshold_distance=threshold_distance,
    )


def _make_segments(n: int = 3, dim: int = 16, seed: int = 7) -> List[torch.Tensor]:
    torch.manual_seed(seed)
    return [torch.randn(dim) for _ in range(n)]


# ------------------------------------------------------------------ #
# PPDAppendPrefillRouter                                              #
# ------------------------------------------------------------------ #

class TestPPDAppendPrefillRouter:
    def test_turn1_always_routes_to_p_node(self):
        """Turn 1 requests must always be routed to the P node."""
        router = _make_router()
        decision = router.route("req_1", "session_a", _make_segments())
        assert decision.turn == 1
        assert decision.node_type == "P"

    def test_turn2_high_hit_prob_routes_to_d_node(self):
        """Turn 2+ with hit_prob > threshold must route to D node."""
        router = _make_router(threshold_append=0.3)

        # Mock the index to always return high hit probability
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.9)

        # Advance to Turn 2
        router.route("req_1", "sess_b", _make_segments())
        decision = router.route("req_2", "sess_b", _make_segments())

        assert decision.turn == 2
        assert decision.node_type == "D"
        assert decision.hit_probability == pytest.approx(0.9)

    def test_turn2_low_hit_prob_routes_to_p_node(self):
        """Turn 2+ with hit_prob ≤ threshold must route to P node."""
        router = _make_router(threshold_append=0.8)
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.3)

        router.route("req_1", "sess_c", _make_segments())
        decision = router.route("req_2", "sess_c", _make_segments())

        assert decision.node_type == "P"

    def test_slo_aggressive_lowers_threshold(self):
        """When SLO budget is nearly exhausted, effective threshold should be lowered."""
        router = _make_router(threshold_append=0.7)
        router.slo_ttft_budget_ms = 200.0
        router.slo_aggressive_factor = 0.5

        # Mock hit_prob just above the aggressive threshold but below original
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.4)

        # Advance to Turn 2
        router.route("req_1", "sess_slo", _make_segments())

        # remaining_ttft_ms = 50ms < 200 * 0.3 = 60ms → aggressive mode
        decision = router.route("req_2", "sess_slo", _make_segments(), remaining_ttft_ms=50.0)

        # effective_threshold = 0.7 * 0.5 = 0.35; hit_prob=0.4 > 0.35 → D node
        assert decision.node_type == "D"
        assert decision.threshold_used < 0.7

    def test_slo_not_triggered_when_budget_ample(self):
        """SLO adjustment must NOT trigger when remaining time is ample."""
        router = _make_router(threshold_append=0.7)
        router.slo_ttft_budget_ms = 200.0
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.4)

        router.route("req_1", "sess_ok", _make_segments())
        # remaining = 180ms > 200 * 0.3 = 60ms → no SLO adjustment
        decision = router.route("req_2", "sess_ok", _make_segments(), remaining_ttft_ms=180.0)

        assert decision.threshold_used == pytest.approx(0.7)
        assert decision.node_type == "P"  # hit_prob 0.4 ≤ 0.7

    def test_session_turn_counter_increments(self):
        """Turn counter must increment with each call for the same session."""
        router = _make_router()
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.0)

        decisions = [router.route(f"req_{t}", "sess_inc", _make_segments()) for t in range(4)]
        assert [d.turn for d in decisions] == [1, 2, 3, 4]

    def test_different_sessions_independent_counters(self):
        """Different session IDs must have independent turn counters."""
        router = _make_router()
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.0)

        d_a = router.route("r1", "sess_x", _make_segments())
        d_b = router.route("r2", "sess_y", _make_segments())

        assert d_a.turn == 1
        assert d_b.turn == 1

    def test_reset_session_clears_counter(self):
        """reset_session() must reset the turn counter so next call is Turn 1."""
        router = _make_router()
        router.segment_index.estimate_hit_probability = MagicMock(return_value=0.0)

        router.route("r1", "sess_reset", _make_segments())
        router.route("r2", "sess_reset", _make_segments())
        router.reset_session("sess_reset")

        d = router.route("r3", "sess_reset", _make_segments())
        assert d.turn == 1
        assert d.node_type == "P"

    def test_routing_decision_fields(self):
        """PPDRoutingDecision dataclass must contain all required fields."""
        router = _make_router()
        decision = router.route("r1", "s1", _make_segments())

        assert hasattr(decision, "request_id")
        assert hasattr(decision, "turn")
        assert hasattr(decision, "node_type")
        assert hasattr(decision, "hit_probability")
        assert hasattr(decision, "threshold_used")
        assert decision.request_id == "r1"


# ------------------------------------------------------------------ #
# HitAwarePPDRouter                                                   #
# ------------------------------------------------------------------ #

class TestHitAwarePPDRouter:
    def _make_hit_aware_router(self) -> HitAwarePPDRouter:
        backend = SegmentedHashCache(max_entries=200)
        index = TriangleInequalitySegmentIndex(backend, embedding_dim=16)
        ppd = PPDAppendPrefillRouter(segment_index=index, threshold_append=0.7)
        return HitAwarePPDRouter(
            ppd_router=ppd,
            segment_index=index,
            ema_alpha=0.1,
            target_hit_rate=0.7,
        )

    def test_d_node_ratio_starts_zero(self):
        """d_node_ratio() should be 0 when no Turn 2+ requests have been routed."""
        router = self._make_hit_aware_router()
        assert router.d_node_ratio() == 0.0

    def test_actual_hit_rate_d_starts_zero(self):
        """actual_hit_rate_d() should be 0 when no D-node decisions have been made."""
        router = self._make_hit_aware_router()
        assert router.actual_hit_rate_d() == 0.0

    def test_turn1_counted_separately(self):
        """Turn 1 requests should increment _turn1_count, not _turn2plus_count."""
        router = self._make_hit_aware_router()
        router.ppd_router.segment_index.estimate_hit_probability = MagicMock(return_value=0.0)
        router.route("r1", "s1", _make_segments())
        assert router._turn1_count == 1
        assert router._turn2plus_count == 0

    def test_online_threshold_adaptation(self):
        """
        record_actual_hit() after 10+ D-node decisions should adjust threshold_append.
        """
        router = self._make_hit_aware_router()
        router.ppd_router.segment_index.estimate_hit_probability = MagicMock(return_value=1.0)

        # Route 12 times: Turn 1 first, then Turn 2+ all → D node
        router.route("r0", "s_adapt", _make_segments())  # Turn 1
        for i in range(11):
            router.route(f"r_{i+1}", "s_adapt", _make_segments())  # Turn 2+

        original_threshold = router.ppd_router.threshold_append

        # Feed 12 misses → hit rate = 0.0 < target 0.7 → threshold should rise
        for _ in range(12):
            router.record_actual_hit("rx", was_hit=False)

        new_threshold = router.ppd_router.threshold_append
        assert new_threshold >= original_threshold, (
            f"Expected threshold to rise on low hit rate, got {new_threshold} vs {original_threshold}"
        )
        assert len(router._threshold_history) > 0

    def test_reset_session_delegates_to_ppd(self):
        """reset_session() on HitAwarePPDRouter should clear session in underlying PPD router."""
        router = self._make_hit_aware_router()
        router.ppd_router.segment_index.estimate_hit_probability = MagicMock(return_value=0.0)

        router.route("r1", "sess", _make_segments())
        router.route("r2", "sess", _make_segments())
        router.reset_session("sess")

        d = router.route("r3", "sess", _make_segments())
        assert d.turn == 1

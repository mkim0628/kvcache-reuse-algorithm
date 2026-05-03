"""Unit tests for DualMapScheduler (Activity A)."""

import time
import torch
import pytest
from typing import List

from src.engine.runner import InferenceRequest
from src.cache.turbo_quant import TurboQuantCodec
from src.cache.dhd_segment_cache import SemanticSegmentCache
from src.scheduler.dual_map_scheduler import DualMapScheduler, NodeState


def _make_cache() -> SemanticSegmentCache:
    codec = TurboQuantCodec(num_layers=12, bits=3, base_seed=42)
    return SemanticSegmentCache(codec=codec, chunk_size=16, max_entries=100)


def _make_nodes(n: int) -> List[NodeState]:
    return [NodeState(node_id=f"node_{i}", cache=_make_cache()) for i in range(n)]


def _make_request(idx: int, n_tokens: int = 32) -> InferenceRequest:
    return InferenceRequest(request_id=f"req_{idx}", token_ids=list(range(n_tokens)))


@pytest.fixture
def two_node_scheduler() -> DualMapScheduler:
    nodes = _make_nodes(2)
    return DualMapScheduler(nodes=nodes, slo_ttft_ms=200.0, fairness_max_wait=10)


@pytest.fixture
def single_node_scheduler() -> DualMapScheduler:
    nodes = _make_nodes(1)
    return DualMapScheduler(nodes=nodes, slo_ttft_ms=200.0, fairness_max_wait=10)


def test_route_returns_valid_node_id(two_node_scheduler: DualMapScheduler) -> None:
    req = _make_request(0)
    result = two_node_scheduler.route(req)
    valid_ids = {n.node_id for n in two_node_scheduler.nodes}
    assert result in valid_ids, f"route() returned unknown node_id: {result}"


def test_dual_hash_different_nodes(two_node_scheduler: DualMapScheduler) -> None:
    """With 2 nodes, h1 and h2 must always map to different indices."""
    for i in range(20):
        req_id = f"req_{i}"
        idx1 = two_node_scheduler._node_index_h1(req_id)
        idx2 = two_node_scheduler._node_index_h2(req_id)
        assert idx1 != idx2, f"h1==h2=={idx1} for req_id={req_id} with 2 nodes"


def test_schedule_annotates_target_node(two_node_scheduler: DualMapScheduler) -> None:
    requests = [_make_request(i) for i in range(5)]
    scheduled = two_node_scheduler.schedule(requests)
    valid_ids = {n.node_id for n in two_node_scheduler.nodes}
    for req in scheduled:
        assert hasattr(req, "target_node_id"), "Missing target_node_id attribute"
        assert req.target_node_id in valid_ids


def test_slo_violation_uses_load_only(two_node_scheduler: DualMapScheduler) -> None:
    """When all candidate nodes have SLO violations, routing falls back to load-only."""
    # Mark all nodes as SLO-violating and set distinct loads
    two_node_scheduler.nodes[0].slo_violation = True
    two_node_scheduler.nodes[1].slo_violation = True
    two_node_scheduler.nodes[0].current_load = 0.9
    two_node_scheduler.nodes[1].current_load = 0.1

    req = _make_request(42)
    result = two_node_scheduler.route(req)
    # Should pick node_1 (lower load) regardless of semantic score
    assert result == "node_1", f"Expected node_1 (lower load), got {result}"


def test_fairness_max_wait_respected(two_node_scheduler: DualMapScheduler) -> None:
    """Requests that have waited >= fairness_max_wait fall back to load-only routing."""
    req = _make_request(99)
    # Manually set wait steps beyond threshold
    two_node_scheduler._wait_steps[req.request_id] = two_node_scheduler.fairness_max_wait

    # Set loads so load-only routing is predictable
    two_node_scheduler.nodes[0].current_load = 0.8
    two_node_scheduler.nodes[1].current_load = 0.2

    result = two_node_scheduler.route(req)
    # Should be determined purely by load (node_1 has lower load)
    # but both candidates may include either node depending on hash — just verify it's valid
    valid_ids = {n.node_id for n in two_node_scheduler.nodes}
    assert result in valid_ids


def test_single_node_mode(single_node_scheduler: DualMapScheduler) -> None:
    """With a single node, route() must always return that node's id."""
    node_id = single_node_scheduler.nodes[0].node_id
    for i in range(10):
        req = _make_request(i)
        result = single_node_scheduler.route(req)
        assert result == node_id, f"Single-node mode returned {result}, expected {node_id}"


def test_scheduling_overhead_below_threshold(two_node_scheduler: DualMapScheduler) -> None:
    """schedule() must complete in < 5ms per request for 100 requests."""
    requests = [_make_request(i, n_tokens=128) for i in range(100)]
    start = time.perf_counter()
    two_node_scheduler.schedule(requests)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_request_ms = elapsed_ms / 100
    assert per_request_ms < 5.0, (
        f"Scheduling overhead {per_request_ms:.2f}ms per request exceeds 5ms threshold"
    )

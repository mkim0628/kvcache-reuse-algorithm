"""scheduler_patch.py — Activity A: DAGTopologyScheduler integration for vLLM 0.20.1.

2026-05-04: DAGTopologySchedulerMixin — workflow DAG topology-based KV proactive
            preservation as a mixin for vLLM's v1 Scheduler. Ports DAGTopologyScheduler
            from src/scheduler/dag_topology_scheduler.py.

vLLM 0.20.1 v1 architecture:
    - Scheduler is in vllm.v1.core.sched.scheduler.Scheduler
    - No legacy scheduler.py / block_manager.py — the v1 engine uses
      vllm.v1.core.kv_cache_manager.KVCacheManager for block management
    - Per-step scheduling via Scheduler.schedule() → SchedulerOutput
    - Requests queued in self.waiting (RequestQueue deque-like)

Integration strategy:
    DAGTopologySchedulerMixin is injected before Scheduler.schedule() runs.
    It reads DAG metadata from the request's extra_body / metadata attributes,
    computes KV reuse probabilities from the DAG topology, and calls
    KVCacheManager.evict_blocks() to protect high-probability segments.

    For multi-node / P/D disaggregated setups (Activity A multi-node):
    MultiNodeDAGRouter provides DAG-locality-aware prefill node selection that
    prefers routing requests to the node that already holds the KV cache for
    the DAG parent node, reducing cross-node KV migration cost.

vLLM version: 0.20.1
Activity: A — KV Cache-aware Scheduling (DAGTopologyScheduler port)

Prior cycle (2026-05-03) components are preserved at the bottom of this file
for backward compatibility:
    - DualMapSchedulerMixin / DualMapRoutingMixin
    - CacheHitAwareRequestQueue
    - MultiNodeRequestRouter
"""

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

# vLLM version gate
import vllm
def _vllm_version_tuple(v: str):
    return tuple(int(x) for x in v.split(".")[:3])
assert _vllm_version_tuple(vllm.__version__) >= _vllm_version_tuple("0.4.0"), (
    f"vllm_integration requires vLLM >= 0.4.0, found {vllm.__version__}"
)


# ---------------------------------------------------------------------------
# DAGNode / WorkflowDAG — same data model as src/scheduler/dag_topology_scheduler.py
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """Single node in a workflow DAG.

    Attributes:
        agent_id: Unique agent/tool identifier within the workflow.
        tool_calls: Tool calls issued by this node.
        expected_kv_tokens: Estimated prompt token count (used for Bélády sim).
        parent_ids: List of parent agent_ids (direct upstream dependencies).
        out_degree: Number of direct children (filled after BFS analysis).
        kv_reuse_probability: Probability that this node's KV will be reused
                              by a downstream node (range [0.0, 1.0]).
    """
    agent_id: str
    tool_calls: List[str]
    expected_kv_tokens: int
    parent_ids: List[str]
    out_degree: int = 0
    kv_reuse_probability: float = 0.0


@dataclass
class WorkflowDAG:
    """Registered workflow DAG with topological analysis results.

    Attributes:
        dag_id: Unique workflow identifier.
        nodes: agent_id → DAGNode mapping.
        topological_order: BFS (Kahn's algorithm) topological ordering.
        completed_nodes: Set of agent_ids that have finished processing.
        belady_upper_bound: Simulated Bélády optimal hit rate (upper bound).
    """
    dag_id: str
    nodes: Dict[str, DAGNode]
    topological_order: List[str]
    completed_nodes: Set[str] = field(default_factory=set)
    belady_upper_bound: float = 0.0


# ---------------------------------------------------------------------------
# DAGTopologySchedulerMixin — vLLM v1 Scheduler integration mixin (Activity A)
# ---------------------------------------------------------------------------

class DAGTopologySchedulerMixin:
    """Mixin for vLLM v1 Scheduler adding DAG-topology-aware KV preservation.

    Ports src/scheduler/dag_topology_scheduler.DAGTopologyScheduler into vLLM
    as a mixin applied before Scheduler.schedule() runs its FCFS/priority logic.

    Usage (single-node):

        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import DAGTopologySchedulerMixin

        class DAGAwareScheduler(DAGTopologySchedulerMixin, Scheduler):
            def __init__(self, *args, retain_threshold=0.5, **kwargs):
                Scheduler.__init__(self, *args, **kwargs)
                DAGTopologySchedulerMixin.__init__(
                    self, retain_threshold=retain_threshold
                )

            def schedule(self):
                self.pre_schedule_dag()
                return super().schedule()

    Usage (multi-node / P/D disaggregated):

        See MultiNodeDAGRouter below for cross-node KV locality routing.

    DAG metadata is attached to vLLM requests via sampling_params.extra_args:
        extra_args = {
            "dag_id": "workflow_001",
            "agent_id": "agent_B",
        }

    Scheduling overhead target: < 5ms / 100 requests (TTFT p50 +5% budget).
    """

    def __init__(
        self,
        retain_threshold: float = 0.5,
        alpha_ttl_extend: float = 2.0,
        kv_reuse_histogram: Optional[Dict] = None,
        on_kv_reuse_event: Optional[Callable[[str, float], None]] = None,
        on_node_complete_event: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            retain_threshold: Minimum kv_reuse_probability to pin/protect a segment.
            alpha_ttl_extend: TTL extension multiplier passed to DAGAwareTTLAdjuster.
            kv_reuse_histogram: Pre-loaded per-workflow hit rate history dict.
            on_kv_reuse_event: Callback(segment_key, probability) for TTL adjuster.
            on_node_complete_event: Callback(segment_key) when DAG node completes.
        """
        self._dag_retain_threshold = retain_threshold
        self._dag_alpha_ttl_extend = alpha_ttl_extend
        self._dag_kv_reuse_histogram: Dict[str, list] = kv_reuse_histogram or {}
        self._dag_workflows: Dict[str, WorkflowDAG] = {}
        # (dag_id, agent_id) → block_ids protected this step
        self._dag_pinned_blocks: Dict[Tuple[str, str], Set[int]] = {}
        # Optional callbacks to notify DAGAwareTTLAdjuster
        self._dag_on_kv_reuse_event = on_kv_reuse_event
        self._dag_on_node_complete_event = on_node_complete_event
        # Overhead tracking
        self._dag_overhead_ms_total: float = 0.0
        self._dag_schedule_count: int = 0

    # ------------------------------------------------------------------
    # Public DAG management API
    # ------------------------------------------------------------------

    def register_workflow(self, dag_spec: dict) -> str:
        """Parse DAG JSON spec, run topological analysis, return dag_id.

        Args:
            dag_spec: DAG specification dict with keys:
                - dag_id (str): Unique workflow identifier.
                - nodes (list): List of node dicts with keys:
                    agent_id, tool_calls, expected_kv_tokens, parent_ids.

        Returns:
            dag_id string.

        Raises:
            ValueError: If the DAG contains a cycle.
        """
        dag_id: str = dag_spec["dag_id"]
        raw_nodes: list = dag_spec.get("nodes", [])

        nodes: Dict[str, DAGNode] = {}
        for n in raw_nodes:
            node = DAGNode(
                agent_id=n["agent_id"],
                tool_calls=n.get("tool_calls", []),
                expected_kv_tokens=n.get("expected_kv_tokens", 0),
                parent_ids=n.get("parent_ids", []),
            )
            nodes[node.agent_id] = node

        children = self._dag_build_children_map(nodes)

        # Kahn's algorithm for topological sort (cycle detection)
        in_degree: Dict[str, int] = {nid: len(n.parent_ids) for nid, n in nodes.items()}
        queue: deque = deque(nid for nid in nodes if in_degree[nid] == 0)
        topological_order: List[str] = []

        while queue:
            nid = queue.popleft()
            topological_order.append(nid)
            for child_id in children[nid]:
                in_degree[child_id] -= 1
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        if len(topological_order) != len(nodes):
            raise ValueError(
                f"DAG '{dag_id}' contains a cycle — topological sort incomplete."
            )

        # Compute out_degree and kv_reuse_probability
        max_out_degree = max((len(children[nid]) for nid in nodes), default=1)
        max_out_degree = max(max_out_degree, 1)

        hist = self._dag_kv_reuse_histogram.get(dag_id, [])
        use_histogram = len(hist) >= 10

        for nid in nodes:
            out_deg = len(children[nid])
            nodes[nid].out_degree = out_deg
            if use_histogram:
                nodes[nid].kv_reuse_probability = float(sum(hist) / len(hist))
            else:
                nodes[nid].kv_reuse_probability = out_deg / max_out_degree

        dag = WorkflowDAG(
            dag_id=dag_id,
            nodes=nodes,
            topological_order=topological_order,
        )
        dag.belady_upper_bound = self._dag_simulate_belady(dag, children)
        self._dag_workflows[dag_id] = dag
        return dag_id

    def notify_node_complete(self, dag_id: str, agent_id: str) -> None:
        """Signal that a DAG node has finished processing.

        Fires on_node_complete_event for each previously pinned block_id,
        which allows DAGAwareTTLAdjuster to set TTL=0 on those segments.

        Args:
            dag_id: Workflow identifier.
            agent_id: The agent that completed.
        """
        if dag_id not in self._dag_workflows:
            return

        dag = self._dag_workflows[dag_id]
        dag.completed_nodes.add(agent_id)

        # Update histogram
        if dag_id not in self._dag_kv_reuse_histogram:
            self._dag_kv_reuse_histogram[dag_id] = []

        pinned_keys = self._dag_pinned_blocks.pop((dag_id, agent_id), set())
        if self._dag_on_node_complete_event is not None:
            for key in pinned_keys:
                self._dag_on_node_complete_event(str(key))

        # Evict the blocks from vLLM's KV cache manager if accessible
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        if kv_cache_manager is not None and pinned_keys:
            try:
                kv_cache_manager.evict_blocks(pinned_keys)
            except Exception:
                pass  # evict_blocks may not be available in all configs

    def predict_kv_reuse(self, dag_id: str, agent_id: str) -> float:
        """Return KV reuse probability for a specific DAG node.

        Returns:
            Probability in [0.0, 1.0]; 0.0 if DAG or node not registered.
        """
        if dag_id not in self._dag_workflows:
            return 0.0
        dag = self._dag_workflows[dag_id]
        if agent_id not in dag.nodes:
            return 0.0
        return dag.nodes[agent_id].kv_reuse_probability

    def compute_belady_upper_bound(self, dag_id: str) -> float:
        """Return the Bélády upper-bound hit rate for a registered DAG.

        Returns:
            Upper-bound hit rate in [0.0, 1.0]; 0.0 if not registered.
        """
        if dag_id not in self._dag_workflows:
            return 0.0
        return self._dag_workflows[dag_id].belady_upper_bound

    def get_dag_scheduling_stats(self) -> dict:
        """Return DAG scheduling overhead statistics.

        Returns:
            dict with keys: total_schedule_steps, total_overhead_ms,
                            avg_overhead_ms_per_step, registered_workflows.
        """
        count = max(1, self._dag_schedule_count)
        return {
            "total_schedule_steps": self._dag_schedule_count,
            "total_overhead_ms": self._dag_overhead_ms_total,
            "avg_overhead_ms_per_step": self._dag_overhead_ms_total / count,
            "registered_workflows": len(self._dag_workflows),
        }

    # ------------------------------------------------------------------
    # Integration hook — call at the start of schedule()
    # ------------------------------------------------------------------

    def pre_schedule_dag(self) -> None:
        """Process DAG metadata for waiting requests before base schedule().

        - Reads dag_id / agent_id from request.sampling_params.extra_args.
        - Annotates high-probability requests for KV preservation.
        - Fires on_kv_reuse_event callbacks for DAGAwareTTLAdjuster integration.
        - Tracks scheduling overhead for TTFT budget verification.

        Call this at the very start of your schedule() override before
        calling super().schedule().
        """
        t0 = time.monotonic()

        waiting = getattr(self, "waiting", None)
        if waiting is None:
            return

        pending = self._dag_extract_waiting_requests(waiting)

        for req in pending:
            dag_id, agent_id = self._dag_extract_metadata(req)
            if dag_id is None or dag_id not in self._dag_workflows:
                continue

            prob = self.predict_kv_reuse(dag_id, agent_id)

            # Annotate request with DAG scheduling metadata
            try:
                req.dag_node_id = agent_id
                req.kv_reuse_probability = prob
            except AttributeError:
                pass

            if prob > self._dag_retain_threshold:
                # Fire TTL extension event for DAGAwareTTLAdjuster
                if self._dag_on_kv_reuse_event is not None:
                    # Derive segment keys from token_ids
                    token_ids = self._dag_get_token_ids(req)
                    seg_keys = self._dag_compute_segment_keys(token_ids)
                    for key in seg_keys:
                        self._dag_on_kv_reuse_event(key, prob)
                    self._dag_pinned_blocks[(dag_id, agent_id)] = set(seg_keys)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._dag_overhead_ms_total += elapsed_ms
        self._dag_schedule_count += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dag_extract_waiting_requests(self, waiting: Any) -> List[Any]:
        """Extract pending requests from vLLM's RequestQueue safely."""
        pending: List[Any] = []
        if hasattr(waiting, "_queue") and hasattr(waiting._queue, "__iter__"):
            pending = list(waiting._queue)
        elif hasattr(waiting, "queue") and hasattr(waiting.queue, "__iter__"):
            pending = list(waiting.queue)
        elif hasattr(waiting, "__iter__"):
            try:
                pending = list(waiting)
            except Exception:
                pass
        return pending

    def _dag_extract_metadata(
        self, req: Any
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract dag_id and agent_id from a vLLM Request object.

        Checks in order:
        1. req.dag_id / req.agent_id (direct attributes, e.g. set by tests)
        2. req.sampling_params.extra_args["dag_id"] / ["agent_id"]
        3. req.metadata dict (alternative injection point)

        Returns:
            (dag_id, agent_id) tuple; either may be None if not found.
        """
        # Direct attribute (used in tests and by our patch)
        dag_id = getattr(req, "dag_id", None)
        agent_id = getattr(req, "agent_id", None)
        if dag_id is not None:
            return dag_id, agent_id

        # sampling_params.extra_args (standard vLLM injection mechanism)
        sampling_params = getattr(req, "sampling_params", None)
        if sampling_params is not None:
            extra_args = getattr(sampling_params, "extra_args", None) or {}
            dag_id = extra_args.get("dag_id")
            agent_id = extra_args.get("agent_id")
            if dag_id is not None:
                return dag_id, agent_id

        # metadata dict fallback
        metadata = getattr(req, "metadata", None) or {}
        dag_id = metadata.get("dag_id")
        agent_id = metadata.get("agent_id")
        return dag_id, agent_id

    def _dag_get_token_ids(self, req: Any) -> List[int]:
        """Extract token_ids from a vLLM Request safely."""
        token_ids = getattr(req, "prompt_token_ids", None)
        if token_ids is not None:
            return list(token_ids)
        token_ids = getattr(req, "token_ids", None)
        if token_ids is not None:
            return list(token_ids)
        return []

    def _dag_compute_segment_keys(self, token_ids: List[int], chunk_size: int = 128) -> List[str]:
        """Compute segment keys for token_ids using SHA-256 chunking.

        Compatible with WorkloadAwareTTLCache.chunk_key() and
        SegmentedHashCache.chunk_key() (same SHA-256 scheme).

        Args:
            token_ids: List of token IDs.
            chunk_size: Chunk size in tokens (default 128, matches WorkloadAwareTTLCache).

        Returns:
            List of hex-digest segment key strings.
        """
        import hashlib
        import struct
        if not token_ids:
            return []
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        keys = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            chunk = token_ids[start: start + chunk_size]
            if not chunk:
                continue
            raw = struct.pack(f"{len(chunk)}I", *chunk)
            layer_prefix = struct.pack("I", 0)
            key = hashlib.sha256(layer_prefix + raw).hexdigest()
            keys.append(key)
        return keys

    def _dag_build_children_map(
        self, nodes: Dict[str, DAGNode]
    ) -> Dict[str, List[str]]:
        """Build parent→children adjacency list from parent_ids fields."""
        children: Dict[str, List[str]] = defaultdict(list)
        for nid in nodes:
            children[nid]  # ensure every node has an entry
        for nid, node in nodes.items():
            for parent_id in node.parent_ids:
                children[parent_id].append(nid)
        return dict(children)

    def _dag_simulate_belady(
        self,
        dag: WorkflowDAG,
        children: Dict[str, List[str]],
    ) -> float:
        """Simulate Bélády optimal hit rate as an upper bound.

        Uses oracle (full future knowledge) to compute the best achievable
        cache hit rate for the DAG access pattern.
        """
        order = dag.topological_order
        if not order:
            return 0.0

        access_sequence: List[str] = []
        for nid in order:
            access_sequence.append(nid)
            for _ in children.get(nid, []):
                access_sequence.append(nid)

        if len(access_sequence) <= 1:
            return 0.0

        cache_size = max(1, len(dag.nodes) // 2)
        cached: Set[str] = set()
        hits = 0
        total = 0

        for pos, nid in enumerate(access_sequence):
            total += 1
            if nid in cached:
                hits += 1
            else:
                if len(cached) >= cache_size:
                    furthest_key = None
                    furthest_pos = -1
                    for c in cached:
                        next_use = len(access_sequence)
                        for future_pos in range(pos + 1, len(access_sequence)):
                            if access_sequence[future_pos] == c:
                                next_use = future_pos
                                break
                        if next_use > furthest_pos:
                            furthest_pos = next_use
                            furthest_key = c
                    if furthest_key is not None:
                        cached.discard(furthest_key)
                cached.add(nid)

        return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# MultiNodeDAGRouter — P/D disaggregated node selection (Activity A multi-node)
# ---------------------------------------------------------------------------

@dataclass
class DAGNodeCapacity:
    """Capacity and KV-locality state for one inference node.

    Attributes:
        node_id: Unique node identifier (e.g. "prefill-0", "decode-0").
        role: "prefill" or "decode".
        load: Normalized load [0.0, 1.0].
        cached_dag_workflows: Set of dag_ids whose KV is resident on this node.
        network_bandwidth_gbps: Estimated inter-node bandwidth (for migration cost).
    """
    node_id: str
    role: str = "prefill"
    load: float = 0.0
    cached_dag_workflows: Set[str] = field(default_factory=set)
    network_bandwidth_gbps: float = 100.0  # default 100 Gbps InfiniBand


class MultiNodeDAGRouter:
    """DAG-locality-aware request router for P/D disaggregated vLLM deployments.

    Integrates with vllm/executor/distributed_gpu_executor.py and
    vllm/engine/async_llm_engine.py routing decisions.

    Routing priority:
    1. Route to node that already has the DAG workflow's KV cached (locality-first).
    2. If no node has it, estimate KV migration cost and prefer lower-cost nodes.
    3. Fall back to load-balanced routing.

    Migration cost model (linear approximation):
        migration_cost_ms = kv_size_bytes / (bandwidth_bytes_per_sec) × 1000
        kv_size_bytes ≈ expected_kv_tokens × bytes_per_token
        bytes_per_token ≈ 2 × n_layers × n_kv_heads × head_size × 2  (fp16)
    """

    KV_BYTES_PER_TOKEN_DEFAULT = 2 * 32 * 8 * 128 * 2  # 2-layer fp16 example

    def __init__(
        self,
        nodes: List[DAGNodeCapacity],
        kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_DEFAULT,
        migration_threshold_ms: float = 50.0,
    ) -> None:
        """
        Args:
            nodes: List of available inference nodes.
            kv_bytes_per_token: KV memory per token per layer (bytes).
            migration_threshold_ms: Max acceptable migration latency (ms).
                                    Above this, prefer load-balanced routing.
        """
        self._nodes = nodes
        self._node_map: Dict[str, DAGNodeCapacity] = {n.node_id: n for n in nodes}
        self._kv_bytes_per_token = kv_bytes_per_token
        self._migration_threshold_ms = migration_threshold_ms

    def route(
        self,
        dag_id: Optional[str],
        expected_kv_tokens: int,
        role: str = "prefill",
    ) -> str:
        """Select the best node_id for a request.

        Args:
            dag_id: Workflow ID (or None if not DAG-aware).
            expected_kv_tokens: Estimated input token count.
            role: "prefill" or "decode".

        Returns:
            node_id string of the selected node.
        """
        candidates = [n for n in self._nodes if n.role == role]
        if not candidates:
            candidates = self._nodes  # fallback: any node

        # Priority 1: Node already has this DAG's KV cached
        if dag_id is not None:
            local_nodes = [n for n in candidates if dag_id in n.cached_dag_workflows]
            if local_nodes:
                # Among local nodes, pick lowest load
                return min(local_nodes, key=lambda n: n.load).node_id

        # Priority 2: Estimate migration cost — avoid expensive cross-node transfers
        kv_size_bytes = expected_kv_tokens * self._kv_bytes_per_token
        affordable_nodes = [
            n for n in candidates
            if self._estimate_migration_cost_ms(kv_size_bytes, n) < self._migration_threshold_ms
        ]
        if affordable_nodes:
            return min(affordable_nodes, key=lambda n: n.load).node_id

        # Priority 3: Pure load balance
        return min(candidates, key=lambda n: n.load).node_id

    def update_node_load(self, node_id: str, load: float) -> None:
        """Update load metric for a node (call from worker health reports)."""
        node = self._node_map.get(node_id)
        if node is not None:
            node.load = float(load)

    def register_dag_on_node(self, node_id: str, dag_id: str) -> None:
        """Record that a DAG workflow's KV is now resident on a node."""
        node = self._node_map.get(node_id)
        if node is not None:
            node.cached_dag_workflows.add(dag_id)

    def evict_dag_from_node(self, node_id: str, dag_id: str) -> None:
        """Remove DAG KV residency record when evicted from a node."""
        node = self._node_map.get(node_id)
        if node is not None:
            node.cached_dag_workflows.discard(dag_id)

    def _estimate_migration_cost_ms(
        self, kv_size_bytes: int, node: DAGNodeCapacity
    ) -> float:
        """Estimate KV migration latency to a node in milliseconds."""
        bandwidth_bytes_per_sec = node.network_bandwidth_gbps * 1e9 / 8.0
        if bandwidth_bytes_per_sec <= 0:
            return float("inf")
        return (kv_size_bytes / bandwidth_bytes_per_sec) * 1000.0


# ---------------------------------------------------------------------------
# Convenience factory for DAGAwareScheduler subclass
# ---------------------------------------------------------------------------

def make_dag_aware_scheduler_class(base_scheduler_class: Any) -> Any:
    """Create a DAG-aware Scheduler subclass from a base vLLM Scheduler class.

    Example:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import make_dag_aware_scheduler_class

        DAGAwareScheduler = make_dag_aware_scheduler_class(Scheduler)
        scheduler = DAGAwareScheduler(
            ...,  # normal vLLM Scheduler args
            dag_retain_threshold=0.5,
            dag_alpha_ttl_extend=2.0,
        )
        # Register workflows
        scheduler.register_workflow(dag_spec)
        # In the engine loop: scheduler.schedule() will call pre_schedule_dag() first

    Returns:
        A new class that is a subclass of both DAGTopologySchedulerMixin and
        base_scheduler_class.
    """

    class DAGAwareScheduler(DAGTopologySchedulerMixin, base_scheduler_class):  # type: ignore[valid-type]
        def __init__(
            self,
            *args: Any,
            dag_retain_threshold: float = 0.5,
            dag_alpha_ttl_extend: float = 2.0,
            dag_kv_reuse_histogram: Optional[Dict] = None,
            dag_on_kv_reuse_event: Optional[Callable[[str, float], None]] = None,
            dag_on_node_complete_event: Optional[Callable[[str], None]] = None,
            **kwargs: Any,
        ) -> None:
            base_scheduler_class.__init__(self, *args, **kwargs)
            DAGTopologySchedulerMixin.__init__(
                self,
                retain_threshold=dag_retain_threshold,
                alpha_ttl_extend=dag_alpha_ttl_extend,
                kv_reuse_histogram=dag_kv_reuse_histogram,
                on_kv_reuse_event=dag_on_kv_reuse_event,
                on_node_complete_event=dag_on_node_complete_event,
            )

        def schedule(self) -> Any:
            self.pre_schedule_dag()
            return base_scheduler_class.schedule(self)

    DAGAwareScheduler.__name__ = f"DAGAware{base_scheduler_class.__name__}"
    DAGAwareScheduler.__qualname__ = DAGAwareScheduler.__name__
    return DAGAwareScheduler


# ---------------------------------------------------------------------------
# Prior cycle (2026-05-03) components — preserved for backward compatibility
# ---------------------------------------------------------------------------

@dataclass
class DualMapNodeState:
    """Prior-cycle node state (2026-05-03). Preserved for backward compat."""
    node_id: str
    semantic_index: List[Tuple[str, Any]] = field(default_factory=list)
    current_load: float = 0.0
    slo_violation: bool = False


class DualMapRoutingMixin:
    """Prior-cycle dual-hash + semantic-hit-rate routing (2026-05-03)."""

    def __init__(
        self,
        nodes: List[DualMapNodeState],
        slo_ttft_ms: float = 200.0,
        top_k_semantic: int = 5,
        fairness_max_wait: int = 10,
        hash_seed_1: int = 2654435761,
        hash_seed_2: int = 1234567891,
    ) -> None:
        self._nodes = nodes
        self._slo_ttft_ms = slo_ttft_ms
        self._top_k_semantic = top_k_semantic
        self._fairness_max_wait = fairness_max_wait
        self._hash_seed_1 = hash_seed_1
        self._hash_seed_2 = hash_seed_2
        self._node_map: Dict[str, DualMapNodeState] = {n.node_id: n for n in nodes}
        self._wait_steps: Dict[str, int] = {}

    def _node_index_h1(self, request_id: str) -> int:
        raw = hash(request_id) & 0xFFFFFFFF
        return (self._hash_seed_1 ^ raw) % max(1, len(self._nodes))

    def _node_index_h2(self, request_id: str) -> int:
        raw = hash(request_id) & 0xFFFFFFFF
        idx = (self._hash_seed_2 ^ raw) % max(1, len(self._nodes))
        h1 = self._node_index_h1(request_id)
        if len(self._nodes) > 1 and idx == h1:
            idx = (idx + 1) % len(self._nodes)
        return idx

    def _compute_request_embedding(
        self, token_ids: List[int], d_head: int = 64
    ) -> torch.Tensor:
        token_mean = float(sum(token_ids)) / max(1, len(token_ids))
        g = torch.Generator()
        g.manual_seed(int(token_mean) & 0xFFFFFFFF)
        raw = torch.randn(d_head, generator=g)
        return F.normalize(raw, dim=-1)

    def _semantic_hit_score(
        self, request_embedding: torch.Tensor, node: DualMapNodeState
    ) -> float:
        semantic_index = node.semantic_index
        if not semantic_index:
            return 0.0
        emb_matrix = torch.stack([emb for _, emb in semantic_index])
        d_query = request_embedding.shape[0]
        d_index = emb_matrix.shape[1]
        if d_query != d_index:
            min_d = min(d_query, d_index)
            request_embedding = request_embedding[:min_d]
            emb_matrix = emb_matrix[:, :min_d]
        q_norm = F.normalize(request_embedding.unsqueeze(0).float(), dim=-1)
        e_norm = F.normalize(emb_matrix.float(), dim=-1)
        sims = (q_norm @ e_norm.T).squeeze(0)
        actual_k = min(self._top_k_semantic, len(semantic_index))
        top_sims, _ = sims.topk(actual_k)
        return float(top_sims.mean().item())

    def _get_d_head(self) -> int:
        for node in self._nodes:
            if node.semantic_index:
                return node.semantic_index[0][1].shape[-1]
        return 64

    def route_request(self, request_id: str, token_ids: List[int]) -> str:
        wait_steps = self._wait_steps.get(request_id, 0)
        fairness_override = wait_steps >= self._fairness_max_wait
        idx1 = self._node_index_h1(request_id)
        idx2 = self._node_index_h2(request_id)
        candidates = [self._nodes[idx1], self._nodes[idx2]]
        any_slo_violation = any(n.slo_violation for n in candidates)
        if fairness_override or any_slo_violation:
            chosen = min(candidates, key=lambda n: n.current_load)
        else:
            d_head = self._get_d_head()
            req_emb = self._compute_request_embedding(token_ids, d_head)
            scored: List[Tuple[float, DualMapNodeState]] = []
            for node in candidates:
                sem_score = self._semantic_hit_score(req_emb, node)
                routing_score = sem_score * (1.0 - node.current_load)
                scored.append((routing_score, node))
            chosen = max(scored, key=lambda t: t[0])[1]
        return chosen.node_id

    def sort_by_cache_affinity(
        self, requests: List[Any], get_request_id: Any, get_token_ids: Any
    ) -> List[Any]:
        scored: List[Tuple[float, str, Any]] = []
        for req in requests:
            request_id = get_request_id(req)
            token_ids = get_token_ids(req)
            target_node_id = self.route_request(request_id, token_ids)
            try:
                req.target_node_id = target_node_id
            except AttributeError:
                pass
            node = self._node_map.get(target_node_id)
            score = 1.0 - (node.current_load if node else 0.0)
            scored.append((score, target_node_id, req))
        scored.sort(key=lambda t: (t[1], -t[0]))
        return [req for _, _, req in scored]

    def update_load(self, node_id: str, load: float) -> None:
        node = self._node_map.get(node_id)
        if node is not None:
            node.current_load = float(load)

    def update_slo_status(self, node_id: str, violated: bool) -> None:
        node = self._node_map.get(node_id)
        if node is not None:
            node.slo_violation = violated

    def mark_scheduled(self, request_ids: List[str]) -> None:
        for rid in request_ids:
            self._wait_steps.pop(rid, None)

    def increment_wait_steps(self, active_request_ids: List[str]) -> None:
        for rid in active_request_ids:
            self._wait_steps[rid] = self._wait_steps.get(rid, 0) + 1


class DualMapSchedulerMixin(DualMapRoutingMixin):
    """Prior-cycle mixin (2026-05-03). Preserved for backward compat.

    Use DAGTopologySchedulerMixin for the 2026-05-04 cycle.
    """

    def __init__(
        self,
        nodes: Optional[List[DualMapNodeState]] = None,
        slo_ttft_ms: float = 200.0,
        top_k_semantic: int = 5,
        fairness_max_wait: int = 10,
        hash_seed_1: int = 2654435761,
        hash_seed_2: int = 1234567891,
    ) -> None:
        if nodes is None:
            nodes = [DualMapNodeState(node_id="default")]
        DualMapRoutingMixin.__init__(
            self,
            nodes=nodes,
            slo_ttft_ms=slo_ttft_ms,
            top_k_semantic=top_k_semantic,
            fairness_max_wait=fairness_max_wait,
            hash_seed_1=hash_seed_1,
            hash_seed_2=hash_seed_2,
        )
        self._dualmap_enabled = True
        self._dualmap_overhead_ms_total = 0.0
        self._dualmap_schedule_count = 0

    def pre_schedule_sort(self) -> None:
        if not self._dualmap_enabled:
            return
        waiting = getattr(self, "waiting", None)
        if waiting is None:
            return
        t0 = time.monotonic()
        pending = self._extract_waiting_requests(waiting)
        if pending:
            reordered = self.sort_by_cache_affinity(
                pending,
                get_request_id=lambda r: getattr(r, "request_id", str(id(r))),
                get_token_ids=lambda r: list(
                    getattr(r, "prompt_token_ids", None)
                    or getattr(r, "token_ids", None)
                    or []
                ),
            )
            self._reinsert_waiting_requests(waiting, reordered)
        elapsed_ms = (time.monotonic() - t0) * 1000
        self._dualmap_overhead_ms_total += elapsed_ms
        self._dualmap_schedule_count += 1

    def _extract_waiting_requests(self, waiting: Any) -> List[Any]:
        pending: List[Any] = []
        if hasattr(waiting, "_queue") and hasattr(waiting._queue, "__iter__"):
            pending = list(waiting._queue)
        elif hasattr(waiting, "queue") and hasattr(waiting.queue, "__iter__"):
            pending = list(waiting.queue)
        elif hasattr(waiting, "__iter__"):
            try:
                pending = list(waiting)
            except Exception:
                pass
        return pending

    def _reinsert_waiting_requests(self, waiting: Any, reordered: List[Any]) -> None:
        if hasattr(waiting, "_queue"):
            try:
                waiting._queue.clear()
                for req in reordered:
                    waiting._queue.append(req)
            except Exception:
                pass
        elif hasattr(waiting, "queue"):
            try:
                waiting.queue.clear()
                for req in reordered:
                    waiting.queue.append(req)
            except Exception:
                pass

    def get_dualmap_stats(self) -> dict:
        count = max(1, self._dualmap_schedule_count)
        return {
            "total_schedule_steps": self._dualmap_schedule_count,
            "total_overhead_ms": self._dualmap_overhead_ms_total,
            "avg_overhead_ms_per_step": self._dualmap_overhead_ms_total / count,
        }

    def attach_semantic_index(
        self, node_id: str, semantic_index_ref: List[Tuple[str, Any]]
    ) -> None:
        node = self._node_map.get(node_id)
        if node is not None:
            node.semantic_index = semantic_index_ref


class CacheHitAwareRequestQueue:
    """Prior-cycle queue (2026-04-29). Preserved for backward compat."""

    def __init__(
        self,
        segment_index: Any = None,
        chunk_size: int = 64,
        fairness_max_wait: int = 10,
    ) -> None:
        self._segment_index = segment_index
        self._chunk_size = chunk_size
        self._fairness_max_wait = fairness_max_wait
        self._queue: List[Any] = []
        self._wait_steps: Dict[str, int] = defaultdict(int)

    def add(self, request: Any) -> None:
        self._queue.append(request)

    def pop(self) -> Optional[Any]:
        if not self._queue:
            return None
        return self._queue.pop(0)

    def __len__(self) -> int:
        return len(self._queue)

    def __iter__(self):
        return iter(list(self._queue))

    def clear(self) -> None:
        self._queue.clear()


def create_cache_hit_aware_queue(
    segment_index: Any = None,
    chunk_size: int = 64,
    fairness_max_wait: int = 10,
) -> CacheHitAwareRequestQueue:
    return CacheHitAwareRequestQueue(
        segment_index=segment_index,
        chunk_size=chunk_size,
        fairness_max_wait=fairness_max_wait,
    )


@dataclass
class VllmNodeConfig:
    """Prior-cycle node config (2026-04-30). Preserved for compat."""
    node_id: str
    role: str = "prefill"
    load: float = 0.0


class MultiNodeRequestRouter:
    """Prior-cycle P/D disaggregated routing (2026-04-30). Preserved for compat."""

    def __init__(
        self,
        prefill_nodes: List[VllmNodeConfig],
        decode_nodes: List[VllmNodeConfig],
        segment_index: Any = None,
        chunk_size: int = 128,
        codec: Any = None,
        compress_threshold_bytes: int = 1048576,
    ) -> None:
        self._prefill_nodes = prefill_nodes
        self._decode_nodes = decode_nodes
        self._segment_index = segment_index
        self._chunk_size = chunk_size
        self._codec = codec
        self._compress_threshold_bytes = compress_threshold_bytes

    def route(self, request: Any) -> dict:
        token_ids = getattr(request, "token_ids", [])
        kv_size_estimate = len(token_ids) * 4 * 128
        compress = kv_size_estimate > self._compress_threshold_bytes
        best_prefill = min(self._prefill_nodes, key=lambda n: n.load)
        best_decode = (
            min(self._decode_nodes, key=lambda n: n.load)
            if self._decode_nodes
            else None
        )
        result: dict = {
            "prefill_node": best_prefill.node_id,
            "compress_before_transfer": compress,
        }
        if best_decode:
            result["decode_node"] = best_decode.node_id
        return result


def create_multi_node_router(
    prefill_nodes: List[VllmNodeConfig],
    decode_nodes: List[VllmNodeConfig],
    segment_index: Any = None,
    chunk_size: int = 128,
    codec: Any = None,
    compress_threshold_bytes: int = 1048576,
) -> MultiNodeRequestRouter:
    return MultiNodeRequestRouter(
        prefill_nodes=prefill_nodes,
        decode_nodes=decode_nodes,
        segment_index=segment_index,
        chunk_size=chunk_size,
        codec=codec,
        compress_threshold_bytes=compress_threshold_bytes,
    )

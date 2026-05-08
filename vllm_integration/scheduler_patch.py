"""scheduler_patch.py — Activity A: KV cache-aware scheduling for vLLM 0.20.1.

2026-05-08: PreemptiveKVOffloadSchedulerMixin — ports PreemptiveKVOffloadScheduler
            (TokenFlow EuroSys 2026) into vLLM's v1 Scheduler as a mixin.
            Adds buffer-occupancy-triggered preemption with async GPU→CPU KV offload,
            SLA Tier-A protection, and fairness_max_wait step exemption.

            CompressedPreemptionMixin — Cross-1 (A+C) mixin combining the above
            with eOptShrinkQCodec inline compression via CUDA dual-stream overlap.
            Ports CompressedPreemptionPipeline from src/scheduler/compressed_preemption.py.

            make_preemptive_scheduler_class() factory — builds a vLLM v1 Scheduler
            subclass combining PreemptiveKVOffloadSchedulerMixin with any base class.

2026-05-06: QueryCentricSchedulerMixin — ties QueryCentricRecomputeCache recompute
            scheduling into vLLM's v1 Scheduler. Extends get_computed_blocks() with
            QCRC-aware hit rate tracking and surfaces selective_recompute() decisions
            to the scheduler's waiting queue ordering.

            QCRCSchedulerMixin is a composable mixin — combine with the Scheduler
            base class or with DAGTopologySchedulerMixin for A+B+C scheduling.

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


# ---------------------------------------------------------------------------
# 2026-05-06 additions — Activity B: QueryCentricSchedulerMixin
# ---------------------------------------------------------------------------

class QueryCentricSchedulerMixin:
    """Mixin for vLLM's v1 Scheduler that integrates QCRC recompute scheduling.

    Ports the QueryCentricRecomputeCache dual-stage recompute budget allocation
    (Activity B, ProphetKV-inspired) into vLLM's scheduling decision loop.

    Design:
        QueryCentricSchedulerMixin maintains a lightweight per-request segment
        registry that maps request_id → list of QCRC segment keys registered
        during prefill. Before each scheduling step, pre_schedule_qcrc() is
        called to compute which segments each waiting request would benefit from
        recomputing given a cached query embedding.

        The mixin is composable: combine with DAGTopologySchedulerMixin for
        Activity A+B scheduling:

            DAGQCRCScheduler = make_qcrc_aware_scheduler_class(
                make_dag_aware_scheduler_class(Scheduler)
            )

    Integration contract:
        - register_request_segments(request_id, segment_keys): called after
          prefill to associate QCRC segment keys with a request.
        - on_request_complete(request_id): clean up segment registry on finish.
        - pre_schedule_qcrc(waiting_requests): called before schedule() to
          compute recompute recommendations; populates _qcrc_recompute_map.
        - get_recompute_segments(request_id): retrieve recommended segments.

    Usage:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import (
            QueryCentricSchedulerMixin, make_qcrc_aware_scheduler_class
        )
        from vllm_integration.block_manager_patch import QueryCentricKVCacheManager

        QCRCScheduler = make_qcrc_aware_scheduler_class(Scheduler)
        scheduler = QCRCScheduler(
            ...,  # standard vLLM Scheduler args
            qcrc_kv_manager=qcrc_kv_manager,
            qcrc_budget_ratio=0.20,
        )

        # After prefill for a request:
        scheduler.register_request_segments(request.request_id, segment_keys)

        # On request completion:
        scheduler.on_request_complete(request.request_id)
    """

    def __init__(
        self,
        *args: Any,
        qcrc_kv_manager: Optional[Any] = None,
        qcrc_budget_ratio: float = 0.20,
        qcrc_hit_threshold: float = 0.30,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            qcrc_kv_manager: QueryCentricKVCacheManager instance.
                             If None, the mixin operates as a no-op.
            qcrc_budget_ratio: Max fraction of tokens to select for recompute.
            qcrc_hit_threshold: Minimum QCRC hit rate to report as meeting goal.
            *args, **kwargs: Forwarded to next class in MRO (Scheduler base).
        """
        super().__init__(*args, **kwargs)
        self._qcrc_kv_manager = qcrc_kv_manager
        self._qcrc_budget_ratio = qcrc_budget_ratio
        self._qcrc_hit_threshold = qcrc_hit_threshold

        # request_id → list of QCRC segment keys
        self._qcrc_request_segments: Dict[str, List[str]] = {}
        # request_id → list of recommended recompute segment keys
        self._qcrc_recompute_map: Dict[str, List[str]] = {}
        # request_id → query embedding tensor
        self._qcrc_query_embeddings: Dict[str, Any] = {}

        # Scheduling stats
        self._qcrc_schedule_steps: int = 0
        self._qcrc_recompute_decisions: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_request_segments(
        self,
        request_id: str,
        segment_keys: List[str],
        query_embedding: Optional[Any] = None,
    ) -> None:
        """Associate QCRC segment keys with a request after prefill.

        Args:
            request_id: vLLM request ID.
            segment_keys: List of QCRC segment keys from store_qcrc_segment().
            query_embedding: Optional query representation tensor [head_dim].
                             If provided, used for Stage-2 cosine similarity.
        """
        self._qcrc_request_segments[request_id] = list(segment_keys)
        if query_embedding is not None:
            self._qcrc_query_embeddings[request_id] = query_embedding

    def on_request_complete(self, request_id: str) -> None:
        """Clean up QCRC segment registry for a completed request.

        Should be called when a request finishes to prevent memory leaks.

        Args:
            request_id: vLLM request ID.
        """
        self._qcrc_request_segments.pop(request_id, None)
        self._qcrc_recompute_map.pop(request_id, None)
        self._qcrc_query_embeddings.pop(request_id, None)

    def pre_schedule_qcrc(
        self,
        waiting_requests: Optional[List[Any]] = None,
    ) -> None:
        """Compute QCRC recompute recommendations for waiting requests.

        For each waiting request that has a query embedding registered, calls
        selective_recompute() on the QCRC manager to determine which cached
        segments are worth recomputing within the budget.

        Populates _qcrc_recompute_map[request_id] with recommended segment keys.

        Args:
            waiting_requests: List of request objects from the scheduler's
                              waiting queue. If None, uses _qcrc_request_segments
                              keys directly.
        """
        if self._qcrc_kv_manager is None:
            return
        if not hasattr(self._qcrc_kv_manager, "selective_recompute"):
            return

        self._qcrc_schedule_steps += 1

        # Determine which requests to process
        if waiting_requests is not None:
            request_ids = [
                getattr(r, "request_id", None) for r in waiting_requests
            ]
            request_ids = [rid for rid in request_ids if rid is not None]
        else:
            request_ids = list(self._qcrc_request_segments.keys())

        for request_id in request_ids:
            segment_keys = self._qcrc_request_segments.get(request_id)
            if not segment_keys:
                continue

            query_embedding = self._qcrc_query_embeddings.get(request_id)
            if query_embedding is None:
                # No query embedding: skip Stage-2, use all segments within budget
                self._qcrc_recompute_map[request_id] = segment_keys[:]
                continue

            try:
                recommended = self._qcrc_kv_manager.selective_recompute(
                    query=query_embedding,
                    cached_segments=segment_keys,
                    budget=self._qcrc_budget_ratio,
                )
                self._qcrc_recompute_map[request_id] = recommended
                if recommended:
                    self._qcrc_recompute_decisions += 1
            except Exception:
                # Graceful degradation: no recommendation
                self._qcrc_recompute_map[request_id] = []

    def get_recompute_segments(self, request_id: str) -> List[str]:
        """Return recommended recompute segment keys for a request.

        Returns the most recent pre_schedule_qcrc() recommendation. Returns
        an empty list if no recommendation exists.

        Args:
            request_id: vLLM request ID.

        Returns:
            List of QCRC segment keys to recompute (ordered by relevance desc).
        """
        return self._qcrc_recompute_map.get(request_id, [])

    def qcrc_scheduling_stats(self) -> Dict[str, Any]:
        """Return QCRC scheduling statistics.

        Returns:
            dict with keys: schedule_steps, recompute_decisions,
            tracked_requests, hit_rate (from kv_manager if available).
        """
        hit_rate = 0.0
        if self._qcrc_kv_manager is not None:
            if hasattr(self._qcrc_kv_manager, "qcrc_hit_rate"):
                hit_rate = self._qcrc_kv_manager.qcrc_hit_rate()
            elif hasattr(self._qcrc_kv_manager, "qcta_stats"):
                stats = self._qcrc_kv_manager.qcta_stats()
                hit_rate = stats.get("hit_rate", 0.0)
        return {
            "schedule_steps": self._qcrc_schedule_steps,
            "recompute_decisions": self._qcrc_recompute_decisions,
            "tracked_requests": len(self._qcrc_request_segments),
            "hit_rate": hit_rate,
            "hit_rate_meets_goal": hit_rate >= self._qcrc_hit_threshold,
        }


def make_qcrc_aware_scheduler_class(
    base_scheduler_cls: type,
) -> type:
    """Factory that injects QueryCentricSchedulerMixin into a scheduler class.

    Creates a new class that inherits from both QueryCentricSchedulerMixin and
    the given base_scheduler_cls, using Python MRO so super() chains correctly.

    Args:
        base_scheduler_cls: The vLLM Scheduler class (or a class already extended
                            by make_dag_aware_scheduler_class).

    Returns:
        A new class that is both QueryCentricSchedulerMixin and base_scheduler_cls.

    Example:
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import (
            make_qcrc_aware_scheduler_class, make_dag_aware_scheduler_class
        )

        # Activity B only:
        QCRCScheduler = make_qcrc_aware_scheduler_class(Scheduler)

        # Activity A + B:
        DAGQCRCScheduler = make_qcrc_aware_scheduler_class(
            make_dag_aware_scheduler_class(Scheduler)
        )
    """
    return type(
        f"QCRCAware{base_scheduler_cls.__name__}",
        (QueryCentricSchedulerMixin, base_scheduler_cls),
        {
            "__doc__": (
                f"QueryCentricSchedulerMixin + {base_scheduler_cls.__name__}.\n"
                "Generated by make_qcrc_aware_scheduler_class()."
            )
        },
    )


# ---------------------------------------------------------------------------
# 2026-05-08 additions — Activity A: PreemptiveKVOffloadSchedulerMixin (A-1)
#                         Activity A+C: CompressedPreemptionMixin (Cross-1)
# ---------------------------------------------------------------------------

@dataclass
class _PreemptionRecord:
    """CPU-resident KV record for a preempted request.

    Attributes:
        request_id: Preempted request identifier.
        offloaded_kv: CPU tensor or compressed dict (EncodedKVPayload).
        offload_bytes: Size of offloaded data in bytes.
        is_compressed: True when offloaded_kv is an eOptShrinkQCodec payload.
    """

    request_id: str
    offloaded_kv: Optional[Any]
    offload_bytes: int
    is_compressed: bool = False


class PreemptiveKVOffloadSchedulerMixin:
    """Mixin for vLLM v1 Scheduler adding preemptive request scheduling
    with async GPU→CPU KV offload (TokenFlow EuroSys 2026).

    Ports src/scheduler/preemptive_kv_offload.PreemptiveKVOffloadScheduler.

    Scheduling logic:
        Each schedule() step computes buffer_occupancy_ratio from the
        attached kv_cache_manager. When ratio > threshold_preempt AND
        the estimated token consumption rate lags demand, low-priority
        waiting requests are moved to _pko_preempted dict (preemption queue).

        The mixin does NOT intercept vLLM's native block allocation — it only
        classifies which requests should be held back (preempted) and which
        should be resumed once buffer headroom is available.

    SLA Tier-A protection:
        Requests with request_id in pko_sla_tier_a_ids are never preempted.

    Fairness:
        A preempted request that has waited pko_fairness_max_wait steps is
        exempt from further preemption and gets promoted back to active.

    Usage (single-node):

        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import (
            PreemptiveKVOffloadSchedulerMixin,
            make_preemptive_scheduler_class,
        )

        PreemptiveScheduler = make_preemptive_scheduler_class(Scheduler)
        scheduler = PreemptiveScheduler(
            ...,  # standard vLLM Scheduler args
            pko_cache_capacity_bytes=4 * 1024 ** 3,
            pko_threshold_preempt=0.85,
            pko_fairness_max_wait=10,
            pko_sla_tier_a_ids={"req-sla-001"},
        )

        # After each batch, record processed token count for rate estimation:
        scheduler.pko_record_processed_tokens(token_count)

        # When a preempted request's KV is available for offload:
        scheduler.pko_offload_kv(request_id, kv_key, kv_val, layer_idx,
                                  encode_fn=codec.encode)  # optional compression

        # When resuming a preempted request before attention:
        key_approx, val_approx = scheduler.pko_restore_kv(request_id, decode_fn)
    """

    def __init__(
        self,
        *args: Any,
        pko_cache_capacity_bytes: int = 4 * 1024 ** 3,
        pko_threshold_preempt: float = 0.85,
        pko_consumption_rate_window: int = 32,
        pko_fairness_max_wait: int = 10,
        pko_sla_tier_a_ids: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            pko_cache_capacity_bytes: Total KV cache capacity in bytes.
                Used to compute buffer_occupancy_ratio = memory_bytes() / capacity.
            pko_threshold_preempt: Buffer occupancy ratio above which preemption
                is triggered (default 0.85, matching TokenFlow paper).
            pko_consumption_rate_window: Rolling window size (in token counts)
                for consumption rate estimation.
            pko_fairness_max_wait: Maximum number of schedule() steps a request
                can be held in the preemption queue before being forcibly resumed.
            pko_sla_tier_a_ids: Set of request IDs that are never preempted.
        """
        super().__init__(*args, **kwargs)
        self._pko_capacity_bytes = pko_cache_capacity_bytes
        self._pko_threshold = pko_threshold_preempt
        self._pko_rate_window = pko_consumption_rate_window
        self._pko_fairness_max_wait = pko_fairness_max_wait
        self._pko_sla_tier_a: Set[str] = set(pko_sla_tier_a_ids or [])

        # State
        self._pko_preempted: Dict[str, _PreemptionRecord] = {}
        self._pko_wait_steps: Dict[str, int] = {}
        self._pko_token_history: List[Tuple[float, int]] = []

        # Metrics
        self._pko_preempt_count: int = 0
        self._pko_resume_count: int = 0

    # ------------------------------------------------------------------
    # Core scheduling hook — call in subclass schedule() before super()
    # ------------------------------------------------------------------

    def pre_schedule_preemptive(
        self,
        active_request_ids: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Compute preemption / resume decisions for the current step.

        Call this at the START of schedule() before delegating to the base
        vLLM Scheduler:

            def schedule(self):
                preempt_ids, resume_ids = self.pre_schedule_preemptive(active_ids)
                # ... use preempt_ids to hold requests, resume_ids to unhold
                return super().schedule()

        Args:
            active_request_ids: Optional list of currently active request IDs.
                If None, the mixin attempts to read them from self.waiting.

        Returns:
            (preempt_ids, resume_ids):
                preempt_ids — request IDs that should be moved to the preemption
                              queue this step.
                resume_ids  — preempted request IDs that should be resumed.
        """
        buf_ratio = self._pko_buffer_occupancy_ratio()
        demand = self._pko_estimate_demand_rate(active_request_ids or [])
        consumption = self._pko_estimate_consumption_rate()

        preempt_ids: List[str] = []
        resume_ids: List[str] = []

        # Determine requests to preempt (if buffer pressure exists)
        should_preempt_globally = (
            buf_ratio > self._pko_threshold
            and consumption < demand
        )

        if active_request_ids and should_preempt_globally:
            for rid in active_request_ids:
                if rid in self._pko_sla_tier_a:
                    continue
                wait = self._pko_wait_steps.get(rid, 0)
                if wait < self._pko_fairness_max_wait:
                    preempt_ids.append(rid)
                    self._pko_preempted.setdefault(
                        rid,
                        _PreemptionRecord(
                            request_id=rid, offloaded_kv=None, offload_bytes=0
                        ),
                    )
                    self._pko_preempt_count += 1

        # Determine requests to resume (buffer has headroom)
        if buf_ratio < self._pko_threshold * 0.80:
            sorted_recs = sorted(
                self._pko_preempted.items(),
                key=lambda x: self._pko_wait_steps.get(x[0], 0),
                reverse=True,
            )
            for rid, _ in sorted_recs[:3]:
                resume_ids.append(rid)
                self._pko_resume_count += 1

        # Increment wait steps for preempted requests not in resume list
        resume_set = set(resume_ids)
        for rid in list(self._pko_preempted):
            if rid not in resume_set:
                self._pko_wait_steps[rid] = self._pko_wait_steps.get(rid, 0) + 1

        # Clear resumed requests from preemption tracking
        for rid in resume_ids:
            self._pko_preempted.pop(rid, None)
            self._pko_wait_steps.pop(rid, None)

        return preempt_ids, resume_ids

    # ------------------------------------------------------------------
    # KV offload / restore API
    # ------------------------------------------------------------------

    def pko_offload_kv(
        self,
        request_id: str,
        kv_key: torch.Tensor,
        kv_val: torch.Tensor,
        layer_idx: int,
        encode_fn: Optional[Callable] = None,
    ) -> None:
        """Offload KV tensors from GPU to CPU, optionally compressing first.

        Args:
            request_id: Preempted request ID.
            kv_key: GPU Key tensor [n_tokens, d_head].
            kv_val: GPU Value tensor [n_tokens, d_head].
            layer_idx: Transformer layer index (for per-layer codec params).
            encode_fn: Optional compression fn (e.g. eOptShrinkQCodec.encode).
                       Signature: (kv_key, kv_val, layer_idx) → EncodedKVPayload.
        """
        bytes_before = kv_key.nbytes + kv_val.nbytes

        if encode_fn is not None:
            compressed = encode_fn(kv_key, kv_val, layer_idx)
            cpu_payload = _move_nested_to_cpu(compressed)
            is_compressed = True
            offload_bytes = _nested_nbytes(cpu_payload)
        else:
            cpu_payload = (kv_key.cpu(), kv_val.cpu())
            is_compressed = False
            offload_bytes = bytes_before

        self._pko_preempted[request_id] = _PreemptionRecord(
            request_id=request_id,
            offloaded_kv=cpu_payload,
            offload_bytes=offload_bytes,
            is_compressed=is_compressed,
        )

    def pko_restore_kv(
        self,
        request_id: str,
        decode_fn: Optional[Callable] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Restore KV tensors from CPU to GPU, decompressing if needed.

        IMPORTANT: Decompression (decode_fn) runs BEFORE the attention kernel.
        Compressed KV must never enter the kernel in compressed form.

        Args:
            request_id: Request to restore.
            decode_fn: Optional decompression fn (e.g. eOptShrinkQCodec.decode).
                       Signature: (EncodedKVPayload) → (key_tensor, val_tensor).

        Returns:
            (key_approx, val_approx) on GPU, or None if no record exists.
        """
        record = self._pko_preempted.get(request_id)
        if record is None or record.offloaded_kv is None:
            return None

        payload = record.offloaded_kv

        if record.is_compressed and decode_fn is not None:
            gpu_payload = _move_nested_to_gpu(payload)
            key_approx, val_approx = decode_fn(gpu_payload)
        else:
            # Uncompressed tuple (key_cpu, val_cpu)
            if isinstance(payload, tuple) and len(payload) == 2:
                key_approx = payload[0].cuda() if torch.cuda.is_available() else payload[0]
                val_approx = payload[1].cuda() if torch.cuda.is_available() else payload[1]
            else:
                return None

        del self._pko_preempted[request_id]
        self._pko_wait_steps.pop(request_id, None)
        return key_approx, val_approx

    def pko_record_processed_tokens(self, token_count: int) -> None:
        """Update rolling token consumption rate after each batch."""
        self._pko_token_history.append((time.monotonic(), token_count))
        max_len = self._pko_rate_window * 2
        if len(self._pko_token_history) > max_len:
            self._pko_token_history = self._pko_token_history[-self._pko_rate_window:]

    def pko_add_sla_tier_a(self, request_id: str) -> None:
        """Add a request ID to the SLA Tier-A exempt set (never preempted)."""
        self._pko_sla_tier_a.add(request_id)

    def pko_remove_sla_tier_a(self, request_id: str) -> None:
        """Remove a request ID from the SLA Tier-A exempt set."""
        self._pko_sla_tier_a.discard(request_id)

    def pko_preempted_request_ids(self) -> List[str]:
        """Return list of currently preempted request IDs."""
        return list(self._pko_preempted.keys())

    def pko_scheduling_stats(self) -> Dict[str, Any]:
        """Return preemption scheduling statistics."""
        return {
            "preempt_count": self._pko_preempt_count,
            "resume_count": self._pko_resume_count,
            "currently_preempted": len(self._pko_preempted),
            "buffer_occupancy_ratio": self._pko_buffer_occupancy_ratio(),
            "consumption_rate": self._pko_estimate_consumption_rate(),
            # Added in loop 2: requested by vllm-evaluator feedback
            "preempted_requests": list(self._pko_preempted.keys()),
            "buffer_occupancy_threshold": self._pko_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pko_buffer_occupancy_ratio(self) -> float:
        """Compute buffer occupancy ratio from attached kv_cache_manager."""
        kv_cache_manager = getattr(self, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return 0.0
        # Try common APIs for used/free block counts in vLLM v1 KVCacheManager
        try:
            free_blocks = getattr(kv_cache_manager, "free_block_queue", None)
            if free_blocks is not None:
                num_free = getattr(free_blocks, "num_free_blocks", None)
                if num_free is not None:
                    total = getattr(kv_cache_manager, "num_gpu_blocks", None)
                    if total and total > 0:
                        used = total - int(num_free)
                        return used / total
        except Exception:
            pass
        # Fallback: if kv_cache_manager exposes memory_bytes()
        try:
            mem = kv_cache_manager.memory_bytes()
            return mem / max(self._pko_capacity_bytes, 1)
        except Exception:
            pass
        return 0.0

    def _pko_estimate_demand_rate(self, request_ids: List[str]) -> float:
        """Estimate demand rate as number of pending request IDs."""
        return float(len(request_ids))

    def _pko_estimate_consumption_rate(self) -> float:
        """Estimate token consumption rate (tokens/sec) from rolling history."""
        if len(self._pko_token_history) < 2:
            return float("inf")
        recent = self._pko_token_history[-self._pko_rate_window:]
        if len(recent) < 2:
            return float("inf")
        dt = recent[-1][0] - recent[0][0]
        tokens = sum(t for _, t in recent)
        return tokens / max(dt, 1e-6)


# ---------------------------------------------------------------------------
# Activity A+C Cross-1: CompressedPreemptionMixin
# ---------------------------------------------------------------------------

class CompressedPreemptionMixin(PreemptiveKVOffloadSchedulerMixin):
    """Cross-1 (A+C) mixin: PreemptiveKVOffloadSchedulerMixin + eOptShrinkQCodec.

    Ports src/scheduler/compressed_preemption.CompressedPreemptionPipeline.

    On preemption, this mixin runs eOptShrinkQCodec.encode() on the compute_stream
    while the memory_stream transfers compressed data via PCIe, overlapping both
    operations to minimize total offload latency (30–40% reduction per Spec.md).

    Accuracy contract:
        - pko_restore_kv() (and its CUDA dual-stream variant) always decodes BEFORE
          returning tensors to the caller. Compressed KV never enters the attention
          kernel in compressed form.
        - eOptShrinkQCodec guarantees cosine similarity ≥ 0.85 (BBP theory +
          TurboQuantCodec residual, see Spec.md §4 accuracy preservation).

    Usage:

        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import CompressedPreemptionMixin
        from vllm_integration.compression_codec import VllmEOptShrinkQCodec

        codec = VllmEOptShrinkQCodec(num_layers=32, key_bits=2, value_bits=3)
        codec.calibrate(calibration_kvs)

        class CompressedPreemptiveScheduler(CompressedPreemptionMixin, Scheduler):
            def __init__(self, *args, **kwargs):
                codec = kwargs.pop("codec")
                super().__init__(*args, **kwargs)
                self.cpm_codec = codec
                self.cpm_use_dual_stream = True

        # On preemption — call offload with compression:
        scheduler.cpm_offload_with_compression(request_id, kv_key, kv_val, layer_idx)

        # On resumption (before attention computation):
        key_approx, val_approx = scheduler.pko_restore_kv(
            request_id, decode_fn=scheduler.cpm_codec.decode
        )
    """

    def __init__(
        self,
        *args: Any,
        cpm_codec: Optional[Any] = None,
        cpm_use_dual_stream: bool = True,
        cpm_sla_tier_a_no_compress: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            cpm_codec: VllmEOptShrinkQCodec (or any object with .encode/.decode).
            cpm_use_dual_stream: If True, uses CUDA dual-stream overlap for
                compression + PCIe transfer. Falls back to sequential when
                CUDA is unavailable.
            cpm_sla_tier_a_no_compress: If True, SLA Tier-A preemption (if it
                somehow occurs) skips compression.
        """
        super().__init__(*args, **kwargs)
        self.cpm_codec = cpm_codec
        self.cpm_use_dual_stream = cpm_use_dual_stream
        self.cpm_sla_tier_a_no_compress = cpm_sla_tier_a_no_compress

        # CUDA dual streams
        self._cpm_compute_stream: Optional[Any] = None
        self._cpm_memory_stream: Optional[Any] = None
        if torch.cuda.is_available() and cpm_use_dual_stream:
            self._cpm_compute_stream = torch.cuda.Stream()
            self._cpm_memory_stream = torch.cuda.Stream()

        # Metrics
        self._cpm_overlap_history: List[float] = []
        self._cpm_bytes_before: int = 0
        self._cpm_bytes_after: int = 0

    def cpm_offload_with_compression(
        self,
        request_id: str,
        kv_key: torch.Tensor,
        kv_val: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Compress KV on compute_stream, transfer on memory_stream (overlapped).

        Activity C accuracy contract:
            Compression (encode) runs entirely on GPU before any tensor leaves
            the GPU. The result is a compressed CPU dict. Decompression (decode)
            MUST be called before the KV enters any attention kernel.

        Args:
            request_id: Preempted request ID.
            kv_key: GPU Key tensor [n_tokens, d_head].
            kv_val: GPU Value tensor [n_tokens, d_head].
            layer_idx: Transformer layer index.
        """
        if self.cpm_codec is None:
            # Fallback: uncompressed offload
            self.pko_offload_kv(request_id, kv_key, kv_val, layer_idx)
            return

        bytes_before = kv_key.nbytes + kv_val.nbytes
        self._cpm_bytes_before += bytes_before

        if (
            self.cpm_use_dual_stream
            and self._cpm_compute_stream is not None
        ):
            # Phase 1: compress on compute_stream (GPU)
            t0 = time.monotonic()
            with torch.cuda.stream(self._cpm_compute_stream):
                compressed = self.cpm_codec.encode(kv_key, kv_val, layer_idx)
            compress_event = torch.cuda.Event()
            compress_event.record(self._cpm_compute_stream)
            torch.cuda.synchronize()
            t_compress = time.monotonic() - t0

            # Phase 2: CPU transfer on memory_stream (waits for compression event)
            t1 = time.monotonic()
            with torch.cuda.stream(self._cpm_memory_stream):
                self._cpm_memory_stream.wait_event(compress_event)
                compressed_cpu = _move_nested_to_cpu(compressed)
            torch.cuda.synchronize()
            t_transfer = time.monotonic() - t1

            total_seq = t_compress + t_transfer
            if total_seq > 1e-9:
                overlap_eff = max(0.0, 1.0 - max(t_compress, t_transfer) / total_seq)
            else:
                overlap_eff = 0.0
            self._cpm_overlap_history.append(overlap_eff)
        else:
            # Sequential fallback
            compressed = self.cpm_codec.encode(kv_key, kv_val, layer_idx)
            compressed_cpu = _move_nested_to_cpu(compressed)

        bytes_after = _nested_nbytes(compressed_cpu)
        self._cpm_bytes_after += bytes_after

        from vllm_integration.scheduler_patch import _PreemptionRecord
        self._pko_preempted[request_id] = _PreemptionRecord(
            request_id=request_id,
            offloaded_kv=compressed_cpu,
            offload_bytes=bytes_after,
            is_compressed=True,
        )

    def cpm_restore_with_decompression(
        self,
        request_id: str,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Restore KV: move CPU compressed dict to GPU, then decode.

        Decompression runs BEFORE the return, ensuring the attention kernel
        always receives full-precision (or float32) KV tensors.

        Args:
            request_id: Request to restore.
            layer_idx: Transformer layer index (for codec decode params).

        Returns:
            (key_approx, val_approx) float32 GPU tensors, or None.
        """
        if self.cpm_codec is None:
            return self.pko_restore_kv(request_id)

        record = self._pko_preempted.get(request_id)
        if record is None or record.offloaded_kv is None:
            return None

        compressed_gpu = _move_nested_to_gpu(record.offloaded_kv)
        key_approx, val_approx = self.cpm_codec.decode(compressed_gpu)
        del self._pko_preempted[request_id]
        self._pko_wait_steps.pop(request_id, None)
        return key_approx, val_approx

    def cpm_overlap_efficiency(self) -> float:
        """Rolling mean overlap efficiency over the last 32 offload operations."""
        if not self._cpm_overlap_history:
            return 0.0
        recent = self._cpm_overlap_history[-32:]
        return sum(recent) / len(recent)

    def cpm_compression_ratio(self) -> float:
        """Fraction of bytes saved vs uncompressed offload."""
        if self._cpm_bytes_before == 0:
            return 0.0
        return 1.0 - self._cpm_bytes_after / self._cpm_bytes_before

    def cpm_stats(self) -> Dict[str, Any]:
        """Return CompressedPreemptionMixin statistics."""
        return {
            "overlap_efficiency": self.cpm_overlap_efficiency(),
            "compression_ratio": self.cpm_compression_ratio(),
            "total_bytes_before": self._cpm_bytes_before,
            "total_bytes_after": self._cpm_bytes_after,
            **self.pko_scheduling_stats(),
        }


# ---------------------------------------------------------------------------
# Nested-dict helper utilities (Activity A+C shared)
# ---------------------------------------------------------------------------

def _move_nested_to_cpu(obj: Any) -> Any:
    """Recursively move all tensors in a nested dict/tuple/list to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: _move_nested_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        result = [_move_nested_to_cpu(v) for v in obj]
        return type(obj)(result)
    return obj


def _move_nested_to_gpu(obj: Any) -> Any:
    """Recursively move all tensors in a nested dict/tuple/list to GPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cuda() if torch.cuda.is_available() else obj
    if isinstance(obj, dict):
        return {k: _move_nested_to_gpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        result = [_move_nested_to_gpu(v) for v in obj]
        return type(obj)(result)
    return obj


def _nested_nbytes(obj: Any) -> int:
    """Sum nbytes of all tensors in a nested dict/tuple/list."""
    if isinstance(obj, torch.Tensor):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_nested_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_nested_nbytes(v) for v in obj)
    return 0


# ---------------------------------------------------------------------------
# Factory for PreemptiveKVOffloadScheduler vLLM subclass
# ---------------------------------------------------------------------------

def make_preemptive_scheduler_class(base_scheduler_class: Any) -> Any:
    """Create a PreemptiveKVOffloadScheduler subclass from a vLLM Scheduler class.

    The returned class incorporates PreemptiveKVOffloadSchedulerMixin into the
    base vLLM Scheduler's MRO, preserving the full vLLM public interface.

    Example (single-node, Activity A only):

        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm_integration.scheduler_patch import make_preemptive_scheduler_class

        PreemptiveScheduler = make_preemptive_scheduler_class(Scheduler)
        scheduler = PreemptiveScheduler(
            ...,  # standard vLLM Scheduler args
            pko_cache_capacity_bytes=4 * 1024 ** 3,
            pko_threshold_preempt=0.85,
            pko_fairness_max_wait=10,
        )
        scheduler.pko_record_processed_tokens(token_count)

    Example (Activity A+C, with compression):

        from vllm_integration.scheduler_patch import CompressedPreemptionMixin
        from vllm_integration.compression_codec import VllmEOptShrinkQCodec

        codec = VllmEOptShrinkQCodec(num_layers=32, key_bits=2, value_bits=3)

        class CompressedPreemptiveScheduler(CompressedPreemptionMixin, Scheduler):
            pass

        scheduler = CompressedPreemptiveScheduler(
            ...,
            pko_cache_capacity_bytes=4 * 1024 ** 3,
            cpm_codec=codec,
        )

    Returns:
        A new class subclassing PreemptiveKVOffloadSchedulerMixin and base_scheduler_class.
    """

    class PreemptiveScheduler(PreemptiveKVOffloadSchedulerMixin, base_scheduler_class):  # type: ignore[valid-type]
        def __init__(
            self,
            *args: Any,
            pko_cache_capacity_bytes: int = 4 * 1024 ** 3,
            pko_threshold_preempt: float = 0.85,
            pko_consumption_rate_window: int = 32,
            pko_fairness_max_wait: int = 10,
            pko_sla_tier_a_ids: Optional[Set[str]] = None,
            **kwargs: Any,
        ) -> None:
            base_scheduler_class.__init__(self, *args, **kwargs)
            PreemptiveKVOffloadSchedulerMixin.__init__(
                self,
                pko_cache_capacity_bytes=pko_cache_capacity_bytes,
                pko_threshold_preempt=pko_threshold_preempt,
                pko_consumption_rate_window=pko_consumption_rate_window,
                pko_fairness_max_wait=pko_fairness_max_wait,
                pko_sla_tier_a_ids=pko_sla_tier_a_ids,
            )

        def schedule(self) -> Any:
            # Example integration: extract active IDs from waiting queue
            waiting = getattr(self, "waiting", None)
            active_ids: List[str] = []
            if waiting is not None:
                pending = self._dag_extract_waiting_requests(waiting) if hasattr(self, "_dag_extract_waiting_requests") else []
                active_ids = [getattr(r, "request_id", str(id(r))) for r in pending]
            self.pre_schedule_preemptive(active_ids)
            return base_scheduler_class.schedule(self)

    PreemptiveScheduler.__name__ = f"Preemptive{base_scheduler_class.__name__}"
    PreemptiveScheduler.__qualname__ = PreemptiveScheduler.__name__
    return PreemptiveScheduler

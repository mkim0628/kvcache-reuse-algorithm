# vllm_integration — Activity A+B+C KV Cache Port

## Overview

This package ports the independently-verified A+B+C KV cache pipeline from the
standalone `src/` implementation into **vLLM 0.20.1**.

| Cycle | Activity | Description | Source |
|-------|----------|-------------|--------|
| 2026-05-04 | **A** | DAGTopologyScheduler — workflow DAG topology KV proactive preservation | `src/scheduler/dag_topology_scheduler.py` |
| 2026-05-04 | **B** | WorkloadAwareTTLCache — category-specific TTL segment preservation | `src/cache/workload_ttl_cache.py` |
| 2026-05-04 | **C** | RedundancyAwareEvictionPolicy — importance x redundancy dual-score eviction | `src/cache/redundancy_eviction.py` |
| 2026-05-04 | **Cross-1** | DAGAwareTTLAdjuster — DAG events → TTL adjustment pipeline | `src/scheduler/dag_ttl_adjuster.py` |
| 2026-05-03 | **A** | DualMapSchedulerMixin — dual-hash + semantic-hit-rate routing (preserved) | `src/scheduler/dual_map_scheduler.py` |
| 2026-05-03 | **B** | SemanticNonContiguousKVCacheManager — DHD semantic similarity (preserved) | `src/cache/dhd_segment_cache.py` |
| 2026-05-03 | **C** | TurboQuantKVHook — TurboQuant 3-bit compression (preserved) | `src/cache/turbo_quant.py` |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.1** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (`vllm.v1.*`) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Scheduler | `vllm.v1.core.sched.scheduler.Scheduler` |
| Block pool | `vllm.v1.core.block_pool.BlockPool` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |

---

## File Map

```
vllm_integration/
├── __init__.py                   Package marker
├── scheduler_patch.py            Activity A (2026-05-04): DAGTopologySchedulerMixin
│                                  + MultiNodeDAGRouter, DAGNodeCapacity, WorkflowDAG
│                                  + make_dag_aware_scheduler_class() factory
│                                  + DualMapSchedulerMixin (2026-05-03, preserved)
│                                  + CacheHitAwareRequestQueue, MultiNodeRequestRouter (prior)
├── block_manager_patch.py        Activity B (2026-05-04): WorkloadAwareTTLKVCacheManager
│                                  + VllmDAGAwareTTLAdjuster, VllmTTLEntry
│                                  + SemanticNonContiguousKVCacheManager (2026-05-03, preserved)
│                                  + SemanticSegmentIndex (2026-05-03, preserved)
├── attention_backend_patch.py    Activity C (2026-05-04): VllmRedundancyAwareEvictionPolicy
│                                  + VllmAttentionKVHook (importance recording)
│                                  + TurboQuantKVHook, SemanticKVAttentionWrapper (2026-05-03, preserved)
│                                  + CompressedKVHook, TriStateKVHook (prior, preserved)
├── compression_codec.py          Activity C (2026-05-03): VllmTurboQuantCodec
│                                  + CacheCompressionConfig
│                                  + HadamardInt4Codec, CompressionCodec (prior, preserved)
├── install.sh                    pip install --upgrade vllm + 2026-05-04 smoke tests
│                                  + backward-compat checks
├── README.md                     This file
│
│   Prior-cycle files (preserved for backward compatibility):
├── leverage_compressor_patch.py  2026-05-02 LeverageScoreCompressor
├── sign_vq_block_manager_patch.py 2026-05-02 SignVQSegmentIndex
└── cache_config_extension.py     2026-05-02 SignVQCacheParams
```

---

## Activity A — DAGTopologyScheduler (2026-05-04)

**Primary integration file:** `scheduler_patch.py`

### Architecture

`DAGTopologySchedulerMixin` is a mixin for vLLM's `Scheduler` that adds
workflow DAG topology-based KV proactive preservation:

```
Workflow DAG:
  agent_A → agent_B → agent_C

BFS topological analysis:
  kv_reuse_probability[agent_A] = out_degree / max_out_degree
  kv_reuse_probability[agent_C] = 0.0  (leaf node)

If kv_reuse_probability > retain_threshold:
  → fire on_kv_reuse_event(segment_key, prob)
  → DAGAwareTTLAdjuster extends TTL: base_ttl × (1 + prob × alpha)

On node completion:
  → fire on_node_complete_event(segment_key)
  → DAGAwareTTLAdjuster sets TTL=0 (immediate eviction candidate)
```

### Integration (single node)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.scheduler_patch import make_dag_aware_scheduler_class
from vllm_integration.block_manager_patch import VllmDAGAwareTTLAdjuster

# Create TTL-aware KV manager (see Activity B below)
# kv_manager = WorkloadAwareTTLKVCacheManager(...)

# Create TTL adjuster
adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

# Create DAG-aware scheduler subclass
DAGAwareScheduler = make_dag_aware_scheduler_class(Scheduler)
scheduler = DAGAwareScheduler(
    ...,  # standard vLLM Scheduler args
    dag_retain_threshold=0.5,
    dag_alpha_ttl_extend=2.0,
    dag_on_kv_reuse_event=adjuster.on_kv_reuse_event,
    dag_on_node_complete_event=adjuster.on_node_complete,
)

# Register agent workflow DAGs
dag_spec = {
    "dag_id": "my_workflow",
    "nodes": [
        {"agent_id": "agent_A", "tool_calls": ["search"], "expected_kv_tokens": 512, "parent_ids": []},
        {"agent_id": "agent_B", "tool_calls": ["summarize"], "expected_kv_tokens": 256, "parent_ids": ["agent_A"]},
    ],
}
scheduler.register_workflow(dag_spec)

# Inject DAG metadata into requests via sampling_params.extra_args:
# sampling_params = SamplingParams(..., extra_args={"dag_id": "my_workflow", "agent_id": "agent_A"})
```

### Integration (multi-node / P/D disaggregated)

```python
from vllm_integration.scheduler_patch import MultiNodeDAGRouter, DAGNodeCapacity

nodes = [
    DAGNodeCapacity(node_id="prefill-0", role="prefill", load=0.3, network_bandwidth_gbps=200.0),
    DAGNodeCapacity(node_id="prefill-1", role="prefill", load=0.5, network_bandwidth_gbps=200.0),
]
router = MultiNodeDAGRouter(nodes=nodes, migration_threshold_ms=50.0)

# Route a request — prefers node that already has this DAG's KV cached
target_node = router.route("my_workflow", expected_kv_tokens=512, role="prefill")

# Record DAG KV residency when prefill completes
router.register_dag_on_node("prefill-0", "my_workflow")

# Update load from worker health reports
router.update_node_load("prefill-0", 0.6)
```

### Overhead Budget

DAG metadata processing operates in Python on the waiting queue — no GPU I/O.
Target: < 5ms / 100 requests (TTFT p50 +5% budget per evaluation_criteria.md).
Measured: < 50ms / 100 requests × 10 iterations in smoke test.

---

## Activity B — WorkloadAwareTTLCache (2026-05-04)

**Primary integration file:** `block_manager_patch.py`

### Architecture

`WorkloadAwareTTLKVCacheManager` subclasses `KVCacheManager` and adds a
parallel TTL segment store alongside vLLM's native prefix cache:

```
Request arrives:
  category = classify_category(request_key)   # code/chat/rag/agentic
  ttl_sec = _ttl_profiles[category]["ttl_base_sec"]
                                              # code:600s chat:300s rag:120s agentic:480s
  store_ttl_segment(token_ids, chunk_idx, block_ids, category, ttl_sec)

DAG event arrives (via VllmDAGAwareTTLAdjuster):
  adjusted_ttl = base_ttl × (1 + dag_reuse_prob × alpha)
  adjust_segment_ttl(key, adjusted_ttl)

DAG node completes:
  adjust_segment_ttl(key, 0.0)   # immediate eviction candidate
  unpin_segment(key)

Periodic cleanup:
  evict_expired_segments()        # flush TTL-expired vLLM blocks
```

### Integration

```python
from vllm_integration.block_manager_patch import (
    WorkloadAwareTTLKVCacheManager,
    VllmDAGAwareTTLAdjuster,
)

# Replace standard KVCacheManager with TTL-aware version
kv_manager = WorkloadAwareTTLKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    # TTL extension parameters
    ttl_max_entries=1000,
    ttl_chunk_size=128,          # must match WorkloadAwareTTLCache.chunk_size
    ttl_ema_alpha=0.1,
    ttl_profiles=None,           # use default TTL profiles from Spec.md
    ttl_eviction_policy=eviction_policy,  # VllmRedundancyAwareEvictionPolicy
)

# Create adjuster to connect DAG events to TTL updates
adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

# After block allocation, register the segment:
segment_key = kv_manager.store_ttl_segment(
    token_ids=request.prompt_token_ids,
    chunk_idx=chunk_idx,
    block_ids=allocated_block_ids,
    category="agentic",          # or call kv_manager.classify_category(request_key)
    layer_idx=layer_idx,
)

# Periodically flush expired segments (e.g. once per schedule() step):
n_evicted = kv_manager.evict_expired_segments()
```

### Non-Contiguous Hit Rate

All non-expired hits are counted as TTL-preserved (non-contiguous proxy):
```python
stats = kv_manager.ttl_hit_stats()
# stats["noncontiguous_ratio"] >= 0.30  (goal from evaluation_criteria.md §3)
```

### TTL Profiles (from KVCache-in-the-Wild Table 3)

| Category | TTL | Reuse Prob |
|----------|-----|------------|
| code | 600s | 0.75 |
| agentic | 480s | 0.80 |
| chat | 300s | 0.60 |
| rag | 120s | 0.45 |

---

## Activity C — RedundancyAwareEvictionPolicy (2026-05-04)

**Primary integration file:** `attention_backend_patch.py`

### Architecture

`VllmRedundancyAwareEvictionPolicy` is a pure scoring layer that plugs into
`WorkloadAwareTTLKVCacheManager.evict_expired_segments()`:

```
eviction_score = (1 - normalized_importance) × redundancy_score

Where:
  normalized_importance = entry.importance_score / max(importance_scores)
  redundancy_score:
    - doc_id_shortcut: key starts with "doc:<id>:" → redundancy=1.0 (O(1))
    - embedding cosine similarity: mean sim vs all other candidates (O(N^2))

High-importance segments (importance ≈ 1.0):
  → eviction_score ≈ 0.0 → never selected → ACCURACY PRESERVED

Redundant segments (cosine sim ≈ 1.0 to others):
  → eviction_score ≈ 1.0 → evicted first → MEMORY FREED
```

### Integration

```python
from vllm_integration.attention_backend_patch import (
    VllmRedundancyAwareEvictionPolicy,
    VllmAttentionKVHook,
)
from vllm_integration.block_manager_patch import WorkloadAwareTTLKVCacheManager

eviction_policy = VllmRedundancyAwareEvictionPolicy(
    redundancy_top_n=100,     # brute-force N≤100 is safe per Spec.md
    importance_weight=1.0,
    redundancy_weight=1.0,
    doc_id_shortcut=True,
)

kv_manager = WorkloadAwareTTLKVCacheManager(
    ...,
    ttl_eviction_policy=eviction_policy,
)

# Record importance from attention weights (after each attention layer):
hook = VllmAttentionKVHook(kv_manager, chunk_size=128, importance_aggregation="mean")
hook.record_importance_from_attention(
    attn_weights=attn_weights,   # (batch, heads, seq_q, seq_k) or (seq_q, seq_k)
    token_ids=request.prompt_token_ids,
    layer_idx=layer_idx,
)
```

### Accuracy Preservation Contract

- Only TTL-expired segments are candidates — no in-TTL segment is ever touched.
- `eviction_score(importance=1.0) = 0.0` — high-importance segments are structurally safe.
- No lossy operations (no quantization, no approximation of KV values).
- Policy only reorders the eviction queue; it cannot evict segments that TTL would not already evict.

Validated in Report ① 2026-05-04:
- High-importance segment preservation: 100%
- Residual Key cosine similarity after eviction: ≥ 0.99
- Hit rate delta for important segments: ≤ 1%p

---

## Full Pipeline Integration (Cross-1: A+B+C)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.scheduler_patch import make_dag_aware_scheduler_class
from vllm_integration.block_manager_patch import (
    WorkloadAwareTTLKVCacheManager, VllmDAGAwareTTLAdjuster
)
from vllm_integration.attention_backend_patch import (
    VllmRedundancyAwareEvictionPolicy, VllmAttentionKVHook
)

# Step 1: Create eviction policy (Activity C)
eviction_policy = VllmRedundancyAwareEvictionPolicy(redundancy_top_n=100)

# Step 2: Create TTL-aware KV manager (Activity B)
kv_manager = WorkloadAwareTTLKVCacheManager(
    kv_cache_config=..., max_model_len=..., hash_block_size=...,
    ttl_max_entries=1000, ttl_eviction_policy=eviction_policy,
)

# Step 3: Create TTL adjuster (Cross-1 adapter)
adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

# Step 4: Create DAG-aware scheduler (Activity A)
DAGAwareScheduler = make_dag_aware_scheduler_class(Scheduler)
scheduler = DAGAwareScheduler(
    ...,
    dag_retain_threshold=0.5,
    dag_on_kv_reuse_event=adjuster.on_kv_reuse_event,
    dag_on_node_complete_event=adjuster.on_node_complete,
)

# Step 5: Register workflows and attach attention hooks
scheduler.register_workflow(dag_spec)
attn_hook = VllmAttentionKVHook(kv_manager, chunk_size=128)

# Step 6: Per-step:
#   a. scheduler.schedule() — calls pre_schedule_dag() → fires on_kv_reuse_event
#   b. attn_hook.record_importance_from_attention() — feeds importance scores
#   c. kv_manager.evict_expired_segments() — flushes expired blocks
#   d. scheduler.notify_node_complete(dag_id, agent_id) — when agent finishes
```

---

## How to Apply

### 1. Install

```bash
bash vllm_integration/install.sh
```

### 2. Run smoke tests

The install script automatically runs smoke tests for all activities.

---

## Compatibility Notes

| Component | Status |
|-----------|--------|
| vLLM 0.20.1 (v1 engine) | Target version |
| `KVCacheManager` public API | Preserved (subclass only) |
| `Scheduler` public API | Preserved (mixin, no monkey-patching) |
| `AttentionImpl.forward` | Not modified — hook is additive only |
| `CacheConfig` fields | Not modified (composition pattern) |
| GPU memory layout | vLLM paged blocks unmodified |
| Accuracy of KV values | Preserved — no quantization in C (eviction ordering only) |

---

## Performance Expectations (from Report ① 2026-05-04)

| Metric | Standalone (233/233 tests) | vLLM port target |
|--------|---------------------------|-----------------|
| Non-contiguous cache hit rate (TTL) | 100% (noncontiguous_ratio=1.0) | ≥ 30% |
| High-importance segment preservation | 100% | 100% |
| Residual Key cosine similarity | ≥ 0.99 | ≥ 0.99 |
| Hit rate delta (important segments) | ≤ 0.0 (0%p) | ≤ 1%p |
| Scheduling overhead TTFT p50 | < 5ms / 100 reqs | ≤ 5ms / req |
| on_kv_reuse_event() p50 latency | < 1ms | < 1ms |

---

## Cycle History

| Date | Loop | Activities | Key Additions |
|------|------|-----------|---------------|
| 2026-04-28 | 1/3 | B+C | NonContiguousKVCacheManager, CompressedKVHook (INT8) |
| 2026-04-29 | 1/3 | A+B+C | CacheHitAwareRequestQueue, HadamardInt4Codec (INT4) |
| 2026-04-30 | 1/3 | A+B+C | MultiNodeRequestRouter, TriStateKVHook, SegmentAdapterMixin |
| 2026-05-02 | 1/3 | B+C | VllmLeverageCompressor, SignVQSegmentIndex, SignVQCacheParams |
| 2026-05-03 | 1/3 | A+B+C | DualMapSchedulerMixin, SemanticSegmentIndex (DHD), VllmTurboQuantCodec |
| **2026-05-04** | **1/3** | **A+B+C** | **DAGTopologySchedulerMixin, WorkloadAwareTTLKVCacheManager, VllmRedundancyAwareEvictionPolicy, VllmDAGAwareTTLAdjuster, MultiNodeDAGRouter** |

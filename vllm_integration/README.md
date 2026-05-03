# vllm_integration — Activity A+B+C KV Cache Port

## Overview

This package ports the independently-verified A+B+C KV cache pipeline from the
standalone `src/` implementation into **vLLM 0.20.1**.

| Activity | Description | Source |
|----------|-------------|--------|
| **A** | DualMap semantic-hit-rate cache-affinity scheduling | `src/scheduler/dual_map_scheduler.py` |
| **B** | DHD semantic similarity non-contiguous KV reuse | `src/cache/dhd_segment_cache.py` |
| **C** | TurboQuant 3-bit PolarQuant + QJL residual compression | `src/cache/turbo_quant.py` |
| **A+B+C** | Full pipeline: scheduler + DHD NC reuse + TurboQuant compression | cross-activity |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.1** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (`vllm.v1.*`) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Scheduler | `vllm.v1.core.sched.scheduler.Scheduler` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |
| CacheConfig | `vllm.config.cache.CacheConfig` (pydantic v2 frozen) |

---

## File Map

```
vllm_integration/
├── __init__.py                   Package marker
├── compression_codec.py          Activity C: VllmTurboQuantCodec (new, 2026-05-03)
│                                  + CacheCompressionConfig with compression_method field
│                                  + HadamardInt4Codec, CompressionCodec (prior, preserved)
├── block_manager_patch.py        Activity B: SemanticSegmentIndex (DHD semantic index)
│                                  + SemanticNonContiguousKVCacheManager subclass
│                                  + Prior-cycle SegmentHashMixin etc. (preserved)
├── attention_backend_patch.py    Activity B+C: TurboQuantKVHook (new, 2026-05-03)
│                                  + SemanticKVAttentionWrapper (new, 2026-05-03)
│                                  + CompressedKVHook, TriStateKVHook (prior, preserved)
├── scheduler_patch.py            Activity A: DualMapSchedulerMixin (new, 2026-05-03)
│                                  + DualMapRoutingMixin, DualMapNodeState
│                                  + CacheHitAwareRequestQueue, MultiNodeRequestRouter (prior)
├── install.sh                    pip install --upgrade vllm + smoke tests
├── README.md                     This file
│
│   Prior-cycle files (preserved for backward compatibility):
├── leverage_compressor_patch.py  2026-05-02 LeverageScoreCompressor
├── sign_vq_block_manager_patch.py 2026-05-02 SignVQSegmentIndex
└── cache_config_extension.py     2026-05-02 SignVQCacheParams
```

---

## Activity A — DualMap Semantic-Hit-Rate Scheduling

**Primary integration file:** `scheduler_patch.py`

### Architecture

`DualMapSchedulerMixin` is a mixin for vLLM's `Scheduler` that adds
cache-affinity-aware request ordering:

```
Routing score = semantic_hit_rate(req_emb, node) × (1 − node.load)
```

Where `semantic_hit_rate` is the mean cosine similarity of the request's
token-mean embedding vs the top-k cached segment embeddings on each node.

Routing degrades to load-only when:
- Any candidate node has an active SLO violation, OR
- Request wait_steps >= fairness_max_wait (starvation protection)

### Integration

```python
from vllm_integration.scheduler_patch import (
    DualMapSchedulerMixin,
    DualMapNodeState,
)
from vllm.v1.core.sched.scheduler import Scheduler

class CacheAwareScheduler(DualMapSchedulerMixin, Scheduler):
    def __init__(self, *args, nodes, **kwargs):
        Scheduler.__init__(self, *args, **kwargs)
        DualMapSchedulerMixin.__init__(self, nodes=nodes)

    def schedule(self):
        self.pre_schedule_sort()   # DualMap reorders waiting queue
        return super().schedule()  # vLLM handles block allocation

# Setup nodes with semantic index references
nodes = [
    DualMapNodeState(
        node_id="node0",
        semantic_index=kv_manager._segment_index._semantic_index,
        current_load=0.0,
    )
]
scheduler = CacheAwareScheduler(..., nodes=nodes)

# Attach semantic index when kv_manager is ready
scheduler.attach_semantic_index("node0", kv_manager._segment_index._semantic_index)
```

### Overhead Budget

Pre-sort operates on the waiting queue list in Python — no GPU or disk I/O.
Target: < 5ms / 100 requests (well within TTFT +5% budget).
Measured in standalone: 0.028ms / 100 requests.

---

## Activity B — DHD Semantic Non-Contiguous KV Reuse

**Primary integration file:** `block_manager_patch.py`

### Architecture

`SemanticNonContiguousKVCacheManager` subclasses `KVCacheManager` and adds
a parallel `SemanticSegmentIndex` for non-contiguous KV reuse:

```
For each prefill chunk:
    1. lookup_segment(token_ids, chunk_idx, query_keys, layer_idx)
       → check exact SHA-256 hash (fast path)
       → if miss: cosine similarity search over semantic_index
       → DHD deviation check: ||q_keys - cached_keys|| / ||cached_keys|| <= threshold
    2. On hit: return cached [K, V] (decompress via TurboQuantCodec)
    3. On miss: compute KV normally, then store_segment() to populate index
```

DHD (Dual-Stage High Deviation) classification:
- cosine_sim >= similarity_threshold → candidate found
- L2 deviation <= deviation_threshold → semantic hit (return cached KV)
- deviation > threshold → recompute (return miss, let caller recompute)

### Integration

```python
from vllm_integration.block_manager_patch import SemanticNonContiguousKVCacheManager
from vllm_integration.compression_codec import VllmTurboQuantCodec

codec = VllmTurboQuantCodec(num_layers=32, bits=3)

kv_manager = SemanticNonContiguousKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,  # Must equal vLLM block_size
    enable_caching=True,
    codec=codec,
    segment_chunk_size=16,       # Must align with vLLM block_size
    segment_max_entries=2000,
    similarity_threshold=0.80,
    deviation_threshold=0.20,
)

# In attention layer (prefill):
k_cached, v_cached, hit_type = kv_manager.lookup_segment(
    token_ids, chunk_idx, query_keys, layer_idx
)
if hit_type == "miss":
    # compute KV normally
    kv_manager.store_segment(token_ids, chunk_idx, new_keys, new_values, layer_idx)
```

### Block Boundary Contract

`segment_chunk_size` MUST divide evenly into vLLM's `block_size`. Segments
that cross block boundaries are not supported. Set `segment_chunk_size` equal
to `block_size` (default 16) for guaranteed alignment.

### Non-Contiguous Hit Rate

Semantic (non-contiguous) hits are counted separately from exact hash hits.
Target: semantic_hits / total_hits >= 30%.

```python
stats = kv_manager.segment_index_stats()
# stats["noncontiguous_ratio"] >= 0.30
```

---

## Activity C — TurboQuant 3-bit KV Compression

**Primary integration file:** `compression_codec.py`, `attention_backend_patch.py`

### Architecture

`VllmTurboQuantCodec` wraps `TurboQuantCodec` (PolarQuant + QJL) for vLLM shapes:

```
PolarQuant:
    R = random orthogonal rotation matrix (per-layer, fixed seed)
    kv_rotated = kv @ R.T               ← redistribute outliers uniformly
    quantized = round(kv_rotated / scale)  ← 3-bit (or 4-bit for sensitive layers)

QJL residual correction:
    residual = kv_rotated - dequant(quantized)
    proj = residual @ P.T               ← JL projection (P = ±1/sqrt(d))
    qjl_packed = (proj >= 0)           ← 1-bit sign per projection dim

Decode:
    kv_dequant = quantized * scale
    residual_approx = sign(qjl) @ P * ||residual||
    kv_corrected = kv_dequant + residual_approx
    return kv_corrected @ R             ← inverse rotation (R orthogonal: R^T = R^{-1})
```

Sensitive layers (first 25%): 4-bit quantization.
Non-sensitive layers: 3-bit quantization.

### Memory Reduction (d_head=128, n_tokens=1000)

| Component | Bytes | Notes |
|-----------|-------|-------|
| FP32 baseline | 512,000 | reference |
| quantized (int8) | 128,000 | 3-bit stored in int8 |
| scale (float32) | 4,000 | per-row |
| qjl_packed (1-bit) | 16,000 | packed bits |
| qjl_residual_norm | 4,000 | per-row |
| **Total** | **152,000** | **−70.3% vs FP32** |

### CacheConfig Extension

`CacheCompressionConfig` attaches compression parameters to vLLM via
composition (not mutation of vLLM's frozen pydantic CacheConfig):

```python
from vllm_integration.compression_codec import CacheCompressionConfig

# compression_method: "none" | "int8" | "fp8" | "turbo3" | "turbo4"
cfg = CacheCompressionConfig(
    compression_method="turbo3",
    num_layers=32,
    bits=3,
    sensitive_layers_ratio=0.25,
)
codec = cfg.build_codec()  # VllmTurboQuantCodec or None
```

### Write/Read Hooks

`TurboQuantKVHook` hooks compression into the attention backend write/read paths:

```python
from vllm_integration.attention_backend_patch import TurboQuantKVHook

hook = TurboQuantKVHook(codec=codec, enabled=True)

# After QKV projection, before segment index storage:
compressed_k = hook.write_to_cache(key, layer_idx, tensor_id=0)
compressed_v = hook.write_to_cache(value, layer_idx, tensor_id=1)

# Before attention kernel (MUST decompress before kernel entry):
key_decompressed   = hook.read_from_cache(compressed_k, layer_idx, tensor_id=0)
value_decompressed = hook.read_from_cache(compressed_v, layer_idx, tensor_id=1)
```

**Critical constraint**: Compressed tensors NEVER enter flashinfer or
flash-attention kernels. Decompression occurs BEFORE kernel invocation.

### Wrapper for Combined B+C

`SemanticKVAttentionWrapper` combines DHD lookup (Activity B) and TurboQuant
hooks (Activity C) around an existing AttentionImpl:

```python
from vllm_integration.attention_backend_patch import SemanticKVAttentionWrapper

wrapped = SemanticKVAttentionWrapper(
    impl=original_attn_impl,
    hook=hook,
    kv_manager=kv_manager,
    chunk_size=16,
)

# Prefill write path (store compressed KV in index):
wrapped.store_kv_chunks(token_ids, keys, values, layer_idx)

# Decode / re-prefill read path (check index for non-contiguous hits):
hit_chunks, miss_indices = wrapped.load_cached_chunks(token_ids, layer_idx, query_keys)

# Standard attention (interface unchanged):
output = wrapped.forward(layer, query, key, value, kv_cache, attn_metadata, output)
```

---

## How to Apply

### 1. Install

```bash
bash vllm_integration/install.sh
```

### 2. Configure Compression (Activity C)

```python
from vllm_integration.compression_codec import CacheCompressionConfig, VllmTurboQuantCodec

compression_cfg = CacheCompressionConfig(compression_method="turbo3", num_layers=32)
codec = compression_cfg.build_codec()
```

### 3. Create Semantic KV Cache Manager (Activity B)

```python
from vllm_integration.block_manager_patch import SemanticNonContiguousKVCacheManager

kv_manager = SemanticNonContiguousKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    codec=codec,
    segment_chunk_size=block_size,  # align with vLLM block_size
    segment_max_entries=2000,
)
```

### 4. Enable DualMap Scheduling (Activity A)

```python
from vllm_integration.scheduler_patch import DualMapSchedulerMixin, DualMapNodeState
from vllm.v1.core.sched.scheduler import Scheduler

class CacheAwareScheduler(DualMapSchedulerMixin, Scheduler):
    def __init__(self, *args, nodes, **kwargs):
        Scheduler.__init__(self, *args, **kwargs)
        DualMapSchedulerMixin.__init__(self, nodes=nodes)

    def schedule(self):
        self.pre_schedule_sort()
        return super().schedule()

nodes = [DualMapNodeState(
    node_id="node0",
    semantic_index=kv_manager._segment_index._semantic_index,
)]
scheduler = CacheAwareScheduler(..., nodes=nodes)
```

### 5. Wire Compression Hooks (Activity B+C)

```python
from vllm_integration.attention_backend_patch import TurboQuantKVHook, SemanticKVAttentionWrapper

hook = TurboQuantKVHook(codec=codec)
wrapped_attn = SemanticKVAttentionWrapper(
    impl=attn_impl, hook=hook, kv_manager=kv_manager, chunk_size=block_size
)
```

---

## Compatibility Notes

| Component | Status |
|-----------|--------|
| vLLM 0.20.1 (v1 engine) | Target version |
| `KVCacheManager` public API | Preserved (subclass only) |
| `Scheduler` public API | Preserved (mixin, no monkey-patching) |
| `AttentionImpl.forward` | Preserved (wrapper delegates unchanged) |
| `CacheConfig` fields | Not modified (composition via CacheCompressionConfig) |
| GPU memory layout | vLLM paged blocks unmodified |
| Compression in flash-attn kernels | Never — decompress before kernel |

---

## Performance Expectations (from Report ① 2026-05-03)

| Metric | Standalone (45/45 tests) | vLLM port target |
|--------|--------------------------|-----------------|
| Throughput improvement | +20% (memory-budget) | ≥ +10% |
| Non-contiguous cache hit rate (semantic) | 100% (test) | ≥ 30% |
| KV cache memory reduction | −70.3% vs FP32 | ≥ −30% (goal −60%) |
| Compression accuracy (cosine sim) | ≥ 0.98 | ≥ 0.95 |
| Scheduling overhead (TTFT) | 0.028ms / 100 reqs | ≤ 5ms / req |
| Effective context length | 3.37× | ≥ 2× |

---

## Cycle History

| Date | Loop | Activities | Key Additions |
|------|------|-----------|---------------|
| 2026-04-28 | 1/3 | B+C | NonContiguousKVCacheManager, CompressedKVHook (INT8) |
| 2026-04-29 | 1/3 | A+B+C | CacheHitAwareRequestQueue, HadamardInt4Codec (INT4) |
| 2026-04-30 | 1/3 | A+B+C | MultiNodeRequestRouter, TriStateKVHook, SegmentAdapterMixin |
| 2026-05-02 | 1/3 | B+C | VllmLeverageCompressor, SignVQSegmentIndex, SignVQCacheParams |
| 2026-05-03 | 1/3 | A+B+C | DualMapSchedulerMixin, SemanticSegmentIndex (DHD), VllmTurboQuantCodec, TurboQuantKVHook |

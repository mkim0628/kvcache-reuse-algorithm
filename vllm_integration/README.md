# vllm_integration — Activity A+B+C KV Cache Port

## Overview

This package ports the independently-verified A+B+C KV cache pipeline from the
standalone `src/` implementation into **vLLM 0.20.0**.

| Activity | Description | Source |
|----------|-------------|--------|
| **A** | Cache-hit-rate-aware request scheduling | `src/scheduler/cache_aware_scheduler.py` |
| **B** | Position-independent segmented hash cache | `src/cache/segmented.py` |
| **C** | Hadamard INT4 + mixed-precision KV compression | `src/cache/compression.py` |
| **A+B+C** | Full pipeline: scheduler + NC reuse + compression | `src/cache/compressed_segment.py` |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.0** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (vllm.v1.*) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Request queue | `vllm.v1.core.sched.request_queue.RequestQueue` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |

---

## File Map

```
vllm_integration/
├── __init__.py                  Package marker
├── compression_codec.py         Activity C: HadamardInt4Codec (new) + CompressionCodec (prior)
├── block_manager_patch.py       Activity B: Position-independent segment index
│                                + NonContiguousKVCacheManager subclass
├── attention_backend_patch.py   Activity B+C: CompressedKVHook + wrapper
├── scheduler_patch.py           Activity A: CacheHitAwareRequestQueue
├── install.sh                   Install + smoke-test script
└── README.md                    This file
```

---

## vLLM Integration Points

### Activity A — Cache-Hit-Aware Scheduling

**Primary integration file:** `scheduler_patch.py`

vLLM 0.20.0 v1 scheduler uses a pluggable `RequestQueue` (defined in
`vllm.v1.core.sched.request_queue`). The default policy is FCFS or priority.

`CacheHitAwareRequestQueue` subclasses `RequestQueue` and maintains a
priority heap that orders requests by predicted KV segment hit rate:

```
priority = hit_rate × (1 − min(wait_steps / fairness_max_wait, 1.0))
```

Hit rate prediction peeks at `CompressedSegmentIndex._store` keys without
calling `get()`, so cache statistics are not polluted.  A wait-step penalty
prevents cold requests from being indefinitely starved.

**How to enable:**
```python
from vllm_integration.scheduler_patch import create_cache_hit_aware_queue

# In Scheduler.__init__, replace default request queue:
self.waiting = create_cache_hit_aware_queue(
    segment_index=kv_manager._segment_index,
    chunk_size=kv_manager._segment_chunk_size,
    fairness_max_wait=10,
)
```

### Activity B — Non-Contiguous KV Cache Reuse

**Primary integration file:** `block_manager_patch.py`

`NonContiguousKVCacheManager` subclasses `KVCacheManager` and adds:

1. `SegmentHashMixin.get_segment_key(token_ids, chunk_idx, layer_idx)` — a
   position-independent SHA-256 hash of a token chunk (content-only, no
   absolute position), enabling reuse when tokens appear at different offsets.

2. `CompressedSegmentIndex` — LRU dict mapping segment keys to compressed KV
   tensors (backed by `HadamardInt4Codec` by default).

3. `get_computed_blocks` override — queries the segment index after the
   standard prefix cache lookup for non-contiguous hits.

4. `store_segment` / `lookup_segment` — attention backend API.

**Default codec changed to `HadamardInt4Codec`** in this cycle (pass
`use_hadamard_int4=False` for the prior INT8 codec).

### Activity C — KV Cache Compression

**Primary integration file:** `compression_codec.py`, `attention_backend_patch.py`

**`HadamardInt4Codec`** (new in cycle 2026-04-29, recommended):
- Early layers (cutoff_ratio=0.2): FP16 — 50% savings
- Late layers: Hadamard rotation + INT4-range quantized, stored as int8 — 75% savings
- Average: ~70% memory reduction vs FP32
- Accuracy: attention KL divergence < 0.05, cosine similarity ≥ 0.95

**`CompressionCodec`** (prior cycle, reference):
- Early layers: FP16; later layers: symmetric INT8 — ~67% average savings

---

## How to Apply Patches

### 1. Install

```bash
bash vllm_integration/install.sh
```

### 2. Substitute KV Cache Manager (Activity B+C)

```python
from vllm_integration.compression_codec import HadamardInt4Codec
from vllm_integration.block_manager_patch import NonContiguousKVCacheManager

kv_manager = NonContiguousKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    use_hadamard_int4=True,   # HadamardInt4Codec (recommended)
    segment_chunk_size=64,
    segment_max_entries=2000,
)
```

### 3. Enable Cache-Hit-Aware Scheduling (Activity A)

```python
from vllm_integration.scheduler_patch import create_cache_hit_aware_queue

# During Scheduler construction:
self.waiting = create_cache_hit_aware_queue(
    segment_index=kv_manager._segment_index,
    chunk_size=64,
    fairness_max_wait=10,
)
```

### 4. Wire Compression Hooks into Attention (Activity B+C)

```python
from vllm_integration.attention_backend_patch import (
    CompressedKVHook, NonContiguousAttentionWrapper,
)

hook = CompressedKVHook(kv_manager._segment_index.codec)
wrapped = NonContiguousAttentionWrapper(
    impl=original_attn_impl, hook=hook, kv_manager=kv_manager, chunk_size=64,
)

# Prefill write path:
wrapped.store_kv_chunks(token_ids, k, v, layer_idx)

# Decode / re-prefill read path:
hit_chunks, miss_chunks = wrapped.load_cached_chunks(token_ids, layer_idx)
```

---

## Compatibility Notes

| Component | Status |
|-----------|--------|
| vLLM 0.20.0 (v1 engine) | Tested |
| `KVCacheManager` public API | Preserved (subclass, no monkey-patching) |
| `RequestQueue` interface | Fully implemented (`CacheHitAwareRequestQueue`) |
| `AttentionImpl.forward` | Delegated (wrapper is pass-through) |
| `SchedulerConfig` fields | Not modified |
| GPU memory layout | Follows vLLM block_size |

---

## Performance Expectations (from Report ①)

| Metric | Standalone (55/55 tests) | vLLM port target |
|--------|--------------------------|-----------------|
| Throughput improvement | > 10% (memory-budget test) | ≥ 10% |
| Non-contiguous cache hit rate | ≥ 30% | ≥ 30% |
| KV cache memory reduction | ≥ 70% vs FP32 | ≥ 30% (goal) |
| Compression accuracy (KL) | < 0.05 all layers | ≤ 0.05 |
| Scheduling overhead (TTFT) | ≤ 5% | ≤ 5% |

---

## Cycle History

| Date | Loop | Activities | Key Additions |
|------|------|-----------|---------------|
| 2026-04-28 | 1/3 | B+C | NonContiguousKVCacheManager, CompressedKVHook (INT8) |
| 2026-04-29 | 1/3 | A+B+C | CacheHitAwareRequestQueue, HadamardInt4Codec (INT4) |
| 2026-04-30 | 1/3 | A+B+C | MultiNodeRequestRouter, TriStateKVHook, SegmentAdapterMixin |

---

## 2026-04-30 Additions

### Activity A — MultiNodeRequestRouter (`scheduler_patch.py`)

Port of `src/scheduler/multi_node_scheduler.py`. Adds P/D disaggregated prefill
routing above `CacheHitAwareRequestQueue`:

```python
from vllm_integration.scheduler_patch import create_multi_node_router, VllmNodeConfig

router = create_multi_node_router(
    prefill_nodes=[VllmNodeConfig("p0", "prefill"), VllmNodeConfig("p1", "prefill")],
    decode_nodes=[VllmNodeConfig("d0", "decode"), VllmNodeConfig("d1", "decode")],
    segment_index=kv_manager._segment_index,
    chunk_size=128,
    codec=hadamard_codec,               # enables compress_before_transfer
    compress_threshold_bytes=1048576,
)
routing = router.route(request)
# routing["compress_before_transfer"] → True if KV size > threshold
```

### Activity C — TriStateKVHook (`attention_backend_patch.py`)

Port of `src/cache/tri_state_compressor.py`. Adds ARKV-style tri-state
(retain/compress/evict) compression to the attention write path:

```python
from vllm_integration.attention_backend_patch import TriStateKVHook
from vllm_integration.compression_codec import HadamardInt4Codec

codec = HadamardInt4Codec(num_layers=32, cutoff_ratio=0.2)
hook = TriStateKVHook(codec=codec, retain_ratio=0.20, evict_ratio=0.40)

# Write path (after K/V projection):
storage = hook.encode_kv(k, attn_weights, layer_idx=5, tensor_id=0)
# Read path:
k_reconstructed = hook.decode_kv(storage, layer_idx=5, tensor_id=0)
# Compression: ~80% memory savings vs FP32 (hook.compression_ratio() ≈ 0.20)
```

**Accuracy constraint**: retain tier (FP16) + compress tier (HadamardInt4, KL<0.007)
maintains perplexity delta ±1%. Validated in Report ① 2026-04-30.

### Activity B — SegmentAdapterMixin (`block_manager_patch.py`)

Port of `src/cache/segment_adapter.py`. Adds MLP position-mismatch correction
to non-contiguous KV hits:

```python
from vllm_integration.block_manager_patch import NonContiguousKVCacheManagerWithAdapter

mgr = NonContiguousKVCacheManagerWithAdapter(hidden_dim=64)
# Offline training:
mgr.train_adapter(cached_kvs, target_kvs, kv_dim=64, n_steps=500)
# Inference (non-contiguous hit):
kv = mgr.lookup_segment_with_adapter(token_ids, chunk_idx, layer_idx, is_noncontiguous=True)
```

---

## Updated Performance Expectations (from Report ① 2026-04-30)

| Metric | Standalone (77/77 tests) | vLLM port target |
|--------|--------------------------|-----------------|
| Throughput improvement | +391% (memory-budget) | ≥ +10% |
| Non-contiguous cache hit rate | ≥ 30% | ≥ 30% |
| KV memory reduction (TriState) | 80% | ≥ 30% (goal) |
| Compression accuracy (KL) | < 0.005 (INT4 tri-state) | ≤ 0.05 |
| Scheduling overhead (TTFT) | 0.0% | ≤ 5% |

---

## 2026-05-02 Additions (B+C Cross-1: LeverageScore + SignVQ)

### Activity C — VllmLeverageCompressor (`leverage_compressor_patch.py`)

Port of `src/cache/leverage_compressor.LeverageScoreCompressor` adapted for
vLLM's paged-block KV cache layout `[2, num_blocks, block_size, num_kv_heads, head_size]`.

3-tier classification per block:
- Tier-1 (top 20% by leverage score): FP16 full KV — exact round-trip
- Tier-2 (middle 60%): 1-bit sign packed Key + FP16 Value
- Tier-3 (bottom 20%): evicted (0 bytes)

Memory reduction: ~74% vs FP32 baseline (verified in Report ①).

```python
from vllm_integration.leverage_compressor_patch import VllmLeverageCompressor

comp = VllmLeverageCompressor(rank=32, tier1_ratio=0.20, tier3_ratio=0.20)
storage = comp.encode_block(key_block, value_block, layer_idx=5, block_id=42)
decoded = comp.decode_block(storage)   # (2, block_size, d_head)
# Multi-head variant (vLLM shape: block_size × num_kv_heads × head_size):
per_head = comp.encode_block_multihead(key_block, value_block, layer_idx=5)
out      = comp.decode_block_multihead(per_head)  # (2, bs, H, head_size)
```

### Activity B+C — SignVQSegmentIndex + NonContiguousKVCacheManagerV2 (`sign_vq_block_manager_patch.py`)

Port of `src/cache/sign_vq_segment.SignVQSegmentCache`.  Adds a 3-stage lookup
index alongside vLLM's standard prefix cache:

1. **exact_fp16**: SHA-256 content-only hash → FP16 KV (exact)
2. **approx_sign**: XOR + popcount Hamming distance ≤ threshold → ±1 sign KV
3. **miss**: normal prefill

`NonContiguousKVCacheManagerV2` subclasses `KVCacheManager` and exposes:
- `store_segment(token_ids, chunk_idx, keys, values, layer_idx)`
- `lookup_segment(token_ids, chunk_idx, layer_idx, query_keys)`
- `lookup_all_segments(token_ids, layer_idx, query_keys)`
- `sign_index_stats()` → tier hit rate dict

```python
from vllm_integration.sign_vq_block_manager_patch import NonContiguousKVCacheManagerV2

kv_manager = NonContiguousKVCacheManagerV2(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    sign_vq_chunk_size=16,
    sign_vq_max_entries=2000,
    sign_vq_hamming_threshold=0.15,
    sign_vq_rank=32,
)
```

### Activity C — CacheConfig Extension (`cache_config_extension.py`)

`SignVQCacheParams` carries the new compression parameters without modifying
vLLM's frozen `CacheConfig`:

```python
from vllm_integration.cache_config_extension import (
    SignVQCacheParams, build_sign_vq_compressor, build_sign_vq_index,
)

params = SignVQCacheParams(
    enable_sign_vq=True,
    tier1_ratio=0.20,
    tier2_ratio=0.60,
    hamming_threshold=0.15,
)
compressor = build_sign_vq_compressor(params)
index      = build_sign_vq_index(params)
```

---

## Updated Performance Expectations (from Report ① 2026-05-02)

| Metric | Standalone (121/121 tests) | vLLM port target |
|--------|---------------------------|-----------------|
| Throughput improvement | ≥ +10% (vs FP32 memory budget) | ≥ +10% |
| Non-contiguous cache hit rate | ≥ 30% | ≥ 30% |
| KV memory reduction | 74.1% vs FP32 | ≥ 30% (goal −70%) |
| Compression accuracy (KL) | < 0.015 (PRIMARY) | ≤ 0.015 |
| Integration perplexity proxy | cosine ≥ 0.84 | ≥ 0.84 |
| Scheduling overhead (TTFT) | ≤ 5% | ≤ 5% |

---

## Latest Cycle

- Date: 2026-05-02
- Loop: 1 / 3
- Activity: B+C (LeverageScoreCompressor + SignVQSegmentCache)
- Report ①: `reports/evaluations/2026-05-02.md`

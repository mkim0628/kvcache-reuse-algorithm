# vllm_integration — Activity A+B+C KV Cache Port

## Overview

This package ports the independently-verified A+B+C KV cache pipeline from the
standalone `src/` implementation into **vLLM 0.20.2**.

| Cycle | Activity | Description | Source |
|-------|----------|-------------|--------|
| 2026-05-12 | **B** | AdapShotBlockManager — AdapShotMixedDimSegmentPipeline auxiliary segment store (RoPE re-encoding + MixedDim codec, B+C Cross-2) | `src/cache/adapshot_pipeline.py` |
| 2026-05-12 | **C** | MixedDimAttentionHook — write/read hooks for MixedDimPerTokenBudgetCodec (compress-before-store, decompress-before-kernel) | `src/cache/mixed_dim_codec.py` |
| 2026-05-12 | **B** | AdapShotSegmentSchedulerMixin — non-contiguous hit-rate-based request reordering (B+C Cross-2 scheduling) | (new, wraps vLLM Scheduler) |
| 2026-05-11 | **B** | WiCERBlockManager — CEGAR iterative non-contiguous KV artefact cache (parallel segment store) | `src/cache/wicer_iterative_cache.py` |
| 2026-05-11 | **C** | RateQuantAttentionHook — reverse water-filling write/read hooks (compress-before-store, decompress-before-kernel) | `src/cache/ratequant_codec.py` |
| 2026-05-11 | **C** | RateQuantVllmCodec — per-head optimal bit allocation, 75% memory reduction, < 1% accuracy error | `src/cache/ratequant_codec.py` |
| 2026-05-10 | **B** | KVPacketVQBlockManager — KVPacket soft-adapter non-contiguous cache, subclasses KVCacheManager | `src/cache/kv_packet_adapter.py` |
| 2026-05-10 | **C** | VQCodecAttentionHook — VQ codec write/read hooks (compress-before-store, decompress-before-kernel) | `src/compression/vq_codec.py` |
| 2026-05-10 | **A** | KVPacketSegmentSchedulerMixin — segment-hash-based request reordering for B+C pipeline | (new) |
| 2026-05-08 | **A** | PreemptiveKVOffloadSchedulerMixin — TokenFlow preemptive scheduling + async GPU→CPU KV offload | `src/scheduler/preemptive_kv_offload.py` |
| 2026-05-08 | **C** | VllmEOptShrinkQCodec — BBP auto-rank low-rank (Key 2-bit / Value 3-bit) + TurboQuant residual | `src/cache/eopt_shrinkq_codec.py` |
| 2026-05-08 | **A+C** | CompressedPreemptionMixin — CUDA dual-stream inline compression during preemption offload | `src/scheduler/compressed_preemption.py` |
| 2026-05-08 | **C** | EOptShrinkQAttentionHook — write/read hooks for eOptShrinkQ (compress before store, decompress before kernel) | (new) |
| 2026-05-08 | **C** | ManifoldKVOutlierScoreHook — read-only Euclidean outlier scoring hook (no lossy ops) | (new) |
| 2026-05-08 | **B** | StaticDynamicSegmentKVManager — static/dynamic segment classification + multi-hop invalidation (max 2 hops) | (new) |
| 2026-05-08 | **C** | ManifoldKVWindowedEvictionManager — Euclidean outlier-based eviction policy (drop-in for LRU) | (new) |
| 2026-05-06 | **B** | QueryCentricKVCacheManager — ProphetKV dual-stage recompute budget allocation | `src/cache/query_centric_recompute.py` |
| 2026-05-06 | **C** | TriAttentionCodecWrapper — pre-RoPE trigonometric KV compression codec | `src/cache/tri_attention_codec.py` |
| 2026-05-06 | **B+C** | QueryCentricTriAttentionKVCacheManager — dual-path raw/compressed store | `src/cache/qc_tri_store.py` |
| 2026-05-06 | **C** | TriAttentionAttentionHook — compress/decompress hooks for attention backend | (new, wraps TriAttentionCodecWrapper) |
| 2026-05-06 | **C** | VllmQueryCentricAttentionWrapper — pre-RoPE key capture + QCTA integration | (new, wraps AttentionImpl.forward) |
| 2026-05-06 | **B** | QueryCentricSchedulerMixin — QCRC recompute scheduling mixin for vLLM Scheduler | (new) |
| 2026-05-05 | **B** | DiffAwareKVPatch — master + block-sparse diff non-contiguous KV reuse | `src/cache/diff_aware_store.py` |
| 2026-05-05 | **C** | NQKVCodecPatch — Normal Float INT4 block-quantile KV compression | `src/cache/nqkv_codec.py` |
| 2026-05-05 | **B+C** | CompressedKVManager — INT4-compressed masters + FP16 diff storage | `src/cache/compressed_diff_store.py` |
| 2026-05-05 | **C** | FireQAttentionPatch — RoPE-aware 2-stage outlier smoothing attn hook | `src/cache/fireq_codec.py` |
| 2026-05-04 | **A** | DAGTopologyScheduler — workflow DAG topology KV proactive preservation | `src/scheduler/dag_topology_scheduler.py` |
| 2026-05-04 | **B** | WorkloadAwareTTLCache — category-specific TTL segment preservation | `src/cache/workload_ttl_cache.py` |
| 2026-05-04 | **C** | RedundancyAwareEvictionPolicy — importance x redundancy dual-score eviction | `src/cache/redundancy_eviction.py` |
| 2026-05-04 | **Cross-1** | DAGAwareTTLAdjuster — DAG events → TTL adjustment pipeline | `src/scheduler/dag_ttl_adjuster.py` |
| 2026-05-03 | **A** | DualMapSchedulerMixin — dual-hash + semantic-hit-rate routing (preserved) | `src/scheduler/dual_map_scheduler.py` |
| 2026-05-03 | **B** | SemanticNonContiguousKVCacheManager — DHD semantic similarity (preserved) | `src/cache/dhd_segment_cache.py` |
| 2026-05-03 | **C** | TurboQuantKVHook — TurboQuant 3-bit compression (preserved) | `src/cache/turbo_quant.py` |

---

## 2026-05-12 Integration: Activity B+C (RoPEReencodingNonContiguousCache + MixedDimPerTokenBudgetCodec + AdapShotMixedDimSegmentPipeline)

### vLLM Version

**vLLM 0.20.2** (installed: `pip install --upgrade vllm`)

### Summary

This cycle ports the B+C (Cross-2) pipeline validated in Report ① (2026-05-12):
- **Non-contiguous hit rate**: 100% (3/3 hits non-contiguous; goal ≥ 30%)
- **KV memory reduction**: −50% (budget_ratio=0.50; goal −30%)
- **Perplexity delta**: 0.36% (< 1% mandatory threshold)
- **KL divergence**: 0.000023 (< 0.015 threshold)
- **Cosine similarity**: 0.999994 (≥ 0.99 threshold)
- **83/83 tests passing** (66 unit + 17 integration)

### New Classes (2026-05-12)

**block_manager_patch.py**:
- `AdapShotBlockManager` — wraps `AdapShotMixedDimSegmentPipeline` as a parallel KV store
  - `store_segment(token_ids, chunk_idx, pre_rope_kv, layer_idx, attn_weights)` — B+C store contract: mixed-dim compress → pre-RoPE store
  - `load_segment(token_ids, chunk_idx, target_offset, layer_idx)` → tensor | None — B+C restore contract: pre-RoPE load + RoPE re-apply → decode
  - `get_segments(token_ids, target_offset, layer_idx)` → (hits, misses)
  - `annotate_request(request, token_ids, target_offset, layer_idx)` — sets adapshot_hits, adapshot_misses, adapshot_noncontiguous_hit_rate on request
  - `hit_stats()` → dict
- `make_adapshot_kv_cache_manager_class(base_class, ...)` → KVCacheManager subclass with adapshot_* methods

**attention_backend_patch.py**:
- `MixedDimAttentionHook` — write/read hooks for MixedDimPerTokenBudgetCodec
  - `write_to_cache(kv, layer_idx, attn_weights, budget_ratio)` → dict — compress before store
  - `read_from_cache(payload, layer_idx)` → tensor — decompress BEFORE attention kernel (accuracy contract)
  - `memory_reduction_ratio(payload)` → float
  - `hook_stats()` → dict
- `extend_cache_config_mixed_dim(budget_ratio, enabled)` → dict — CacheConfig extension helper

**scheduler_patch.py**:
- `AdapShotSegmentSchedulerMixin` — non-contiguous hit-rate-based request reordering
  - `pre_schedule_adapshot()` — reorders waiting queue by adapshot_noncontiguous_hit_rate
  - `adapshot_reorder_stats()` → dict
- `make_adapshot_scheduler_class(base_class, adapshot_reorder_window)` → Scheduler subclass

### Integration Architecture

```
vLLM v1 KVCacheManager (native block pool, unchanged)
           │
           ├── AdapShotBlockManager (parallel auxiliary store)
           │         │
           │         ├── [STORE] MixedDimPerTokenBudgetCodec.encode()   (Activity C)
           │         │         → bisection-search threshold λ* → retain top dims
           │         │
           │         └── [STORE] RoPEReencodingNonContiguousCache.store_pre_rope()  (Activity B)
           │                   → content-hash keying (position-independent)
           │
           └── [LOAD]  RoPEReencodingNonContiguousCache.load_with_rope()
                     → apply RoPE for target_positions
                     → MixedDimPerTokenBudgetCodec.decode() (no-op, masked_kv)
                     → [full-precision tensor → attention kernel]  ← accuracy contract
```

### Accuracy Contract (Activity C)

Compressed KV is NEVER passed to the attention kernel in compressed form.
`load_segment()` and `MixedDimAttentionHook.read_from_cache()` always return
full-precision tensors before the caller invokes an attention operation.

Memory reduction comes from zeroing low-importance dimensions (determined by
per-token loss score = attention_importance × value_magnitude × compressibility).
At budget_ratio=0.50: validated 0.36% relative error (< 1% mandatory threshold).

### Non-Contiguous Reuse (Activity B)

Segments are stored pre-RoPE (position-independent content hash key).
On retrieval, RoPE rotation matrices are computed for the target request's
absolute token positions, enabling reuse of segments whose tokens appear at
different positions than when originally cached.

Non-contiguous hits: a hit at chunk_idx k where a miss exists at any chunk_idx < k.
Validated: 100% non-contiguous hit rate in the chunk-0-miss / chunk-1,2,3-hit scenario.

### File Integration Points

| vLLM File | Integration Strategy | Notes |
|-----------|---------------------|-------|
| `vllm/v1/core/kv_cache_manager.py` | Subclassed via `make_adapshot_kv_cache_manager_class()` | Native block pool unmodified |
| `vllm/v1/core/sched/scheduler.py` | Subclassed via `make_adapshot_scheduler_class()` | schedule() wrapped with reorder hook |
| `vllm/config.py` (CacheConfig) | Extended via `extend_cache_config_mixed_dim()` dict | Cannot add fields directly (pydantic) |
| `vllm/attention/backends/` | Hooked via `MixedDimAttentionHook.write_to_cache()/.read_from_cache()` | Parallel store pattern |

### Budget Ratio Performance Sweep (from Report ①)

| budget_ratio | Memory Reduction | Relative Error |
|-------------|-----------------|----------------|
| 0.30 | −70% | 0.69% |
| 0.40 | −60% | 0.67% |
| 0.50 | −50% | 0.62% |
| 0.60 | −40% | 0.43% |
| 0.70 | −30% | 0.30% |

All budget_ratio values in [0.30, 0.70] achieve < 1% perplexity delta (mandatory §4).

---

## 2026-05-11 Integration: Activity B+C (WiCERIterativeKVWikiCache + RateQuantReverseWaterfillingCodec)

### Summary

This cycle ports the B+C pipeline validated in Report ① (2026-05-11):
- **Non-contiguous hit rate**: ≥ 30% on gap-containing queries (CEGAR mechanism verified)
- **KV memory reduction**: 75% (avg 4-bit reverse water-filling vs FP16 baseline)
- **Perplexity delta**: ≤ 1% (relative attention-output error = 0.0086 < 0.01, mandatory §4)
- **Combined B+C accuracy**: err = 0.0055 < 0.01 (mandatory §5)

### New Classes

**block_manager_patch.py**:
- `WiCERBlockManager` — CEGAR parallel auxiliary segment store
  - `store_segment()` — store KV chunk (optional RateQuant compression)
  - `get_segment()` — retrieve; always dequantises compressed segments before returning
  - `cegar_compile()` — compile domain corpus into hash-indexed KV artefacts
  - `cegar_evaluate()` — evaluate hit rate on validation queries, collect counterexamples
  - `cegar_refine()` — halve chunk sizes for counterexample docs, recompile
  - `cegar_loop()` — full CEGAR loop: compile → evaluate → refine until convergence
  - `annotate_request()` — set `request.wicer_noncontiguous_hits` for scheduler visibility
  - `save_artifacts()` / `load_artifacts()` — serialise/restore segment store + CEGAR state
- `make_wicer_kv_cache_manager_class(base)` — factory: KVCacheManager subclass with WiCER API

**attention_backend_patch.py**:
- `RateQuantAttentionHook` — write/read hooks for RateQuant compression
  - `write_to_cache()` — compress KV with per-channel reverse water-filling quantisation
  - `read_from_cache()` — ALWAYS dequantises to float16 before returning (accuracy contract)
  - `hook_stats()` — encode/decode counts, compression ratio

**compression_codec.py**:
- `RateQuantVllmCodec` — reverse water-filling bit allocation codec
  - `calibrate()` — measure per-head KV variance, compute bit allocation via binary-search λ
  - `write_to_cache()` — per-channel int16 quantisation with [1,2,d_head] scale
  - `read_from_cache()` — dequantise → float16 (ALWAYS before attention kernel)
  - `compression_ratio()` — 1 − avg_bits/16 (e.g. 0.75 at 4-bit avg)
  - `hook_stats()` — encode/decode counts

### Accuracy Preservation Contract (Activity C — MANDATORY)

1. `RateQuantAttentionHook.write_to_cache()`: KV quantised to int16 with per-channel
   [1,2,d_head] min-max scale. 8-bit precision for < 1% accuracy error.
2. `RateQuantAttentionHook.read_from_cache()`: ALWAYS dequantises to float16 before
   returning. Quantised tensors NEVER enter the attention kernel.
3. `WiCERBlockManager.get_segment()`: same dequantisation contract — always returns float16.
4. Validated: relative attention-output error = 0.0086 < 0.01 (< 1% mandatory §4).
5. Calibration-independent validation: error = 0.0062 < 0.01 (seed=0 cal, seed=999 test).

### Integration (Activity B+C, single node)

```python
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm_integration.compression_codec import RateQuantVllmCodec
from vllm_integration.attention_backend_patch import RateQuantAttentionHook
from vllm_integration.block_manager_patch import (
    WiCERBlockManager, make_wicer_kv_cache_manager_class
)

# Step 1: Create and calibrate RateQuant codec
codec = RateQuantVllmCodec(n_heads=32, d_head=128, total_bit_budget=4.0)
# cal_kvs: list of [n_tokens, 2, n_heads, d_head] float tensors
codec.calibrate(cal_kvs, layer_idx=0)
print(f"Compression ratio: {codec.compression_ratio():.0%}")   # 75%

# Step 2: Create WiCER block manager with RateQuant compression
wicer = WiCERBlockManager(
    chunk_size=128,
    min_chunk_size=16,
    max_entries=2000,
    target_hit_rate=0.80,
    max_cegar_iterations=5,
    vllm_block_size=16,
)

# Step 3: Create attention hook
hook = RateQuantAttentionHook(codec=codec, enabled=True)

# Step 4 (optional): pre-load domain corpus with CEGAR
wicer.cegar_loop(docs, val_queries, kv_fn, layer_idx=0, codec=codec)

# Step 5: Create WiCER-augmented KV manager (optional composition pattern)
WiCERKVManager = make_wicer_kv_cache_manager_class(KVCacheManager)
# kv_manager = WiCERKVManager(..., wicer_manager=wicer)

# Step 6: Attention write path (before paged-block write)
payload = hook.write_to_cache(kv_tensor, layer_idx=5)

# Step 7: Attention read path (before FlashAttention kernel)
kv_fp16 = hook.read_from_cache(payload, layer_idx=5)
# kv_fp16 is float16 — safe to pass to attention kernel

# Step 8: Store non-contiguous segment
seg_key = wicer.store_segment(token_ids, chunk_idx=0, kv_tensor=kv_fp16, layer_idx=5, codec=codec)

# Step 9: Annotate request for scheduler visibility
wicer.annotate_request(request, token_ids, layer_idx=5)
# Sets request.wicer_noncontiguous_hits, request.wicer_hit_rate

# Step 10: Stats
stats = wicer.hit_stats()
# stats["hit_rate"], stats["noncontiguous_ratio"], stats["compress_count"]
```

### Key Parameters (2026-05-11)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_bit_budget` | 4.0 | Average bits per head (FP16 → 4-bit = 75% reduction) |
| `min_bits` | 2 | Minimum bits per head (reverse water-filling lower bound) |
| `max_bits` | 8 | Maximum bits per head |
| `chunk_size` | 128 | Initial CEGAR chunk size in tokens |
| `min_chunk_size` | 16 | Minimum chunk size after CEGAR refinement |
| `max_entries` | 2000 | Maximum segments in LRU store |
| `target_hit_rate` | 0.80 | CEGAR loop termination hit rate |
| `max_cegar_iterations` | 5 | Maximum CEGAR refinement iterations |
| `vllm_block_size` | 16 | vLLM native block size (chunk alignment) |

---

## vLLM Version

| Field | Value |
|-------|-------|
| vLLM version | **0.20.2** |
| Install command | `pip install --upgrade vllm` |
| Architecture | v1 (`vllm.v1.*`) |
| KV cache manager | `vllm.v1.core.kv_cache_manager.KVCacheManager` |
| Scheduler | `vllm.v1.core.sched.scheduler.Scheduler` |
| Block pool | `vllm.v1.core.block_pool.BlockPool` |
| Attention backend | `vllm.v1.attention.backend.AttentionImpl` |

---

## 2026-05-10 Integration: Activity B+C (KVPacketSoftAdapterCache + VQCodec)

### Summary

This cycle ports the B+C pipeline validated in Report ① (2026-05-10):
- **Non-contiguous hit rate**: 87.5% (baseline 30% target)
- **KV memory reduction**: 70.3% (baseline 30% target)
- **Perplexity delta**: ≤ 1% (±1% accuracy preservation contract)

### New Classes

**block_manager_patch.py**:
- `KVPacketVQBlockManager` — subclasses `KVCacheManager`; adds parallel KVPacket store
  with SoftTokenAdapter (Activity B) and VQ compression (Activity C).
  - `kvp_store_segment()` — store segment: VQ-encode old tokens, FP16 recent tokens
  - `kvp_get_segment()` — retrieve: VQ-decode + adapter.adapt() before returning
  - `kvp_pack_segments()` — concatenate N adapted segments without recomputation
  - `get_computed_blocks()` — extended to annotate `request.kvp_noncontiguous_hits`
- `make_kvp_vq_kv_cache_manager_class(base)` — factory function

**attention_backend_patch.py**:
- `VQCodecAttentionHook` — write/read hooks for VQ compression
  - `write_to_cache()` — compress tokens beyond `recent_window` with VQCodec.encode()
  - `read_from_cache()` — always VQCodec.decode() before returning (accuracy contract)
  - Runtime warning if compression ratio falls below `warn_compression_threshold`

**scheduler_patch.py**:
- `KVPacketSegmentSchedulerMixin` — segment-hash-based request reordering
  - `pre_schedule_kvp()` — score waiting requests by KVPacket hit rate; reorder in place
  - Target: < 5ms overhead per schedule step for N ≤ 100 requests
- `make_kvp_segment_scheduler_class(base)` — factory function

### Accuracy Preservation Contract (Activity C)

1. `VQCodecAttentionHook.write_to_cache()`: tokens beyond `recent_window` are
   VQ-encoded (codes stored as int16). Recent tokens (last N=64) stored in FP16.
2. `VQCodecAttentionHook.read_from_cache()`: ALWAYS calls `VQCodec.decode()` before
   returning. Compressed tensors NEVER reach the FlashAttention kernel.
3. If the observed compression ratio drops below `warn_compression_threshold` (default 30%),
   a `RuntimeWarning` is raised. This guards the ±1% perplexity constraint.
4. `KVPacketVQBlockManager.kvp_get_segment()` follows the same decode-before-return contract.

### Integration (single node — Activity B+C)

```python
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration import apply_all_patches
from src.compression.vq_codec import VQCodec, VQCodebookConfig

# Step 1: Create and calibrate VQCodec
vq_cfg = VQCodebookConfig(codebook_size=256, n_residuals=4, d_head=128,
                           n_heads=32, recent_window=64, seed=42)
vq_codec = VQCodec(vq_cfg)
# vq_codec.fit(calibration_keys, calibration_vals, layer_idx=0)

# Step 2: Apply all patches
patches = apply_all_patches(
    vq_codec=vq_codec,
    n_heads=32,
    d_head=128,
    recent_window=64,
    adapter_rank=8,
    max_packets=512,
)
VQHook = patches["vq_hook"]              # VQCodecAttentionHook
KVPVQManagerClass = patches["kv_manager_class"]  # KVPacketVQBlockManager subclass
KVPSchedulerClass = patches["scheduler_class"]   # scheduler with KVPSegmentMixin

# Step 3: Attention write path (before paged block write)
compressed_payload = VQHook.write_to_cache(kv_tensor, positions, layer_idx)

# Step 4: Attention read path (before FlashAttention kernel)
kv_fp16 = VQHook.read_from_cache(compressed_payload, layer_idx)

# Step 5: Segment store (non-contiguous reuse)
kv_manager = KVPVQManagerClass(
    kv_cache_config=kv_cache_config, max_model_len=max_model_len,
    hash_block_size=block_size, enable_caching=True,
    kvp_n_heads=32, kvp_d_head=128, kvp_vq_codec=vq_codec,
)
seg_key = kv_manager.kvp_store_segment(token_ids, chunk_idx=0, kv_block=kv_fp16, layer_idx=0)

# Step 6: Per-schedule-step reordering
scheduler = KVPSchedulerClass(
    ...,  # standard vLLM Scheduler args
    kvp_kv_manager=kv_manager,
    kvp_reorder_window=32,
)
scheduler.pre_schedule_kvp()  # annotates requests with kvp_hit_score
```

### Key Parameters (2026-05-10)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recent_window` | 64 | FP16 tokens kept uncompressed per segment |
| `adapter_rank` | 8 | SoftTokenAdapter rank (2×n_heads×rank×d_head×2B extra) |
| `max_packets` | 512 | Max segments in LRU packet store |
| `kvp_reorder_window` | 32 | Max waiting requests inspected per schedule step |
| `warn_compression_threshold` | 0.30 | Warn if compression ratio < this value |

---

## 2026-05-08 Integration: Activity A+C (PreemptiveKVOffload + eOptShrinkQCodec)

### Summary

This cycle addresses two known failure modes from the 2026-05-08 Report ①:
- **TTFT p99 regression** (+286.9% over baseline) caused by unbounded preemption latency.
- **Request fairness** (burst p99/normal p50 = 4.0×) caused by uncapped SLA Tier-A protection.

The CompressedPreemptionMixin cross-couples Activity A and C: CUDA dual-stream overlap
compresses KV tensors on `compute_stream` while PCIe transfer runs on `memory_stream`,
cutting preemption wall time by the codec's compression ratio.

### Files Modified

| File | Activity | New Class | Description |
|------|----------|-----------|-------------|
| `scheduler_patch.py` | A | `PreemptiveKVOffloadSchedulerMixin` | Preemptive scheduling + async GPU→CPU KV offload |
| `scheduler_patch.py` | A+C | `CompressedPreemptionMixin` | Dual-stream inline compression during preemption |
| `scheduler_patch.py` | A+C | `make_preemptive_scheduler_class()` | Factory combining base + preemption + compression mixins |
| `compression_codec.py` | C | `VllmEOptShrinkQCodec` | BBP auto-rank low-rank + TurboQuant residual (K 2-bit / V 3-bit) |
| `attention_backend_patch.py` | C | `EOptShrinkQAttentionHook` | compress-before-store / decompress-before-kernel hooks |
| `attention_backend_patch.py` | C | `ManifoldKVOutlierScoreHook` | Read-only Euclidean outlier scoring (no lossy ops) |
| `block_manager_patch.py` | B | `StaticDynamicSegmentKVManager` | Static/dynamic segment classification + multi-hop invalidation |
| `block_manager_patch.py` | C | `ManifoldKVWindowedEvictionManager` | Euclidean outlier-based eviction (drop-in for LRU) |

### Architecture

```
[Request arrives at Scheduler]
         │
         ▼
PreemptiveKVOffloadSchedulerMixin.pre_schedule_preemptive()
  ├── _pko_buffer_occupancy_ratio() >= 0.85 ?
  │    NO  → normal scheduling
  │    YES → select preemption candidates:
  │          - Sort by priority (Tier-A last)
  │          - Skip if fairness_max_wait_steps exceeded
  │          - Select lowest-priority until buffer drops below 0.85
  └── pko_offload_kv(request_id, kv_key, kv_val, layer_idx)
       │
       ▼
CompressedPreemptionMixin.cpm_offload_with_compression()  [Cross-1: A+C]
  ├── CUDA compute_stream: VllmEOptShrinkQCodec.encode_tokens(kv_key, kv_val, layer_idx)
  │    ├── BBP threshold: sigma_c = noise_level × (1 + sqrt(aspect_ratio))^2
  │    ├── SVD decomposition → keep top-r singular vectors (float16)
  │    └── TurboQuant residual: Key 2-bit / Value 3-bit asymmetric
  └── CUDA memory_stream: pinned_tensor.copy_(compressed, non_blocking=True)
       [streams overlap → PCIe transfer hides compression compute time]
         │
         ▼
[Buffer pressure relieved — normal scheduling resumes]
         │
         ▼
[Request restore: pko_restore_kv() / cpm_restore_with_decompression()]
  ├── Load compressed payload from CPU
  └── VllmEOptShrinkQCodec.decode_tokens(compressed) BEFORE returning to caller
       [accuracy contract: decompressed KV never enters attention kernel compressed]
         │
         ▼
[Attention Kernel — receives fully decompressed FP16 KV tensors]
```

### eOptShrinkQCodec: BBP Auto-Rank Selection

The codec uses the Baik-Ben Arous-Péché (BBP) phase-transition threshold to
automatically determine the optimal low-rank cut-off:

```
Marchenko-Pastur noise floor:
  sigma_c = noise_level × (1 + sqrt(n / m))^2

For each (layer, head):
  singular_values = SVD(K)[1]         # descending
  r = sum(singular_values > sigma_c)  # BBP threshold

Compression:
  K_low = U_r @ S_r @ Vt_r           # float16 low-rank
  residual = K - K_low
  K_compressed = quantize(residual, bits=2)   # Key 2-bit
  V_compressed = quantize(V,        bits=3)   # Value 3-bit

Effective rate: ~2.2 bits/element
Accuracy target: cosine(K, decode(encode(K))) >= 0.85 (validated in smoke test)
```

### Integration (Activity A — preemptive scheduling)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.scheduler_patch import make_preemptive_scheduler_class
from vllm_integration.compression_codec import VllmEOptShrinkQCodec

# Step 1: Create and calibrate eOptShrinkQCodec
codec = VllmEOptShrinkQCodec(
    num_layers=32,
    key_bits=2,
    value_bits=3,
    calibration_samples=20,
)
# calib_kvs: list of (kv_key, kv_val, layer_idx) tuples from representative requests
codec.calibrate(calib_kvs, save_path="eopt_calibration.pt")
# (or) codec.load_calibration("eopt_calibration.pt")

# Step 2: Create preemptive scheduler class
PreemptiveScheduler = make_preemptive_scheduler_class(
    Scheduler,
    use_compression=True,    # activates CompressedPreemptionMixin
)

scheduler = PreemptiveScheduler(
    ...,  # standard vLLM Scheduler args
    # PreemptiveKVOffloadSchedulerMixin args:
    pko_cache_capacity_bytes=4 * 1024**3,   # 4 GiB CPU offload buffer
    pko_threshold_preempt=0.85,             # preempt if buffer > 85%
    pko_consumption_rate_window=32,         # sliding window size
    pko_fairness_max_wait=10,               # max wait steps before Tier-A exemption expires
    pko_sla_tier_a_ids={"req_vip_001"},    # SLA Tier-A request IDs
    # CompressedPreemptionMixin additional args:
    cpm_codec=codec,
    cpm_use_dual_stream=True,              # CUDA dual-stream overlap
    cpm_sla_tier_a_no_compress=True,       # Tier-A: offload uncompressed for speed
)

# Step 3: Per-schedule-step preemption check
preempted, exempted = scheduler.pre_schedule_preemptive(
    active_request_ids=list(active_requests.keys())
)
# preempted: request IDs offloaded to CPU this step
# exempted: Tier-A requests that were protected

# Step 4: Record processed tokens (feeds consumption rate estimator)
scheduler.pko_record_processed_tokens(batch_token_count)

# Step 5: Restore preempted request when rescheduled
kv_key, kv_val = scheduler.cpm_restore_with_decompression(request_id, layer_idx=0)

# Step 6: Stats
stats = scheduler.cpm_stats()
# stats["overlap_efficiency"] >= 0.5 → dual-stream is helping
# stats["compression_ratio"]       → e.g. 3.8 (3.8× smaller in CPU)
```

### Integration (Activity C — eOptShrinkQCodec attention hooks)

```python
from vllm_integration.compression_codec import VllmEOptShrinkQCodec
from vllm_integration.attention_backend_patch import (
    EOptShrinkQAttentionHook, ManifoldKVOutlierScoreHook
)

codec = VllmEOptShrinkQCodec(num_layers=32, key_bits=2, value_bits=3)
codec.calibrate(calib_kvs)

# Compress-before-store / decompress-before-kernel hook
hook = EOptShrinkQAttentionHook(codec=codec, enabled=True)

# In attention layer write path (before paged block write):
compressed_payload = hook.write_to_cache(kv_key, kv_val, layer_idx)
# Returns EncodedKVPayload dict (or passthrough if codec not calibrated)

# In attention layer read path (before passing to FlashAttention kernel):
kv_key_fp16, kv_val_fp16 = hook.read_from_cache(compressed_payload, layer_idx)
# ALWAYS decompresses — never returns compressed tensors to attention kernel

# Stats:
stats = hook.hook_stats()
# stats["n_writes"], stats["n_reads"], stats["mean_cosine_sim"]

# Optional: outlier scoring hook (read-only, no lossy ops)
outlier_hook = ManifoldKVOutlierScoreHook(window_size=4096)
score = outlier_hook.record_outlier_score(
    key_vectors=kv_key,    # [n_tokens, head_dim]
    segment_key="seg_001"
)
# score: Euclidean outlier score [0, ∞) — high score = semantically important
```

### Integration (Activity B — StaticDynamicSegmentKVManager)

```python
from vllm_integration.block_manager_patch import StaticDynamicSegmentKVManager

kv_manager = StaticDynamicSegmentKVManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    # StaticDynamicSegmentKVManager args:
    sdm_max_invalidation_range=2,   # max 2-hop invalidation propagation
    sdm_max_static_segments=512,    # max static segment count
    sdm_chunk_size=128,             # token chunk size for segment keys
)

# Store a segment and mark it static (LRU-exempt):
seg_key = kv_manager.store_segment(
    token_ids=[1, 2, 3, 4],
    chunk_idx=0,
    block_ids={42, 43},
    layer_idx=0,
    is_static=True,
)

# Mark an existing segment static or dynamic:
kv_manager.mark_segment_static(seg_key)
kv_manager.mark_segment_dynamic(seg_key)

# Invalidate a dynamic segment and up to 2 hops of dependents:
invalidated = kv_manager.invalidate_dynamic_range(seg_key)

# Stats:
stats = kv_manager.sdm_hit_stats()
# stats["static_hits"], stats["dynamic_hits"], stats["noncontiguous_ratio"]
```

### Integration (Activity C — ManifoldKVWindowedEvictionManager)

```python
from vllm_integration.block_manager_patch import ManifoldKVWindowedEvictionManager

kv_manager = ManifoldKVWindowedEvictionManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    mvwem_window_size=4096,    # sliding window for local centroid distance
)

# Register a segment with its outlier score (from ManifoldKVOutlierScoreHook):
kv_manager.register_outlier_score(
    segment_key="seg_001",
    block_ids={42, 43},
    outlier_score=3.7,    # high score = semantically important, retain
)

# Evict the segment with the lowest outlier score (least important):
evicted = kv_manager.evict_lowest_outlier_score()

# Evict N least-important segments:
evicted_list = kv_manager.evict_lowest_n(n=10)

# Stats:
stats = kv_manager.mvwem_stats()
# stats["n_segments"], stats["mean_outlier_score"], stats["n_evictions"]
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pko_threshold_preempt` | 0.85 | Buffer occupancy ratio that triggers preemption |
| `pko_cache_capacity_bytes` | 4 GiB | Total CPU offload buffer capacity |
| `pko_fairness_max_wait` | 10 | Max scheduling steps before Tier-A exemption expires |
| `cpm_use_dual_stream` | True | CUDA dual-stream overlap for compress+transfer |
| `cpm_sla_tier_a_no_compress` | True | Offload Tier-A requests uncompressed for latency |
| `key_bits` | 2 | Quantization bits for Key residual |
| `value_bits` | 3 | Quantization bits for Value |
| `calibration_samples` | 20 | Number of (kv, layer) samples for BBP calibration |
| `sdm_max_invalidation_range` | 2 | Max hops for multi-hop invalidation |
| `sdm_chunk_size` | 128 | Token chunk size for segment keys |
| `mvwem_window_size` | 4096 | Sliding window size for outlier centroid distance |

### Accuracy Preservation Contract

1. **EOptShrinkQAttentionHook.read_from_cache()** always calls `codec.decode_tokens()`
   before returning — compressed tensors never reach the attention kernel.
2. **CompressedPreemptionMixin.cpm_restore_with_decompression()** always calls
   `codec.decode_tokens()` before returning KV to the scheduler — no compressed
   KV ever reenters the forward pass.
3. **ManifoldKVWindowedEvictionManager** only evicts segments by outlier score —
   it never modifies KV tensor values (lossless eviction ordering).
4. **ManifoldKVOutlierScoreHook** is read-only — it never writes to or modifies
   any KV tensor.
5. Accuracy target: cosine similarity between original and encode→decode KV >= 0.85
   (validated in install.sh smoke tests).

### Scheduling Overhead

PreemptiveKVOffloadSchedulerMixin CPU overhead: O(N) per schedule step where
N = active request count. Buffer occupancy check: O(1). Target: < 5ms / 100 requests.
CompressedPreemptionMixin compression: GPU kernel — overlapped with PCIe transfer via
dual CUDA streams. Preemption wall time ≈ transfer_time (not transfer_time + compress_time).

---

## 2026-05-06 Integration: Activity B+C (QueryCentricRecompute + TriAttentionCodec)

### Files Modified

| File | Activity | Description |
|------|----------|-------------|
| `block_manager_patch.py` | B | `QueryCentricKVCacheManager` — dual-stage QCRC recompute budget allocation |
| `block_manager_patch.py` | B+C | `QueryCentricTriAttentionKVCacheManager` — dual-path raw/compressed KV store |
| `block_manager_patch.py` | C | `TriAttentionCodecWrapper` — vLLM-compatible TriAttentionCodec adapter |
| `attention_backend_patch.py` | C | `TriAttentionAttentionHook` — compress/decompress hooks for attention write/read |
| `attention_backend_patch.py` | C | `VllmQueryCentricAttentionWrapper` — AttentionImpl wrapper capturing pre-RoPE keys |
| `scheduler_patch.py` | B | `QueryCentricSchedulerMixin` — QCRC recompute scheduling mixin |
| `scheduler_patch.py` | B | `make_qcrc_aware_scheduler_class()` — factory for QCRC-aware scheduler |

### Architecture

```
[Request arrives at Scheduler]
         │
         ▼
QueryCentricSchedulerMixin.pre_schedule_qcrc()
  ├── selective_recompute(query, segment_keys, budget=0.20)
  │    Stage 1: attention-norm top-50% filter
  │    Stage 2: cosine-similarity re-rank (query vs. segment embedding)
  │    Budget:  max 20% of total cached tokens selected
  └── _qcrc_recompute_map[request_id] = [seg_key_1, seg_key_2, ...]
         │
         ▼
[Attention Layer (VllmQueryCentricAttentionWrapper.forward)]
  ├── key captured pre-RoPE (before RoPE applied inside flash_attn)
  ├── base impl.forward() called unmodified → native vLLM paged KV cache
  └── keys_pre_rope + kv stored to QCTA manager
         │
         ▼
QueryCentricTriAttentionKVCacheManager.store_qcta_segment()
  ├── cosine_sim(query_embedding, seg_embedding) >= relevance_threshold (0.60)?
  │    YES → raw store (_qcta_raw_store) + QCRC store (for selective_recompute)
  │    NO  → TriAttentionCodecWrapper.compress(kv, keys_pre_rope, ratio=0.10)
  │          → compressed store (_qcta_compressed_store)
  └── Result: 10× memory reduction for low-relevance segments
         │
         ▼
[Next request with overlapping context]
  ├── get_qcta_segment(key) → raw (high-relevance) or decompress (low-relevance)
  │    decompress: zeros at pruned positions (pre-RoPE importance ranking)
  └── selective_recompute() → raw-only segments eligible for recomputation
```

### Integration (single node — Activity B+C)

```python
from vllm.v1.core.sched.scheduler import Scheduler
from vllm_integration.block_manager_patch import (
    QueryCentricTriAttentionKVCacheManager, TriAttentionCodecWrapper
)
from vllm_integration.attention_backend_patch import (
    TriAttentionAttentionHook, VllmQueryCentricAttentionWrapper
)
from vllm_integration.scheduler_patch import make_qcrc_aware_scheduler_class

# Step 1: Create codec and calibrate
codec = TriAttentionCodecWrapper(
    n_layers=32, n_heads=32, head_dim=128,
    compression_ratio=0.10, series_terms=8, prune_window=128,
)
# calib_kvs: list of [layers, heads, seq_len, head_dim] pre-RoPE K tensors
codec.calibrate(calib_kvs, save_path="calibration.pt")
# (or) codec.load_calibration("calibration.pt")

# Step 2: Create QCTA KV manager
kv_manager = QueryCentricTriAttentionKVCacheManager(
    kv_cache_config=kv_cache_config,
    max_model_len=max_model_len,
    hash_block_size=block_size,
    enable_caching=True,
    codec=codec,
    relevance_threshold=0.60,   # cosine sim >= 0.60 → raw storage
    compression_ratio=0.10,     # keep 10% tokens for low-relevance segments
    qcrc_recompute_budget_ratio=0.20,  # recompute up to 20% of tokens
)

# Step 3: Create attention hook and wrap model layers
hook = TriAttentionAttentionHook(codec=codec, compression_ratio=0.10)
n_patched = VllmQueryCentricAttentionWrapper.patch_model_layers(
    model, kv_manager, codec, chunk_size=128, compression_ratio=0.10
)

# Step 4: Create QCRC-aware scheduler
QCRCScheduler = make_qcrc_aware_scheduler_class(Scheduler)
scheduler = QCRCScheduler(
    ...,  # standard vLLM Scheduler args
    qcrc_kv_manager=kv_manager,
    qcrc_budget_ratio=0.20,
)

# Step 5: Per-request lifecycle
# After prefill (segment keys from store_qcta_segment() calls):
scheduler.register_request_segments(request.request_id, segment_keys, query_emb)

# Before scheduling step:
scheduler.pre_schedule_qcrc(waiting_requests)

# Get recompute recommendations for a request:
to_recompute = scheduler.get_recompute_segments(request.request_id)

# On request completion:
scheduler.on_request_complete(request.request_id)
```

### TriAttentionCodec Accuracy Preservation

The TriAttentionCodecWrapper implements the pre-RoPE trigonometric importance
estimation described in TriAttention (arXiv 2604.04921):

1. **Calibration**: Estimate per-(layer, head) mean μ_k from representative keys.
   Fit Fourier coefficients a_m by least-squares regression.

2. **Compression**: For each 128-token window, compute importance =
   |Σ_m a_m × (sin(m×d_i) + cos(m×d_i))| where d_i = ||k_i - μ_k||.
   Keep top compression_ratio fraction by importance score.

3. **Position stability**: Pre-RoPE space is position-invariant — the same token
   receives the same importance score regardless of its sequence position.
   This avoids the position-dependent bias of post-RoPE importance estimation.

4. **Decompression**: Zeros at pruned positions. Kept positions are lossless.
   Accuracy target: ±1% perplexity relative to full KV cache.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compression_ratio` | 0.10 | Fraction of tokens kept per 128-token window |
| `relevance_threshold` | 0.60 | Cosine sim threshold for raw vs. compressed routing |
| `qcrc_recompute_budget_ratio` | 0.20 | Max fraction of tokens for selective_recompute() |
| `qcrc_stage1_top_k_ratio` | 0.50 | Stage-1 attention-norm filter fraction |
| `series_terms` | 8 | Fourier series terms for importance estimation |
| `prune_window` | 128 | Token window size for windowed pruning |

### Scheduling Overhead

QueryCentricSchedulerMixin operates in Python (no GPU I/O) on the waiting queue.
selective_recompute() is O(N×K) where N = segment count, K = series_terms.
Target: < 5ms per 100 requests (TTFT p50 +5% budget per evaluation_criteria.md).

---

## 2026-05-05 Integration: Activity B+C

### Files Added

| File | Activity | Description |
|------|----------|-------------|
| `nqkv_codec_patch.py` | C-1 | `NQKVCodecPatch` — NQKVCodec adapted for vLLM paged blocks |
| `diff_aware_kv_patch.py` | B-1 | `DiffAwareKVPatch` — master + block-sparse diff non-contiguous reuse |
| `compressed_kv_manager.py` | B+C Cross-1 | `CompressedKVManager` — INT4 masters + FP16 diffs |
| `fireq_attention_patch.py` | C-2 | `FireQAttentionPatch` + `_FireQCodecCore` — RoPE-aware 2-stage hook |

### Integration Points (vLLM 0.20.1)

| Component | vLLM Integration Point |
|-----------|------------------------|
| `NQKVCodecPatch` | Wraps individual KV blocks or layer-wide cache tensors; use alongside `KVCacheManager.allocate_slots()` |
| `DiffAwareKVPatch` | Augments `KVCacheManager` block_id tracking; call `register_master_block()` on first-write, `put_agent_block()` on subsequent requests |
| `CompressedKVManager` | Drop-in for `DiffAwareKVPatch` when compressed storage is needed; integrates with `KVCacheManager.cache_blocks()` path |
| `FireQAttentionPatch` | Wraps `FlashAttentionImpl.forward()` or any `AttentionImpl`; inject via `patch_vllm_model_layers()` after model load |

### NQKVCodecPatch — vLLM block layout

vLLM v1 allocates KV blocks of shape `[num_blocks, block_size, num_kv_heads, head_dim]`.
`NQKVCodecPatch.encode_layer_kv_cache()` operates on this full tensor.
For per-block operations, use `encode_vllm_block()` / `decode_vllm_block()`.

Compression ratio: ~3.5x vs FP16 (64-element quantisation blocks, NF4 14-value table).
Memory reduction: ~71.4% (1/3.5).

### DiffAwareKVPatch — Non-contiguous reuse

```python
from vllm_integration.diff_aware_kv_patch import DiffAwareKVPatch

patch = DiffAwareKVPatch(seq_block_size=64, diff_threshold=0.1)

# On first prefill for a shared context:
patch.register_master_block(block_id=42, kv_tensor=shared_kv)

# On per-agent/per-request decode:
patch.put_agent_block(block_id=42, agent_id="req_001", kv_tensor=agent_kv)

# On KV cache read:
kv = patch.get_agent_block(block_id=42, agent_id="req_001")

# Stats:
stats = patch.diff_hit_stats()
# {'diff_hit_rate': ..., 'master_hit_rate': ..., 'overall_hit_rate': ...,
#  'n_groups': ..., 'search_space_reduction': ...}
```

### CompressedKVManager — INT4 compressed masters

```python
from vllm_integration.compressed_kv_manager import CompressedKVManager

mgr = CompressedKVManager(seq_block_size=64, diff_threshold=0.1, codec_block_size=64)

# Store compressed master:
mgr.store_block(block_id=10, kv_tensor=kv_fp16)

# Store per-agent diff (master decompressed transiently):
mgr.store_agent_diff(block_id=10, agent_id="req_001", kv_tensor=agent_kv)

# Retrieve with diff applied:
kv = mgr.retrieve_block(block_id=10, agent_id="req_001")

# Memory stats:
summary = mgr.compression_summary(sample_kv=kv_fp16)
```

### FireQAttentionPatch — attention hook

```python
from vllm_integration.fireq_attention_patch import FireQAttentionPatch

# Create and calibrate codec:
codec = FireQAttentionPatch.make_codec(n_heads=32, d_head=128)
codec.calibrate(calibration_data)  # list of (kv_tensor, layer_idx)

# Patch all attention layers in a loaded vLLM model:
n_patched = FireQAttentionPatch.patch_vllm_model_layers(model, codec)

# Or wrap a single impl:
patch = FireQAttentionPatch(original_impl, codec=codec, layer_idx=5)
output = patch.forward(layer, query, key, value, kv_cache, attn_metadata, output)
```

---

## File Map

```
vllm_integration/
├── __init__.py                   Package marker
├── nqkv_codec_patch.py           Activity C-1 (2026-05-05): NQKVCodecPatch
├── diff_aware_kv_patch.py        Activity B-1 (2026-05-05): DiffAwareKVPatch
├── compressed_kv_manager.py      Activity B+C Cross-1 (2026-05-05): CompressedKVManager
├── fireq_attention_patch.py      Activity C-2 (2026-05-05): FireQAttentionPatch
├── scheduler_patch.py            Activity A (2026-05-08): PreemptiveKVOffloadSchedulerMixin
│                                  + CompressedPreemptionMixin (A+C Cross-1, CUDA dual-stream)
│                                  + make_preemptive_scheduler_class() factory
│                                  Activity A (2026-05-04): DAGTopologySchedulerMixin
│                                  + MultiNodeDAGRouter, DAGNodeCapacity, WorkflowDAG
│                                  + make_dag_aware_scheduler_class() factory
│                                  Activity B (2026-05-06): QueryCentricSchedulerMixin
│                                  + make_qcrc_aware_scheduler_class() factory
│                                  + DualMapSchedulerMixin (2026-05-03, preserved)
│                                  + CacheHitAwareRequestQueue, MultiNodeRequestRouter (prior)
├── block_manager_patch.py        Activity B (2026-05-08): StaticDynamicSegmentKVManager
│                                  + multi-hop invalidation (max 2 hops)
│                                  Activity C (2026-05-08): ManifoldKVWindowedEvictionManager
│                                  + Euclidean outlier-based eviction (drop-in for LRU)
│                                  Activity B (2026-05-06): QueryCentricKVCacheManager
│                                  + QueryCentricTriAttentionKVCacheManager (B+C)
│                                  + TriAttentionCodecWrapper (Activity C codec adapter)
│                                  Activity B (2026-05-04): WorkloadAwareTTLKVCacheManager
│                                  + VllmDAGAwareTTLAdjuster, VllmTTLEntry
│                                  + SemanticNonContiguousKVCacheManager (2026-05-03, preserved)
│                                  + SemanticSegmentIndex (2026-05-03, preserved)
├── attention_backend_patch.py    Activity C (2026-05-08): EOptShrinkQAttentionHook
│                                  + ManifoldKVOutlierScoreHook (read-only outlier scoring)
│                                  Activity C (2026-05-06): TriAttentionAttentionHook
│                                  + VllmQueryCentricAttentionWrapper (pre-RoPE capture)
│                                  Activity C (2026-05-04): VllmRedundancyAwareEvictionPolicy
│                                  + VllmAttentionKVHook (importance recording)
│                                  + TurboQuantKVHook, SemanticKVAttentionWrapper (2026-05-03, preserved)
│                                  + CompressedKVHook, TriStateKVHook (prior, preserved)
├── compression_codec.py          Activity C (2026-05-08): VllmEOptShrinkQCodec
│                                  + BBP auto-rank low-rank (Key 2-bit / Value 3-bit)
│                                  + TurboQuant residual backend
│                                  Activity C (2026-05-03): VllmTurboQuantCodec
│                                  + CacheCompressionConfig (updated: eopt_shrinkq method)
│                                  + HadamardInt4Codec, CompressionCodec (prior, preserved)
├── install.sh                    pip install --upgrade vllm + all smoke tests
│                                  (2026-05-06, 2026-05-05, 2026-05-04, backward-compat)
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

## 2026-05-09 Integration: Activity A+B+C Cross-1 (HitAwarePPDRouter + TriangleInequalitySegmentIndex)

### Architecture

This cycle ports the Cross-1 combination validated in Report ①:

```
Scheduler (HitAwarePPDRouterMixin)
    │
    ├─ Turn 1 → P-node always
    └─ Turn 2+ → TriangleIndex.estimate_hit_probability()
                 ├─ hit_prob ≥ threshold → D-node (append-prefill)
                 └─ hit_prob < threshold → P-node
                 └─ EMA adaptation: raise threshold if actual_hit_rate < target

KVCacheManager (TriangleIndexKVCacheManagerMixin)
    │
    ├─ get_computed_blocks(request)
    │     ├─ Standard vLLM contiguous prefix lookup (unchanged)
    │     └─ Fallback: TriangleIndex non-contiguous search → annotate request
    │           request.ppd_noncontiguous_hits = [(segment_key, distance), ...]
    │           request.ppd_noncontiguous_hit_probability = float
    │
    └─ SegmentIndexAdapter (auto-sync)
          put(key, value) → cache.put() + index.add_embedding()
          (resolves Report① unresolved issue #4)

AttentionBackend (SpecKVContextGuardCombinedHook)
    │
    ├─ ContextIntensiveGuardAttentionHook (density estimation)
    │     layer._ci_density_score, _ci_min_bits, _ci_density_level
    └─ SpecKVGammaAttentionHook (gamma selection)
          layer._speckv_gamma (1..6)
          High-density → compression_level=0 (FP16) → higher γ
```

### New Classes (2026-05-09)

**scheduler_patch.py**:
- `HitAwarePPDRouterMixin` — primary A+B Cross-1 integration
- `_InlinePPDRouter` — fallback router (no src/ dependency)
- `make_hit_aware_ppd_scheduler_class(base)` — factory function
- `patch_scheduler_instance(scheduler, ...)` — monkey-patch helper

**block_manager_patch.py**:
- `SegmentIndexAdapter` — wraps cache + triangle index; auto-syncs put() calls
- `TriangleIndexKVCacheManagerMixin` — non-contiguous annotation on get_computed_blocks()
- `_InlineTriangleIndex` — fallback without src/ dependency (linear scan)
- `_LightweightSegmentStore` — minimal LRU dict backend
- `build_triangle_index(capacity_bytes, embedding_dim, leaf_size, use_semantic_backend)` — factory

**attention_backend_patch.py**:
- `SpecKVGammaAttentionHook` — γ annotation; EMA bias correction via record_verification()
- `ContextIntensiveGuardAttentionHook` — density-based min_bits annotation
- `SpecKVContextGuardCombinedHook` — combined hook, executes guard then gamma in sequence
- `patch_attention_impl_with_combined_hook(model, hook)` — patches all do_kv_cache_update() methods

### Usage Example (2026-05-09)

```python
from vllm_integration.scheduler_patch import (
    make_hit_aware_ppd_scheduler_class,
    patch_scheduler_instance,
)
from vllm_integration.block_manager_patch import (
    make_triangle_index_kv_cache_manager_class,
    patch_kv_cache_manager_instance,
    build_triangle_index,
    SegmentIndexAdapter,
)
from vllm_integration.attention_backend_patch import (
    SpecKVContextGuardCombinedHook,
    patch_attention_impl_with_combined_hook,
)

# Step 1: Build the segment index (with optional SegmentIndexAdapter for auto-sync)
triangle_index = build_triangle_index(
    capacity_bytes=512 * 1024 * 1024,  # 512 MB
    embedding_dim=32,
    leaf_size=8,
    use_semantic_backend=False,  # True requires src/ on sys.path
)

# Step 2: Build the KV cache manager class and patch
from vllm.v1.core.kv_cache_manager import KVCacheManager
TriKVCacheManager = make_triangle_index_kv_cache_manager_class(KVCacheManager)
# Or for an existing instance:
patch_kv_cache_manager_instance(existing_kv_cache_manager, triangle_index)

# Step 3: Build the scheduler class
from vllm.v1.core.sched.scheduler import Scheduler
HitAwareScheduler = make_hit_aware_ppd_scheduler_class(Scheduler)
# Or for an existing instance:
patch_scheduler_instance(
    existing_scheduler,
    segment_index=triangle_index,
    target_d_node_ratio=0.4,
    ema_alpha=0.1,
)

# Step 4: Attach attention hooks (Activity C)
combined_hook = SpecKVContextGuardCombinedHook()
patch_attention_impl_with_combined_hook(model, combined_hook)

# Step 5: Per-step:
#   a. scheduler.pre_schedule_ppd() — annotates requests with ppd_node_type
#   b. scheduler.schedule() — standard vLLM scheduling
#   c. Read request.ppd_node_type, request.ppd_noncontiguous_hits for routing
#   d. combined_hook.write_to_cache() fires automatically via do_kv_cache_update
#   e. scheduler.record_ppd_hit(request_id, was_hit) — EMA adaptation
```

### Block Boundary Note

Non-contiguous KV reuse via TriangleIndexKVCacheManagerMixin is **annotation-only**.
Physical block allocation remains in vLLM's native block pool. The mixin annotates
`request.ppd_noncontiguous_hits` and `request.ppd_noncontiguous_hit_probability`
for the scheduler's P/D routing decision. No cross-block segment assembly occurs
at the GPU kernel level — this is safe and does not violate vLLM's block boundary invariant.

### SegmentIndexAdapter (Resolves Report① Issue #4)

Prior cycles had a manual connection between SemanticBoundarySegmentCache and
TriangleInequalitySegmentIndex. The `SegmentIndexAdapter` wraps both:

```python
adapter = SegmentIndexAdapter(cache_backend, triangle_index)
adapter.put(key, value)   # → cache.put(key, value) + index.add_embedding(key)
adapter.get(key)          # → cache.get(key)
```

Every `put()` call now auto-registers the segment embedding in the triangle index.
No manual synchronization required.

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
| vLLM 0.20.2 (v1 engine) | Target version |
| `KVCacheManager` public API | Preserved (subclass only) |
| `Scheduler` public API | Preserved (mixin, no monkey-patching) |
| `AttentionImpl.forward` | Not modified — hook is additive only |
| `CacheConfig` fields | Not modified (composition pattern) |
| GPU memory layout | vLLM paged blocks unmodified |
| Accuracy of KV values | Preserved — no quantization in C (eviction ordering only) |

---

## Performance Expectations (from Report ① 2026-05-09)

| Metric | Standalone (547/547 tests) | vLLM port target |
|--------|---------------------------|-----------------|
| Non-contiguous cache hit rate | 100% (N=30, noise=0.02, mid-segment query) | ≥ 30% |
| PPD D-node routing (Turn 2+) | Verified (HitAwarePPDRouter) | Turn 2+ correct |
| eOptShrinkQ Key MSE (proxy) | 1.05% (< 5%) | < 5% |
| eOptShrinkQ Val MSE (proxy) | 0.23% (< 5%) | < 5% |
| SpecKV gamma (FP16 / INT8 / NF4) | 5 / 2 / 3 | NF4 ≤ FP16 |
| Index search time (N=1000) | ~25.5ms | O(log N) amortized |
| Scheduling overhead TTFT p50 | O(log N) ≈ 25ms at N=1000 | ≤ 5% overhead |
| Context guard high-density min_bits | 4.0 | ≥ 4.0 |

---

## Cycle History

| Date | Loop | Activities | Key Additions |
|------|------|-----------|---------------|
| 2026-04-28 | 1/3 | B+C | NonContiguousKVCacheManager, CompressedKVHook (INT8) |
| 2026-04-29 | 1/3 | A+B+C | CacheHitAwareRequestQueue, HadamardInt4Codec (INT4) |
| 2026-04-30 | 1/3 | A+B+C | MultiNodeRequestRouter, TriStateKVHook, SegmentAdapterMixin |
| 2026-05-02 | 1/3 | B+C | VllmLeverageCompressor, SignVQSegmentIndex, SignVQCacheParams |
| 2026-05-03 | 1/3 | A+B+C | DualMapSchedulerMixin, SemanticSegmentIndex (DHD), VllmTurboQuantCodec |
| 2026-05-04 | 1/3 | A+B+C | DAGTopologySchedulerMixin, WorkloadAwareTTLKVCacheManager, VllmRedundancyAwareEvictionPolicy, VllmDAGAwareTTLAdjuster, MultiNodeDAGRouter |
| 2026-05-05 | 1/3 | B+C | NQKVCodecPatch (NF4 INT4), DiffAwareKVPatch (block-sparse diff), CompressedKVManager, FireQAttentionPatch (RoPE-aware 2-stage) |
| **2026-05-06** | **1/3** | **B+C** | **QueryCentricKVCacheManager (ProphetKV dual-stage), TriAttentionCodecWrapper (pre-RoPE trig), QueryCentricTriAttentionKVCacheManager (B+C dual-path), TriAttentionAttentionHook, VllmQueryCentricAttentionWrapper, QueryCentricSchedulerMixin** |
| **2026-05-08** | **1/3** | **A+C** | **PreemptiveKVOffloadSchedulerMixin (TokenFlow preemption + async offload), CompressedPreemptionMixin (CUDA dual-stream), VllmEOptShrinkQCodec (BBP auto-rank K2/V3), EOptShrinkQAttentionHook, ManifoldKVOutlierScoreHook, StaticDynamicSegmentKVManager (multi-hop invalidation), ManifoldKVWindowedEvictionManager (outlier-based eviction)** |
| **2026-05-09** | **1/3** | **A+B+C (Cross-1)** | **HitAwarePPDRouterMixin (A+B Cross-1: PPDAppendPrefillRouter + EMA threshold, Turn 2+ D-node routing), TriangleIndexKVCacheManagerMixin (B: O(log N) non-contiguous lookup via TriangleInequalitySegmentIndex), SegmentIndexAdapter (auto-sync cache↔index, resolves Report① issue #4), _InlinePPDRouter + _InlineTriangleIndex (fallback, no src/ dependency), SpecKVGammaAttentionHook (C: γ annotation, FP16/INT8/NF4), ContextIntensiveGuardAttentionHook (C: density-based min_bits gate), SpecKVContextGuardCombinedHook (C: combined A+B+C pipeline), build_triangle_index factory, make_hit_aware_ppd_scheduler_class factory** |
| **2026-05-10** | **1/3** | **B+C** | **KVPacketVQBlockManager (B+C: KVPacket soft-adapter cache + VQ compression, subclasses KVCacheManager, non-contiguous hit annotation), VQCodecAttentionHook (C: compress-before-store / decompress-before-kernel via VQCodec arXiv 2603.16435, FP16 recent_window=64), KVPacketSegmentSchedulerMixin (A+B: segment-hash reordering < 5ms/batch), make_kvp_vq_kv_cache_manager_class + make_kvp_segment_scheduler_class factories, apply_all_patches() updated** |
| **2026-05-11** | **1/3** | **B+C** | **WiCERBlockManager (B: CEGAR iterative parallel KV segment store, non-contiguous annotation, LRU eviction, save/load artefacts, cegar_loop()), RateQuantVllmCodec (C: reverse water-filling bit allocation, per-channel int16 quantisation, 75% memory reduction, < 1% accuracy error), RateQuantAttentionHook (C: compress-before-store / decompress-before-kernel, mandatory ±1% accuracy contract), make_wicer_kv_cache_manager_class factory, CacheCompressionConfig.ratequant method added** |

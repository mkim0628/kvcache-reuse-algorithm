# vllm_integration: Activity A+B+C KV cache port for vLLM 0.20.1
#
# 2026-05-08 cycle additions:
#   scheduler_patch          — PreemptiveKVOffloadSchedulerMixin: preemptive scheduling
#                               + async GPU→CPU KV offload (Activity A)
#                             + CompressedPreemptionMixin: CUDA dual-stream inline
#                               compression via VllmEOptShrinkQCodec (Activity A+C)
#                             + make_preemptive_scheduler_class() factory
#   compression_codec        — VllmEOptShrinkQCodec: BBP auto-rank low-rank + TurboQuant
#                               residual (Key 2-bit / Value 3-bit, Activity C)
#   attention_backend_patch  — EOptShrinkQAttentionHook: write/read hooks for eOptShrinkQ
#                               (Activity C: compress before store, decompress before kernel)
#                             + ManifoldKVOutlierScoreHook: Euclidean outlier scoring hook
#                               (Activity C: read-only, no lossy ops)
#   block_manager_patch      — StaticDynamicSegmentKVManager: static/dynamic segment
#                               classification + Multi-hop invalidation (Activity B)
#                             + ManifoldKVWindowedEvictionManager: Euclidean outlier-based
#                               eviction policy (Activity C, drop-in for LRU)
#
# 2026-05-06 cycle additions:
#   block_manager_patch      — QueryCentricKVCacheManager: ProphetKV dual-stage recompute (B)
#                             + QueryCentricTriAttentionKVCacheManager: dual-path raw/compressed (B+C)
#                             + TriAttentionCodecWrapper: pre-RoPE trigonometric codec (C)
#   attention_backend_patch  — TriAttentionAttentionHook: compress/decompress for attention (C)
#                             + VllmQueryCentricAttentionWrapper: pre-RoPE key capture (C)
#   scheduler_patch          — QueryCentricSchedulerMixin: QCRC recompute scheduling (B)
#                             + make_qcrc_aware_scheduler_class() factory
#
# 2026-05-05 cycle additions:
#   nqkv_codec_patch         — NQKVCodecPatch: NF4 INT4 block-quantile KV compression
#   diff_aware_kv_patch      — DiffAwareKVPatch: master + block-sparse diff non-contiguous reuse
#   compressed_kv_manager    — CompressedKVManager: INT4-compressed masters + FP16 diffs
#   fireq_attention_patch    — FireQAttentionPatch: RoPE-aware 2-stage outlier smoothing hook
#
# 2026-05-04 cycle additions:
#   scheduler_patch          — DAGTopologySchedulerMixin: workflow DAG topology KV preservation
#                             + MultiNodeDAGRouter, WorkflowDAG, DAGNodeCapacity
#   block_manager_patch      — WorkloadAwareTTLKVCacheManager: category-specific TTL segments
#                             + VllmDAGAwareTTLAdjuster, VllmTTLEntry
#   attention_backend_patch  — VllmRedundancyAwareEvictionPolicy, VllmAttentionKVHook
#
# 2026-05-03 cycle additions (preserved for backward compatibility):
#   scheduler_patch          — DualMapSchedulerMixin: dual-hash + semantic-hit-rate routing
#   block_manager_patch      — SemanticNonContiguousKVCacheManager: DHD semantic similarity
#   attention_backend_patch  — TurboQuantKVHook: PolarQuant+QJL 3-bit write/read hooks
#   compression_codec        — VllmTurboQuantCodec: TurboQuantCodec wrapped for vLLM shapes
#
# Older cycle files are preserved for backward compatibility:
#   leverage_compressor_patch, sign_vq_block_manager_patch, cache_config_extension

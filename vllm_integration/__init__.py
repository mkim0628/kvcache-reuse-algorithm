# vllm_integration: Activity A+B+C KV cache port for vLLM 0.20.1
#
# 2026-05-03 cycle additions:
#   scheduler_patch          — DualMapSchedulerMixin: dual-hash + semantic-hit-rate routing
#   block_manager_patch      — SemanticNonContiguousKVCacheManager: DHD semantic similarity
#                             non-contiguous block mapping on top of KVCacheManager
#   attention_backend_patch  — TurboQuantKVHook: PolarQuant+QJL 3-bit write/read hooks
#   compression_codec        — VllmTurboQuantCodec: TurboQuantCodec wrapped for vLLM shapes
#
# Previous cycle files are preserved for backward compatibility:
#   leverage_compressor_patch, sign_vq_block_manager_patch, cache_config_extension

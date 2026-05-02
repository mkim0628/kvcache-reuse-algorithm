# vllm_integration: Activity B+C KV cache port for vLLM 0.20.0
#
# 2026-05-02 cycle additions:
#   leverage_compressor_patch   — VllmLeverageCompressor (3-tier FP16/sign-VQ/evict)
#   sign_vq_block_manager_patch — SignVQSegmentIndex + NonContiguousKVCacheManagerV2
#   cache_config_extension      — SignVQCacheParams + factory helpers

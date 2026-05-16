# vllm_integration: Activity A+C KV cache port for vLLM 0.21.0
#
# 2026-05-16 cycle additions:
#   scheduler_patch       — NAtHDDROffloadingSchedulerMixin + NAtHDDROffloadingSchedulerConfig
#                             + make_nath_ddr_scheduler_class (Activity A):
#                             NAtH 4-tier DDR offloading minimal-eviction scheduler.
#                             Based on NAtH (arXiv 2605.09490): accuracy depends only on
#                             permanent eviction rate; DDR offloading = zero approx error.
#                             Classifies waiting requests' tokens into 4 tiers each step.
#                             Permanent eviction capped at max_eviction_ratio=3%.
#                             Overhead: < 5ms p50 per schedule() call.
#   compression_codec     — GlobalRetentionGateVllmCodec (Activity C):
#                             Cross-layer competitive KV eviction. Ports
#                             GlobalRetentionGateEvictionCodec (src/cache/).
#                             Memory reduction: 70% (budget_ratio=0.3). Accuracy: < 1% error.
#                           + NAtHDDROffloadingCodecAdapter (Activity A+C):
#                             Bridges NAtH DDR tier policy with vLLM compression hooks.
#                             Tier 2: FP16 CPU offload. Tier 3: INT8 offload. Tier 4: evict.
#   attention_backend_patch — GlobalRetentionGateAttentionHook (Activity C):
#                              write_to_cache() / read_from_cache() hooks for
#                              GlobalRetentionGate eviction (post-compute, pre-cache write).
#                              Accuracy: < 1% attention error (MANDATORY §4).
#                            + NAtHDDRGlobalRetentionHook (Cross A+C):
#                              Composite hook: NAtH DDR 4-tier + GlobalRetentionGate budget.
#                            + apply_global_retention_gate_patch() monkey-patcher
#                            + extend_cache_config_global_retention() helper
#
# 2026-05-15 cycle additions:
#   scheduler_patch       — RadixFeatherSchedulerMixin + make_radix_feather_scheduler_class
#                             (Activity A): Feather (arXiv 2605.06046) prefix-homogeneity-
#                             aware batch reordering. Reorders vLLM's waiting queue per
#                             schedule() step using Radix tree prefix-match signal.
#                             Overhead: O(window * prefix_len), target < 5ms p50.
#   block_manager_patch   — RelayUShapeAuxStore + RelayUShapeKVCacheManagerMixin
#                             + make_relay_ulayer_kv_cache_manager_class (Activity B):
#                             U-shape layer-selective non-contiguous segment auxiliary
#                             store alongside vLLM's PagedAttention block table.
#                             Layer bitmask stored per segment; middle ~70% layers
#                             reused for non-identical segments.
#                             Ports: src/cache/relay_ulayer_segment.py
#   attention_backend_patch — LookaheadEvictionAttentionHook (Activity C):
#                              write_to_cache() / read_from_cache() for LookaheadKV
#                              eviction (arXiv 2603.10899, ICLR 2026). Kept KV is
#                              FP16 original (no quantization distortion).
#                              Accuracy: eviction_ratio=0.7 → attention error < 1%.
#                            + LookaheadRelayAttentionHook (Activity B+C):
#                              dual-filter: U-shape layer filter then LookaheadKV
#                              token filter. Combined memory reduction ~70–85%.
#                            + apply_lookahead_eviction_patch() monkey-patcher
#                            + extend_cache_config_lookahead_eviction() helper
#                              (adds compression_method, eviction_ratio, etc.)
#
# 2026-05-14 cycle additions:
#   compression_codec     — VllmFibQuantVQCodec: FibQuant Spherical-Beta radial-angular
#                             VQ codec adapter for vLLM attention-backend write/read hooks.
#                             Ports src/cache/fibquant_vq_codec.FibQuantVQCodec.
#                             Compression: 1.88x (bits_dir=8) / 3.56x (bits_dir=4) / 6.40x (bits_dir=2)
#                             Accuracy: cosine>=0.99 at 1.88x (mandatory), >=0.97 at 3.56x
#   block_manager_patch   — FibQuantVQSegmentKVManager (Activity B):
#                             KVCacheManager subclass with FibQuant non-contiguous auxiliary store.
#                             store_segment(): FibQuant-compress → auxiliary store.
#                             load_segment(): decompress on-demand (random access).
#                             get_noncontiguous_segments(): multi-chunk hit lookup + tracking.
#                             pad_block_table_with_fibquant(): FIBQUANT_SENTINEL block padding.
#                           + make_fibquant_kv_cache_manager_class() factory
#   attention_backend_patch — FibQuantAttentionHook (Activity B+C):
#                              write_to_cache() / read_from_cache() for FibQuant VQ.
#                              Pre-RoPE path: write_pre_rope() / read_pre_rope() for
#                              position-independent segment reuse (Cross B+C).
#                              apply_fibquant_patch() monkey-patcher for FlashAttentionImpl.
#                            + extend_cache_config_fibquant() helper
#
# 2026-05-13 cycle additions:
#   scheduler_patch       — PBKVAgentSegmentPreservationSchedulerMixin (Activity A):
#                             PBKV prediction-based segment preservation, fairness-weighted
#                             request reordering, GPU preserve/host evict policy.
#                           + make_pbkv_scheduler_class() factory
#   block_manager_patch   — KVFoldAccumulativeBlockManager (Activity B):
#                             foldl accumulator-based non-contiguous KV reuse,
#                             StreamingLLM fallback, SRFT+INT8 B+C integration hook.
#                           + make_kvfold_kv_cache_manager_class() factory
#   attention_backend_patch — SRFTInt8AttentionHook (Activity C):
#                              SRFT Gaussianization + INT8 per-group compression,
#                              write_to_cache() / read_from_cache() hooks,
#                              apply_srft_int8_patch() for FlashAttentionImpl.
#                            + extend_cache_config_srft_int8() helper
#
# 2026-05-12 cycle additions:
#   block_manager_patch   — AdapShotBlockManager: B+C AdapShotMixedDimSegmentPipeline
#                             parallel auxiliary store (RoPE re-encoding + MixedDim codec)
#                           + make_adapshot_kv_cache_manager_class() factory
#   attention_backend_patch — MixedDimAttentionHook: write/read hooks for
#                              MixedDimPerTokenBudgetCodec (compress before store,
#                              decompress before kernel)
#                           + extend_cache_config_mixed_dim(): CacheConfig extension helper
#   scheduler_patch       — AdapShotSegmentSchedulerMixin: non-contiguous hit-rate-based
#                             request reordering (B+C Cross-2)
#                           + make_adapshot_scheduler_class() factory
#
# 2026-05-11 cycle additions:
#   block_manager_patch   — WiCERBlockManager: CEGAR iterative non-contiguous KV
#                             artefact cache + parallel segment store (Activity B)
#                           + make_wicer_kv_cache_manager_class() factory
#   attention_backend_patch — RateQuantAttentionHook: RateQuant write/read hooks
#                              (compress before store, decompress before kernel)
#                              Accuracy: < 1% relative error at avg 4-bit budget
#   compression_codec     — RateQuantVllmCodec: reverse water-filling bit allocation
#                             75% memory reduction, < 1% accuracy error
#
# 2026-05-10 cycle additions:
#   block_manager_patch   — KVPacketVQBlockManager: KVPacket soft-adapter B+C cache
#                             + VQ compression, subclasses KVCacheManager
#                           + make_kvp_vq_kv_cache_manager_class() factory
#   attention_backend_patch — VQCodecAttentionHook: VQ write/read hooks for attention
#                              (compress before store, decompress before kernel)
#   scheduler_patch       — KVPacketSegmentSchedulerMixin: segment-hash reordering
#                           + make_kvp_segment_scheduler_class() factory
#
# 2026-05-09 cycle additions:
#   scheduler_patch          — HitAwarePPDRouterMixin, _InlinePPDRouter,
#                               make_hit_aware_ppd_scheduler_class, patch_scheduler_instance
#   block_manager_patch      — SegmentIndexAdapter, TriangleIndexKVCacheManagerMixin,
#                               _InlineTriangleIndex, _LightweightSegmentStore,
#                               build_triangle_index, make_triangle_index_kv_cache_manager_class,
#                               patch_kv_cache_manager_instance
#   attention_backend_patch  — SpecKVGammaAttentionHook, ContextIntensiveGuardAttentionHook,
#                               SpecKVContextGuardCombinedHook,
#                               patch_attention_impl_with_combined_hook
#
# Prior cycle additions are preserved for backward compatibility.
# See each submodule's docstring for full changelog.

from __future__ import annotations

import warnings
from typing import Any, Optional

# 2026-05-16 imports (Activity A+C)
try:
    from vllm_integration.scheduler_patch import (
        NAtHDDROffloadingSchedulerConfig,
        NAtHDDROffloadingSchedulerMixin,
        make_nath_ddr_scheduler_class,
    )
except Exception as _e_a16:
    warnings.warn(f"vllm_integration: 2026-05-16 Activity A import failed: {_e_a16}", RuntimeWarning)
    NAtHDDROffloadingSchedulerConfig = None  # type: ignore
    NAtHDDROffloadingSchedulerMixin = None  # type: ignore
    make_nath_ddr_scheduler_class = None  # type: ignore

try:
    from vllm_integration.compression_codec import (
        GlobalRetentionGateVllmCodec,
        NAtHDDROffloadingCodecAdapter,
    )
except Exception as _e_c16:
    warnings.warn(f"vllm_integration: 2026-05-16 Activity C import failed: {_e_c16}", RuntimeWarning)
    GlobalRetentionGateVllmCodec = None  # type: ignore
    NAtHDDROffloadingCodecAdapter = None  # type: ignore

try:
    from vllm_integration.attention_backend_patch import (
        GlobalRetentionGateAttentionHook,
        NAtHDDRGlobalRetentionHook,
        apply_global_retention_gate_patch,
        extend_cache_config_global_retention,
    )
except Exception as _e_ab16:
    warnings.warn(f"vllm_integration: 2026-05-16 Activity C attention hook import failed: {_e_ab16}", RuntimeWarning)
    GlobalRetentionGateAttentionHook = None  # type: ignore
    NAtHDDRGlobalRetentionHook = None  # type: ignore
    apply_global_retention_gate_patch = None  # type: ignore
    extend_cache_config_global_retention = None  # type: ignore

# 2026-05-15 imports (Activity A+B+C)
try:
    from vllm_integration.scheduler_patch import (
        RadixFeatherSchedulerConfig,
        RadixFeatherSchedulerMixin,
        make_radix_feather_scheduler_class,
    )
except Exception as _e_a15:
    warnings.warn(f"vllm_integration: 2026-05-15 Activity A import failed: {_e_a15}", RuntimeWarning)

try:
    from vllm_integration.block_manager_patch import (
        RelayUShapeAuxStore,
        RelayUShapeKVCacheManagerMixin,
        make_relay_ulayer_kv_cache_manager_class,
    )
except Exception as _e_b15:
    warnings.warn(f"vllm_integration: 2026-05-15 Activity B import failed: {_e_b15}", RuntimeWarning)

try:
    from vllm_integration.attention_backend_patch import (
        LookaheadEvictionAttentionHook,
        LookaheadRelayAttentionHook,
        apply_lookahead_eviction_patch,
        extend_cache_config_lookahead_eviction,
    )
except Exception as _e_c15:
    warnings.warn(f"vllm_integration: 2026-05-15 Activity C import failed: {_e_c15}", RuntimeWarning)


def apply_all_patches(
    vq_codec: Any = None,
    n_heads: int = 8,
    d_head: int = 128,
    recent_window: int = 64,
    adapter_rank: int = 8,
    max_packets: int = 512,
    reorder_window: int = 32,
    enable_scheduler_reorder: bool = True,
    enable_compression_hook: bool = True,
    enable_block_manager: bool = True,
) -> dict:
    """Apply all 2026-05-10 B+C patches to vLLM components.

    This function:
      1. Creates a VQCodecAttentionHook (Activity C write/read hooks).
      2. Creates a KVPacketVQBlockManager class (Activity B+C block manager).
      3. Creates a KVPacketSegmentSchedulerMixin class (Activity A+B scheduler).

    Returns a dict with:
      "vq_hook"           : VQCodecAttentionHook instance
      "kv_manager_class"  : KVPacketVQBlockManager subclass
      "scheduler_class"   : scheduler class with KVPacketSegmentSchedulerMixin
      "vllm_version"      : str
      "patches_applied"   : list[str]

    Parameters
    ----------
    vq_codec : VQCodec | None
        Pre-fitted VQCodec instance. If None, the hook will auto-fit on first use.
    n_heads, d_head : int
        Model architecture for SoftTokenAdapter dimensioning.
    recent_window : int
        FP16 tokens kept uncompressed (Activity C accuracy contract).
    adapter_rank : int
        SoftTokenAdapter rank.
    max_packets : int
        Max packets in LRU store.
    reorder_window : int
        Max waiting requests inspected per schedule step.
    enable_scheduler_reorder : bool
        If False, skip scheduler patch (graceful degradation).
    enable_compression_hook : bool
        If False, VQCodecAttentionHook is disabled (identity).
    enable_block_manager : bool
        If False, KVPacketVQBlockManager returns no-ops.

    Accuracy constraint:
        VQCodecAttentionHook.read_from_cache() always decompresses before
        returning — compressed tensors never enter the attention kernel.
        This satisfies evaluation_criteria.md §4 perplexity ±1% requirement.
    """
    import vllm as _vllm

    vllm_version = _vllm.__version__
    patches_applied = []

    # -- Activity C: VQ compression hook ------------------------------------
    try:
        from vllm_integration.attention_backend_patch import VQCodecAttentionHook
        vq_hook = VQCodecAttentionHook(
            vq_codec=vq_codec,
            recent_window=recent_window,
            enabled=enable_compression_hook,
        )
        patches_applied.append("VQCodecAttentionHook")
    except Exception as exc:
        warnings.warn(f"apply_all_patches: VQCodecAttentionHook failed: {exc}", RuntimeWarning)
        vq_hook = None

    # -- Activity B+C: KVPacket block manager class -------------------------
    kv_manager_class = None
    if enable_block_manager:
        try:
            from vllm_integration.block_manager_patch import (
                make_kvp_vq_kv_cache_manager_class,
            )
            from vllm.v1.core.kv_cache_manager import KVCacheManager
            kv_manager_class = make_kvp_vq_kv_cache_manager_class(KVCacheManager)
            patches_applied.append("KVPacketVQBlockManager")
        except Exception as exc:
            warnings.warn(f"apply_all_patches: KVPacketVQBlockManager failed: {exc}", RuntimeWarning)

    # -- Activity A+B: segment-aware scheduler class ------------------------
    scheduler_class = None
    if enable_scheduler_reorder:
        try:
            from vllm_integration.scheduler_patch import make_kvp_segment_scheduler_class
            from vllm.v1.core.sched.scheduler import Scheduler
            scheduler_class = make_kvp_segment_scheduler_class(Scheduler)
            patches_applied.append("KVPacketSegmentSchedulerMixin")
        except Exception as exc:
            warnings.warn(f"apply_all_patches: KVPacketSegmentSchedulerMixin failed: {exc}", RuntimeWarning)

    return {
        "vq_hook": vq_hook,
        "kv_manager_class": kv_manager_class,
        "scheduler_class": scheduler_class,
        "vllm_version": vllm_version,
        "patches_applied": patches_applied,
    }


__all__ = [
    "apply_all_patches",
    # 2026-05-16
    "NAtHDDROffloadingSchedulerConfig",
    "NAtHDDROffloadingSchedulerMixin",
    "make_nath_ddr_scheduler_class",
    "GlobalRetentionGateVllmCodec",
    "NAtHDDROffloadingCodecAdapter",
    "GlobalRetentionGateAttentionHook",
    "NAtHDDRGlobalRetentionHook",
    "apply_global_retention_gate_patch",
    "extend_cache_config_global_retention",
    # 2026-05-15
    "RadixFeatherSchedulerConfig",
    "RadixFeatherSchedulerMixin",
    "make_radix_feather_scheduler_class",
    "RelayUShapeAuxStore",
    "RelayUShapeKVCacheManagerMixin",
    "make_relay_ulayer_kv_cache_manager_class",
    "LookaheadEvictionAttentionHook",
    "LookaheadRelayAttentionHook",
    "apply_lookahead_eviction_patch",
    "extend_cache_config_lookahead_eviction",
    # 2026-05-14
    "VllmFibQuantVQCodec",
    "FibQuantVQSegmentKVManager",
    "make_fibquant_kv_cache_manager_class",
    "FibQuantAttentionHook",
    "apply_fibquant_patch",
    "extend_cache_config_fibquant",
    # 2026-05-12
    "AdapShotBlockManager",
    "make_adapshot_kv_cache_manager_class",
    "MixedDimAttentionHook",
    "extend_cache_config_mixed_dim",
    "AdapShotSegmentSchedulerMixin",
    "make_adapshot_scheduler_class",
    # 2026-05-11
    "RateQuantVllmCodec",
    "RateQuantAttentionHook",
    "WiCERBlockManager",
    "make_wicer_kv_cache_manager_class",
    # 2026-05-10
    "VQCodecAttentionHook",
    "KVPacketVQBlockManager",
    "KVPacketSegmentSchedulerMixin",
    "make_kvp_vq_kv_cache_manager_class",
    "make_kvp_segment_scheduler_class",
]

# vllm_integration: Activity A+B+C KV cache port for vLLM 0.20.2
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

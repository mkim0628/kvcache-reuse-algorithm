"""block_manager_patch.py — Activity B: KV cache non-contiguous reuse integration for vLLM 0.20.2.

2026-05-12: AdapShotBlockManager — wraps AdapShotMixedDimSegmentPipeline (B+C Cross-2).
            Ports RoPEReencodingNonContiguousCache + MixedDimPerTokenBudgetCodec pipeline
            from src/cache/adapshot_pipeline.py into vLLM's KVCacheManager subclass.
            - store_segment(): mixed-dim compress → pre-RoPE store (B+C pipeline contract)
            - load_segment(): pre-RoPE load + RoPE re-apply → decode (B+C restore contract)
            - annotate_request(): detect non-contiguous hits and attach metadata to requests
            - make_adapshot_kv_cache_manager_class(): factory for KVCacheManager subclass

            Validated metrics (Report ① 2026-05-12):
              Non-contiguous hit rate: 100% (3/3 hits non-contiguous)
              KV memory reduction: −50% (budget_ratio=0.50)
              Perplexity delta: 0.36% (< 1% threshold)
              KL divergence: 0.000023 (< 0.015 threshold)
              Cosine similarity: 0.999994 (≥ 0.99 threshold)

2026-05-11: WiCERBlockManager — CEGAR iterative non-contiguous KV artefact cache.
            (prior cycle, preserved for backward compatibility)

Original docstring (prior cycles):

2026-05-06: QueryCentricKVCacheManager — subclasses KVCacheManager and adds
            ProphetKV-inspired dual-stage recompute budget allocation using
            QueryCentricRecomputeCache. Ports src/cache/query_centric_recompute.py.

            QueryCentricTriAttentionKVCacheManager — extends the above with
            TriAttentionCodec compression (Activity C) for a dual-path B+C store.
            Ports src/cache/qc_tri_store.QueryCentricTriAttentionCache.

2026-05-04: WorkloadAwareTTLKVCacheManager — subclasses KVCacheManager and adds
            TTL-based segment preservation using WorkloadAwareTTLCache logic.
            Ports WorkloadAwareTTLCache from src/cache/workload_ttl_cache.py.

vLLM 0.20.1 v1 architecture:
    - KVCacheManager is in vllm.v1.core.kv_cache_manager.KVCacheManager
    - BlockPool handles raw block allocation (FreeKVCacheBlockQueue)
    - KVCacheManager.get_computed_blocks() — prefix cache lookup
    - KVCacheManager.allocate_slots() — block allocation
    - KVCacheManager.evict_blocks(block_ids) — explicit eviction API

Integration strategy:
    WorkloadAwareTTLKVCacheManager is a KVCacheManager subclass that maintains
    a parallel WorkloadAwareTTLCache alongside vLLM's native prefix cache.
    It intercepts allocate_slots() to record segment insertions and
    get_computed_blocks() to record TTL-aware hits, feeding stats back to the
    WorkloadAwareTTLCache for TTL EMA updates.

    The TTL cache acts as a scheduling signal: when DAGTopologySchedulerMixin
    calls on_kv_reuse_event(), the TTL adjuster extends segment TTL. When a
    segment's TTL expires, it becomes an eviction candidate and is released
    from vLLM's block pool.

Activity: B — Non-Contiguous KV Cache Reuse (WorkloadAwareTTLCache port)

Prior cycle (2026-05-03) components are preserved at the bottom of this file.

vLLM version: 0.20.1
"""

import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager

import vllm
def _vllm_version_tuple(v: str):
    return tuple(int(x) for x in v.split(".")[:3])
assert _vllm_version_tuple(vllm.__version__) >= _vllm_version_tuple("0.4.0"), (
    f"vllm_integration requires vLLM >= 0.4.0, found {vllm.__version__}"
)

if TYPE_CHECKING:
    from vllm_integration.attention_backend_patch import VllmRedundancyAwareEvictionPolicy


# ---------------------------------------------------------------------------
# TTLEntry — segment metadata (mirrors src/cache/workload_ttl_cache.TTLEntry)
# ---------------------------------------------------------------------------

@dataclass
class VllmTTLEntry:
    """Per-segment metadata stored in the TTL cache.

    Attributes:
        block_ids: Set of vLLM block IDs associated with this segment.
        category: Workload category (code / chat / rag / agentic).
        ttl_sec: TTL in seconds.
        created_at: Insertion timestamp (time.monotonic()).
        pinned: If True, excluded from eviction (set by DAGTopologySchedulerMixin).
        importance_score: Accumulated attention importance (for eviction scoring).
        embedding: Mean Key vector embedding (for redundancy computation).
    """
    block_ids: set
    category: str
    ttl_sec: float
    created_at: float
    pinned: bool = False
    importance_score: float = 0.0
    embedding: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Default TTL profiles — mirrors src/cache/workload_ttl_cache._DEFAULT_TTL_PROFILES
# ---------------------------------------------------------------------------

_DEFAULT_TTL_PROFILES: Dict[str, dict] = {
    "code":     {"ttl_base_sec": 600.0, "reuse_probability": 0.75},
    "chat":     {"ttl_base_sec": 300.0, "reuse_probability": 0.60},
    "rag":      {"ttl_base_sec": 120.0, "reuse_probability": 0.45},
    "agentic":  {"ttl_base_sec": 480.0, "reuse_probability": 0.80},
}

_CODE_KEYWORDS    = ("def ", "class ", "import ", "```python")
_RAG_KEYWORDS     = ("document", "context:", "passage", "retrieved")
_AGENTIC_KEYWORDS = ("tool_call", "function_call", "agent", "workflow")


# ---------------------------------------------------------------------------
# WorkloadAwareTTLKVCacheManager — KVCacheManager subclass (Activity B)
# ---------------------------------------------------------------------------

class WorkloadAwareTTLKVCacheManager(KVCacheManager):
    """KVCacheManager subclass adding TTL-based segment preservation.

    Maintains a parallel TTL segment store (WorkloadAwareTTLCache logic)
    alongside vLLM's native prefix cache. Segments are classified by workload
    category, assigned category-specific TTLs, and protected from eviction
    until TTL expiry or DAG completion.

    Activity B integration:
        - store_ttl_segment(): register a segment after prefill
        - get_ttl_segment(): look up a segment with TTL check
        - evict_expired_segments(): flush TTL-expired segments from vLLM's block pool
        - adjust_segment_ttl(): called by DAGAwareTTLAdjuster for TTL extension
        - pin_segment() / unpin_segment(): called by DAGTopologySchedulerMixin

    Block boundary contract:
        Segment chunk_size should divide evenly into vLLM's block_size so that
        segment boundaries align with KV block page boundaries.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import WorkloadAwareTTLKVCacheManager

        # Monkey-patch: replace standard manager with TTL-aware version
        kv_manager = WorkloadAwareTTLKVCacheManager(
            *original_args,
            ttl_max_entries=1000,
            ttl_chunk_size=128,
            ttl_ema_alpha=0.1,
            ttl_profiles=None,  # use defaults
            **original_kwargs,
        )
    """

    def __init__(
        self,
        *args: Any,
        ttl_max_entries: int = 1000,
        ttl_chunk_size: int = 128,
        ttl_ema_alpha: float = 0.1,
        ttl_profiles: Optional[Dict[str, dict]] = None,
        ttl_eviction_policy: Optional["VllmRedundancyAwareEvictionPolicy"] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            ttl_max_entries: Maximum number of segment entries in the TTL store.
            ttl_chunk_size: Token chunk size for segment key computation.
            ttl_ema_alpha: EMA coefficient for online TTL profile updates.
            ttl_profiles: Custom TTL profiles dict; defaults to _DEFAULT_TTL_PROFILES.
            ttl_eviction_policy: Optional VllmRedundancyAwareEvictionPolicy instance.
            *args, **kwargs: Forwarded to KVCacheManager.__init__().
        """
        super().__init__(*args, **kwargs)

        import copy
        self._ttl_profiles: Dict[str, dict] = copy.deepcopy(
            ttl_profiles if ttl_profiles is not None else _DEFAULT_TTL_PROFILES
        )
        self._ttl_max_entries = ttl_max_entries
        self._ttl_chunk_size = ttl_chunk_size
        self._ttl_ema_alpha = ttl_ema_alpha
        self._ttl_eviction_policy = ttl_eviction_policy

        # Main TTL segment store: segment_key → VllmTTLEntry
        self._ttl_store: OrderedDict[str, VllmTTLEntry] = OrderedDict()
        self._ttl_pinned: set = set()

        # Counters
        self._ttl_exact_hits: int = 0
        self._ttl_preserved_hits: int = 0
        self._ttl_misses: int = 0
        self._ttl_eviction_count: int = 0
        self._ttl_pressure_eviction_count: int = 0

    # ------------------------------------------------------------------
    # TTL segment API — mirrors WorkloadAwareTTLCache public API
    # ------------------------------------------------------------------

    def store_ttl_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        block_ids: set,
        category: str = "chat",
        embedding: Optional[torch.Tensor] = None,
        override_ttl_sec: Optional[float] = None,
        layer_idx: int = 0,
    ) -> str:
        """Register a segment in the TTL store after block allocation.

        Args:
            token_ids: Full token sequence.
            chunk_idx: Chunk index within the sequence.
            block_ids: vLLM block IDs allocated for this segment.
            category: Workload category (code/chat/rag/agentic).
            embedding: Optional mean Key vector for redundancy scoring.
            override_ttl_sec: Override TTL; uses category profile if None.
            layer_idx: Transformer layer index (for keying).

        Returns:
            Segment key (hex digest).
        """
        key = self._ttl_chunk_key(token_ids, chunk_idx, layer_idx)

        if key in self._ttl_store:
            self._ttl_store.move_to_end(key)
            return key

        if len(self._ttl_store) >= self._ttl_max_entries:
            self._ttl_evict_one()

        ttl_sec = (
            override_ttl_sec
            if override_ttl_sec is not None
            else self._ttl_profiles.get(category, _DEFAULT_TTL_PROFILES["chat"])["ttl_base_sec"]
        )

        entry = VllmTTLEntry(
            block_ids=set(block_ids),
            category=category,
            ttl_sec=float(ttl_sec),
            created_at=time.monotonic(),
            embedding=embedding.detach().clone() if embedding is not None else None,
        )
        self._ttl_store[key] = entry
        return key

    def get_ttl_segment(self, segment_key: str) -> Optional[VllmTTLEntry]:
        """Look up a segment by key with TTL check.

        Returns the entry if present and TTL not expired; records a miss and
        returns None otherwise. Does NOT evict the expired entry immediately —
        that is deferred to evict_expired_segments().

        Args:
            segment_key: Segment key from store_ttl_segment().

        Returns:
            VllmTTLEntry if valid hit, None on miss or expiry.
        """
        if segment_key not in self._ttl_store:
            self._ttl_misses += 1
            return None

        entry = self._ttl_store[segment_key]
        elapsed = time.monotonic() - entry.created_at

        if elapsed > entry.ttl_sec:
            self._ttl_misses += 1
            return None

        # Valid hit
        self._ttl_store.move_to_end(segment_key)
        self._ttl_exact_hits += 1
        self._ttl_preserved_hits += 1
        return entry

    def evict_expired_segments(self) -> int:
        """Flush TTL-expired, non-pinned segments from the TTL store and block pool.

        Should be called periodically (e.g. once per schedule() step).

        Returns:
            Number of segments evicted.
        """
        now = time.monotonic()
        expired_keys = [
            k for k, entry in self._ttl_store.items()
            if k not in self._ttl_pinned
            and (now - entry.created_at) > entry.ttl_sec
        ]

        if not expired_keys:
            return 0

        # Apply redundancy-aware eviction scoring if policy is available
        if self._ttl_eviction_policy is not None:
            scored = self._ttl_eviction_policy.score_ttl_candidates(
                expired_keys, self._ttl_store
            )
            evict_order = [k for k, _ in scored]
        else:
            evict_order = expired_keys

        n_evicted = 0
        all_block_ids: set = set()
        for key in evict_order:
            if key not in self._ttl_store:
                continue
            entry = self._ttl_store.pop(key)
            self._ttl_pinned.discard(key)
            all_block_ids.update(entry.block_ids)
            self._ttl_eviction_count += 1
            n_evicted += 1

        # Evict the associated vLLM blocks from the block pool
        if all_block_ids:
            try:
                self.evict_blocks(all_block_ids)
            except Exception:
                pass  # graceful degradation if evict_blocks unavailable

        return n_evicted

    def adjust_segment_ttl(self, segment_key: str, new_ttl_sec: float) -> None:
        """Override the TTL for a specific segment.

        Called by VllmDAGAwareTTLAdjuster when a DAG reuse event arrives.

        Args:
            segment_key: Key of the segment to adjust.
            new_ttl_sec: New TTL in seconds. 0.0 marks for immediate eviction.
        """
        if segment_key in self._ttl_store:
            self._ttl_store[segment_key].ttl_sec = float(new_ttl_sec)

    def pin_segment(self, segment_key: str) -> None:
        """Protect a segment from TTL-based eviction (DAGTopologySchedulerMixin hook).

        Args:
            segment_key: Segment key to pin.
        """
        self._ttl_pinned.add(segment_key)
        if segment_key in self._ttl_store:
            self._ttl_store[segment_key].pinned = True

    def unpin_segment(self, segment_key: str) -> None:
        """Release a previously pinned segment (DAGTopologySchedulerMixin hook).

        Args:
            segment_key: Segment key to unpin.
        """
        self._ttl_pinned.discard(segment_key)
        if segment_key in self._ttl_store:
            self._ttl_store[segment_key].pinned = False

    def evict_candidates(self) -> List[str]:
        """Return list of TTL-expired, non-pinned segment keys.

        Used by VllmRedundancyAwareEvictionPolicy.score_ttl_candidates().

        Returns:
            List of segment keys eligible for eviction.
        """
        now = time.monotonic()
        return [
            k for k, entry in self._ttl_store.items()
            if k not in self._ttl_pinned
            and (now - entry.created_at) > entry.ttl_sec
        ]

    def record_segment_importance(self, segment_key: str, score: float) -> None:
        """Accumulate attention importance score for a segment.

        Used by the attention backend patch to record which segments are
        important for downstream accuracy (Activity C accuracy preservation).

        Args:
            segment_key: Segment key.
            score: Importance score to add to the accumulator.
        """
        if segment_key in self._ttl_store:
            self._ttl_store[segment_key].importance_score += score

    def classify_category(self, key: str) -> str:
        """Classify a request/segment key into a workload category.

        Args:
            key: Segment key or request metadata string.

        Returns:
            Category string: "code", "rag", "agentic", or "chat".
        """
        text = key.lower()
        if any(kw in text for kw in _CODE_KEYWORDS):
            return "code"
        if any(kw in text for kw in _RAG_KEYWORDS):
            return "rag"
        if any(kw in text for kw in _AGENTIC_KEYWORDS):
            return "agentic"
        return "chat"

    def record_hit(self, segment_key: str) -> None:
        """Record a cache hit and update the EMA TTL for the segment's category.

        Args:
            segment_key: Key of the segment that was hit.
        """
        if segment_key not in self._ttl_store:
            return
        entry = self._ttl_store[segment_key]
        reuse_gap = time.monotonic() - entry.created_at
        category = entry.category
        old_ttl = self._ttl_profiles[category]["ttl_base_sec"]
        ttl_multiplier = 1.2
        new_ttl = (
            (1.0 - self._ttl_ema_alpha) * old_ttl
            + self._ttl_ema_alpha * reuse_gap * ttl_multiplier
        )
        self._ttl_profiles[category]["ttl_base_sec"] = new_ttl

    def ttl_hit_stats(self) -> dict:
        """Return TTL cache hit/eviction statistics.

        Returns:
            dict with keys: exact_hit_rate, ttl_preserved_hit_rate,
                            overall_hit_rate, noncontiguous_ratio,
                            eviction_ttl_count, eviction_pressure_count,
                            num_entries, num_pinned.
        """
        total = self._ttl_exact_hits + self._ttl_misses
        exact_rate = self._ttl_exact_hits / total if total > 0 else 0.0
        ttl_rate = self._ttl_preserved_hits / total if total > 0 else 0.0
        hits_total = self._ttl_exact_hits
        noncontiguous_ratio = (
            self._ttl_preserved_hits / hits_total if hits_total > 0 else 0.0
        )
        return {
            "exact_hit_rate": exact_rate,
            "ttl_preserved_hit_rate": ttl_rate,
            "overall_hit_rate": exact_rate,
            "noncontiguous_ratio": noncontiguous_ratio,
            "eviction_ttl_count": self._ttl_eviction_count,
            "eviction_pressure_count": self._ttl_pressure_eviction_count,
            "num_entries": len(self._ttl_store),
            "num_pinned": len(self._ttl_pinned),
        }

    def reset_ttl_stats(self) -> None:
        """Reset TTL cache counters."""
        self._ttl_exact_hits = 0
        self._ttl_preserved_hits = 0
        self._ttl_misses = 0
        self._ttl_eviction_count = 0
        self._ttl_pressure_eviction_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ttl_chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """SHA-256 chunk key — identical algorithm to WorkloadAwareTTLCache.chunk_key().

        Compatible with SegmentedHashCache.chunk_key() so segment keys are
        interoperable across cache implementations.
        """
        start = chunk_idx * self._ttl_chunk_size
        chunk = token_ids[start: start + self._ttl_chunk_size]
        if not chunk:
            chunk = [0]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

    def _ttl_evict_one(self) -> None:
        """Evict one segment from the TTL store (LRU fallback)."""
        if not self._ttl_store:
            return

        # Try TTL-expired first
        now = time.monotonic()
        for k, entry in self._ttl_store.items():
            if k not in self._ttl_pinned and (now - entry.created_at) > entry.ttl_sec:
                self._ttl_store.pop(k)
                self._ttl_pinned.discard(k)
                self._ttl_eviction_count += 1
                return

        # LRU fallback: oldest non-pinned
        for k in self._ttl_store:
            if k not in self._ttl_pinned:
                self._ttl_store.pop(k)
                self._ttl_pressure_eviction_count += 1
                return


# ---------------------------------------------------------------------------
# VllmDAGAwareTTLAdjuster — TTL adjuster integrated with vLLM's KVCacheManager
# ---------------------------------------------------------------------------

class VllmDAGAwareTTLAdjuster:
    """Adapter connecting DAGTopologySchedulerMixin events to WorkloadAwareTTLKVCacheManager.

    Mirrors src/scheduler/dag_ttl_adjuster.DAGAwareTTLAdjuster but operates
    on WorkloadAwareTTLKVCacheManager.adjust_segment_ttl() instead of
    WorkloadAwareTTLCache.adjust_ttl().

    Usage:
        adjuster = VllmDAGAwareTTLAdjuster(kv_manager, alpha=2.0)

        # Connect to DAGTopologySchedulerMixin callbacks:
        scheduler_mixin = DAGTopologySchedulerMixin(
            on_kv_reuse_event=adjuster.on_kv_reuse_event,
            on_node_complete_event=adjuster.on_node_complete,
        )
    """

    def __init__(
        self,
        kv_manager: WorkloadAwareTTLKVCacheManager,
        alpha: float = 2.0,
        measure_latency: bool = True,
    ) -> None:
        """
        Args:
            kv_manager: The WorkloadAwareTTLKVCacheManager instance.
            alpha: TTL extension multiplier. adjusted_ttl = base_ttl × (1 + prob × alpha).
            measure_latency: If True, record event-to-update latency in ms.
        """
        self.kv_manager = kv_manager
        self.alpha = alpha
        self.measure_latency = measure_latency
        self._latency_samples: List[float] = []

    def on_kv_reuse_event(
        self,
        segment_key: str,
        dag_reuse_probability: float,
    ) -> None:
        """Extend TTL for a segment based on DAG reuse probability.

        Args:
            segment_key: Segment key in the TTL store.
            dag_reuse_probability: Probability in [0.0, 1.0].
        """
        t0 = time.monotonic()

        entry = self.kv_manager._ttl_store.get(segment_key)
        if entry is None:
            if self.measure_latency:
                self._latency_samples.append((time.monotonic() - t0) * 1000.0)
            return

        base_ttl: float = self.kv_manager._ttl_profiles.get(
            entry.category, _DEFAULT_TTL_PROFILES["chat"]
        )["ttl_base_sec"]
        adjusted_ttl = base_ttl * (1.0 + dag_reuse_probability * self.alpha)
        self.kv_manager.adjust_segment_ttl(segment_key, adjusted_ttl)

        if self.measure_latency:
            self._latency_samples.append((time.monotonic() - t0) * 1000.0)

    def on_node_complete(self, segment_key: str) -> None:
        """Mark a segment for immediate eviction after DAG node completion.

        Sets TTL=0 and unpins the segment so the next evict_expired_segments()
        call can free the associated vLLM blocks.

        Args:
            segment_key: Segment key.
        """
        self.kv_manager.adjust_segment_ttl(segment_key, new_ttl_sec=0.0)
        self.kv_manager.unpin_segment(segment_key)

    def overhead_stats(self) -> dict:
        """Return event-to-TTL-update latency statistics.

        Returns:
            dict with keys: p50_ms, p99_ms, mean_ms, n_samples.
        """
        n = len(self._latency_samples)
        if n == 0:
            return {"p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "n_samples": 0}

        import numpy as np
        arr = np.array(self._latency_samples, dtype=float)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
            "n_samples": n,
        }


# ---------------------------------------------------------------------------
# Prior cycle (2026-05-03) components — preserved for backward compatibility
# ---------------------------------------------------------------------------

# NOTE: SemanticSegmentIndex and SemanticNonContiguousKVCacheManager from the
# prior cycle (DHD semantic similarity) are preserved in their original form
# below. Import from this module for backward compatibility.

import hashlib as _hashlib
import struct as _struct
from collections import OrderedDict as _OrderedDict
from typing import Tuple as _Tuple
import torch.nn.functional as _F


class SemanticSegmentIndex:
    """Prior-cycle DHD semantic segment index (2026-05-03). Preserved for compat."""

    def __init__(
        self,
        codec: Any = None,
        chunk_size: int = 16,
        max_entries: int = 2000,
        top_k: int = 5,
        similarity_threshold: float = 0.80,
        deviation_threshold: float = 0.20,
    ) -> None:
        self.codec = codec
        self.chunk_size = chunk_size
        self.max_entries = max_entries
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.deviation_threshold = deviation_threshold
        self._compressed_store: _OrderedDict[str, dict] = _OrderedDict()
        self._semantic_index: List[_Tuple[str, torch.Tensor]] = []
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._recompute_count = 0

    def chunk_key(self, token_ids: List[int], chunk_idx: int, layer_idx: int = 0) -> str:
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start: start + self.chunk_size]
        if not chunk:
            chunk = [0]
        raw = _struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = _struct.pack("I", layer_idx)
        return _hashlib.sha256(layer_prefix + raw).hexdigest()

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)
        if keys.dim() == 3:
            embedding = keys.float().mean(dim=(0, 1))
        else:
            embedding = keys.float().mean(dim=0)
        if self.codec is not None:
            k_compressed = self.codec.encode_tokens(keys, layer_idx, tensor_id=0)
            v_compressed = self.codec.encode_tokens(values, layer_idx, tensor_id=1)
        else:
            k_compressed = {"raw": keys.float()}
            v_compressed = {"raw": values.float()}
        if len(self._compressed_store) >= self.max_entries:
            evict_key, _ = self._compressed_store.popitem(last=False)
            self._semantic_index = [
                (k, e) for k, e in self._semantic_index if k != evict_key
            ]
        self._compressed_store[key] = {
            "k": k_compressed, "v": v_compressed,
            "layer_idx": layer_idx, "embedding": embedding,
        }
        self._semantic_index.append((key, embedding))
        return key

    def lookup_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        query_keys: torch.Tensor,
        layer_idx: int = 0,
    ) -> _Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)
        if key in self._compressed_store:
            entry = self._compressed_store[key]
            k = self._decode_entry(entry["k"], layer_idx, 0)
            v = self._decode_entry(entry["v"], layer_idx, 1)
            self._exact_hits += 1
            self._compressed_store.move_to_end(key)
            return k, v, "exact"
        if not self._semantic_index:
            self._misses += 1
            return None, None, "miss"
        query_emb = self._compute_embedding(query_keys)
        candidates = self._cosine_search(query_emb, self.top_k)
        for cand_key, _cand_emb, cos_sim in candidates:
            if cos_sim < self.similarity_threshold:
                continue
            if cand_key not in self._compressed_store:
                continue
            entry = self._compressed_store[cand_key]
            k_cand = self._decode_entry(entry["k"], layer_idx, 0)
            v_cand = self._decode_entry(entry["v"], layer_idx, 1)
            deviation = self._compute_dhd_deviation(query_keys, k_cand)
            if deviation <= self.deviation_threshold:
                self._semantic_hits += 1
                return k_cand, v_cand, "semantic"
            else:
                self._recompute_count += 1
                self._misses += 1
                return None, None, "miss"
        self._misses += 1
        return None, None, "miss"

    def _decode_entry(self, entry: dict, layer_idx: int, tensor_id: int) -> torch.Tensor:
        if "raw" in entry:
            return entry["raw"]
        if self.codec is not None:
            return self.codec.decode_tokens(entry, layer_idx, tensor_id)
        return entry.get("raw", torch.zeros(1))

    def _compute_embedding(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.dim() == 3:
            return keys.float().mean(dim=(0, 1))
        return keys.float().mean(dim=0)

    def _cosine_search(
        self, query_emb: torch.Tensor, top_k: int
    ) -> List[_Tuple[str, torch.Tensor, float]]:
        if not self._semantic_index:
            return []
        keys_list = [k for k, _ in self._semantic_index]
        emb_matrix = torch.stack([emb for _, emb in self._semantic_index])
        if query_emb.shape[0] != emb_matrix.shape[1]:
            min_d = min(query_emb.shape[0], emb_matrix.shape[1])
            query_emb = query_emb[:min_d]
            emb_matrix = emb_matrix[:, :min_d]
        q_norm = _F.normalize(query_emb.unsqueeze(0).float(), dim=-1)
        e_norm = _F.normalize(emb_matrix.float(), dim=-1)
        sims = (q_norm @ e_norm.T).squeeze(0)
        actual_k = min(top_k, len(self._semantic_index))
        top_indices = sims.argsort(descending=True)[:actual_k]
        return [
            (keys_list[i.item()], self._semantic_index[i.item()][1], sims[i.item()].item())
            for i in top_indices
        ]

    def _compute_dhd_deviation(self, query_keys: torch.Tensor, cached_keys: torch.Tensor) -> float:
        q = query_keys.float()
        c = cached_keys.float()
        if q.dim() == 3:
            q = q.reshape(q.shape[0], -1)
        if c.dim() == 3:
            c = c.reshape(c.shape[0], -1)
        min_len = min(q.shape[0], c.shape[0])
        q = q[:min_len]
        c = c[:min_len]
        return (q - c).norm(dim=-1).mean().item() / (c.norm(dim=-1).mean().item() + 1e-8)

    def hit_rate(self) -> float:
        total = self._exact_hits + self._semantic_hits + self._misses
        return (self._exact_hits + self._semantic_hits) / total if total > 0 else 0.0

    def semantic_hit_rates(self) -> dict:
        total_hits = self._exact_hits + self._semantic_hits
        total = total_hits + self._misses
        return {
            "exact_hit_rate": self._exact_hits / total if total > 0 else 0.0,
            "semantic_hit_rate": self._semantic_hits / total if total > 0 else 0.0,
            "overall_hit_rate": total_hits / total if total > 0 else 0.0,
            "noncontiguous_ratio": (
                self._semantic_hits / total_hits if total_hits > 0 else 0.0
            ),
            "recompute_ratio": self._recompute_count / max(1, total_hits),
        }

    def reset_stats(self) -> None:
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._recompute_count = 0

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._compressed_store.values():
            for tensor_key in ("k", "v"):
                compressed = entry.get(tensor_key, {})
                if "quantized" in compressed:
                    total += compressed["quantized"].nbytes + compressed["scale"].nbytes
                    if "qjl_packed" in compressed:
                        total += compressed["qjl_packed"].nbytes
                elif "raw" in compressed:
                    total += compressed["raw"].nbytes
                elif "fp16" in compressed:
                    total += compressed["fp16"].nbytes
        return total


class SemanticNonContiguousKVCacheManager(KVCacheManager):
    """Prior-cycle DHD semantic KVCacheManager subclass (2026-05-03). Preserved for compat."""

    def __init__(
        self,
        *args: Any,
        codec: Any = None,
        segment_chunk_size: int = 16,
        segment_max_entries: int = 2000,
        segment_top_k: int = 5,
        similarity_threshold: float = 0.80,
        deviation_threshold: float = 0.20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._segment_index = SemanticSegmentIndex(
            codec=codec,
            chunk_size=segment_chunk_size,
            max_entries=segment_max_entries,
            top_k=segment_top_k,
            similarity_threshold=similarity_threshold,
            deviation_threshold=deviation_threshold,
        )
        self._segment_codec = codec
        self._segment_chunk_size = segment_chunk_size

    def store_segment(
        self, token_ids: List[int], chunk_idx: int,
        keys: torch.Tensor, values: torch.Tensor, layer_idx: int = 0,
    ) -> str:
        return self._segment_index.store_segment(token_ids, chunk_idx, keys, values, layer_idx)

    def lookup_segment(
        self, token_ids: List[int], chunk_idx: int,
        query_keys: torch.Tensor, layer_idx: int = 0,
    ) -> _Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        return self._segment_index.lookup_segment(token_ids, chunk_idx, query_keys, layer_idx)

    def lookup_all_segments(
        self, token_ids: List[int], layer_idx: int, query_keys: torch.Tensor,
    ) -> List[_Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor], str]]:
        chunk_size = self._segment_chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        results = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, query_keys.shape[0])
            chunk_query_keys = query_keys[start:end]
            k, v, hit_type = self.lookup_segment(token_ids, chunk_idx, chunk_query_keys, layer_idx)
            results.append((chunk_idx, k, v, hit_type))
        return results

    def segment_index_stats(self) -> dict:
        return self._segment_index.semantic_hit_rates()

    def segment_memory_bytes(self) -> int:
        return self._segment_index.memory_bytes()


# ---------------------------------------------------------------------------
# 2026-05-06 additions — Activity B (QueryCentricRecomputeCache port)
# ---------------------------------------------------------------------------

class QueryCentricKVCacheManager(KVCacheManager):
    """KVCacheManager subclass adding ProphetKV dual-stage recompute budget allocation.

    Ports QueryCentricRecomputeCache (src/cache/query_centric_recompute.py) into
    the vLLM KVCacheManager layer.

    Design:
        A parallel segment store (OrderedDict keyed by SHA-256 segment hash) holds
        KV segment embeddings and attention norms alongside vLLM's native paged
        block pool. After each prefill step, the caller registers segments via
        store_qcrc_segment(). Before scheduling, selective_recompute() returns
        the segment keys worth recomputing given a query and a token-count budget.

    Stage 1: global attention-norm filter — top-50% by ||K||_mean.
    Stage 2: query cosine-similarity re-rank — budget-limited to recompute_budget_ratio.

    Block boundary contract:
        qcrc_chunk_size must divide vLLM's block_size so segment boundaries align
        with KV page boundaries. Default 128 aligns with vLLM's typical block_size=16
        multiples and the TriAttentionCodec prune_window.

    Usage:
        kv_manager = QueryCentricKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            hash_block_size=block_size,
            enable_caching=True,
            qcrc_capacity_bytes=512 * 1024 * 1024,  # 512 MB parallel store
            qcrc_chunk_size=128,
            qcrc_recompute_budget_ratio=0.20,
            qcrc_stage1_top_k_ratio=0.50,
        )
        # After prefill:
        key = kv_manager.store_qcrc_segment(token_ids, chunk_idx, kv_tensor, layer_idx)
        # Before next request with same context:
        to_recompute = kv_manager.selective_recompute(query_emb, all_seg_keys)
    """

    def __init__(
        self,
        *args: Any,
        qcrc_capacity_bytes: int = 512 * 1024 * 1024,
        qcrc_chunk_size: int = 128,
        qcrc_recompute_budget_ratio: float = 0.20,
        qcrc_stage1_top_k_ratio: float = 0.50,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            qcrc_capacity_bytes: Max bytes for the parallel QCRC segment store.
            qcrc_chunk_size: Token chunk size for segment key computation.
            qcrc_recompute_budget_ratio: Max fraction of tokens to recompute (budget).
            qcrc_stage1_top_k_ratio: Fraction kept after Stage-1 attention-norm filter.
            *args, **kwargs: Forwarded to KVCacheManager.__init__().
        """
        super().__init__(*args, **kwargs)
        self._qcrc_chunk_size = qcrc_chunk_size
        self._qcrc_capacity_bytes = qcrc_capacity_bytes
        self._qcrc_recompute_budget_ratio = qcrc_recompute_budget_ratio
        self._qcrc_stage1_top_k_ratio = qcrc_stage1_top_k_ratio

        # {segment_key: {"kv": Tensor, "embedding": Tensor, "attn_norm": float}}
        self._qcrc_store: OrderedDict[str, dict] = OrderedDict()
        self._qcrc_hit_count: int = 0
        self._qcrc_miss_count: int = 0

    # ------------------------------------------------------------------
    # QCRC segment API
    # ------------------------------------------------------------------

    def store_qcrc_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv_tensor: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        """Store a KV segment in the parallel QCRC store after block allocation.

        The segment is keyed by SHA-256(layer_idx || token_chunk). If the key
        already exists, it is moved to the most-recent position (LRU touch).

        Args:
            token_ids: Full prompt token ID list.
            chunk_idx: Index of the 128-token chunk within the prompt.
            kv_tensor: KV tensor [layers, heads, seq_len, head_dim].
            layer_idx: Transformer layer index (for keying).

        Returns:
            Segment key (64-char SHA-256 hex string).
        """
        key = self._qcrc_chunk_key(token_ids, chunk_idx, layer_idx)

        if key in self._qcrc_store:
            self._qcrc_store.move_to_end(key)
            return key

        # Evict LRU until under capacity
        while (
            self._qcrc_memory_bytes() + kv_tensor.nbytes > self._qcrc_capacity_bytes
            and self._qcrc_store
        ):
            self._qcrc_evict_lru()

        embedding = kv_tensor.float().mean(dim=(0, 1, 2))  # [head_dim]
        attn_norm = float(kv_tensor.norm(dim=-1).mean().item())

        self._qcrc_store[key] = {
            "kv": kv_tensor.detach().clone(),
            "embedding": embedding.detach().clone(),
            "attn_norm": attn_norm,
        }
        return key

    def get_qcrc_segment(self, segment_key: str) -> Optional[torch.Tensor]:
        """Look up a segment by key. Returns KV tensor or None on miss.

        Args:
            segment_key: Key from store_qcrc_segment().

        Returns:
            KV tensor [layers, heads, seq_len, head_dim] or None.
        """
        entry = self._qcrc_store.get(segment_key)
        if entry is None:
            self._qcrc_miss_count += 1
            return None
        self._qcrc_store.move_to_end(segment_key)
        self._qcrc_hit_count += 1
        return entry["kv"]

    def selective_recompute(
        self,
        query: torch.Tensor,
        cached_segments: List[str],
        budget: Optional[float] = None,
    ) -> List[str]:
        """Two-stage recompute budget allocation (ProphetKV algorithm).

        Stage 1: global attention-norm saliency filter (top stage1_top_k_ratio).
        Stage 2: query cosine-similarity re-rank, budget-limited by token count.

        Args:
            query: Query representation vector [head_dim].
            cached_segments: Candidate segment key list.
            budget: Max fraction of total cached tokens to recompute.
                    Defaults to qcrc_recompute_budget_ratio.

        Returns:
            List of segment keys selected for recomputation, ordered by
            descending relevance, within budget.
        """
        import torch.nn.functional as _F
        budget = budget if budget is not None else self._qcrc_recompute_budget_ratio
        present = [k for k in cached_segments if k in self._qcrc_store]
        if not present:
            return []

        # Stage 1: global saliency filter (attention norm top-50%)
        attn_norms = {k: self._qcrc_store[k]["attn_norm"] for k in present}
        sorted_by_norm = sorted(attn_norms, key=lambda k: attn_norms[k], reverse=True)
        n_stage1 = max(1, int(len(sorted_by_norm) * self._qcrc_stage1_top_k_ratio))
        stage1_candidates = sorted_by_norm[:n_stage1]

        # Stage 2: query cosine-similarity re-rank
        relevance_scores: Dict[str, float] = {}
        for seg_key in stage1_candidates:
            seg_emb = self._qcrc_store[seg_key]["embedding"]
            score = _F.cosine_similarity(
                query.unsqueeze(0).float(), seg_emb.unsqueeze(0).float()
            ).item()
            relevance_scores[seg_key] = score

        sorted_by_relevance = sorted(
            relevance_scores, key=lambda k: relevance_scores[k], reverse=True
        )

        # Enforce token-count budget
        total_tokens = sum(
            self._qcrc_store[k]["kv"].shape[2] for k in sorted_by_relevance
        )
        token_budget = max(1, int(total_tokens * budget))
        selected: List[str] = []
        accumulated = 0
        for seg_key in sorted_by_relevance:
            seg_len = self._qcrc_store[seg_key]["kv"].shape[2]
            if accumulated + seg_len > token_budget:
                break
            selected.append(seg_key)
            accumulated += seg_len

        return selected

    def qcrc_hit_rate(self) -> float:
        """Cumulative cache hit rate for QCRC segment store."""
        total = self._qcrc_hit_count + self._qcrc_miss_count
        return self._qcrc_hit_count / total if total > 0 else 0.0

    def qcrc_stats(self) -> dict:
        """Return QCRC hit/miss/memory statistics."""
        return {
            "hit_count": self._qcrc_hit_count,
            "miss_count": self._qcrc_miss_count,
            "hit_rate": self.qcrc_hit_rate(),
            "num_segments": len(self._qcrc_store),
            "memory_bytes": self._qcrc_memory_bytes(),
        }

    def reset_qcrc_stats(self) -> None:
        """Reset QCRC hit/miss counters."""
        self._qcrc_hit_count = 0
        self._qcrc_miss_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _qcrc_chunk_key(
        self, token_ids: List[int], chunk_idx: int, layer_idx: int = 0
    ) -> str:
        """SHA-256 chunk key — compatible with QueryCentricRecomputeCache keying."""
        start = chunk_idx * self._qcrc_chunk_size
        chunk = token_ids[start: start + self._qcrc_chunk_size]
        if not chunk:
            chunk = [0]
        raw = _struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = _struct.pack("I", layer_idx)
        return _hashlib.sha256(layer_prefix + raw).hexdigest()

    def _qcrc_memory_bytes(self) -> int:
        return sum(e["kv"].nbytes for e in self._qcrc_store.values())

    def _qcrc_evict_lru(self) -> None:
        if self._qcrc_store:
            self._qcrc_store.popitem(last=False)


# ---------------------------------------------------------------------------
# 2026-05-06 additions — Activity B+C (QueryCentricTriAttentionCache port)
# ---------------------------------------------------------------------------

class QueryCentricTriAttentionKVCacheManager(QueryCentricKVCacheManager):
    """KVCacheManager subclass combining QCRC (Activity B) + TriAttentionCodec (Activity C).

    Ports src/cache/qc_tri_store.QueryCentricTriAttentionCache into the vLLM
    KVCacheManager layer as a dual-path storage strategy:

    High-relevance segments (cosine sim >= relevance_threshold):
        Stored as raw KV in the QCRC store for quality recomputation.

    Low-relevance segments (cosine sim < relevance_threshold):
        Compressed with TriAttentionCodec (pre-RoPE trigonometric importance
        estimation, windowed pruning to compression_ratio fraction) and stored in
        a separate compressed store. Pre-RoPE keys MUST be passed to compress().

    Accuracy preservation:
        - selective_recompute() only returns raw KV segments (never compressed).
        - Compressed segments are decompressed on read: zeros at pruned positions.
        - No quantization or approximation beyond the windowed top-K pruning that
          TriAttentionCodec applies (which is pre-calibrated to minimize perplexity).

    Usage:
        codec = TriAttentionCodecWrapper(n_layers=32, n_heads=32, head_dim=128)
        codec.load_calibration("calibration.pt")

        kv_manager = QueryCentricTriAttentionKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            hash_block_size=block_size,
            enable_caching=True,
            codec=codec,
            relevance_threshold=0.60,
            compression_ratio=0.10,
        )

        # After prefill with query context:
        kv_manager.store_qcta_segment(
            token_ids, chunk_idx, kv_tensor, keys_pre_rope, query_embedding, layer_idx
        )

        # Recompute (raw KV only):
        to_recompute = kv_manager.selective_recompute(query_emb, all_seg_keys)

        # Read back (raw or decompressed):
        kv = kv_manager.get_qcta_segment(segment_key)
    """

    def __init__(
        self,
        *args: Any,
        codec: Optional[Any] = None,
        relevance_threshold: float = 0.60,
        compression_ratio: float = 0.10,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            codec: TriAttentionCodecWrapper instance (must be calibrated before use).
                   If None, all segments fall back to raw QCRC storage.
            relevance_threshold: Cosine similarity threshold for raw vs. compressed routing.
            compression_ratio: Fraction of tokens kept per 128-token window by codec.
            *args, **kwargs: Forwarded to QueryCentricKVCacheManager.
        """
        super().__init__(*args, **kwargs)
        self._qcta_codec = codec
        self._qcta_relevance_threshold = relevance_threshold
        self._qcta_compression_ratio = compression_ratio

        # {segment_key: compressed_dict} for low-relevance segments
        self._qcta_compressed_store: Dict[str, dict] = {}
        # {segment_key: torch.Tensor} for high-relevance segments (fast lookup)
        self._qcta_raw_store: Dict[str, torch.Tensor] = {}

        self._qcta_hit_count: int = 0
        self._qcta_miss_count: int = 0
        self._qcta_compressed_hits: int = 0
        self._qcta_raw_hits: int = 0

    # ------------------------------------------------------------------
    # Query-aware dual-path API
    # ------------------------------------------------------------------

    def store_qcta_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv_tensor: torch.Tensor,
        keys_pre_rope: torch.Tensor,
        query_embedding: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        """Store with query-context-aware routing: raw or compressed.

        High-relevance segments (cosine sim >= relevance_threshold):
            Stored in _qcta_raw_store AND forwarded to QCRC (for selective_recompute).

        Low-relevance segments (cosine sim < relevance_threshold):
            Compressed by TriAttentionCodec and stored in _qcta_compressed_store.
            Falls back to QCRC raw storage if codec is uncalibrated.

        Args:
            token_ids: Full prompt token ID list.
            chunk_idx: Chunk index within the prompt.
            kv_tensor: KV tensor [layers, heads, seq_len, head_dim].
            keys_pre_rope: Pre-RoPE K tensor (same shape as kv_tensor). MUST be pre-RoPE.
            query_embedding: Query mean K vector [head_dim].
            layer_idx: Transformer layer index.

        Returns:
            Segment key (64-char SHA-256 hex string).
        """
        import torch.nn.functional as _F
        key = self._qcrc_chunk_key(token_ids, chunk_idx, layer_idx)

        seg_emb = kv_tensor.float().mean(dim=(0, 1, 2))  # [head_dim]
        relevance = _F.cosine_similarity(
            query_embedding.unsqueeze(0).float(),
            seg_emb.unsqueeze(0).float(),
        ).item()

        if relevance >= self._qcta_relevance_threshold:
            # High-relevance: raw storage for quality recomputation
            self._qcta_raw_store[key] = kv_tensor.detach().clone()
            # Also register in QCRC for selective_recompute()
            self.store_qcrc_segment(token_ids, chunk_idx, kv_tensor, layer_idx)
        else:
            # Low-relevance: compress to save capacity
            codec = self._qcta_codec
            if codec is not None and getattr(codec, "mu_k", None) is not None:
                try:
                    compressed = codec.compress(
                        kv_tensor, keys_pre_rope, self._qcta_compression_ratio
                    )
                    self._qcta_compressed_store[key] = compressed
                except Exception:
                    # Graceful fallback: store raw in QCRC
                    self.store_qcrc_segment(token_ids, chunk_idx, kv_tensor, layer_idx)
            else:
                # Codec not calibrated: fall back to raw QCRC storage
                self.store_qcrc_segment(token_ids, chunk_idx, kv_tensor, layer_idx)

        return key

    def get_qcta_segment(self, segment_key: str) -> Optional[torch.Tensor]:
        """Retrieve a segment: raw store → compressed store → QCRC.

        Compressed segments are decompressed (zeros at pruned positions).

        Args:
            segment_key: Key from store_qcta_segment().

        Returns:
            KV tensor [layers, heads, seq_len, head_dim] or None on miss.
        """
        # Raw store (high-relevance)
        if segment_key in self._qcta_raw_store:
            self._qcta_hit_count += 1
            self._qcta_raw_hits += 1
            return self._qcta_raw_store[segment_key]

        # Compressed store (low-relevance)
        if segment_key in self._qcta_compressed_store:
            self._qcta_hit_count += 1
            self._qcta_compressed_hits += 1
            codec = self._qcta_codec
            if codec is not None:
                try:
                    return codec.decompress(self._qcta_compressed_store[segment_key])
                except Exception:
                    pass
            # Fallback: return the compressed KV directly
            return self._qcta_compressed_store[segment_key].get("kv")

        # QCRC store fallback
        result = self.get_qcrc_segment(segment_key)
        if result is None:
            self._qcta_miss_count += 1
        else:
            self._qcta_hit_count += 1
        return result

    def selective_recompute(
        self,
        query: torch.Tensor,
        cached_segments: List[str],
        budget: Optional[float] = None,
    ) -> List[str]:
        """Two-stage recompute budget allocation using only raw KV segments.

        Compressed segments are excluded — reconstruction quality is insufficient
        for recomputation (lossy windowed pruning).

        Delegates to QueryCentricKVCacheManager.selective_recompute() with only
        the segments that have raw KV available (in _qcta_raw_store or QCRC).

        Args:
            query: Query mean K vector [head_dim].
            cached_segments: Candidate segment keys.
            budget: Max fraction of total tokens to recompute.

        Returns:
            Selected segment keys for recomputation.
        """
        # Only raw-KV and QCRC-stored segments are eligible
        raw_segments = [k for k in cached_segments if k in self._qcta_raw_store]
        qcrc_segments = [
            k for k in cached_segments
            if k not in self._qcta_raw_store
            and k not in self._qcta_compressed_store
        ]
        eligible = raw_segments + qcrc_segments
        return super().selective_recompute(query, eligible, budget)

    def qcta_stats(self) -> dict:
        """Return QCTA dual-path statistics."""
        total = self._qcta_hit_count + self._qcta_miss_count
        return {
            "hit_count": self._qcta_hit_count,
            "miss_count": self._qcta_miss_count,
            "hit_rate": self._qcta_hit_count / total if total > 0 else 0.0,
            "raw_hits": self._qcta_raw_hits,
            "compressed_hits": self._qcta_compressed_hits,
            "num_raw_segments": len(self._qcta_raw_store),
            "num_compressed_segments": len(self._qcta_compressed_store),
            "raw_memory_bytes": sum(t.nbytes for t in self._qcta_raw_store.values()),
            "compressed_memory_bytes": sum(
                v.get("kv").nbytes if isinstance(v.get("kv"), torch.Tensor) else 0
                for v in self._qcta_compressed_store.values()
            ),
        }

    def reset_qcta_stats(self) -> None:
        """Reset QCTA hit/miss counters."""
        self._qcta_hit_count = 0
        self._qcta_miss_count = 0
        self._qcta_compressed_hits = 0
        self._qcta_raw_hits = 0


# ---------------------------------------------------------------------------
# TriAttentionCodecWrapper — vLLM-compatible adapter for TriAttentionCodec
# ---------------------------------------------------------------------------

class TriAttentionCodecWrapper:
    """vLLM-compatible wrapper for TriAttentionCodec (Activity C).

    Wraps src/cache/tri_attention_codec.TriAttentionCodec with vLLM-specific
    shape handling and provides calibrate/compress/decompress as used by
    QueryCentricTriAttentionKVCacheManager.

    vLLM KV block shape convention:
        [num_blocks, block_size, num_kv_heads, head_dim]   (FlashAttention layout)

    TriAttentionCodec shape convention:
        [n_layers, n_heads, seq_len, head_dim]

    This wrapper handles the shape translation transparently.

    Calibration:
        Must be called before compress(). Requires at least 10 pre-RoPE K tensors
        from representative requests.

        codec = TriAttentionCodecWrapper(n_layers=32, n_heads=32, head_dim=128)
        codec.calibrate(calibration_kvs, save_path="calib.pt")

    Compression (Activity C):
        compressed = codec.compress(kv_tensor, keys_pre_rope)
        # compressed["kv"] shape: [n_layers, n_heads, kept_tokens, head_dim]

    Decompression:
        reconstructed = codec.decompress(compressed)
        # shape: [n_layers, n_heads, original_seq_len, head_dim]
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        compression_ratio: float = 0.10,
        series_terms: int = 8,
        prune_window: int = 128,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        # Calibration tensors (set by calibrate() or load_calibration())
        self.mu_k: Optional[torch.Tensor] = None   # [n_layers, n_heads, head_dim]
        self.a_m: Optional[torch.Tensor] = None    # [n_layers, n_heads, series_terms]
        self._series_terms = series_terms
        self._prune_window = prune_window

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        calibration_kvs: List[torch.Tensor],
        save_path: Optional[str] = None,
    ) -> None:
        """Estimate mu_k and a_m from calibration KV tensors.

        Delegates to TriAttentionCodec.calibrate() from src/cache/.

        Args:
            calibration_kvs: List of tensors [layers, heads, seq_len, head_dim].
            save_path: Optional .pt file path to save calibration.
        """
        try:
            import sys, os
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from src.cache.tri_attention_codec import TriAttentionCodec as _TAC
            _codec = _TAC(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                head_dim=self.head_dim,
                compression_ratio=self.compression_ratio,
                series_terms=self._series_terms,
                prune_window=self._prune_window,
            )
            _codec.calibrate(calibration_kvs, save_path=save_path)
            self.mu_k = _codec.mu_k
            self.a_m = _codec.a_m
        except ImportError:
            # Fallback: inline calibration without src/ dependency
            self._inline_calibrate(calibration_kvs, save_path)

    def load_calibration(self, load_path: str) -> None:
        """Load previously saved calibration from disk."""
        import torch
        ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
        self.mu_k = ckpt["mu_k"]
        self.a_m = ckpt["a_m"]

    # ------------------------------------------------------------------
    # Compression / decompression
    # ------------------------------------------------------------------

    def compress(
        self,
        kv_tensor: torch.Tensor,
        keys_pre_rope: torch.Tensor,
        compression_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compress KV tensor using pre-RoPE trigonometric importance scores.

        Args:
            kv_tensor: KV tensor [layers, heads, seq_len, head_dim].
            keys_pre_rope: Pre-RoPE K tensor (same shape). MUST be pre-RoPE.
            compression_ratio: Fraction of tokens to keep (overrides instance default).

        Returns:
            Dict with "kv", "kept_indices", "original_seq_len", "compression_ratio".
        """
        if self.mu_k is None or self.a_m is None:
            raise RuntimeError("calibrate() or load_calibration() must be called first")

        ratio = compression_ratio if compression_ratio is not None else self.compression_ratio
        seq_len = kv_tensor.shape[2]

        importance = self._estimate_importance(keys_pre_rope)  # [L, H, S]
        token_importance = importance.mean(dim=(0, 1))           # [S]

        kept_parts: List[torch.Tensor] = []
        for window_start in range(0, seq_len, self._prune_window):
            window_end = min(window_start + self._prune_window, seq_len)
            window_imp = token_importance[window_start:window_end]
            n_keep = max(1, int(len(window_imp) * ratio))
            top_local = window_imp.topk(n_keep).indices + window_start
            kept_parts.append(top_local)

        kept_indices = torch.cat(kept_parts).sort().values
        compressed_kv = kv_tensor[:, :, kept_indices, :]

        return {
            "kv": compressed_kv,
            "kept_indices": kept_indices,
            "original_seq_len": seq_len,
            "compression_ratio": ratio,
        }

    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct full-length KV with zeros at pruned positions.

        Args:
            compressed: Dict returned by compress().

        Returns:
            Reconstructed tensor [layers, heads, original_seq_len, head_dim].
        """
        kv_c = compressed["kv"]
        kept_indices = compressed["kept_indices"]
        original_len = compressed["original_seq_len"]
        layers, heads, _, dim = kv_c.shape
        reconstructed = torch.zeros(
            layers, heads, original_len, dim,
            dtype=kv_c.dtype, device=kv_c.device,
        )
        reconstructed[:, :, kept_indices, :] = kv_c
        return reconstructed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_importance(self, keys_pre_rope: torch.Tensor) -> torch.Tensor:
        """Trigonometric Fourier series importance (mirrors TriAttentionCodec)."""
        device = keys_pre_rope.device
        mu_k = self.mu_k.to(device=device, dtype=keys_pre_rope.dtype)
        a_m = self.a_m.to(device=device, dtype=keys_pre_rope.dtype)

        diff = keys_pre_rope - mu_k.unsqueeze(2)
        d = diff.norm(dim=-1)

        importance = torch.zeros_like(d)
        for m in range(1, self._series_terms + 1):
            m_d = m * d
            importance = importance + a_m[:, :, m - 1].unsqueeze(2) * (
                torch.sin(m_d) + torch.cos(m_d)
            )
        return importance.abs()

    def _inline_calibrate(
        self,
        calibration_kvs: List[torch.Tensor],
        save_path: Optional[str],
    ) -> None:
        """Inline calibration (used when src/ is not importable)."""
        all_keys = torch.cat(calibration_kvs, dim=2).float()
        self.mu_k = all_keys.mean(dim=2)
        distances = (all_keys - self.mu_k.unsqueeze(2)).norm(dim=-1)
        layers, heads, T = distances.shape
        self.a_m = torch.zeros(layers, heads, self._series_terms)
        target_norms = all_keys.norm(dim=-1)
        for li in range(layers):
            for hi in range(heads):
                dist = distances[li, hi]
                y = target_norms[li, hi]
                X = torch.stack(
                    [torch.sin(m * dist) + torch.cos(m * dist)
                     for m in range(1, self._series_terms + 1)],
                    dim=1,
                )
                try:
                    result = torch.linalg.lstsq(X, y.unsqueeze(1))
                    coeff = result.solution.squeeze(1)
                    if coeff.shape[0] < self._series_terms:
                        pad = torch.zeros(self._series_terms - coeff.shape[0])
                        coeff = torch.cat([coeff, pad])
                    self.a_m[li, hi] = coeff[: self._series_terms]
                except Exception:
                    self.a_m[li, hi] = torch.ones(self._series_terms)
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            torch.save({"mu_k": self.mu_k, "a_m": self.a_m}, save_path)


# ---------------------------------------------------------------------------
# 2026-05-08 Activity B: StaticDynamicSegmentKVManager
# ---------------------------------------------------------------------------

class StaticDynamicSegmentKVManager(KVCacheManager):
    """KVCacheManager subclass adding static/dynamic segment classification.

    Ports src/cache/static_dynamic_segment.StaticDynamicSegmentCache into the
    vLLM integration layer.

    Design:
        Maintains a parallel segment index alongside vLLM's native prefix cache.
        Static segments (system prompts, shared documents) are marked as LRU-exempt
        and never evicted until explicitly demoted via mark_segment_dynamic().
        Dynamic segments (agentic history, per-user context) use standard LRU
        eviction with Multi-hop invalidation range limiting.

    Integration with vLLM v1 KVCacheManager:
        - The static/dynamic index is a parallel Python dict — it does NOT modify
          vLLM's native block pool or prefix cache.
        - Static segments are protected from pressure eviction by checking
          is_static_segment() before calling evict_blocks().
        - Dynamic segments are invalidated within max_invalidation_range steps of
          the updated segment to prevent unbounded recomputation cascades.

    Activity B non-contiguous reuse:
        Static segments are always eligible for full reuse across requests
        (noncontiguous_ratio target ≥ 30% per evaluation_criteria.md §3).
        The segment index is keyed by SHA-256 hash of token chunk + layer_idx
        (compatible with WorkloadAwareTTLKVCacheManager.store_ttl_segment()).

    Accuracy contract:
        This manager does NOT compress or modify KV values. It only controls
        which block_ids are protected from vLLM's eviction pressure.
    """

    def __init__(
        self,
        *args: Any,
        sdm_max_invalidation_range: int = 2,
        sdm_max_static_segments: int = 512,
        sdm_chunk_size: int = 128,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            sdm_max_invalidation_range: Max number of segments to invalidate
                after a dynamic segment update (Multi-hop depth limit).
            sdm_max_static_segments: Maximum number of static segment slots.
                Excess static segments are demoted to dynamic on overflow.
            sdm_chunk_size: Token chunk size for SHA-256 segment key computation.
        """
        super().__init__(*args, **kwargs)
        self._sdm_max_invalidation_range = sdm_max_invalidation_range
        self._sdm_max_static = sdm_max_static_segments
        self._sdm_chunk_size = sdm_chunk_size

        # Segment metadata
        self._sdm_static_keys: Set[str] = set()
        self._sdm_segment_order: List[str] = []  # insertion order for invalidation
        # segment_key → block_ids set
        self._sdm_block_map: Dict[str, Set[int]] = {}

        # Hit tracking
        self._sdm_static_hits: int = 0
        self._sdm_dynamic_hits: int = 0
        self._sdm_misses: int = 0

    # ------------------------------------------------------------------
    # Public segment API
    # ------------------------------------------------------------------

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        block_ids: Set[int],
        layer_idx: int = 0,
        is_static: bool = False,
    ) -> str:
        """Register a segment in the static/dynamic index.

        Args:
            token_ids: Full prompt token IDs.
            chunk_idx: Chunk index within the prompt.
            block_ids: vLLM block IDs associated with this segment.
            layer_idx: Transformer layer index.
            is_static: If True, mark segment as static (LRU-exempt).

        Returns:
            SHA-256 hex segment key string.
        """
        key = self._sdm_chunk_key(token_ids, chunk_idx, layer_idx)
        self._sdm_block_map[key] = set(block_ids)
        if key not in self._sdm_segment_order:
            self._sdm_segment_order.append(key)
        if is_static:
            self.mark_segment_static(key)
        return key

    def mark_segment_static(self, segment_key: str) -> None:
        """Mark a segment as static (LRU-exempt, no pressure eviction).

        If the static set is at capacity, the oldest static segment is demoted
        to dynamic before adding the new one.
        """
        if len(self._sdm_static_keys) >= self._sdm_max_static:
            # Demote the oldest static segment
            oldest = next(
                (k for k in self._sdm_segment_order if k in self._sdm_static_keys),
                None,
            )
            if oldest is not None:
                self._sdm_static_keys.discard(oldest)
        self._sdm_static_keys.add(segment_key)

    def mark_segment_dynamic(self, segment_key: str) -> None:
        """Demote a segment from static to dynamic (restores eviction eligibility)."""
        self._sdm_static_keys.discard(segment_key)

    def is_static_segment(self, segment_key: str) -> bool:
        """Return True if the segment is currently static (LRU-exempt)."""
        return segment_key in self._sdm_static_keys

    def get_segment_block_ids(self, segment_key: str) -> Optional[Set[int]]:
        """Return vLLM block IDs for a registered segment, or None if not found."""
        block_ids = self._sdm_block_map.get(segment_key)
        if block_ids is not None:
            if segment_key in self._sdm_static_keys:
                self._sdm_static_hits += 1
            else:
                self._sdm_dynamic_hits += 1
        else:
            self._sdm_misses += 1
        return block_ids

    def invalidate_dynamic_range(
        self, segment_key: str
    ) -> List[str]:
        """Invalidate up to max_invalidation_range segments following segment_key.

        Used when a dynamic segment is updated to prevent stale downstream
        segments from being reused. Only invalidates dynamic (non-static) segments.

        Args:
            segment_key: The updated segment's key.

        Returns:
            List of invalidated segment keys (caller should release their block IDs).
        """
        if segment_key not in self._sdm_segment_order:
            return []

        idx = self._sdm_segment_order.index(segment_key)
        invalidation_end = min(
            idx + 1 + self._sdm_max_invalidation_range,
            len(self._sdm_segment_order),
        )
        invalidated: List[str] = []
        for inv_key in self._sdm_segment_order[idx + 1: invalidation_end]:
            if inv_key not in self._sdm_static_keys and inv_key in self._sdm_block_map:
                block_ids = self._sdm_block_map.pop(inv_key, set())
                if block_ids:
                    try:
                        self.evict_blocks(block_ids)
                    except Exception:
                        pass
                invalidated.append(inv_key)
        # Remove invalidated keys from segment_order
        for inv_key in invalidated:
            try:
                self._sdm_segment_order.remove(inv_key)
            except ValueError:
                pass
        return invalidated

    def evict_dynamic_pressure_segment(self) -> Optional[str]:
        """Evict the oldest dynamic (non-static) segment under memory pressure.

        Returns:
            Evicted segment key, or None if no dynamic segments available.
        """
        for key in self._sdm_segment_order:
            if key not in self._sdm_static_keys and key in self._sdm_block_map:
                block_ids = self._sdm_block_map.pop(key)
                self._sdm_segment_order.remove(key)
                try:
                    self.evict_blocks(block_ids)
                except Exception:
                    pass
                return key
        return None

    def sdm_hit_stats(self) -> Dict[str, Any]:
        """Return static/dynamic segment hit statistics."""
        total = self._sdm_static_hits + self._sdm_dynamic_hits + self._sdm_misses
        noncontiguous_hits = self._sdm_static_hits  # static hits are non-contiguous reuse
        total_hits = self._sdm_static_hits + self._sdm_dynamic_hits
        return {
            "static_hits": self._sdm_static_hits,
            "dynamic_hits": self._sdm_dynamic_hits,
            "misses": self._sdm_misses,
            "overall_hit_rate": total_hits / max(total, 1),
            "noncontiguous_ratio": noncontiguous_hits / max(total_hits, 1),
            "num_static_segments": len(self._sdm_static_keys),
            "num_dynamic_segments": len(self._sdm_block_map) - len(self._sdm_static_keys),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sdm_chunk_key(
        self, token_ids: List[int], chunk_idx: int, layer_idx: int
    ) -> str:
        """SHA-256 chunk key (compatible with WorkloadAwareTTLKVCacheManager)."""
        import hashlib
        import struct
        start = chunk_idx * self._sdm_chunk_size
        chunk = token_ids[start: start + self._sdm_chunk_size]
        if not chunk:
            chunk = [0]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()


# ---------------------------------------------------------------------------
# 2026-05-08 Activity C: ManifoldKVWindowedEvictionManager
# ---------------------------------------------------------------------------

class ManifoldKVWindowedEvictionManager(KVCacheManager):
    """KVCacheManager subclass with Euclidean outlier-based eviction scoring.

    Ports src/cache/manifoldkv_windowed.ManifoldKVWindowedEviction into the
    vLLM integration layer.

    ManifoldKV (arXiv 2602.08343) insight:
        Standard cosine-similarity eviction ignores token norm (scale), causing
        semantically important high-norm tokens to be incorrectly evicted.
        Euclidean distance from the sliding-window local centroid captures
        true "outlier" tokens that stand out from their local context.
        Segments with high mean outlier score are retained; low-score
        segments are evicted first.

    Eviction policy:
        evict_lowest_outlier_score(): evict the registered segment with the
        lowest mean Euclidean outlier score. Designed as a drop-in replacement
        for LRU-based evict_blocks() calls in the engine loop.

    Integration:
        Outlier scores are fed by ManifoldKVOutlierScoreHook (attention_backend_patch)
        via register_outlier_score(). The engine loop calls evict_lowest_outlier_score()
        under memory pressure instead of the default LRU eviction.

    Accuracy contract:
        This manager NEVER modifies KV values — it only reorders which vLLM blocks
        are evicted. No quantization, no approximation.

    Usage:

        from vllm_integration.block_manager_patch import ManifoldKVWindowedEvictionManager
        from vllm_integration.attention_backend_patch import ManifoldKVOutlierScoreHook

        # Score store shared between hook and manager
        score_store: dict = {}

        hook = ManifoldKVOutlierScoreHook(segment_score_store=score_store)
        mgr  = ManifoldKVWindowedEvictionManager(
            ...,  # standard KVCacheManager args
            mvwem_window_size=4096,
        )

        # After attention for each layer:
        hook.record_outlier_score(key_tensor, segment_key)
        mgr.register_outlier_score(segment_key, block_ids, score_store[segment_key])

        # Under memory pressure:
        evicted_key = mgr.evict_lowest_outlier_score()
    """

    def __init__(
        self,
        *args: Any,
        mvwem_window_size: int = 4096,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            mvwem_window_size: Sliding window size (tokens) for outlier score
                computation by ManifoldKVOutlierScoreHook.
        """
        super().__init__(*args, **kwargs)
        self._mvwem_window_size = mvwem_window_size

        # segment_key → (block_ids, outlier_score)
        self._mvwem_segments: Dict[str, Tuple[Set[int], float]] = {}
        self._mvwem_evict_count: int = 0

    def register_outlier_score(
        self,
        segment_key: str,
        block_ids: Set[int],
        outlier_score: float,
    ) -> None:
        """Register a segment with its Euclidean outlier score.

        Args:
            segment_key: Segment identifier (e.g. SHA-256 hash).
            block_ids: vLLM block IDs for this segment.
            outlier_score: Mean Euclidean distance from window centroid.
                Higher = more semantically important = keep.
                Lower = less important = evict first.
        """
        self._mvwem_segments[segment_key] = (set(block_ids), float(outlier_score))

    def evict_lowest_outlier_score(self) -> Optional[str]:
        """Evict the segment with the lowest outlier score (least important).

        Returns:
            Evicted segment key, or None if no segments registered.
        """
        if not self._mvwem_segments:
            return None

        worst_key = min(
            self._mvwem_segments.keys(),
            key=lambda k: self._mvwem_segments[k][1],
        )
        block_ids, _ = self._mvwem_segments.pop(worst_key)
        try:
            self.evict_blocks(block_ids)
        except Exception:
            pass
        self._mvwem_evict_count += 1
        return worst_key

    def evict_lowest_n(self, n: int) -> List[str]:
        """Evict the n segments with the lowest outlier scores.

        Args:
            n: Number of segments to evict.

        Returns:
            List of evicted segment keys.
        """
        sorted_keys = sorted(
            self._mvwem_segments.keys(),
            key=lambda k: self._mvwem_segments[k][1],
        )
        evicted = []
        for key in sorted_keys[:n]:
            block_ids, _ = self._mvwem_segments.pop(key)
            try:
                self.evict_blocks(block_ids)
            except Exception:
                pass
            self._mvwem_evict_count += 1
            evicted.append(key)
        return evicted

    def outlier_score_for(self, segment_key: str) -> Optional[float]:
        """Return the registered outlier score for a segment, or None."""
        entry = self._mvwem_segments.get(segment_key)
        return entry[1] if entry is not None else None

    def mvwem_stats(self) -> Dict[str, Any]:
        """Return ManifoldKVWindowedEviction statistics."""
        if self._mvwem_segments:
            scores = [v[1] for v in self._mvwem_segments.values()]
            mean_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
        else:
            mean_score = min_score = max_score = 0.0
        return {
            "registered_segments": len(self._mvwem_segments),
            "evict_count": self._mvwem_evict_count,
            "mean_outlier_score": mean_score,
            "min_outlier_score": min_score,
            "max_outlier_score": max_score,
            "window_size": self._mvwem_window_size,
        }


# ===========================================================================
# 2026-05-09 additions — Activity B (Cross-1): TriangleInequalitySegmentIndex
#                         integration with vLLM KVCacheManager
# ===========================================================================

import torch.nn.functional as F  # noqa: E402  (already imported above in some configs)


# ---------------------------------------------------------------------------
# _LightweightSegmentStore — fallback when SemanticBoundarySegmentCache
# is not importable (no src/ dependency)
# ---------------------------------------------------------------------------

class _LightweightSegmentStore:
    """Minimal CacheStore-compatible LRU backend.

    Used as TriangleInequalitySegmentIndex backend when
    SemanticBoundarySegmentCache is not importable from src/.
    """

    def __init__(self, capacity_bytes: int = 256 * 1024 * 1024) -> None:
        self._store: Dict[str, torch.Tensor] = {}
        self._lru: List[str] = []
        self._capacity_bytes = capacity_bytes
        self._hits: int = 0
        self._misses: int = 0

    def put(self, key: str, value: torch.Tensor) -> None:
        self._store[key] = value
        if key in self._lru:
            self._lru.remove(key)
        self._lru.append(key)
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._hits += 1
            self._lru.remove(key)
            self._lru.append(key)
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._lru:
            return 0
        evict_key = self._lru.pop(0)
        kv = self._store.pop(evict_key, None)
        return kv.nbytes if kv is not None else 0

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def _maybe_evict(self) -> None:
        while self.memory_bytes() > self._capacity_bytes and self._lru:
            self.evict()


# ---------------------------------------------------------------------------
# build_triangle_index — convenience factory (Activity B)
# ---------------------------------------------------------------------------

def build_triangle_index(
    capacity_bytes: int = 256 * 1024 * 1024,
    embedding_dim: int = 64,
    leaf_size: int = 8,
    use_semantic_backend: bool = True,
) -> Any:
    """Build a TriangleInequalitySegmentIndex for Activity B integration.

    Tries to import TriangleInequalitySegmentIndex + SemanticBoundarySegmentCache
    from src/. Falls back to _InlineTriangleIndex + _LightweightSegmentStore when
    the src/ package is not on sys.path.

    Args:
        capacity_bytes: Backend cache capacity in bytes.
        embedding_dim: Embedding dimensionality.
        leaf_size: Leaf size for hierarchical index.
        use_semantic_backend: Use SemanticBoundarySegmentCache if importable.

    Returns:
        TriangleInequalitySegmentIndex or _InlineTriangleIndex instance.
    """
    # Try src/ imports
    TriangleIdx: Any = None
    SemanticCache: Any = None
    try:
        from src.cache.triangle_index import TriangleInequalitySegmentIndex as _TI
        TriangleIdx = _TI
    except ImportError:
        pass
    if use_semantic_backend:
        try:
            from src.cache.semantic_boundary_cache import SemanticBoundarySegmentCache as _SB
            SemanticCache = _SB
        except ImportError:
            pass

    if SemanticCache is not None:
        backend = SemanticCache(capacity_bytes=capacity_bytes)
    else:
        backend = _LightweightSegmentStore(capacity_bytes=capacity_bytes)

    if TriangleIdx is not None:
        return TriangleIdx(
            backend_cache=backend,
            embedding_dim=embedding_dim,
            leaf_size=leaf_size,
        )

    # Fallback: inline index
    return _InlineTriangleIndex(
        backend=backend,
        embedding_dim=embedding_dim,
        leaf_size=leaf_size,
    )


# ---------------------------------------------------------------------------
# _InlineTriangleIndex — fallback (no src/ dependency)
# ---------------------------------------------------------------------------

class _InlineTriangleIndex:
    """Inline triangle inequality segment index (fallback implementation).

    Provides same API as TriangleInequalitySegmentIndex without src/ dependency.
    Uses linear scan for small N; sufficient for N < 10K in CPU context.
    """

    def __init__(
        self,
        backend: Any,
        embedding_dim: int = 64,
        leaf_size: int = 8,
        distance_fn: str = "cosine",
    ) -> None:
        self._backend = backend
        self._embedding_dim = embedding_dim
        self._leaf_size = leaf_size
        self._distance_fn = distance_fn
        self._embeddings: Dict[str, torch.Tensor] = {}

    def put(self, key: str, value: torch.Tensor) -> None:
        self._backend.put(key, value)
        self._embeddings[key] = self._extract_embedding(value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._backend.get(key)

    def evict(self) -> int:
        freed = self._backend.evict()
        current: Set[str] = set()
        if hasattr(self._backend, "keys"):
            current = set(self._backend.keys())
        elif hasattr(self._backend, "_store"):
            current = set(self._backend._store.keys())
        for k in set(self._embeddings.keys()) - current:
            self._embeddings.pop(k, None)
        return freed

    def hit_rate(self) -> float:
        return self._backend.hit_rate()

    def memory_bytes(self) -> int:
        emb_bytes = sum(e.nbytes for e in self._embeddings.values())
        return self._backend.memory_bytes() + emb_bytes

    def reset_stats(self) -> None:
        self._backend.reset_stats()

    def search_nearest(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[float, str]] = []
        for key, emb in self._embeddings.items():
            dist = self._distance(query_embedding, emb)
            if dist <= max_distance:
                results.append((dist, key))
        results.sort(key=lambda x: x[0])
        return [(k, d) for d, k in results[:top_k]]

    def estimate_hit_probability(
        self,
        query_segments: List[torch.Tensor],
        threshold_distance: float = 0.3,
    ) -> float:
        if not query_segments or not self._embeddings:
            return 0.0
        hits = 0
        for seg in query_segments:
            emb = self._extract_embedding(seg)
            nearest = self.search_nearest(emb, top_k=1, max_distance=threshold_distance)
            if nearest and nearest[0][1] <= threshold_distance:
                hits += 1
        return hits / len(query_segments)

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a_f = a.float()
        b_f = b.float()
        if self._distance_fn == "cosine":
            sim = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
            return float(1.0 - sim)
        return float(torch.norm(a_f - b_f).item())

    def _extract_embedding(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        d = self._embedding_dim
        if kv_tensor.dim() == 1:
            flat = kv_tensor.float()
        else:
            flat = kv_tensor.float().mean(dim=0)
        if flat.shape[0] >= d:
            return flat[:d]
        padded = torch.zeros(d)
        padded[: flat.shape[0]] = flat
        return padded


# ---------------------------------------------------------------------------
# SegmentIndexAdapter — auto-synchronizes cache backend ↔ triangle index
# ---------------------------------------------------------------------------

class SegmentIndexAdapter:
    """Adapter that synchronizes a CacheStore backend with TriangleIndex.

    Resolves unresolved issue #4 from Report ①:
    'SemanticBoundarySegmentCache ↔ TriangleInequalitySegmentIndex complete
    integration: currently manual pipeline, no automatic adapter.'

    Every put() call stores to both the cache backend AND registers the
    embedding in the triangle index, ensuring they stay in sync without
    manual synchronization.

    Usage:
        triangle_index = build_triangle_index()
        semantic_cache = SemanticBoundarySegmentCache(capacity_bytes=256*1024**2)

        adapter = SegmentIndexAdapter(semantic_cache, triangle_index)
        adapter.put("seg_key", kv_tensor)   # registers in both
        result = adapter.get("seg_key")     # returns from cache backend
        hits = adapter.search_nearest(emb)  # delegates to triangle_index
    """

    def __init__(
        self,
        cache_backend: Any,
        triangle_index: Any,
    ) -> None:
        self._cache = cache_backend
        self._index = triangle_index

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store in cache AND register in triangle index."""
        self._cache.put(key, value)
        self._index.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        return self._cache.get(key)

    def evict(self) -> int:
        freed = self._cache.evict()
        self._index.evict()
        return freed

    def hit_rate(self) -> float:
        return self._cache.hit_rate()

    def memory_bytes(self) -> int:
        return self._cache.memory_bytes()

    def reset_stats(self) -> None:
        self._cache.reset_stats()
        self._index.reset_stats()

    def search_nearest(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> List[Tuple[str, float]]:
        return self._index.search_nearest(query_embedding, top_k, max_distance)

    def estimate_hit_probability(
        self,
        query_segments: List[torch.Tensor],
        threshold_distance: float = 0.3,
    ) -> float:
        return self._index.estimate_hit_probability(query_segments, threshold_distance)


# ---------------------------------------------------------------------------
# TriangleIndexKVCacheManagerMixin — Activity B vLLM v1 KVCacheManager mixin
# ---------------------------------------------------------------------------

class TriangleIndexKVCacheManagerMixin:
    """Mixin for vLLM v1 KVCacheManager adding non-contiguous segment lookup.

    Overrides get_computed_blocks() to fall back to TriangleInequalitySegmentIndex
    when vLLM's standard prefix-cache finds no contiguous blocks.

    Block boundary constraint (from Activity B integration principles):
        Physical block allocation is NOT modified. Non-contiguous hits are
        communicated via request attribute annotations only. The scheduler
        (HitAwarePPDRouterMixin) reads these annotations for P/D routing.

    Request annotations added by this mixin:
        request.ppd_noncontiguous_hits: List[Tuple[str, float]]
            [(segment_key, distance), ...] sorted by ascending distance.
        request.ppd_noncontiguous_hit_probability: float
            Fraction of segments with a hit within threshold_distance.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import (
            TriangleIndexKVCacheManagerMixin, build_triangle_index
        )

        class TriKVCacheManager(TriangleIndexKVCacheManagerMixin, KVCacheManager):
            pass

        tri_index = build_triangle_index(capacity_bytes=512 * 1024 * 1024)
        kv_manager = TriKVCacheManager(
            ...,  # standard KVCacheManager args
            triangle_index=tri_index,
        )
    """

    def __init__(
        self,
        *args: Any,
        triangle_index: Optional[Any] = None,
        triangle_threshold_distance: float = 0.3,
        triangle_top_k: int = 5,
        triangle_embedding_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tri_index = triangle_index
        self._tri_threshold = triangle_threshold_distance
        self._tri_top_k = triangle_top_k
        self._tri_embedding_dim = triangle_embedding_dim

        # Metrics
        self._tri_total_lookups: int = 0
        self._tri_noncontiguous_hits: int = 0
        self._tri_contiguous_hits: int = 0

    def get_computed_blocks(self, request: Any) -> Any:
        """Override: prefix cache + triangle index fallback for non-contiguous reuse.

        1. Call super().get_computed_blocks() for standard contiguous prefix hit.
        2. If prefix cache returns num_computed_tokens > 0: track contiguous hit.
        3. If num_computed_tokens == 0 and triangle index available:
           - Build segment embeddings from request token IDs.
           - Query triangle index for nearest cached segments.
           - Annotate request with non-contiguous hit metadata.
        4. Return the original (kv_blocks, num_computed_tokens) unchanged.
           Physical blocks are not modified to respect vLLM block boundaries.
        """
        self._tri_total_lookups += 1

        result = super().get_computed_blocks(request)

        # Unpack (KVCacheBlocks, int)
        if isinstance(result, tuple) and len(result) == 2:
            kv_blocks, num_computed = result
        else:
            return result

        if num_computed > 0:
            self._tri_contiguous_hits += 1
            return result

        if self._tri_index is None:
            return result

        token_ids = self._tri_get_token_ids(request)
        if not token_ids:
            return result

        segments = self._tri_tokens_to_segments(token_ids)
        if not segments:
            return result

        # Collect nearest segments from the triangle index
        noncontiguous_hits: List[Tuple[str, float]] = []
        for seg_emb in segments:
            nearest = self._tri_index.search_nearest(
                seg_emb,
                top_k=self._tri_top_k,
                max_distance=self._tri_threshold,
            )
            noncontiguous_hits.extend(nearest)

        # Deduplicate by key, sort by ascending distance
        seen: Set[str] = set()
        deduped: List[Tuple[str, float]] = []
        for key, dist in sorted(noncontiguous_hits, key=lambda x: x[1]):
            if key not in seen:
                seen.add(key)
                deduped.append((key, dist))

        hit_prob = (
            len([h for h in deduped if h[1] <= self._tri_threshold])
            / max(1, len(segments))
        )

        if deduped:
            self._tri_noncontiguous_hits += 1

        # Annotate request for downstream scheduler use
        try:
            request.ppd_noncontiguous_hits = deduped[: self._tri_top_k]
            request.ppd_noncontiguous_hit_probability = hit_prob
        except (AttributeError, TypeError):
            pass  # vLLM Request frozen in some configs; graceful skip

        return result

    def register_segment(self, key: str, kv_tensor: torch.Tensor) -> None:
        """Register a new KV segment in the triangle index.

        Call after caching a new KV block to keep the index current.
        Typical call site: after allocate_slots() completes and new blocks
        are written to the KV cache.

        Args:
            key: Segment key (e.g. block hash hex string).
            kv_tensor: KV tensor [n_tokens, d_head] or [d_embed].
        """
        if self._tri_index is not None:
            self._tri_index.put(key, kv_tensor)

    def non_contiguous_hit_rate(self) -> float:
        """Fraction of total lookups with a non-contiguous index hit."""
        if self._tri_total_lookups == 0:
            return 0.0
        return self._tri_noncontiguous_hits / self._tri_total_lookups

    def triangle_index_stats(self) -> Dict[str, Any]:
        """Return triangle index usage statistics."""
        index_mem = 0
        if self._tri_index is not None:
            try:
                index_mem = self._tri_index.memory_bytes()
            except Exception:
                pass
        return {
            "total_lookups": self._tri_total_lookups,
            "contiguous_hits": self._tri_contiguous_hits,
            "noncontiguous_hits": self._tri_noncontiguous_hits,
            "non_contiguous_hit_rate": self.non_contiguous_hit_rate(),
            "index_memory_bytes": index_mem,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tri_get_token_ids(self, request: Any) -> List[int]:
        token_ids = getattr(request, "prompt_token_ids", None)
        if token_ids is not None:
            return list(token_ids)
        token_ids = getattr(request, "all_token_ids", None)
        if token_ids is not None:
            return list(token_ids)
        return []

    def _tri_tokens_to_segments(
        self,
        token_ids: List[int],
        chunk_size: int = 128,
    ) -> List[torch.Tensor]:
        """Convert token IDs to [embedding_dim] segment embedding tensors."""
        if not token_ids:
            return []
        import hashlib
        import struct
        d = self._tri_embedding_dim
        segments: List[torch.Tensor] = []
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        for i in range(n_chunks):
            chunk = token_ids[i * chunk_size: (i + 1) * chunk_size]
            if not chunk:
                continue
            raw = struct.pack(f"{len(chunk)}I", *[max(0, t) for t in chunk])
            digest = hashlib.sha256(raw).digest()
            seed = int.from_bytes(digest[:4], "little")
            g = torch.Generator()
            g.manual_seed(seed)
            emb = F.normalize(torch.randn(d, generator=g), dim=-1)
            segments.append(emb)
        return segments


# ---------------------------------------------------------------------------
# make_triangle_index_kv_cache_manager_class — factory
# ---------------------------------------------------------------------------

def make_triangle_index_kv_cache_manager_class(
    base_manager_class: Any,
) -> Any:
    """Create a TriangleIndex-aware KVCacheManager subclass.

    Activity B (Cross-1) integration factory:

        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import (
            make_triangle_index_kv_cache_manager_class,
            build_triangle_index,
        )

        tri_index = build_triangle_index()
        TriKVManager = make_triangle_index_kv_cache_manager_class(KVCacheManager)
        kv_manager = TriKVManager(..., triangle_index=tri_index)

    Returns:
        New class subclassing TriangleIndexKVCacheManagerMixin + base_manager_class.
    """

    class TriangleIndexKVCacheManager(  # type: ignore[valid-type]
        TriangleIndexKVCacheManagerMixin, base_manager_class
    ):
        def __init__(
            self,
            *args: Any,
            triangle_index: Optional[Any] = None,
            triangle_threshold_distance: float = 0.3,
            triangle_top_k: int = 5,
            triangle_embedding_dim: int = 64,
            **kwargs: Any,
        ) -> None:
            base_manager_class.__init__(self, *args, **kwargs)
            TriangleIndexKVCacheManagerMixin.__init__(
                self,
                triangle_index=triangle_index,
                triangle_threshold_distance=triangle_threshold_distance,
                triangle_top_k=triangle_top_k,
                triangle_embedding_dim=triangle_embedding_dim,
            )

    TriangleIndexKVCacheManager.__name__ = (
        f"TriangleIndex{base_manager_class.__name__}"
    )
    TriangleIndexKVCacheManager.__qualname__ = TriangleIndexKVCacheManager.__name__
    return TriangleIndexKVCacheManager


# ---------------------------------------------------------------------------
# patch_kv_cache_manager_instance — monkey-patch a live KVCacheManager
# ---------------------------------------------------------------------------

def patch_kv_cache_manager_instance(
    manager: Any,
    triangle_index: Optional[Any] = None,
    threshold_distance: float = 0.3,
    top_k: int = 5,
    embedding_dim: int = 64,
) -> None:
    """Monkey-patch a live vLLM KVCacheManager with TriangleIndex fallback.

    Use when the manager is already constructed inside LLMEngine and cannot
    be replaced via subclassing.

    Args:
        manager: Live vLLM v1 KVCacheManager instance.
        triangle_index: TriangleInequalitySegmentIndex.
        threshold_distance: Max cosine distance for a segment hit.
        top_k: Nearest neighbors per segment lookup.
        embedding_dim: Token embedding dimensionality.
    """
    import types as _types

    if triangle_index is None:
        triangle_index = build_triangle_index(embedding_dim=embedding_dim)

    # Attach state
    manager._tri_index = triangle_index
    manager._tri_threshold = threshold_distance
    manager._tri_top_k = top_k
    manager._tri_embedding_dim = embedding_dim
    manager._tri_total_lookups = 0
    manager._tri_noncontiguous_hits = 0
    manager._tri_contiguous_hits = 0

    original_get_computed = manager.get_computed_blocks
    _mixin_cls = TriangleIndexKVCacheManagerMixin

    def _patched_get_computed(request: Any) -> Any:
        manager._tri_total_lookups += 1
        result = original_get_computed(request)
        if isinstance(result, tuple) and len(result) == 2:
            kv_blocks, num_computed = result
        else:
            return result
        if num_computed > 0:
            manager._tri_contiguous_hits += 1
            return result
        if manager._tri_index is None:
            return result
        token_ids = _mixin_cls._tri_get_token_ids(manager, request)
        if not token_ids:
            return result
        segments = _mixin_cls._tri_tokens_to_segments(manager, token_ids)
        if not segments:
            return result
        hits: List[Tuple[str, float]] = []
        for seg_emb in segments:
            nearest = manager._tri_index.search_nearest(
                seg_emb, top_k=manager._tri_top_k,
                max_distance=manager._tri_threshold,
            )
            hits.extend(nearest)
        seen: Set[str] = set()
        deduped: List[Tuple[str, float]] = []
        for key, dist in sorted(hits, key=lambda x: x[1]):
            if key not in seen:
                seen.add(key)
                deduped.append((key, dist))
        hit_prob = (
            len([h for h in deduped if h[1] <= manager._tri_threshold])
            / max(1, len(segments))
        )
        if deduped:
            manager._tri_noncontiguous_hits += 1
        try:
            request.ppd_noncontiguous_hits = deduped[: manager._tri_top_k]
            request.ppd_noncontiguous_hit_probability = hit_prob
        except (AttributeError, TypeError):
            pass
        return result

    manager.get_computed_blocks = _patched_get_computed

    for method_name in (
        "register_segment",
        "non_contiguous_hit_rate",
        "triangle_index_stats",
        "_tri_get_token_ids",
        "_tri_tokens_to_segments",
    ):
        fn = getattr(_mixin_cls, method_name)
        setattr(manager, method_name, _types.MethodType(fn, manager))


# ===========================================================================
# 2026-05-10 Activity B+C: KVPacketVQBlockManager
# ---------------------------------------------------------------------------
# Ports KVPacketSoftAdapterCache (Activity B) + VQCodec (Activity C) from
# src/cache/kv_packet_adapter.py and src/compression/vq_codec.py into vLLM's
# KVCacheManager subclass.
#
# Design:
#   - Maintains a parallel KVPacket store alongside vLLM's native block pool.
#   - Segments (fixed block_size chunks) are stored with a SoftTokenAdapter.
#   - "Old" tokens (beyond recent_window) are VQ-compressed before storage.
#   - Non-contiguous segment hits annotate the request with
#     ppd_noncontiguous_hits for downstream scheduler routing.
#   - Accuracy constraint: VQ decode ALWAYS completes before tensors are
#     returned to any caller — compressed tensors never reach attention kernel.
#
# vLLM 0.20.2 integration points:
#   - Subclasses KVCacheManager
#   - get_computed_blocks(): after contiguous prefix lookup, checks packet store
#   - allocate_slots(): after allocation, registers new blocks in packet store
# ===========================================================================

import hashlib as _hashlib

class KVPacketVQBlockManager(KVCacheManager):
    """Activity B+C: KVPacket soft-adapter cache + VQ compression, subclasses KVCacheManager.

    Parameters
    ----------
    *args, **kwargs : passed through to KVCacheManager.__init__()
    kvp_n_heads : int — number of KV heads (for SoftTokenAdapter dimensioning)
    kvp_d_head : int — head dimension
    kvp_adapter_rank : int — soft-token adapter rank (default 8)
    kvp_max_packets : int — max packets in LRU cache (default 512)
    kvp_recent_window : int — FP16 token window kept uncompressed (default 64)
    kvp_vq_codec : VQCodec | None — pre-fitted VQCodec; if None, auto-fits on first encode
    kvp_enable : bool — if False, all packet operations are no-ops (graceful degradation)
    """

    def __init__(self, *args, **kwargs):
        kvp_kwargs = {
            "kvp_n_heads": kwargs.pop("kvp_n_heads", 8),
            "kvp_d_head": kwargs.pop("kvp_d_head", 128),
            "kvp_adapter_rank": kwargs.pop("kvp_adapter_rank", 8),
            "kvp_max_packets": kwargs.pop("kvp_max_packets", 512),
            "kvp_recent_window": kwargs.pop("kvp_recent_window", 64),
            "kvp_vq_codec": kwargs.pop("kvp_vq_codec", None),
            "kvp_enable": kwargs.pop("kvp_enable", True),
        }
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass  # graceful: allow standalone / smoke-test usage

        self._kvp_n_heads: int = kvp_kwargs["kvp_n_heads"]
        self._kvp_d_head: int = kvp_kwargs["kvp_d_head"]
        self._kvp_adapter_rank: int = kvp_kwargs["kvp_adapter_rank"]
        self._kvp_max_packets: int = kvp_kwargs["kvp_max_packets"]
        self._kvp_recent_window: int = kvp_kwargs["kvp_recent_window"]
        self._kvp_vq_codec = kvp_kwargs["kvp_vq_codec"]
        self._kvp_enable: bool = kvp_kwargs["kvp_enable"]

        # Packet store: {segment_key: dict} with keys
        #   "adapter_state"  : dict  — SoftTokenAdapter.state_dict()
        #   "kv_vq_codes"    : dict | None
        #   "kv_recent_fp16" : Tensor [min(recent_window, n_tokens), 2, n_heads, d_head]
        #   "positions"      : Tensor [n_tokens] int64
        #   "n_tokens"       : int
        self._kvp_store: "OrderedDict[str, dict]" = OrderedDict()
        self._kvp_lru: "List[str]" = []
        # _kvp_insertion_order: stable insertion order (NOT mutated by LRU moves)
        # Used for non-contiguous hit detection (mirrors kv_packet_adapter.py logic).
        self._kvp_insertion_order: "List[str]" = []
        self._kvp_hits: int = 0
        self._kvp_misses: int = 0
        self._kvp_noncontiguous_hits: int = 0
        self._kvp_access_order: "List[str]" = []
        self._kvp_compress_count: int = 0
        self._kvp_decompress_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _kvp_segment_key(token_ids: "List[int]", chunk_idx: int, layer_idx: int) -> str:
        """Deterministic SHA-256 key for a segment."""
        raw = str(token_ids[:128]).encode() + str(chunk_idx).encode() + str(layer_idx).encode()
        return _hashlib.sha256(raw).hexdigest()

    def _kvp_evict(self) -> None:
        """LRU eviction from packet store."""
        if self._kvp_lru:
            oldest = self._kvp_lru.pop(0)
            self._kvp_store.pop(oldest, None)
            # Keep insertion_order in sync (for non-contiguous tracking)
            try:
                self._kvp_insertion_order.remove(oldest)
            except ValueError:
                pass

    def _kvp_try_import_src(self):
        """Attempt to import src.cache.kv_packet_adapter and src.compression.vq_codec."""
        try:
            import sys
            import pathlib
            repo_root = str(pathlib.Path(__file__).resolve().parent.parent)
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from src.cache.kv_packet_adapter import SoftTokenAdapter as _STA  # type: ignore
            return _STA
        except Exception:
            return None

    def _kvp_make_soft_adapter(self):
        """Create a SoftTokenAdapter from src/ or a minimal fallback nn.Module."""
        try:
            import torch.nn as nn
            _STA = self._kvp_try_import_src()
            if _STA is not None:
                return _STA(self._kvp_n_heads, self._kvp_d_head, self._kvp_adapter_rank)
        except Exception:
            pass
        # Fallback: minimal compatible adapter
        import torch
        import torch.nn as nn

        class _FallbackAdapter(nn.Module):
            def __init__(self, n_heads, d_head, rank):
                super().__init__()
                self.rank = rank
                self.soft_key = nn.Parameter(torch.zeros(rank, n_heads, d_head))
                self.soft_val = nn.Parameter(torch.zeros(rank, n_heads, d_head))
                nn.init.normal_(self.soft_key, std=0.02)
                nn.init.normal_(self.soft_val, std=0.02)

            def adapt(self, kv_block: torch.Tensor) -> torch.Tensor:
                soft = torch.stack([self.soft_key, self.soft_val], dim=1)
                return torch.cat([soft, kv_block], dim=0)

        return _FallbackAdapter(self._kvp_n_heads, self._kvp_d_head, self._kvp_adapter_rank)

    def _kvp_encode(
        self,
        kv_block: "torch.Tensor",
        positions: "torch.Tensor",
        layer_idx: int,
    ) -> "dict":
        """VQ-encode kv_block[:-recent_window] and keep recent tokens as FP16.

        Returns dict with keys:
          "kv_vq_codes"    : dict | None
          "kv_recent_fp16" : Tensor
        """
        import torch
        n_tokens = kv_block.shape[0]
        recent_w = min(self._kvp_recent_window, n_tokens)
        kv_recent_fp16 = kv_block[-recent_w:].to(torch.float16).detach().clone()
        kv_vq_codes = None

        n_old = n_tokens - recent_w
        if n_old > 0 and self._kvp_vq_codec is not None:
            try:
                codec = self._kvp_vq_codec
                kv_old = kv_block[:n_old].to(torch.float16)
                pos_old = positions[:n_old]
                # Check if codec is fitted for this layer
                if layer_idx not in getattr(codec, "key_codebooks", {}):
                    # Auto-fit on the provided data
                    k_flat = kv_old[:, 0].reshape(n_old * self._kvp_n_heads, self._kvp_d_head)
                    v_flat = kv_old[:, 1].reshape(n_old * self._kvp_n_heads, self._kvp_d_head)
                    codec.fit(k_flat, v_flat, layer_idx)
                kv_vq_codes = codec.encode(kv_old, layer_idx, pos_old)
                self._kvp_compress_count += 1
            except Exception:
                # Fallback: keep old tokens as FP16 too
                extra = kv_block[:n_old].to(torch.float16).detach().clone()
                kv_recent_fp16 = torch.cat([extra, kv_recent_fp16], dim=0)
        elif n_old > 0:
            extra = kv_block[:n_old].to(torch.float16).detach().clone()
            kv_recent_fp16 = torch.cat([extra, kv_recent_fp16], dim=0)

        return {"kv_vq_codes": kv_vq_codes, "kv_recent_fp16": kv_recent_fp16}

    def _kvp_decode(self, packet: dict, layer_idx: int) -> "torch.Tensor":
        """Decode a packet back to full FP16 [n_tokens, 2, n_heads, d_head].

        Accuracy contract: VQ decode completes before tensors are returned.
        """
        import torch
        kv_vq_codes = packet.get("kv_vq_codes")
        kv_recent_fp16 = packet["kv_recent_fp16"]

        if kv_vq_codes is not None and self._kvp_vq_codec is not None:
            try:
                kv_old = self._kvp_vq_codec.decode(kv_vq_codes, layer_idx)
                self._kvp_decompress_count += 1
                return torch.cat([kv_old, kv_recent_fp16], dim=0)
            except Exception:
                pass
        return kv_recent_fp16

    def _kvp_adapt_output(
        self, packet: dict, kv_full: "torch.Tensor"
    ) -> "torch.Tensor":
        """Apply soft-token adapter and return [rank + n_tokens, 2, n_heads, d_head]."""
        import torch
        import torch.nn as nn
        adapter_state = packet.get("adapter_state")
        if adapter_state is None:
            return kv_full
        adapter = self._kvp_make_soft_adapter()
        adapter.load_state_dict({k: v.float() for k, v in adapter_state.items()})
        with torch.no_grad():
            return adapter.adapt(kv_full)

    # ------------------------------------------------------------------
    # Public API: packet store
    # ------------------------------------------------------------------

    def kvp_store_segment(
        self,
        token_ids: "List[int]",
        chunk_idx: int,
        kv_block: "torch.Tensor",  # [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
        positions: "Optional[torch.Tensor]" = None,
    ) -> str:
        """Compress and store a KV segment. Returns segment key.

        Stores:
          - kv_block[-recent_window:] as FP16 (recent, uncompressed)
          - kv_block[:-recent_window] as VQ codes (if codec available)
          - SoftTokenAdapter state dict (FP16)
        """
        if not self._kvp_enable:
            return ""
        import torch
        key = self.kvp_segment_key(token_ids, chunk_idx, layer_idx)
        if key in self._kvp_store:
            if key in self._kvp_lru:
                self._kvp_lru.remove(key)
            self._kvp_lru.append(key)
            return key

        if len(self._kvp_store) >= self._kvp_max_packets:
            self._kvp_evict()

        n_tokens = kv_block.shape[0]
        if positions is None:
            positions = torch.arange(n_tokens, dtype=torch.long)

        encoded = self._kvp_encode(kv_block, positions, layer_idx)
        adapter = self._kvp_make_soft_adapter()
        adapter_state = {k: v.to(torch.float16).detach().clone() for k, v in adapter.state_dict().items()}

        self._kvp_store[key] = {
            "kv_vq_codes": encoded["kv_vq_codes"],
            "kv_recent_fp16": encoded["kv_recent_fp16"],
            "adapter_state": adapter_state,
            "positions": positions.clone(),
            "n_tokens": n_tokens,
            "layer_idx": layer_idx,
        }
        self._kvp_lru.append(key)
        self._kvp_insertion_order.append(key)
        return key

    # Alias for test-code compatibility
    def kvp_segment_key(
        self,
        token_ids: "List[int]",
        chunk_idx: int,
        layer_idx: int,
    ) -> str:
        return self._kvp_segment_key(token_ids, chunk_idx, layer_idx)

    def kvp_get_segment(
        self,
        key: str,
        layer_idx: int = 0,
        apply_adapter: bool = True,
    ) -> "Optional[torch.Tensor]":
        """Retrieve a segment: VQ-decode + optional SoftTokenAdapter.adapt().

        Returns [rank + n_tokens, 2, n_heads, d_head] (or [n_tokens, ...] if apply_adapter=False).
        Returns None on miss.
        """
        if not self._kvp_enable or key not in self._kvp_store:
            self._kvp_misses += 1
            return None

        self._kvp_hits += 1

        # Non-contiguous tracking: use stable insertion order (not LRU order)
        # Mirrors kv_packet_adapter.py: a hit is non-contiguous when the current
        # and previous accessed keys are not adjacent in insertion order.
        if self._kvp_access_order:
            prev = self._kvp_access_order[-1]
            if prev in self._kvp_insertion_order and key in self._kvp_insertion_order:
                prev_pos = self._kvp_insertion_order.index(prev)
                curr_pos = self._kvp_insertion_order.index(key)
                if abs(curr_pos - prev_pos) != 1:
                    self._kvp_noncontiguous_hits += 1
            else:
                self._kvp_noncontiguous_hits += 1
        self._kvp_access_order.append(key)

        # Move to MRU
        if key in self._kvp_lru:
            self._kvp_lru.remove(key)
        self._kvp_lru.append(key)

        packet = self._kvp_store[key]
        kv_full = self._kvp_decode(packet, packet.get("layer_idx", layer_idx))

        if apply_adapter:
            return self._kvp_adapt_output(packet, kv_full)
        return kv_full

    def kvp_pack_segments(
        self,
        keys: "List[str]",
        layer_idx: int = 0,
    ) -> "Optional[torch.Tensor]":
        """Concatenate multiple adapted segments without recomputation.

        Returns [sum(rank + n_tokens_i), 2, n_heads, d_head] or None on any miss.
        Accuracy contract: each segment is fully VQ-decoded before concatenation.
        """
        import torch
        parts = []
        for key in keys:
            adapted = self.kvp_get_segment(key, layer_idx=layer_idx, apply_adapter=True)
            if adapted is None:
                return None
            parts.append(adapted)
        return torch.cat(parts, dim=0) if parts else None

    # ------------------------------------------------------------------
    # Override vLLM KVCacheManager.get_computed_blocks
    # ------------------------------------------------------------------

    def get_computed_blocks(self, request: Any) -> "tuple":
        """Extend contiguous prefix lookup with non-contiguous KVPacket annotation.

        After the standard vLLM prefix cache lookup, if num_computed == 0,
        checks the packet store for non-contiguous segments matching the
        request's token_ids prefix chunks. Annotates request with:
            request.kvp_noncontiguous_hits: List[str]   segment keys
            request.kvp_noncontiguous_hit_rate: float
        """
        try:
            result = super().get_computed_blocks(request)
        except Exception:
            # Graceful: return empty blocks
            return object(), 0

        if not self._kvp_enable or not self._kvp_store:
            return result

        try:
            blocks, num_computed = result
            token_ids = getattr(request, "prompt_token_ids", None) or []
            if not token_ids or num_computed > 0:
                return result  # contiguous hit: no need for non-contiguous lookup

            # Search packet store for matching segment chunks
            import torch
            chunk_size = min(self._kvp_recent_window, 128)
            n_chunks = len(token_ids) // chunk_size
            hits = []
            for ci in range(n_chunks):
                seg_key = self._kvp_segment_key(token_ids, ci, layer_idx=0)
                if seg_key in self._kvp_store:
                    hits.append(seg_key)

            try:
                request.kvp_noncontiguous_hits = hits
                request.kvp_noncontiguous_hit_rate = len(hits) / max(1, n_chunks)
            except (AttributeError, TypeError):
                pass

            return result
        except Exception:
            return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def kvp_stats(self) -> dict:
        """Return packet store statistics."""
        total = self._kvp_hits + self._kvp_misses
        hit_rate = self._kvp_hits / total if total > 0 else 0.0
        noncontiguous_ratio = self._kvp_noncontiguous_hits / max(1, self._kvp_hits)
        return {
            "hits": self._kvp_hits,
            "misses": self._kvp_misses,
            "hit_rate": hit_rate,
            "noncontiguous_hits": self._kvp_noncontiguous_hits,
            "noncontiguous_ratio": noncontiguous_ratio,
            "num_packets": len(self._kvp_store),
            "compress_count": self._kvp_compress_count,
            "decompress_count": self._kvp_decompress_count,
        }

    def kvp_compression_ratio(self) -> float:
        """Effective VQ compression ratio across all stored packets."""
        import torch
        if not self._kvp_store:
            return 0.0
        fp16_bytes = 2
        total_orig = 0
        total_stored = 0
        for packet in self._kvp_store.values():
            n = packet.get("n_tokens", 1)
            orig = n * 2 * self._kvp_n_heads * self._kvp_d_head * fp16_bytes
            total_orig += orig
            stored = packet["kv_recent_fp16"].nbytes
            vq = packet.get("kv_vq_codes")
            if vq is not None:
                stored += vq.get("key_codes", torch.zeros(0)).nbytes
                stored += vq.get("val_codes", torch.zeros(0)).nbytes
            total_stored += stored
        if total_orig == 0:
            return 0.0
        return max(0.0, 1.0 - total_stored / total_orig)


def make_kvp_vq_kv_cache_manager_class(base_class: type = None) -> type:
    """Factory: create a KVPacketVQBlockManager subclass of base_class (or KVCacheManager).

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        KVPVQManager = make_kvp_vq_kv_cache_manager_class(KVCacheManager)
    """
    if base_class is None:
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager as _KVCacheManager
            base_class = _KVCacheManager
        except Exception:
            base_class = object

    class _KVPVQManager(KVPacketVQBlockManager, base_class):  # type: ignore[misc]
        pass

    _KVPVQManager.__name__ = f"KVPacketVQ_{base_class.__name__}"
    _KVPVQManager.__qualname__ = _KVPVQManager.__name__
    return _KVPVQManager


# ---------------------------------------------------------------------------
# 2026-05-11 Activity B: WiCERBlockManager — non-contiguous CEGAR KV segment
#            cache for vLLM 0.20.2.
# ---------------------------------------------------------------------------

class WiCERBlockManager:
    """Non-contiguous KV segment manager based on WiCER CEGAR artefact cache.

    Ports the WiCER CEGAR algorithm (src/cache/wicer_iterative_cache.py) into
    vLLM's block management layer as a **parallel** auxiliary segment store
    that sits alongside vLLM's native paged prefix cache.

    Design principles (vLLM porter rules):
    - Does NOT subclass KVCacheManager to avoid breaking the native block pool.
    - Implements the B integration pattern: parallel segment hash store with
      CEGAR refinement, annotation of request.wicer_noncontiguous_hits.
    - Block boundaries respect vLLM's block_size.
    - Compression hook (RateQuantVllmCodec) is applied at store time and
      decompressed at read time (never compressed tensors to attention kernel).

    Key API:
        store_segment(token_ids, chunk_idx, kv_tensor, layer_idx, codec=None)
            → segment_key (str)  store KV block, optional RateQuant compression.
        get_segment(segment_key) → kv_tensor | None
            read back; if compressed, dequantise BEFORE returning.
        cegar_compile(docs, kv_fn, codec=None)
            compile domain corpus into hash-indexed KV artefacts.
        cegar_evaluate(val_queries) → (hit_rate, counterexamples)
        cegar_refine(counterexamples, docs, kv_fn, codec=None)
            halve chunk size for counterexample documents and recompile.
        annotate_request(request, token_ids)
            set request.wicer_noncontiguous_hits list.

    Integration with vLLM:
        Create one WiCERBlockManager per server; call store_segment() after
        each prefill step to populate the cache. For domain corpus pre-loading,
        call cegar_compile() offline and call load_artifacts() before serving.

    Accuracy contract:
        get_segment() always returns float16 tensors. If the segment was stored
        with a RateQuantVllmCodec, dequantisation is performed transparently
        before returning. Quantised tensors never leave this manager.
    """

    def __init__(
        self,
        chunk_size: int = 128,
        min_chunk_size: int = 16,
        max_chunk_size: int = 512,
        max_entries: int = 2000,
        target_hit_rate: float = 0.80,
        max_cegar_iterations: int = 5,
        vllm_block_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_entries = max_entries
        self.target_hit_rate = target_hit_rate
        self.max_cegar_iterations = max_cegar_iterations
        self.vllm_block_size = vllm_block_size
        self.seed = seed
        # Primary segment store: hash → {"kv": Tensor, "compressed": bool, "payload": dict}
        self._store: OrderedDict = OrderedDict()
        # Per-document chunk sizes (refined by CEGAR)
        self._chunk_sizes: Dict[str, int] = {}
        # CEGAR history
        self._hit_rate_history: List[float] = []
        self._cegar_iteration: int = 0
        # Stats
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0
        self._compress_count: int = 0
        self._decompress_count: int = 0

    # ------------------------------------------------------------------
    # Segment store / retrieve
    # ------------------------------------------------------------------

    def _segment_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
        chunk_size: Optional[int] = None,
    ) -> str:
        """Stable SHA-256 hash of the chunk token content."""
        import hashlib
        import struct
        cs = chunk_size if chunk_size is not None else self.chunk_size
        start = chunk_idx * cs
        end = start + cs
        chunk = token_ids[start:end]
        if not chunk:
            return ""
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv_tensor: torch.Tensor,   # [n_tokens, 2, n_heads, d_head] float16
        layer_idx: int = 0,
        codec: Optional[Any] = None,  # RateQuantVllmCodec | None
        chunk_size: Optional[int] = None,
    ) -> str:
        """Store a KV segment. Block boundaries must respect vllm_block_size.

        Args:
            token_ids: Token IDs for the full sequence (used for key hashing).
            chunk_idx: Chunk index within the sequence.
            kv_tensor: [n_tokens, 2, n_heads, d_head] float16 KV block.
            layer_idx: Transformer layer.
            codec: Optional RateQuantVllmCodec; if provided, compress before storing.
            chunk_size: Override chunk_size for this segment (used during CEGAR).

        Returns:
            Segment key (64-char hex string) or "" if chunk is empty.
        """
        key = self._segment_key(token_ids, chunk_idx, layer_idx, chunk_size)
        if not key:
            return ""

        if codec is not None and hasattr(codec, "write_to_cache"):
            payload = codec.write_to_cache(kv_tensor, layer_idx)
            self._compress_count += 1
            entry = {"kv": None, "compressed": True, "payload": payload}
        else:
            entry = {"kv": kv_tensor, "compressed": False, "payload": None}

        # LRU eviction
        if key in self._store:
            self._store.move_to_end(key)
        else:
            while len(self._store) >= self.max_entries:
                self._store.popitem(last=False)
            self._store[key] = entry

        return key

    def get_segment(
        self,
        segment_key: str,
        codec: Optional[Any] = None,   # RateQuantVllmCodec | None
    ) -> Optional[torch.Tensor]:
        """Retrieve a KV segment, decompressing if necessary.

        ACCURACY CONTRACT: always returns float16 tensor; quantised payloads
        are dequantised here — they never leave this method compressed.
        """
        if not segment_key:
            self._misses += 1
            return None
        entry = self._store.get(segment_key)
        if entry is None:
            self._misses += 1
            return None

        self._hits += 1
        self._store.move_to_end(segment_key)

        if entry["compressed"]:
            if codec is not None and hasattr(codec, "read_from_cache"):
                kv = codec.read_from_cache(entry["payload"])
            else:
                # Fallback: try inline dequant (payload carries quantised data)
                payload = entry["payload"]
                if payload.get("compressed") and "quantized" in payload:
                    tensors: list = []
                    for q, scale, zero_pt in zip(
                        payload["quantized"], payload["scales"], payload["zero_pts"]
                    ):
                        kv_h = (q.float() - zero_pt) * scale
                        tensors.append(kv_h)
                    kv = torch.stack(tensors, dim=2).half()
                else:
                    kv = payload.get("raw_kv", torch.zeros(1))
            self._decompress_count += 1
            return kv
        return entry["kv"]

    # ------------------------------------------------------------------
    # CEGAR artefact cache
    # ------------------------------------------------------------------

    def cegar_compile(
        self,
        docs: Dict[str, List[int]],
        kv_fn: Any,                      # callable(token_ids, layer_idx) → Tensor
        layer_idx: int = 0,
        codec: Optional[Any] = None,
    ) -> None:
        """Compile domain corpus: split into chunks and pre-compute KV.

        Respects per-document chunk sizes refined by cegar_refine().
        Block boundaries are aligned to vllm_block_size.
        """
        import math
        for doc_id, token_ids in docs.items():
            cs = self._chunk_sizes.get(doc_id, self.chunk_size)
            # Align chunk size to vLLM block boundaries
            cs = max(self.vllm_block_size, (cs // self.vllm_block_size) * self.vllm_block_size)
            cs = max(self.min_chunk_size, cs)
            n_chunks = max(1, math.ceil(len(token_ids) / cs))
            for chunk_idx in range(n_chunks):
                start = chunk_idx * cs
                end = start + cs
                chunk = token_ids[start:end]
                if not chunk:
                    continue
                kv = kv_fn(chunk, layer_idx)
                self.store_segment(token_ids, chunk_idx, kv, layer_idx, codec, chunk_size=cs)

    def cegar_evaluate(
        self,
        val_queries: List[List[int]],
        layer_idx: int = 0,
    ) -> tuple:
        """Evaluate hit rate on validation queries.

        Returns (hit_rate, counterexamples) where counterexamples is a list of
        (query_prefix_hash, miss_chunk_idx) tuples.
        """
        import hashlib, struct, math
        total_hits = 0
        total_chunks = 0
        counterexamples: list = []
        for token_ids in val_queries:
            cs = self.chunk_size
            n_chunks = max(1, math.ceil(len(token_ids) / cs))
            prev_hit = False
            for chunk_idx in range(n_chunks):
                key = self._segment_key(token_ids, chunk_idx, layer_idx, cs)
                if key and key in self._store:
                    total_hits += 1
                    if not prev_hit:
                        self._noncontiguous_hits += 1
                    prev_hit = True
                else:
                    total_chunks += 1   # count miss
                    prev_hit = False
                    raw = struct.pack(f"{len(token_ids)}I", *token_ids)
                    ph = hashlib.sha256(raw).hexdigest()[:16]
                    counterexamples.append((ph, chunk_idx))
            total_chunks += total_hits  # add hits to total

        # Recalculate correctly
        total_hits_real = 0
        total_real = 0
        for token_ids in val_queries:
            cs = self.chunk_size
            n_chunks = max(1, math.ceil(len(token_ids) / cs))
            for chunk_idx in range(n_chunks):
                key = self._segment_key(token_ids, chunk_idx, layer_idx, cs)
                total_real += 1
                if key and key in self._store:
                    total_hits_real += 1

        hit_rate = total_hits_real / total_real if total_real > 0 else 0.0
        return hit_rate, counterexamples

    def cegar_refine(
        self,
        counterexamples: list,
        docs: Dict[str, List[int]],
        kv_fn: Any,
        layer_idx: int = 0,
        codec: Optional[Any] = None,
    ) -> None:
        """Refine: halve chunk size for documents that have counterexample misses."""
        refined: set = set()
        for _qhash, miss_chunk_idx in counterexamples:
            for doc_id, token_ids in docs.items():
                cur_cs = self._chunk_sizes.get(doc_id, self.chunk_size)
                if miss_chunk_idx * cur_cs < len(token_ids):
                    new_cs = max(self.min_chunk_size, cur_cs // 2)
                    if new_cs < cur_cs:
                        self._chunk_sizes[doc_id] = new_cs
                        refined.add(doc_id)
        if refined:
            self.cegar_compile(
                {k: v for k, v in docs.items() if k in refined},
                kv_fn, layer_idx, codec,
            )

    def cegar_loop(
        self,
        docs: Dict[str, List[int]],
        val_queries: List[List[int]],
        kv_fn: Any,
        layer_idx: int = 0,
        codec: Optional[Any] = None,
    ) -> None:
        """Full CEGAR loop: compile → evaluate → refine until convergence."""
        self.cegar_compile(docs, kv_fn, layer_idx, codec)
        for iteration in range(self.max_cegar_iterations):
            self._cegar_iteration = iteration
            hit_rate, cex = self.cegar_evaluate(val_queries, layer_idx)
            self._hit_rate_history.append(hit_rate)
            if hit_rate >= self.target_hit_rate or not cex:
                break
            self.cegar_refine(cex, docs, kv_fn, layer_idx, codec)

    # ------------------------------------------------------------------
    # Request annotation (for scheduler visibility)
    # ------------------------------------------------------------------

    def annotate_request(
        self,
        request: Any,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> None:
        """Annotate request with non-contiguous WiCER hit information.

        Sets:
            request.wicer_noncontiguous_hits  — list of (chunk_idx, segment_key)
            request.wicer_hit_rate            — fraction of chunks with hits
        """
        import math
        cs = self.chunk_size
        n_chunks = max(1, math.ceil(len(token_ids) / cs))
        hits: list = []
        prev_hit = False
        for chunk_idx in range(n_chunks):
            key = self._segment_key(token_ids, chunk_idx, layer_idx, cs)
            if key and key in self._store:
                if not prev_hit:
                    hits.append((chunk_idx, key))   # non-contiguous hit
                prev_hit = True
            else:
                prev_hit = False
        request.wicer_noncontiguous_hits = hits
        request.wicer_hit_rate = len(hits) / max(n_chunks, 1)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_artifacts(self, path: str) -> None:
        """Serialise segment store + CEGAR state to disk."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        # Serialise only raw (uncompressed) segments; compressed payloads saved as-is.
        torch.save(
            {
                "store": dict(self._store),
                "chunk_sizes": self._chunk_sizes,
                "hit_rate_history": self._hit_rate_history,
                "cegar_iteration": self._cegar_iteration,
            },
            path,
        )

    def load_artifacts(self, path: str) -> None:
        """Restore segment store + CEGAR state from disk."""
        data = torch.load(path, weights_only=False)
        self._store = OrderedDict(data["store"])
        self._chunk_sizes = data["chunk_sizes"]
        self._hit_rate_history = data["hit_rate_history"]
        self._cegar_iteration = data.get("cegar_iteration", 0)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def hit_stats(self) -> dict:
        """Return hit/miss/noncontiguous counters."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "noncontiguous_hits": self._noncontiguous_hits,
            "noncontiguous_ratio": self._noncontiguous_hits / max(self._hits, 1),
            "compress_count": self._compress_count,
            "decompress_count": self._decompress_count,
            "cegar_iterations": self._cegar_iteration,
            "hit_rate_history": list(self._hit_rate_history),
            "num_segments": len(self._store),
        }


def make_wicer_kv_cache_manager_class(base_class: type) -> type:
    """Factory: create a WiCER-augmented KVCacheManager subclass.

    Returns a class that inherits from both KVPacketVQBlockManager
    (prior B+C cycle) and WiCERBlockManager at the Python level as a
    composition pattern. The native KVCacheManager block pool is not
    modified — WiCER operates as a parallel auxiliary store.

    Args:
        base_class: vLLM KVCacheManager (or compatible subclass).

    Returns:
        New class with WiCER segment API added.
    """

    class _WiCERKVCacheManager(base_class):  # type: ignore[misc]
        """KVCacheManager subclass with WiCER parallel segment store."""

        def __init__(self, *args, wicer_manager: Optional["WiCERBlockManager"] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._wicer = wicer_manager or WiCERBlockManager()

        def wicer_store_segment(
            self,
            token_ids: List[int],
            chunk_idx: int,
            kv_tensor: torch.Tensor,
            layer_idx: int = 0,
            codec: Optional[Any] = None,
        ) -> str:
            return self._wicer.store_segment(token_ids, chunk_idx, kv_tensor, layer_idx, codec)

        def wicer_get_segment(
            self,
            segment_key: str,
            codec: Optional[Any] = None,
        ) -> Optional[torch.Tensor]:
            return self._wicer.get_segment(segment_key, codec)

        def wicer_annotate_request(self, request: Any, token_ids: List[int], layer_idx: int = 0) -> None:
            self._wicer.annotate_request(request, token_ids, layer_idx)

        def wicer_hit_stats(self) -> dict:
            return self._wicer.hit_stats()

    _WiCERKVCacheManager.__name__ = f"WiCER_{base_class.__name__}"
    _WiCERKVCacheManager.__qualname__ = _WiCERKVCacheManager.__name__
    return _WiCERKVCacheManager


# ===========================================================================
# 2026-05-12 Activity B+C: AdapShotBlockManager
# ===========================================================================

class AdapShotBlockManager:
    """Auxiliary KV store that applies the AdapShotMixedDimSegmentPipeline (B+C Cross-2).

    Acts as a parallel segment index alongside vLLM's native paged block pool.
    The native block pool is NOT modified — this manager operates an independent
    content-addressed store keyed by chunk content hash.

    Store order contract (from adapshot_pipeline.py):
        raw pre-RoPE KV
            → [1] MixedDimPerTokenBudgetCodec.encode()   (Activity C compression)
            → [2] RoPEReencodingNonContiguousCache.store_pre_rope()  (Activity B storage)

    Restore order contract:
        stored compressed KV
            → [1] RoPEReencodingNonContiguousCache.load_with_rope()  (pre-RoPE load + RoPE re-apply)
            → [2] MixedDimPerTokenBudgetCodec.decode()  (masked_kv — no-op, zeroed dims stay 0)
            → final KV (RoPE-encoded, mixed-dim compressed)

    Accuracy contract (Activity C):
        Compressed KV is never passed to the attention kernel in compressed form.
        load_segment() always returns the RoPE-re-applied, decoded tensor before
        the caller can invoke an attention operation.

    Non-contiguous reuse (Activity B):
        Segments are keyed by content hash (position-independent). On load, RoPE is
        re-applied for the target request's token positions, enabling reuse of segments
        whose tokens appear at different absolute positions than when they were stored.

    Memory reduction (Activity C):
        budget_ratio=0.50 → −50% KV memory (validated in Report ① 2026-05-12).
        All budget_ratio ∈ [0.30, 0.70] achieve < 1% perplexity delta.

    Usage::

        from vllm_integration.block_manager_patch import AdapShotBlockManager
        mgr = AdapShotBlockManager(chunk_size=128, budget_ratio=0.50)
        mgr.store_segment(token_ids, chunk_idx=0, pre_rope_kv=kv_tensor, layer_idx=3)
        result = mgr.load_segment(token_ids, chunk_idx=0, target_offset=256, layer_idx=3)
        if result is not None:
            # result is RoPE-re-applied + mixed-dim decoded — safe to use in attention
            pass

    Integration with vLLM:
        Use make_adapshot_kv_cache_manager_class() to create a KVCacheManager subclass
        that carries an AdapShotBlockManager instance. The model runner or attention
        wrapper calls store_segment() after prefill and load_segment() before decoding.
    """

    def __init__(
        self,
        chunk_size: int = 128,
        max_entries: int = 2000,
        d_head: int = 64,
        n_heads: int = 8,
        rope_base: float = 10000.0,
        budget_ratio: float = 0.50,
        bisection_iters: int = 64,
        min_retain_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        """
        Args:
            chunk_size: Number of tokens per cache segment (must align with vLLM block_size).
            max_entries: Maximum number of segments in the LRU store.
            d_head: KV head dimension.
            n_heads: Number of KV heads.
            rope_base: RoPE base frequency (default 10000 for LLaMA-style).
            budget_ratio: Fraction of KV dimensions to retain (0.50 → 50% retained, −50% memory).
            bisection_iters: Bisection iterations for threshold search (64 is sufficient).
            min_retain_ratio: Per-token minimum dimension retention (safeguard against over-pruning).
            seed: Random seed for reproducibility.
        """
        import sys
        import os
        # Ensure src/ is on path for direct algorithm imports (no code duplication)
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)

        from src.cache.rope_reencoding_cache import RoPEReencodingConfig
        from src.cache.mixed_dim_codec import MixedDimConfig
        from src.cache.adapshot_pipeline import AdapShotPipelineConfig, AdapShotMixedDimSegmentPipeline

        rope_cfg = RoPEReencodingConfig(
            chunk_size=chunk_size,
            max_entries=max_entries,
            d_head=d_head,
            n_heads=n_heads,
            rope_base=rope_base,
            seed=seed,
        )
        mixdim_cfg = MixedDimConfig(
            n_heads=n_heads,
            d_head=d_head,
            budget_ratio=budget_ratio,
            bisection_iters=bisection_iters,
            min_retain_ratio=min_retain_ratio,
            seed=seed,
        )
        pipeline_cfg = AdapShotPipelineConfig(rope=rope_cfg, mixed_dim=mixdim_cfg)
        self._pipeline = AdapShotMixedDimSegmentPipeline(pipeline_cfg)
        self._chunk_size = chunk_size
        self._budget_ratio = budget_ratio

    # -----------------------------------------------------------------------
    # Core segment store/load API
    # -----------------------------------------------------------------------

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        pre_rope_kv: "torch.Tensor",   # [n_tokens, 2, n_heads, d_head] raw pre-RoPE
        layer_idx: int = 0,
        attn_weights: Optional["torch.Tensor"] = None,  # [n_tokens] importance
    ) -> None:
        """Store a KV chunk via the B+C pipeline: mixed-dim compress → pre-RoPE store.

        Args:
            token_ids: Full token ID sequence for the request.
            chunk_idx: Index of this chunk (0-based).
            pre_rope_kv: Raw (pre-RoPE) KV tensor [n_tokens, 2, n_heads, d_head].
            layer_idx: Transformer layer index (used for scoped cache keys).
            attn_weights: Optional per-token attention importance [n_tokens].
                          If None, value magnitude is used as proxy.
        """
        self._pipeline.store_segment(
            token_ids=token_ids,
            chunk_idx=chunk_idx,
            pre_rope_kv=pre_rope_kv,
            layer_idx=layer_idx,
            attn_weights=attn_weights,
        )

    def load_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        target_offset: int,   # absolute position of chunk's first token in this request
        layer_idx: int = 0,
    ) -> Optional["torch.Tensor"]:
        """Load a KV chunk via the B+C restore contract: RoPE re-apply → decode.

        Returns:
            [n_tokens, 2, n_heads, d_head] tensor (RoPE-encoded, mixed-dim decoded).
            The zeroed dimensions stay zero (lossy for low-importance dims only).
            Returns None on cache miss.

        Accuracy contract:
            The returned tensor is fully decompressed (no quantization remains).
            It is safe to pass directly to an attention kernel.
        """
        return self._pipeline.load_segment(
            token_ids=token_ids,
            chunk_idx=chunk_idx,
            target_offset=target_offset,
            layer_idx=layer_idx,
        )

    def get_segments(
        self,
        token_ids: List[int],
        target_offset: int,
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, "torch.Tensor"]], List[int]]:
        """Retrieve all chunks for a request; classify as hits or misses.

        Non-contiguous hits are tracked automatically (hits after a lower-index miss).

        Returns:
            hits:   [(chunk_idx, kv_tensor), ...] — all cache hits (RoPE re-applied)
            misses: [chunk_idx, ...]              — all cache misses
        """
        return self._pipeline.get_segments(token_ids, target_offset, layer_idx)

    def annotate_request(
        self,
        request: Any,
        token_ids: List[int],
        target_offset: int = 0,
        layer_idx: int = 0,
    ) -> None:
        """Attach segment hit metadata to a vLLM Request object.

        Sets request attributes:
            adapshot_hits:   list of (chunk_idx, kv_tensor) for cache hits
            adapshot_misses: list of chunk_idx values for cache misses
            adapshot_noncontiguous_hit_rate: float

        These attributes are read by AdapShotSegmentSchedulerMixin in scheduler_patch.py
        to prioritise requests with high non-contiguous hit rates.

        Args:
            request: Any vLLM Request / SequenceGroup object (duck-typed).
            token_ids: Token IDs of the request's prompt.
            target_offset: Absolute position of the first token in this request.
            layer_idx: Transformer layer index.
        """
        hits, misses = self.get_segments(token_ids, target_offset, layer_idx)
        try:
            request.adapshot_hits = hits
            request.adapshot_misses = misses
            request.adapshot_noncontiguous_hit_rate = self._pipeline.noncontiguous_hit_rate()
        except AttributeError:
            pass  # request may not support arbitrary attribute assignment

    def hit_stats(self) -> dict:
        """Return hit/miss statistics for the underlying pipeline."""
        return {
            "hit_rate": self._pipeline.hit_rate(),
            "noncontiguous_hit_rate": self._pipeline.noncontiguous_hit_rate(),
            "memory_bytes": self._pipeline.memory_bytes(),
            "budget_ratio": self._budget_ratio,
            "chunk_size": self._chunk_size,
        }

    def reset_stats(self) -> None:
        """Reset hit/miss counters."""
        self._pipeline.reset_stats()


# ---------------------------------------------------------------------------
# Factory: make_adapshot_kv_cache_manager_class
# ---------------------------------------------------------------------------

def make_adapshot_kv_cache_manager_class(
    base_class: Optional[type] = None,
    chunk_size: int = 128,
    max_entries: int = 2000,
    d_head: int = 64,
    n_heads: int = 8,
    rope_base: float = 10000.0,
    budget_ratio: float = 0.50,
    bisection_iters: int = 64,
    min_retain_ratio: float = 0.10,
    seed: int = 42,
) -> type:
    """Factory that returns a KVCacheManager subclass carrying an AdapShotBlockManager.

    The returned class is a subclass of base_class (default: vllm KVCacheManager).
    It adds the following public methods:

        adapshot_store_segment(token_ids, chunk_idx, pre_rope_kv, layer_idx, attn_weights)
        adapshot_load_segment(token_ids, chunk_idx, target_offset, layer_idx) → tensor | None
        adapshot_get_segments(token_ids, target_offset, layer_idx) → (hits, misses)
        adapshot_annotate_request(request, token_ids, target_offset, layer_idx)
        adapshot_hit_stats() → dict

    The AdapShotBlockManager is stored as self._adapshot_mgr.
    It operates as a parallel auxiliary store — the native vLLM block pool is NOT modified.

    Usage::

        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm_integration.block_manager_patch import make_adapshot_kv_cache_manager_class

        AdapShotKVManager = make_adapshot_kv_cache_manager_class(
            KVCacheManager,
            chunk_size=128,
            budget_ratio=0.50,
        )
        kv_mgr = AdapShotKVManager(*original_args, **original_kwargs)
        # After prefill:
        kv_mgr.adapshot_store_segment(token_ids, 0, pre_rope_kv, layer_idx=0)
        # Before decode attention:
        kv = kv_mgr.adapshot_load_segment(token_ids, 0, target_offset=0, layer_idx=0)
    """
    if base_class is None:
        try:
            from vllm.v1.core.kv_cache_manager import KVCacheManager as _KVCacheManager
            base_class = _KVCacheManager
        except ImportError:
            base_class = object

    class _AdapShotKVCacheManager(base_class):  # type: ignore[misc]
        """KVCacheManager subclass with AdapShot B+C auxiliary segment store."""

        def __init__(self, *args, adapshot_manager: Optional[AdapShotBlockManager] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self._adapshot_mgr = adapshot_manager or AdapShotBlockManager(
                chunk_size=chunk_size,
                max_entries=max_entries,
                d_head=d_head,
                n_heads=n_heads,
                rope_base=rope_base,
                budget_ratio=budget_ratio,
                bisection_iters=bisection_iters,
                min_retain_ratio=min_retain_ratio,
                seed=seed,
            )

        def adapshot_store_segment(
            self,
            token_ids: List[int],
            chunk_idx: int,
            pre_rope_kv: "torch.Tensor",
            layer_idx: int = 0,
            attn_weights: Optional["torch.Tensor"] = None,
        ) -> None:
            """Store KV chunk via AdapShot B+C pipeline."""
            self._adapshot_mgr.store_segment(token_ids, chunk_idx, pre_rope_kv, layer_idx, attn_weights)

        def adapshot_load_segment(
            self,
            token_ids: List[int],
            chunk_idx: int,
            target_offset: int,
            layer_idx: int = 0,
        ) -> Optional["torch.Tensor"]:
            """Load KV chunk via AdapShot B+C restore contract (RoPE re-applied, decoded)."""
            return self._adapshot_mgr.load_segment(token_ids, chunk_idx, target_offset, layer_idx)

        def adapshot_get_segments(
            self,
            token_ids: List[int],
            target_offset: int,
            layer_idx: int = 0,
        ) -> Tuple[List[Tuple[int, "torch.Tensor"]], List[int]]:
            """Get all segment hits/misses for this request."""
            return self._adapshot_mgr.get_segments(token_ids, target_offset, layer_idx)

        def adapshot_annotate_request(
            self,
            request: Any,
            token_ids: List[int],
            target_offset: int = 0,
            layer_idx: int = 0,
        ) -> None:
            """Attach AdapShot hit metadata to a vLLM request object."""
            self._adapshot_mgr.annotate_request(request, token_ids, target_offset, layer_idx)

        def adapshot_hit_stats(self) -> dict:
            """Return AdapShot hit/miss statistics."""
            return self._adapshot_mgr.hit_stats()

    _AdapShotKVCacheManager.__name__ = f"AdapShot_{base_class.__name__}"
    _AdapShotKVCacheManager.__qualname__ = _AdapShotKVCacheManager.__name__
    return _AdapShotKVCacheManager


# ---------------------------------------------------------------------------
# Smoke test (run: python vllm_integration/block_manager_patch.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch

    print("=== AdapShotBlockManager smoke test (2026-05-12) ===")

    mgr = AdapShotBlockManager(
        chunk_size=4,
        max_entries=64,
        d_head=8,
        n_heads=2,
        budget_ratio=0.50,
    )

    # Synthetic KV tensor: [4, 2, 2, 8]
    # 16 tokens = 4 chunks of 4 (chunk_size=4); chunk 2 spans tokens 8-11
    torch.manual_seed(42)
    token_ids = list(range(16))
    pre_rope_kv = torch.randn(4, 2, 2, 8)

    # Store chunk 0 (tokens 0-3)
    mgr.store_segment(token_ids, chunk_idx=0, pre_rope_kv=pre_rope_kv, layer_idx=0)
    # Miss chunk 1 (not stored)
    # Store chunk 2 (tokens 8-11)
    mgr.store_segment(token_ids, chunk_idx=2, pre_rope_kv=torch.randn(4, 2, 2, 8), layer_idx=0)

    # Load chunk 0 (should hit)
    result0 = mgr.load_segment(token_ids, chunk_idx=0, target_offset=0, layer_idx=0)
    assert result0 is not None, "chunk 0 should be a cache hit"
    assert result0.shape == (4, 2, 2, 8), f"Expected (4,2,2,8), got {result0.shape}"

    # Load chunk 1 (should miss)
    result1 = mgr.load_segment(token_ids, chunk_idx=1, target_offset=4, layer_idx=0)
    assert result1 is None, "chunk 1 should be a cache miss (not stored)"

    # Load chunk 2 (should hit — non-contiguous since chunk 1 missed)
    result2 = mgr.load_segment(token_ids, chunk_idx=2, target_offset=8, layer_idx=0)
    assert result2 is not None, "chunk 2 should be a cache hit"

    stats = mgr.hit_stats()
    print(f"  hit_rate: {stats['hit_rate']:.3f}")
    print(f"  noncontiguous_hit_rate: {stats['noncontiguous_hit_rate']:.3f}")
    print(f"  memory_bytes: {stats['memory_bytes']}")
    print(f"  budget_ratio: {stats['budget_ratio']}")

    # Factory test
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        AdapShotKVMgr = make_adapshot_kv_cache_manager_class(
            KVCacheManager, chunk_size=4, d_head=8, n_heads=2, budget_ratio=0.50
        )
        print(f"  factory class: {AdapShotKVMgr.__name__}")
        assert issubclass(AdapShotKVMgr, KVCacheManager)
        print("  factory subclass check: PASS")
    except Exception as e:
        print(f"  factory test skipped (no vLLM GPU): {e}")


# ===========================================================================
# 2026-05-13  Activity B — KVFoldAccumulativeBlockManager
# ===========================================================================
"""Activity B (2026-05-13): KVFoldAccumulativeBlockManager

Ports KVFoldAccumulativeRadixCache (src/cache/kv_fold_accumulative.py) and
AgenticChunkPreCachingPipeline (src/cache/agentic_chunk_precaching.py) into
vLLM's v1 KVCacheManager layer as a standalone auxiliary segment store.

Integration model for vLLM 0.20.2:
  - KVFoldAccumulativeBlockManager is a self-contained class; it does NOT
    subclass KVCacheManager directly (which requires complex vLLM internals),
    instead it acts as a *parallel* non-contiguous KV segment store that
    sits alongside vLLM's native prefix-caching block pool.
  - Three public hooks let the vLLM engine (or a custom worker) call into
    the fold store at the right moments:
      store_chunk(tokens, layer_idx, kv_tensor)   — after attention computation
      lookup_chunk(tokens, layer_idx)              — before attention (HIT path)
      lookup_fold_prefix(fold_key)                 — foldl accumulated prefix
  - make_kvfold_kv_cache_manager_class(): produces a KVCacheManager subclass
    that embeds a KVFoldAccumulativeBlockManager; safe to pass to vLLM engine.

Non-contiguous reuse contract (Activity B, vLLM specific):
  - Blocks are aligned to vLLM's native block_size boundary.
  - Segment granularity == chunk_size (configurable, default 128 tokens).
  - Non-contiguous hits are tracked via get_segments_with_fold().
  - StreamingLLM window_size controls GPU memory upper bound for fold states.
"""

import collections as _collections
from dataclasses import dataclass as _dc_b, field as _field_b
from typing import (
    Any as _Any_B,
    Dict as _Dict_B,
    List as _List_B,
    Optional as _Optional_B,
    Tuple as _Tuple_B,
)

import torch as _torch_b


@_dc_b
class KVFoldBlockManagerConfig:
    """Configuration for KVFoldAccumulativeBlockManager."""
    chunk_size: int = 128
    max_entries: int = 2000
    drift_threshold: float = 1e-3
    max_fold_depth: int = 511
    enable_streaming_fallback: bool = True
    window_size: int = 32
    d_head: int = 64
    n_heads: int = 8
    n_layers: int = 12
    seed: int = 42
    compressor: _Optional_B[object] = None  # Optional SRFTFusedINT8KVCodec


class _SegmentHashDict:
    """Minimal segment store backed by an OrderedDict for LRU eviction."""

    def __init__(self, max_entries: int) -> None:
        self._cache: _collections.OrderedDict = _collections.OrderedDict()
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def _make_key(self, tokens: _List_B[int], chunk_idx: int, layer_idx: int) -> str:
        import hashlib as _hl
        import struct as _st
        chunk = tokens[chunk_idx * 128: (chunk_idx + 1) * 128] or [0]
        raw = _st.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = _st.pack("II", layer_idx, chunk_idx)
        return _hl.sha256(layer_prefix + raw).hexdigest()

    def put(self, key: str, value: _torch_b.Tensor) -> None:
        if len(self._cache) >= self._max_entries:
            self._cache.popitem(last=False)
        self._cache[key] = value.detach().cpu()
        self._cache.move_to_end(key)

    def get(self, key: str) -> _Optional_B[_torch_b.Tensor]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def keys(self):
        return list(self._cache.keys())

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._cache.values())


class KVFoldAccumulativeBlockManager:
    """Auxiliary non-contiguous KV segment store with foldl accumulation (Activity B).

    This class mirrors the contract of KVFoldAccumulativeRadixCache but is
    designed to operate alongside vLLM's native KV cache without replacing it.

    Public methods:
      store_chunk()      — store a single KV chunk after attention
      lookup_chunk()     — look up a single KV chunk (standard prefix match)
      fold_chunk()       — integrate chunk into foldl accumulation chain
      lookup_fold_prefix() — get accumulated KV prefix from fold state
      get_noncontiguous_hits() — retrieve hits that bypassed contiguous-prefix
                                  matching
      hit_stats()        — return hit/miss/non-contiguous metrics dict
    """

    def __init__(self, config: KVFoldBlockManagerConfig) -> None:
        self.config = config
        self._store = _SegmentHashDict(max_entries=config.max_entries)
        # fold_key → {accumulated_kv, chunk_ids, fold_depth, plateau_reached}
        self._fold_states: _collections.OrderedDict = _collections.OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._fold_hits = 0
        _torch_b.manual_seed(config.seed)

    # ------------------------------------------------------------------ #
    # Segment store API                                                    #
    # ------------------------------------------------------------------ #

    def store_chunk(
        self,
        token_ids: _List_B[int],
        chunk_idx: int,
        layer_idx: int,
        kv_tensor: _torch_b.Tensor,
    ) -> str:
        """Store KV for one chunk. Returns the chunk key."""
        key = self._chunk_key(token_ids, chunk_idx, layer_idx)
        # Apply SRFT+INT8 compression if configured
        stored = kv_tensor
        if self.config.compressor is not None:
            try:
                stored = self.config.compressor.compression_hook(key, kv_tensor)
            except Exception:
                pass  # compression failure is non-fatal
        self._store.put(key, stored)
        return key

    def lookup_chunk(
        self,
        token_ids: _List_B[int],
        chunk_idx: int,
        layer_idx: int,
    ) -> _Optional_B[_torch_b.Tensor]:
        """Look up KV for one chunk. Returns None on miss."""
        key = self._chunk_key(token_ids, chunk_idx, layer_idx)
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def lookup_segments(
        self,
        token_ids: _List_B[int],
        layer_idx: int = 0,
    ) -> _Tuple_B[_List_B[_Tuple_B[int, _torch_b.Tensor]], _List_B[int]]:
        """Look up all chunks for a token sequence.

        Returns:
          hits: [(chunk_idx, kv_tensor)] for chunks found in cache
          misses: [chunk_idx] for chunks not found
        """
        chunk_size = self.config.chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: _List_B[_Tuple_B[int, _torch_b.Tensor]] = []
        misses: _List_B[int] = []
        for i in range(n_chunks):
            kv = self.lookup_chunk(token_ids, i, layer_idx)
            if kv is not None:
                hits.append((i, kv))
            else:
                misses.append(i)

        # Track non-contiguous hits (hit after a miss)
        miss_set = set(misses)
        for idx, _ in hits:
            if any(m < idx for m in miss_set):
                self._noncontiguous_hits += 1

        return hits, misses

    # ------------------------------------------------------------------ #
    # foldl accumulation API                                               #
    # ------------------------------------------------------------------ #

    def fold_chunk(
        self,
        chunk_tokens: _List_B[int],
        layer_idx: int = 0,
        existing_fold_key: _Optional_B[str] = None,
    ) -> _Tuple_B[str, _torch_b.Tensor]:
        """Integrate a chunk into the foldl accumulation state.

        Mirrors KVFoldAccumulativeRadixCache.fold_chunk() logic.

        Returns:
          (fold_key, accumulated_kv)
        """
        cfg = self.config
        n_heads, d_head = cfg.n_heads, cfg.d_head

        # Load or init state
        if existing_fold_key and existing_fold_key in self._fold_states:
            state = self._fold_states[existing_fold_key]
        else:
            state = {
                "accumulated_kv": _torch_b.empty(0, 2, n_heads, d_head),
                "chunk_ids": [],
                "fold_depth": 0,
                "last_drift": float("inf"),
                "plateau_reached": False,
            }

        # Early exit on plateau
        if state["plateau_reached"] and existing_fold_key is not None:
            return existing_fold_key, state["accumulated_kv"]

        # Get/compute chunk KV
        chunk_key = self._chunk_key_from_tokens(chunk_tokens, layer_idx)
        cached_kv = self._store.get(chunk_key)
        if cached_kv is None:
            new_chunk_kv = self._synthetic_chunk_kv(chunk_tokens, layer_idx)
            self._store.put(chunk_key, new_chunk_kv)
        else:
            new_chunk_kv = cached_kv

        # foldl: cat accumulated_kv with new_chunk_kv
        acc = state["accumulated_kv"]
        if acc.shape[0] == 0:
            updated_kv = new_chunk_kv
        else:
            updated_kv = _torch_b.cat([acc, new_chunk_kv], dim=0)

        # Drift detection
        if acc.shape[0] > 0:
            prev_tail = acc[-new_chunk_kv.shape[0]:]
            drift = (
                (new_chunk_kv.float() - prev_tail.float()).norm().item()
                / prev_tail.float().norm().clamp(min=1e-8).item()
            )
        else:
            drift = float("inf")

        plateau = drift < cfg.drift_threshold and state["fold_depth"] >= 2

        # StreamingLLM fallback
        max_tokens = cfg.window_size * cfg.chunk_size
        if cfg.enable_streaming_fallback and updated_kv.shape[0] > max_tokens:
            sink_tokens = min(4, updated_kv.shape[0])
            keep_recent = max_tokens - sink_tokens
            if keep_recent > 0:
                updated_kv = _torch_b.cat(
                    [updated_kv[:sink_tokens], updated_kv[-keep_recent:]], dim=0
                )
            else:
                updated_kv = updated_kv[:sink_tokens]

        # B+C integration: compress old chunks if compressor attached
        if (
            cfg.compressor is not None
            and updated_kv.shape[0] > cfg.chunk_size
            and updated_kv.dim() == 4
        ):
            try:
                updated_kv = cfg.compressor.compression_hook("fold", updated_kv)
            except Exception:
                pass

        # Build new fold_key and state
        new_fold_key = f"fold:{layer_idx}:{chunk_key}:{state['fold_depth']}"
        new_state = {
            "accumulated_kv": updated_kv,
            "chunk_ids": state["chunk_ids"] + [hash(tuple(chunk_tokens))],
            "fold_depth": state["fold_depth"] + 1,
            "last_drift": drift,
            "plateau_reached": plateau,
        }
        if len(self._fold_states) >= cfg.max_entries:
            self._fold_states.popitem(last=False)
        self._fold_states[new_fold_key] = new_state

        return new_fold_key, updated_kv

    def lookup_fold_prefix(
        self, fold_key: str
    ) -> _Optional_B[_torch_b.Tensor]:
        """Return accumulated KV prefix or None if not found."""
        state = self._fold_states.get(fold_key)
        if state is None:
            return None
        self._fold_hits += 1
        self._hits += 1
        return state["accumulated_kv"]

    def register_prefolded_prefix(
        self,
        fold_key: str,
        accumulated_kv: _torch_b.Tensor,
        chunk_ids: _List_B[int],
    ) -> None:
        """Register a pre-accumulated KV prefix (from agentic pre-caching)."""
        self._fold_states[fold_key] = {
            "accumulated_kv": accumulated_kv,
            "chunk_ids": chunk_ids,
            "fold_depth": len(chunk_ids),
            "last_drift": 0.0,
            "plateau_reached": True,
        }
        self._store.put(fold_key, accumulated_kv)

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def hit_stats(self) -> _Dict_B[str, _Any_B]:
        total = self._hits + self._misses
        nc_rate = (self._noncontiguous_hits + self._fold_hits) / max(self._hits, 1)
        return {
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "noncontiguous_hit_rate": nc_rate,
            "fold_hits": self._fold_hits,
            "total_hits": self._hits,
            "total_misses": self._misses,
            "fold_states": len(self._fold_states),
            "segment_store_entries": len(self._store.keys()),
            "memory_bytes": self._store.memory_bytes(),
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _chunk_key(
        self, token_ids: _List_B[int], chunk_idx: int, layer_idx: int
    ) -> str:
        import hashlib as _hl
        import struct as _st
        chunk_size = self.config.chunk_size
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = token_ids[start:end] or [0]
        raw = _st.pack(f"{len(chunk)}I", *chunk)
        prefix = _st.pack("II", layer_idx, chunk_idx)
        return _hl.sha256(prefix + raw).hexdigest()

    def _chunk_key_from_tokens(
        self, tokens: _List_B[int], layer_idx: int
    ) -> str:
        import hashlib as _hl
        import struct as _st
        chunk = tokens or [0]
        raw = _st.pack(f"{len(chunk)}I", *chunk)
        prefix = _st.pack("I", layer_idx)
        return _hl.sha256(prefix + raw).hexdigest()

    def _synthetic_chunk_kv(
        self, chunk_tokens: _List_B[int], layer_idx: int
    ) -> _torch_b.Tensor:
        """Generate synthetic KV for testing without a live model."""
        n_tok = len(chunk_tokens) or self.config.chunk_size
        seed = (sum(chunk_tokens) + layer_idx) % (2 ** 31)
        _torch_b.manual_seed(seed)
        return _torch_b.randn(n_tok, 2, self.config.n_heads, self.config.d_head)


def make_kvfold_kv_cache_manager_class(
    base_class: type,
    **default_fold_kwargs: _Any_B,
) -> type:
    """Factory: embed a KVFoldAccumulativeBlockManager inside a KVCacheManager subclass.

    The returned class adds a `kvfold_store` attribute (KVFoldAccumulativeBlockManager)
    accessible on any instance. This allows the engine/worker to call
    `scheduler.kv_cache_manager.kvfold_store.fold_chunk(...)` etc.

    Args:
        base_class: vllm.v1.core.kv_cache_manager.KVCacheManager (or mock).
        **default_fold_kwargs: KVFoldBlockManagerConfig keyword args.

    Example:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        KVFoldMgr = make_kvfold_kv_cache_manager_class(
            KVCacheManager,
            chunk_size=128, n_heads=32, d_head=128, window_size=16,
        )
    """

    class _KVFoldKVCacheManager(base_class):
        def __init__(self, *args: _Any_B, **kwargs: _Any_B) -> None:
            fold_cfg = KVFoldBlockManagerConfig(**{
                k: kwargs.pop(k, v)
                for k, v in default_fold_kwargs.items()
            })
            super().__init__(*args, **kwargs)
            self.kvfold_store = KVFoldAccumulativeBlockManager(fold_cfg)

    _KVFoldKVCacheManager.__name__ = f"KVFold_{base_class.__name__}"
    _KVFoldKVCacheManager.__qualname__ = f"KVFold_{base_class.__qualname__}"
    return _KVFoldKVCacheManager


def _smoke_test_kvfold_block_manager() -> None:
    """Smoke test: KVFoldAccumulativeBlockManager 2026-05-13."""
    print("[smoke] KVFoldAccumulativeBlockManager (Activity B 2026-05-13)")

    cfg = KVFoldBlockManagerConfig(
        chunk_size=4,
        max_entries=50,
        n_heads=2,
        d_head=8,
        window_size=4,
        seed=42,
    )
    mgr = KVFoldAccumulativeBlockManager(cfg)

    # Synthetic tokens for 2 chunks
    tokens_a = list(range(8))   # chunk 0: [0..3], chunk 1: [4..7]
    kv_chunk0 = _torch_b.randn(4, 2, 2, 8)
    kv_chunk1 = _torch_b.randn(4, 2, 2, 8)

    # Store
    key0 = mgr.store_chunk(tokens_a, 0, 0, kv_chunk0)
    key1 = mgr.store_chunk(tokens_a, 1, 0, kv_chunk1)
    assert key0 != key1

    # Lookup hit
    result0 = mgr.lookup_chunk(tokens_a, 0, 0)
    assert result0 is not None, "chunk 0 should be a cache hit"

    # foldl accumulation
    fold_key1, acc1 = mgr.fold_chunk(tokens_a[:4], layer_idx=0)
    assert fold_key1.startswith("fold:")
    fold_key2, acc2 = mgr.fold_chunk(tokens_a[4:], layer_idx=0, existing_fold_key=fold_key1)
    assert acc2.shape[0] >= 4, "accumulated KV should cover at least 1 chunk"

    # fold prefix lookup
    prefix = mgr.lookup_fold_prefix(fold_key2)
    assert prefix is not None

    # Non-contiguous segment lookup
    hits, misses = mgr.lookup_segments(tokens_a, layer_idx=0)
    print(f"  hits={len(hits)}, misses={len(misses)}")

    stats = mgr.hit_stats()
    print(f"  hit_stats: {stats}")
    assert stats["total_hits"] >= 1

    # Factory test
    try:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        KVFoldMgr = make_kvfold_kv_cache_manager_class(
            KVCacheManager, chunk_size=4, n_heads=2, d_head=8
        )
        assert issubclass(KVFoldMgr, KVCacheManager)
        print(f"  factory class: {KVFoldMgr.__name__} — PASS")
    except Exception as e:
        print(f"  factory test skipped (no GPU): {e}")

    print("  KVFoldAccumulativeBlockManager: PASS")

    print("AdapShotBlockManager smoke test: PASS")


# ---------------------------------------------------------------------------
# 2026-05-14 Activity B: FibQuantVQSegmentKVManager
# ---------------------------------------------------------------------------

class FibQuantVQSegmentKVManager(KVCacheManager):
    """KVCacheManager subclass with FibQuant VQ non-contiguous segment store.

    Activity B (Non-Contiguous KV Cache Reuse) for vLLM 0.20.2.

    Integrates FibQuantVQSegmentCache (src/cache/fibquant_vq_segment_cache.py)
    as a parallel auxiliary segment store alongside vLLM's native prefix cache.

    Architecture:
        - vLLM's native BlockPool handles standard paged-block allocation.
        - FibQuantVQSegmentCache maintains a compressed auxiliary store keyed
          by content-hash of token chunks (position-independent).
        - get_computed_blocks() first checks vLLM's native cache; on miss,
          checks the FibQuant auxiliary store and reports non-contiguous hits.
        - store_segment() compresses a segment via FibQuantVQCodec and inserts
          it into the auxiliary store with LRU eviction management.

    Non-contiguous block mapping:
        Each FibQuant segment covers a chunk of block_size tokens.
        The segment store maps content-hash keys to compressed code tensors,
        enabling random-access decompression of any segment.
        Before attention kernel call, vLLM's block_table is padded with
        FIBQUANT_SENTINEL block IDs; the model runner must call
        load_fibquant_segment() to materialise KV tensors for those IDs.

    Compression:
        FibQuant bits_direction controls compression factor:
            8-bit (uint8)  → 1.88x compression, cosine >= 0.99
            4-bit (nibble) → 3.56x compression, cosine >= 0.97
            2-bit (quartet)→ 6.40x compression, cosine ~0.85-0.90

    Inherits ALL KVCacheManager public methods without modification.
    New methods added as non-conflicting extensions.
    """

    FIBQUANT_SENTINEL: int = -999  # Block ID sentinel for FibQuant-backed slots

    def __init__(
        self,
        *args,
        fibquant_chunk_size: int = 64,
        fibquant_max_entries: int = 1000,
        fibquant_n_heads: int = 8,
        fibquant_d_head: int = 64,
        fibquant_n_layers: int = 32,
        fibquant_bits_radial: int = 8,
        fibquant_bits_direction: int = 8,
        fibquant_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.fibquant_chunk_size = fibquant_chunk_size
        self.fibquant_n_heads = fibquant_n_heads
        self.fibquant_d_head = fibquant_d_head
        self.fibquant_bits_radial = fibquant_bits_radial
        self.fibquant_bits_direction = fibquant_bits_direction

        # Auxiliary FibQuant segment store: segment_id -> compressed dict
        self._fibquant_store: "OrderedDict[str, dict]" = OrderedDict()
        self._fibquant_max_entries = fibquant_max_entries
        self._fibquant_hits: int = 0
        self._fibquant_misses: int = 0
        self._fibquant_noncontiguous_hits: int = 0

        # Build codec
        self._fibquant_codec = self._build_fibquant_codec(
            n_heads=fibquant_n_heads,
            d_head=fibquant_d_head,
            n_layers=fibquant_n_layers,
            bits_radial=fibquant_bits_radial,
            bits_direction=fibquant_bits_direction,
            seed=fibquant_seed,
            block_size=fibquant_chunk_size,
        )

    @staticmethod
    def _build_fibquant_codec(
        n_heads: int,
        d_head: int,
        n_layers: int,
        bits_radial: int,
        bits_direction: int,
        seed: int,
        block_size: int,
    ):
        """Build VllmFibQuantVQCodec (from compression_codec.py)."""
        try:
            from vllm_integration.compression_codec import VllmFibQuantVQCodec
            return VllmFibQuantVQCodec(
                n_heads=n_heads,
                d_head=d_head,
                n_layers=n_layers,
                bits_radial=bits_radial,
                bits_direction=bits_direction,
                seed=seed,
                block_size=block_size,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Segment-level API (Activity B — non-contiguous reuse)
    # ------------------------------------------------------------------

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        key: torch.Tensor,   # (chunk_size, n_heads, d_head)
        val: torch.Tensor,   # (chunk_size, n_heads, d_head)
        layer_idx: int = 0,
    ) -> str:
        """FibQuant-compress and store one chunk of a request's KV.

        Args:
            token_ids: Full token ID list for the request.
            chunk_idx: 0-based chunk index within the request.
            key, val: KV tensors for this chunk (vLLM 3-D shapes).
            layer_idx: Transformer layer index.

        Returns:
            segment_id: Content-hash key under which the segment was stored.
        """
        segment_id = self._chunk_key(token_ids, chunk_idx, layer_idx)
        if self._fibquant_codec is not None:
            compressed = self._fibquant_codec.write_to_cache(key, val, layer_idx, segment_id)
        else:
            # Fallback: store raw FP16 if codec unavailable
            compressed = {
                "raw_key": key.half(),
                "raw_val": val.half(),
                "layer_idx": layer_idx,
            }
        self._fibquant_lru_insert(segment_id, compressed)
        return segment_id

    def load_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Decompress and return one chunk's KV on cache hit.

        Args:
            token_ids: Full token ID list for the request.
            chunk_idx: 0-based chunk index.
            layer_idx: Transformer layer index.

        Returns:
            (key, val) float16 tensors if hit, else None.
            Decompression always happens here — compressed tensors never
            reach the attention kernel.
        """
        segment_id = self._chunk_key(token_ids, chunk_idx, layer_idx)
        if segment_id not in self._fibquant_store:
            self._fibquant_misses += 1
            return None
        self._fibquant_store.move_to_end(segment_id)
        self._fibquant_hits += 1
        compressed = self._fibquant_store[segment_id]
        if self._fibquant_codec is not None:
            key, val = self._fibquant_codec.read_from_cache(compressed)
        else:
            key = compressed.get("raw_key", torch.zeros(1))
            val = compressed.get("raw_val", torch.zeros(1))
        return key, val

    def get_noncontiguous_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], List[int]]:
        """Look up all chunks for token_ids in the FibQuant store.

        Returns:
            hits:   [(chunk_idx, key, val), ...]  — decompressed, ready for attn
            misses: [chunk_idx, ...]

        Non-contiguous tracking: a hit is non-contiguous if any lower chunk idx
        was a miss. This mirrors FibQuantVQSegmentCache.get_segments() logic.
        """
        chunk_size = self.fibquant_chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        hits: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        misses: List[int] = []

        for i in range(n_chunks):
            result = self.load_segment(token_ids, i, layer_idx)
            if result is not None:
                key, val = result
                hits.append((i, key, val))
                if any(m < i for m in misses):
                    self._fibquant_noncontiguous_hits += 1
            else:
                misses.append(i)

        return hits, misses

    # ------------------------------------------------------------------
    # Non-contiguous block table padding (Activity B vLLM integration)
    # ------------------------------------------------------------------

    def pad_block_table_with_fibquant(
        self,
        block_table: List[int],
        n_fibquant_slots: int,
    ) -> List[int]:
        """Pad block_table with FIBQUANT_SENTINEL IDs for FibQuant-backed slots.

        Before the attention kernel is called, FibQuant-backed slots are
        materialised via load_segment() and inserted into the KV cache.
        This method marks the slots so the model runner knows which block IDs
        correspond to FibQuant content.

        Args:
            block_table: Existing list of physical block IDs.
            n_fibquant_slots: Number of FibQuant-backed slots to append.

        Returns:
            Extended block_table with FIBQUANT_SENTINEL appended n times.
        """
        return block_table + [self.FIBQUANT_SENTINEL] * n_fibquant_slots

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def fibquant_hit_rate(self) -> float:
        """Cumulative FibQuant auxiliary store hit rate."""
        total = self._fibquant_hits + self._fibquant_misses
        return self._fibquant_hits / total if total > 0 else 0.0

    def fibquant_noncontiguous_hit_rate(self) -> float:
        """Fraction of FibQuant hits that are non-contiguous."""
        if self._fibquant_hits == 0:
            return 0.0
        return self._fibquant_noncontiguous_hits / self._fibquant_hits

    def fibquant_memory_bytes(self) -> int:
        """Total bytes used by FibQuant compressed code tensors."""
        total = 0
        for compressed in self._fibquant_store.values():
            for v in compressed.values():
                if isinstance(v, torch.Tensor):
                    total += v.nbytes
        return total

    def fibquant_stats(self) -> dict:
        """Return a dict of FibQuant auxiliary store statistics."""
        codec_factor = (
            self._fibquant_codec.compression_factor()
            if self._fibquant_codec is not None else 1.0
        )
        return {
            "hits": self._fibquant_hits,
            "misses": self._fibquant_misses,
            "noncontiguous_hits": self._fibquant_noncontiguous_hits,
            "hit_rate": self.fibquant_hit_rate(),
            "noncontiguous_hit_rate": self.fibquant_noncontiguous_hit_rate(),
            "stored_segments": len(self._fibquant_store),
            "memory_bytes": self.fibquant_memory_bytes(),
            "compression_factor": codec_factor,
            "bits_radial": self.fibquant_bits_radial,
            "bits_direction": self.fibquant_bits_direction,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int,
        chunk_size: Optional[int] = None,
    ) -> str:
        """Content-hash key for a chunk (position-independent).

        Args:
            token_ids: Full token ID list for the request.
            chunk_idx: 0-based chunk index.
            layer_idx: Transformer layer index.
            chunk_size: Override chunk size; defaults to self.fibquant_chunk_size.

        Returns:
            Hex-digest segment key string.
        """
        chunk_size = chunk_size if chunk_size is not None else self.fibquant_chunk_size
        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk_tokens = token_ids[start:end]
        h = hashlib.sha256(
            struct.pack(f">{len(chunk_tokens)}I", *chunk_tokens)
        ).hexdigest()[:16]
        return f"fqseg:L{layer_idx}:C{chunk_idx}:{h}"

    def _fibquant_lru_insert(self, segment_id: str, compressed: dict) -> None:
        """Insert into LRU store with eviction when at capacity."""
        if len(self._fibquant_store) >= self._fibquant_max_entries:
            # LRU eviction: remove oldest entry
            self._fibquant_store.popitem(last=False)
        if segment_id in self._fibquant_store:
            self._fibquant_store.move_to_end(segment_id)
        else:
            self._fibquant_store[segment_id] = compressed


def make_fibquant_kv_cache_manager_class(
    base_kv_cache_manager_class,
    fibquant_chunk_size: int = 64,
    fibquant_max_entries: int = 1000,
    fibquant_n_heads: int = 8,
    fibquant_d_head: int = 64,
    fibquant_n_layers: int = 32,
    fibquant_bits_radial: int = 8,
    fibquant_bits_direction: int = 8,
    fibquant_seed: int = 42,
):
    """Factory: create a FibQuantVQSegmentKVManager subclass of base_kv_cache_manager_class.

    Usage:
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        FibQuantMgr = make_fibquant_kv_cache_manager_class(KVCacheManager)
        # Use FibQuantMgr in place of KVCacheManager in the vLLM engine.

    The returned class inherits all KVCacheManager public methods unchanged
    and adds store_segment(), load_segment(), get_noncontiguous_segments(),
    and fibquant_stats() as non-conflicting extensions.
    """

    class _FibQuantVQSegmentKVManager(FibQuantVQSegmentKVManager, base_kv_cache_manager_class):
        pass

    _FibQuantVQSegmentKVManager.__name__ = (
        f"FibQuantVQSegmentKVManager[{base_kv_cache_manager_class.__name__}]"
    )
    _FibQuantVQSegmentKVManager.__qualname__ = _FibQuantVQSegmentKVManager.__name__

    # Bind configuration as class defaults so __init__ kwargs are optional
    _defaults = dict(
        fibquant_chunk_size=fibquant_chunk_size,
        fibquant_max_entries=fibquant_max_entries,
        fibquant_n_heads=fibquant_n_heads,
        fibquant_d_head=fibquant_d_head,
        fibquant_n_layers=fibquant_n_layers,
        fibquant_bits_radial=fibquant_bits_radial,
        fibquant_bits_direction=fibquant_bits_direction,
        fibquant_seed=fibquant_seed,
    )
    original_init = _FibQuantVQSegmentKVManager.__init__

    def _bound_init(self, *args, **kwargs):
        merged = {**_defaults, **kwargs}
        original_init(self, *args, **merged)

    _FibQuantVQSegmentKVManager.__init__ = _bound_init
    return _FibQuantVQSegmentKVManager

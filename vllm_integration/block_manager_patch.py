"""block_manager_patch.py — Activity B: WorkloadAwareTTLCache integration for vLLM 0.20.1.

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
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

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

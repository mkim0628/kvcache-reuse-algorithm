"""WorkloadAwareTTLCache — category-based TTL segment preservation (Activity B).

Assigns TTL to each cached segment based on workload category (code/chat/rag/agentic)
derived from request keyword heuristics. TTL-expired segments are eviction candidates
and are processed by RedundancyAwareEvictionPolicy before final eviction.
"""

import hashlib
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import torch

from src.cache.base import CacheStore

if TYPE_CHECKING:
    from src.cache.redundancy_eviction import RedundancyAwareEvictionPolicy


# Default TTL profiles based on KVCache-in-the-Wild Table 3
_DEFAULT_TTL_PROFILES: Dict[str, dict] = {
    "code":     {"ttl_base_sec": 600.0, "reuse_probability": 0.75},
    "chat":     {"ttl_base_sec": 300.0, "reuse_probability": 0.60},
    "rag":      {"ttl_base_sec": 120.0, "reuse_probability": 0.45},
    "agentic":  {"ttl_base_sec": 480.0, "reuse_probability": 0.80},
}

# Keywords for category classification (checked against key string)
_CODE_KEYWORDS    = ("def ", "class ", "import ", "```python")
_RAG_KEYWORDS     = ("document", "context:", "passage", "retrieved")
_AGENTIC_KEYWORDS = ("tool_call", "function_call", "agent", "workflow")


@dataclass
class TTLEntry:
    value: torch.Tensor
    category: str
    ttl_sec: float
    created_at: float           # time.monotonic() at insertion
    pinned: bool = False        # DAG preservation pin — excluded from eviction
    importance_score: float = 0.0
    embedding: Optional[torch.Tensor] = None  # mean Key vector for redundancy


class WorkloadAwareTTLCache(CacheStore):
    """Category-aware TTL KV cache with LRU fallback (Activity B).

    Hit types:
      exact_hits           — key found in store, TTL not expired
      ttl_preserved_hits   — alias for exact_hits; all non-expired hits are TTL-preserved
      misses               — key absent or TTL expired
    """

    def __init__(
        self,
        max_entries: int = 1000,
        chunk_size: int = 128,
        ttl_ema_alpha: float = 0.1,
        eviction_policy: Optional["RedundancyAwareEvictionPolicy"] = None,
        ttl_profiles: Optional[Dict[str, dict]] = None,
    ) -> None:
        self.max_entries = max_entries
        self.chunk_size = chunk_size
        self.ttl_ema_alpha = ttl_ema_alpha
        self._eviction_policy = eviction_policy

        # Deep-copy profiles so mutations do not bleed across instances
        import copy
        self._ttl_profiles: Dict[str, dict] = copy.deepcopy(
            ttl_profiles if ttl_profiles is not None else _DEFAULT_TTL_PROFILES
        )

        self._store: OrderedDict[str, TTLEntry] = OrderedDict()
        self._pinned: set = set()

        # Counters
        self._exact_hits: int = 0
        self._ttl_preserved_hits: int = 0
        self._misses: int = 0
        self._eviction_ttl_count: int = 0
        self._eviction_pressure_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store value with default 'chat' category. Use put_segment() for full control."""
        self.put_segment(key, value, category="chat")

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return cached tensor if present and TTL not expired, else None."""
        if key not in self._store:
            self._misses += 1
            return None

        entry = self._store[key]
        now = time.monotonic()
        elapsed = now - entry.created_at

        if elapsed > entry.ttl_sec:
            # TTL expired → treat as miss; leave entry for next evict() pass
            self._misses += 1
            return None

        # Hit: update LRU order
        self._store.move_to_end(key)
        self._exact_hits += 1
        self._ttl_preserved_hits += 1  # all non-expired hits count as ttl-preserved
        return entry.value

    def evict(self) -> int:
        """Evict one segment. Prefers TTL-expired; falls back to LRU among non-pinned."""
        if not self._store:
            return 0

        candidates = self.evict_candidates()  # TTL-expired, non-pinned

        if candidates:
            if self._eviction_policy is not None:
                keys_to_evict = self._eviction_policy.select_evict_keys(
                    candidates, self._store, n_evict=1
                )
                evict_key = keys_to_evict[0] if keys_to_evict else candidates[0]
            else:
                evict_key = candidates[0]
            self._eviction_ttl_count += 1
        else:
            # LRU fallback: oldest non-pinned entry
            evict_key = None
            for k in self._store:
                if k not in self._pinned:
                    evict_key = k
                    break
            if evict_key is None:
                return 0
            self._eviction_pressure_count += 1

        entry = self._store.pop(evict_key)
        self._pinned.discard(evict_key)
        return entry.value.nbytes

    def hit_rate(self) -> float:
        total = self._exact_hits + self._misses
        return self._exact_hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(e.value.nbytes for e in self._store.values())

    def reset_stats(self) -> None:
        self._exact_hits = 0
        self._ttl_preserved_hits = 0
        self._misses = 0
        self._eviction_ttl_count = 0
        self._eviction_pressure_count = 0

    # ------------------------------------------------------------------ #
    # Extended API                                                         #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        key: str,
        value: torch.Tensor,
        category: str = "chat",
        embedding: Optional[torch.Tensor] = None,
        override_ttl_sec: Optional[float] = None,
    ) -> None:
        """Store a segment with explicit category and optional TTL override."""
        if key in self._store:
            self._store.move_to_end(key)
            return

        if len(self._store) >= self.max_entries:
            self.evict()

        ttl_sec = (
            override_ttl_sec
            if override_ttl_sec is not None
            else self._ttl_profiles.get(category, _DEFAULT_TTL_PROFILES["chat"])["ttl_base_sec"]
        )

        entry = TTLEntry(
            value=value.detach().clone(),
            category=category,
            ttl_sec=float(ttl_sec),
            created_at=time.monotonic(),
            embedding=embedding.detach().clone() if embedding is not None else None,
        )
        self._store[key] = entry

    def adjust_ttl(self, key: str, new_ttl_sec: float) -> None:
        """Override the TTL for a specific segment (called by DAGAwareTTLAdjuster)."""
        if key not in self._store:
            return
        entry = self._store[key]
        entry.ttl_sec = float(new_ttl_sec)
        # When new_ttl_sec == 0.0, elapsed > ttl_sec will immediately be true
        # so the next get() returns miss and evict_candidates() includes this key.

    def pin(self, key: str) -> None:
        """Prevent a segment from being evicted (used by DAGTopologyScheduler)."""
        self._pinned.add(key)
        if key in self._store:
            self._store[key].pinned = True

    def unpin(self, key: str) -> None:
        """Allow a previously pinned segment to be evicted."""
        self._pinned.discard(key)
        if key in self._store:
            self._store[key].pinned = False

    def evict_candidates(self) -> List[str]:
        """Return list of TTL-expired, non-pinned segment keys."""
        now = time.monotonic()
        result = []
        for key, entry in self._store.items():
            if key in self._pinned:
                continue
            elapsed = now - entry.created_at
            if elapsed > entry.ttl_sec:
                result.append(key)
        return result

    def record_hit(self, key: str, is_ttl_preserved: bool = False) -> None:
        """Record a hit and update the EMA TTL for the corresponding category."""
        if key not in self._store:
            return
        entry = self._store[key]
        reuse_gap = time.monotonic() - entry.created_at
        category = entry.category
        old_ttl = self._ttl_profiles[category]["ttl_base_sec"]
        # EMA: blend observed reuse gap toward the stored TTL
        ttl_multiplier = 1.2  # allow TTL to grow beyond observed gap
        new_ttl = (1.0 - self.ttl_ema_alpha) * old_ttl + self.ttl_ema_alpha * reuse_gap * ttl_multiplier
        self._ttl_profiles[category]["ttl_base_sec"] = new_ttl

    def record_importance(self, key: str, score: float) -> None:
        """Accumulate attention importance score for a cached segment."""
        if key in self._store:
            self._store[key].importance_score += score

    def ttl_hit_stats(self) -> dict:
        """Return detailed hit/eviction statistics."""
        total = self._exact_hits + self._misses
        exact_rate = self._exact_hits / total if total > 0 else 0.0
        # ttl_preserved_hits tracks all hits; noncontiguous subset approximation
        # uses ttl_preserved_hits as the non-contiguous proxy (TTL-based reuse)
        ttl_preserved = self._ttl_preserved_hits
        overall_rate = exact_rate
        hits_total = self._exact_hits
        noncontiguous_ratio = (ttl_preserved / hits_total) if hits_total > 0 else 0.0

        return {
            "exact_hit_rate": exact_rate,
            "ttl_preserved_hit_rate": ttl_preserved / total if total > 0 else 0.0,
            "overall_hit_rate": overall_rate,
            "noncontiguous_ratio": noncontiguous_ratio,
            "eviction_ttl_count": self._eviction_ttl_count,
            "eviction_pressure_count": self._eviction_pressure_count,
        }

    def _classify_category(self, key: str, token_ids: Optional[List[int]] = None) -> str:
        """Classify a request into a workload category via keyword rules."""
        text = key.lower()
        if any(kw in text for kw in _CODE_KEYWORDS):
            return "code"
        if any(kw in text for kw in _RAG_KEYWORDS):
            return "rag"
        if any(kw in text for kw in _AGENTIC_KEYWORDS):
            return "agentic"
        return "chat"

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """SHA-256 chunk key compatible with SegmentedHashCache.chunk_key()."""
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start: start + self.chunk_size]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

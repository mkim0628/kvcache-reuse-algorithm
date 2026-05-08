"""StaticDynamicSegmentCache — KEEP (arXiv 2602.23592) based static/dynamic segment separation.

Activity B: Agent memory update patterns drive segment classification.
Static segments (system prompts, shared docs) are pinned from LRU eviction.
Dynamic segments (agent actions, conversation history) allow limited Multi-hop invalidation.
"""

from typing import Dict, List, Optional, Set

import torch

from src.cache.base import CacheStore


class StaticDynamicSegmentCache(CacheStore):
    """Static/dynamic segment separation with bounded Multi-hop invalidation propagation.

    - Static segments: LRU-exempt, immediately reusable across requests.
    - Dynamic segments: LRU-managed, updates invalidate at most max_invalidation_range
      subsequent segments (rather than the full suffix), minimizing recomputation cost.

    Implements full CacheStore interface; compatible with InferenceRunner.get()/put().
    """

    def __init__(
        self,
        capacity_bytes: int,
        max_invalidation_range: int = 2,
        max_recompute_hops: int = 2,
    ) -> None:
        self.capacity_bytes = capacity_bytes
        self.max_invalidation_range = max_invalidation_range
        self.max_recompute_hops = max_recompute_hops

        self._static_store: Dict[str, torch.Tensor] = {}
        self._dynamic_store: Dict[str, torch.Tensor] = {}
        self._static_keys: Set[str] = set()
        self._lru_order: List[str] = []
        self._segment_order: List[str] = []

        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Insert into static or dynamic store based on classification."""
        if key in self._static_keys:
            self._static_store[key] = value
        else:
            self._dynamic_store[key] = value
            if key not in self._lru_order:
                self._lru_order.append(key)
            if key not in self._segment_order:
                self._segment_order.append(key)
        self._maybe_evict()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._static_store:
            self._hit_count += 1
            return self._static_store[key]
        if key in self._dynamic_store:
            self._hit_count += 1
            # Promote to MRU position
            if key in self._lru_order:
                self._lru_order.remove(key)
                self._lru_order.append(key)
            return self._dynamic_store[key]
        self._miss_count += 1
        return None

    def evict(self) -> int:
        """Evict LRU dynamic segment (static segments are exempt)."""
        if not self._lru_order:
            return 0
        evict_key = self._lru_order.pop(0)
        kv = self._dynamic_store.pop(evict_key, None)
        if evict_key in self._segment_order:
            self._segment_order.remove(evict_key)
        return kv.nbytes if kv is not None else 0

    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        static_bytes = sum(v.nbytes for v in self._static_store.values())
        dynamic_bytes = sum(v.nbytes for v in self._dynamic_store.values())
        return static_bytes + dynamic_bytes

    def reset_stats(self) -> None:
        self._hit_count = 0
        self._miss_count = 0

    # ------------------------------------------------------------------ #
    # Static/dynamic classification API                                    #
    # ------------------------------------------------------------------ #

    def mark_static(self, key: str) -> None:
        """Promote segment to static (LRU-exempt). Moves from dynamic store if present."""
        self._static_keys.add(key)
        if key in self._dynamic_store:
            self._static_store[key] = self._dynamic_store.pop(key)
            if key in self._lru_order:
                self._lru_order.remove(key)

    def mark_dynamic(self, key: str) -> None:
        """Demote segment back to dynamic (LRU-managed)."""
        self._static_keys.discard(key)
        if key in self._static_store:
            self._dynamic_store[key] = self._static_store.pop(key)
            if key not in self._lru_order:
                self._lru_order.append(key)

    def update_segment(self, key: str, new_value: torch.Tensor) -> List[str]:
        """Update a dynamic segment and propagate bounded invalidation.

        Only the next max_invalidation_range segments after key are invalidated,
        preventing full-suffix recomputation that standard caches suffer from.

        Returns: list of invalidated segment keys (caller must recompute).
        Raises ValueError if key is a static segment.
        """
        if key in self._static_keys:
            raise ValueError(
                f"Static segment '{key}' cannot be updated via update_segment(). "
                "Call mark_dynamic() first."
            )

        self._dynamic_store[key] = new_value

        invalidated: List[str] = []
        if key in self._segment_order:
            idx = self._segment_order.index(key)
            invalidation_end = min(
                idx + 1 + self.max_invalidation_range, len(self._segment_order)
            )
            for inv_key in self._segment_order[idx + 1 : invalidation_end]:
                if inv_key in self._dynamic_store and inv_key not in self._static_keys:
                    del self._dynamic_store[inv_key]
                    invalidated.append(inv_key)
                    if inv_key in self._lru_order:
                        self._lru_order.remove(inv_key)

        return invalidated

    def is_static(self, key: str) -> bool:
        """Return True if key is registered as a static segment."""
        return key in self._static_keys

    def static_count(self) -> int:
        """Number of currently stored static segments."""
        return len(self._static_store)

    def dynamic_count(self) -> int:
        """Number of currently stored dynamic segments."""
        return len(self._dynamic_store)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _maybe_evict(self) -> None:
        """Evict LRU dynamic segments when capacity is exceeded."""
        while self.memory_bytes() > self.capacity_bytes and self._lru_order:
            self.evict()

from collections import OrderedDict
from typing import Optional
import torch

from src.cache.base import CacheStore


class ContiguousCache(CacheStore):
    """Baseline contiguous prefix cache with LRU eviction."""

    def __init__(self, max_entries: int = 1000) -> None:
        self.max_entries = max_entries
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self.max_entries:
                self.evict()
            self._store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        if key in self._store:
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        if not self._store:
            return 0
        _, evicted = self._store.popitem(last=False)
        return evicted.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0

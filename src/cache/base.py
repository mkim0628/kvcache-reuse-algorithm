from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch


class CacheStore(ABC):
    """Abstract base for all KV cache implementations."""

    @abstractmethod
    def put(self, key: str, value: torch.Tensor) -> None:
        """Store a KV tensor under the given key."""

    @abstractmethod
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve a KV tensor; returns None on miss."""

    @abstractmethod
    def evict(self) -> int:
        """Evict one entry; returns number of bytes freed."""

    @abstractmethod
    def hit_rate(self) -> float:
        """Cumulative cache hit rate (0.0–1.0)."""

    @abstractmethod
    def memory_bytes(self) -> int:
        """Current memory footprint in bytes."""

    @abstractmethod
    def reset_stats(self) -> None:
        """Reset hit/miss counters."""

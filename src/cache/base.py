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

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Optional compression hook called before storing in put().

        Default implementation is identity (no compression).
        Subclasses with Activity C compression may override to compress value.
        """
        return value

    def store_pre_rope(
        self,
        key: str,
        value: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Store RoPE-free (position-decoupled) KV under a content hash key.

        Default raises NotImplementedError; only RoPE-capable subclasses override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support pre-RoPE storage."
        )

    def load_with_rope(
        self,
        key: str,
        target_positions: torch.Tensor,
        layer_idx: int = 0,
        rope_dim: int = -1,
    ) -> Optional[torch.Tensor]:
        """Load pre-RoPE KV and apply rotation matrix for target_positions.

        Returns None on cache miss. Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support pre-RoPE loading."
        )

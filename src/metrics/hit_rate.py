from dataclasses import dataclass, field
from typing import List


@dataclass
class HitRateMetrics:
    total_requests: int = 0
    total_chunks: int = 0
    hit_chunks: int = 0
    noncontiguous_hit_chunks: int = 0

    def record(
        self,
        n_hits: int,
        n_misses: int,
        noncontiguous_hits: int,
    ) -> None:
        self.total_requests += 1
        self.total_chunks += n_hits + n_misses
        self.hit_chunks += n_hits
        self.noncontiguous_hit_chunks += noncontiguous_hits

    def overall_hit_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.hit_chunks / self.total_chunks

    def noncontiguous_fraction(self) -> float:
        """Fraction of hits that are non-contiguous (target ≥ 0.30)."""
        if self.hit_chunks == 0:
            return 0.0
        return self.noncontiguous_hit_chunks / self.hit_chunks

    def reset(self) -> None:
        self.total_requests = 0
        self.total_chunks = 0
        self.hit_chunks = 0
        self.noncontiguous_hit_chunks = 0

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "overall_hit_rate": self.overall_hit_rate(),
            "noncontiguous_fraction": self.noncontiguous_fraction(),
            "hit_chunks": self.hit_chunks,
            "miss_chunks": self.total_chunks - self.hit_chunks,
        }

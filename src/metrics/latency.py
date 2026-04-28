import time
from dataclasses import dataclass, field
from typing import List
import statistics


@dataclass
class LatencyMetrics:
    ttft_samples: List[float] = field(default_factory=list)
    tbt_samples: List[float] = field(default_factory=list)

    def record_ttft(self, ms: float) -> None:
        self.ttft_samples.append(ms)

    def record_tbt(self, ms: float) -> None:
        self.tbt_samples.append(ms)

    def ttft_p50(self) -> float:
        return statistics.median(self.ttft_samples) if self.ttft_samples else 0.0

    def ttft_p99(self) -> float:
        if not self.ttft_samples:
            return 0.0
        sorted_s = sorted(self.ttft_samples)
        idx = int(len(sorted_s) * 0.99)
        return sorted_s[min(idx, len(sorted_s) - 1)]

    def tbt_mean(self) -> float:
        return statistics.mean(self.tbt_samples) if self.tbt_samples else 0.0

    def summary(self) -> dict:
        return {
            "ttft_p50_ms": self.ttft_p50(),
            "ttft_p99_ms": self.ttft_p99(),
            "tbt_mean_ms": self.tbt_mean(),
            "num_samples": len(self.ttft_samples),
        }

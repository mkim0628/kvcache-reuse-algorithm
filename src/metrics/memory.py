from dataclasses import dataclass


@dataclass
class MemoryMetrics:
    baseline_bytes: int = 0
    current_bytes: int = 0

    def reduction_ratio(self) -> float:
        """Fractional memory reduction vs baseline (positive = less memory)."""
        if self.baseline_bytes == 0:
            return 0.0
        return 1.0 - self.current_bytes / self.baseline_bytes

    def reduction_percent(self) -> float:
        return self.reduction_ratio() * 100.0

    def summary(self) -> dict:
        return {
            "baseline_bytes": self.baseline_bytes,
            "current_bytes": self.current_bytes,
            "reduction_percent": self.reduction_percent(),
        }

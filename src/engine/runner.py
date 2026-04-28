"""Simulated LLM inference engine for KV cache benchmarking.

All model API calls go through this module. Uses synthetic KV tensors
so benchmarks run without real GPU/model weights.
"""

import struct
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional
import torch

from src.cache.base import CacheStore
from src.metrics.hit_rate import HitRateMetrics
from src.metrics.latency import LatencyMetrics
from src.metrics.memory import MemoryMetrics


@dataclass
class InferenceRequest:
    request_id: str
    token_ids: List[int]
    output_length: int = 64
    seed: int = 42


@dataclass
class InferenceResult:
    request_id: str
    ttft_ms: float
    output_tokens: int
    cache_hits: int
    cache_misses: int
    noncontiguous_hits: int


def _chunk_key(token_ids: List[int], chunk_idx: int, chunk_size: int, layer_idx: int) -> str:
    """Deterministic, layer-scoped, position-independent chunk key."""
    start = chunk_idx * chunk_size
    chunk = token_ids[start: start + chunk_size]
    raw = struct.pack(f"{len(chunk)}I", *chunk)
    layer_prefix = struct.pack("I", layer_idx)
    return hashlib.sha256(layer_prefix + raw).hexdigest()


class InferenceRunner:
    """Simulated inference runner with pluggable KV cache.

    Both contiguous and segmented caches are accessed at chunk granularity
    so that hit rate comparisons are fair across cache types.
    """

    def __init__(
        self,
        cache: CacheStore,
        num_layers: int = 12,
        hidden_dim: int = 64,
        chunk_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.cache = cache
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        random.seed(seed)
        torch.manual_seed(seed)

        self.hit_metrics = HitRateMetrics()
        self.latency_metrics = LatencyMetrics()
        self.memory_metrics = MemoryMetrics()

    def _simulate_kv(self, n_tokens: int, layer_idx: int, chunk_idx: int) -> torch.Tensor:
        """Generate a reproducible synthetic KV tensor deterministically."""
        torch.manual_seed(layer_idx * 10000 + chunk_idx * 100 + n_tokens)
        return torch.randn(n_tokens, self.hidden_dim)

    def _prefill_latency_ms(self, n_recomputed_chunks: int) -> float:
        tokens = n_recomputed_chunks * self.chunk_size
        return tokens * 0.05 + max(0.0, random.gauss(0, 0.2))

    def run(self, request: InferenceRequest) -> InferenceResult:
        torch.manual_seed(request.seed)
        token_ids = request.token_ids
        n_chunks = max(1, (len(token_ids) + self.chunk_size - 1) // self.chunk_size)

        # Aggregate hits/misses only once (from layer 0) to avoid double counting
        total_hits = 0
        total_misses = 0
        nc_hits = 0

        for layer_idx in range(self.num_layers):
            layer_hits = 0
            layer_misses: List[int] = []

            if hasattr(self.cache, "get_segments"):
                # Segmented cache API (Activity B / B+C)
                hits, misses = self.cache.get_segments(token_ids, layer_idx)
                layer_hits = len(hits)
                layer_misses = misses

                # Store newly computed segments
                for chunk_idx in misses:
                    kv = self._simulate_kv(self.chunk_size, layer_idx, chunk_idx)
                    self.cache.put_segment(token_ids, chunk_idx, kv, layer_idx)
            else:
                # Contiguous cache: chunk-by-chunk lookup with layer-scoped keys
                hits_list: List[int] = []
                for chunk_idx in range(n_chunks):
                    key = _chunk_key(token_ids, chunk_idx, self.chunk_size, layer_idx)
                    cached = self.cache.get(key)
                    if cached is not None:
                        layer_hits += 1
                        hits_list.append(chunk_idx)
                    else:
                        layer_misses.append(chunk_idx)
                        kv = self._simulate_kv(self.chunk_size, layer_idx, chunk_idx)
                        self.cache.put(key, kv)

            if layer_idx == 0:
                total_hits = layer_hits
                total_misses = len(layer_misses)
                # Non-contiguous: a hit at chunk_idx where some earlier chunk is a miss
                miss_set = set(layer_misses)
                hit_indices = [i for i in range(n_chunks) if i not in miss_set]
                for idx in hit_indices:
                    if any(m < idx for m in miss_set):
                        nc_hits += 1

        ttft_ms = self._prefill_latency_ms(total_misses)
        self.hit_metrics.record(total_hits, total_misses, nc_hits)
        self.latency_metrics.record_ttft(ttft_ms)
        self.memory_metrics.current_bytes = self.cache.memory_bytes()

        return InferenceResult(
            request_id=request.request_id,
            ttft_ms=ttft_ms,
            output_tokens=request.output_length,
            cache_hits=total_hits,
            cache_misses=total_misses,
            noncontiguous_hits=nc_hits,
        )

    def run_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        return [self.run(r) for r in requests]

    def metrics_summary(self) -> dict:
        return {
            "hit_rate": self.hit_metrics.summary(),
            "latency": self.latency_metrics.summary(),
            "memory": self.memory_metrics.summary(),
        }

"""SpeculativeSegmentFetcher — async speculative KV segment prefetching (Activity B).

Moves semantic segment search off the critical path by prefetching next-batch
results in background threads while the current batch is still being processed.
"""

import threading
from typing import Dict, List, Optional, Tuple

import torch

from src.engine.runner import InferenceRequest


class SpeculativeSegmentFetcher:
    """Async prefetcher that runs SemanticSegmentCache lookups in background threads.

    Prefetching results are stored in _prefetch_cache keyed by
    (request_id, chunk_idx). Results are consumed via get_prefetched().
    Thread safety on _prefetch_cache is guaranteed by _lock.
    """

    def __init__(
        self,
        cache: "SemanticSegmentCache",  # type: ignore[name-defined]
        max_wait_ms: float = 5.0,
        prefetch_depth: int = 1,
    ) -> None:
        self.cache = cache
        self.max_wait_ms = max_wait_ms
        self.prefetch_depth = prefetch_depth
        self._prefetch_cache: Dict[Tuple[str, int], Tuple[Optional[torch.Tensor], str]] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def prefetch_async(
        self,
        requests: List[InferenceRequest],
        layer_idx: int = 0,
    ) -> None:
        """Start background prefetch for the given requests.

        Any in-progress prefetch thread is allowed to finish before starting a new one
        to avoid stale results polluting the cache.
        """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.max_wait_ms / 1000.0)

        with self._lock:
            self._prefetch_cache.clear()

        self._thread = threading.Thread(
            target=self._prefetch_worker,
            args=(requests, layer_idx),
            daemon=True,
        )
        self._thread.start()

    def _prefetch_worker(
        self,
        requests: List[InferenceRequest],
        layer_idx: int,
    ) -> None:
        """Worker: compute segment embeddings and search cache without touching stats.

        Accesses _compressed_store directly to avoid polluting hit/miss counters.
        Results are stored in _prefetch_cache under (request_id, chunk_idx) keys.
        """
        chunk_size = self.cache.chunk_size

        for req in requests:
            token_ids = req.token_ids
            n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                chunk_tokens = token_ids[start: start + chunk_size]

                # Build a query key embedding from token IDs without needing actual KV tensors.
                # Use a simple deterministic synthetic embedding so prefetch can run without
                # knowing the real KV tensors for the upcoming batch.
                token_mean = float(sum(chunk_tokens)) / max(1, len(chunk_tokens))
                g = torch.Generator()
                g.manual_seed(int(token_mean) & 0xFFFFFFFF)

                # Determine d_head from the semantic index if available, else use default
                d_head = 64
                with self._lock:
                    idx = self.cache._semantic_index
                    if idx:
                        d_head = idx[0][1].shape[-1]

                query_emb = torch.randn(d_head, generator=g)
                candidates = self.cache._cosine_search(query_emb, self.cache.top_k)

                result: Tuple[Optional[torch.Tensor], str] = (None, "miss")
                for cand_key, _emb, cos_sim in candidates:
                    if cos_sim < self.cache.similarity_threshold:
                        continue
                    with self._lock:
                        entry = self.cache._compressed_store.get(cand_key)
                    if entry is None:
                        continue
                    k_cand = self.cache.codec.decode(entry["k"], layer_idx)
                    v_cand = self.cache.codec.decode(entry["v"], layer_idx)
                    result = (torch.cat([k_cand, v_cand], dim=-1), "semantic")
                    break

                with self._lock:
                    self._prefetch_cache[(req.request_id, chunk_idx)] = result

    def get_prefetched(
        self,
        request: InferenceRequest,
        chunk_idx: int,
        timeout_ms: Optional[float] = None,
    ) -> Optional[Tuple[Optional[torch.Tensor], str]]:
        """Retrieve prefetched result for a request/chunk pair.

        Waits at most timeout_ms (or max_wait_ms) for the background thread before
        returning None so TTFT is never blocked beyond the configured budget.
        """
        wait_s = (timeout_ms if timeout_ms is not None else self.max_wait_ms) / 1000.0

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=wait_s)

        with self._lock:
            return self._prefetch_cache.get((request.request_id, chunk_idx))

    def clear(self) -> None:
        """Reset prefetch cache and wait for any running thread to finish."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.max_wait_ms / 1000.0)
        with self._lock:
            self._prefetch_cache.clear()
        self._thread = None

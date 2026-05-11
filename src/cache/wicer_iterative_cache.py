"""WiCERIterativeKVWikiCache — Activity B.

CEGAR (Counterexample-Guided Abstraction Refinement) iterative compilation
of domain KV artefacts. Builds a guaranteed-coverage non-contiguous KV cache
by refining chunk splits based on validation-query misses.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from src.cache.base import CacheStore
from src.cache.segmented import SegmentedHashCache


@dataclass
class WiCERConfig:
    chunk_size: int = 128           # initial chunk size in tokens
    min_chunk_size: int = 16        # minimum chunk size after refinement
    max_chunk_size: int = 512       # maximum chunk size after merge
    target_hit_rate: float = 0.80   # CEGAR termination: target hit rate
    max_iterations: int = 5         # maximum CEGAR iterations
    max_entries: int = 2000         # maximum cache entries
    seed: int = 42


class WiCERIterativeKVWikiCache(CacheStore):
    """CEGAR iterative compilation domain KV artefact cache (Activity B).

    Implements the full CacheStore interface by delegating to an internal
    SegmentedHashCache.  The CEGAR loop (compile → evaluate → refine)
    progressively narrows chunk splits until hit-rate coverage is met or
    no counterexamples remain.

    Typical usage:
      1. compile_corpus(docs, kv_fn)         — initial artefact build
      2. cegar_refine(docs, val_queries, kv_fn) — iterative refinement
      3. put / get / get_segments            — standard cache access
    """

    def __init__(self, config: WiCERConfig) -> None:
        self.config = config
        # Per-document chunk sizes (refined independently by CEGAR)
        self._chunk_sizes: Dict[str, int] = {}
        # Internal SegmentedHashCache backend
        self._store = SegmentedHashCache(
            chunk_size=config.chunk_size,
            max_entries=config.max_entries,
        )
        self._hits: int = 0
        self._misses: int = 0
        self._noncontiguous_hits: int = 0
        self._cegar_iteration: int = 0
        self._counterexamples: List[Tuple[str, int]] = []
        # Hit-rate history per CEGAR iteration (for monotonicity verification)
        self._hit_rate_history: List[float] = []

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV block; key = doc_id, value = [n_tokens, 2, n_heads, d_head]."""
        self._store.put(key, value)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve KV block; updates hit/miss counters."""
        result = self._store.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def evict(self) -> int:
        return self._store.evict()

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return self._store.memory_bytes()

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._store.reset_stats()

    # ------------------------------------------------------------------ #
    # CEGAR API                                                            #
    # ------------------------------------------------------------------ #

    def compile_corpus(
        self,
        docs: Dict[str, List[int]],
        kv_fn: Callable[[List[int], int], torch.Tensor],
        layer_idx: int = 0,
        codec: Optional[object] = None,
    ) -> None:
        """Compile phase: split each document into chunks and pre-compute KV.

        doc_id → per-document chunk size (may differ after refinement).
        kv_fn(token_ids, layer_idx) → Tensor [n_tokens, hidden].
        codec (optional): a compression codec; called as compression_hook if provided.
        """
        for doc_id, token_ids in docs.items():
            chunk_size = self._chunk_sizes.get(doc_id, self.config.chunk_size)
            n_chunks = max(1, math.ceil(len(token_ids) / chunk_size))
            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_size
                end = start + chunk_size
                chunk_tokens = token_ids[start:end]
                if not chunk_tokens:
                    continue
                kv = kv_fn(chunk_tokens, layer_idx)
                if codec is not None and hasattr(codec, "compression_hook"):
                    kv = codec.compression_hook(f"{doc_id}_{chunk_idx}", kv)
                # Use the document-specific chunk size for this segment
                self._put_segment_with_chunk_size(token_ids, chunk_idx, kv, layer_idx, chunk_size)

    def evaluate(
        self,
        val_queries: List[List[int]],
        layer_idx: int = 0,
    ) -> Tuple[float, List[Tuple[str, int]]]:
        """Evaluate phase: run validation queries and collect miss counterexamples.

        Returns (hit_rate, counterexamples) where counterexamples is a list of
        (query_prefix_hash, miss_chunk_idx) tuples.
        """
        total_hits = 0
        total_chunks = 0
        counterexamples: List[Tuple[str, int]] = []

        for token_ids in val_queries:
            hits, misses = self._store.get_segments(token_ids, layer_idx)
            total_hits += len(hits)
            total_chunks += len(hits) + len(misses)
            prefix_hash = self._hash_prefix(token_ids)
            for miss_chunk_idx in misses:
                counterexamples.append((prefix_hash, miss_chunk_idx))

        hit_rate = total_hits / total_chunks if total_chunks > 0 else 0.0
        return hit_rate, counterexamples

    def refine(
        self,
        counterexamples: List[Tuple[str, int]],
        docs: Dict[str, List[int]],
        kv_fn: Callable[[List[int], int], torch.Tensor],
        layer_idx: int = 0,
        codec: Optional[object] = None,
    ) -> None:
        """Refinement phase: split counterexample chunks into finer sub-chunks.

        For every document whose chunk boundary aligns with a counterexample,
        halve the chunk size (floor to min_chunk_size) and recompile that document.
        """
        refined_docs: Dict[str, bool] = {}

        for _query_hash, miss_chunk_idx in counterexamples:
            for doc_id, token_ids in docs.items():
                cur_chunk_size = self._chunk_sizes.get(doc_id, self.config.chunk_size)
                # Heuristic: refine if the miss chunk index falls within this document
                if miss_chunk_idx * cur_chunk_size < len(token_ids):
                    new_chunk_size = max(
                        self.config.min_chunk_size, cur_chunk_size // 2
                    )
                    if new_chunk_size < cur_chunk_size:
                        self._chunk_sizes[doc_id] = new_chunk_size
                        refined_docs[doc_id] = True

        docs_to_recompile = {k: v for k, v in docs.items() if k in refined_docs}
        if docs_to_recompile:
            self.compile_corpus(docs_to_recompile, kv_fn, layer_idx, codec)

    def cegar_refine(
        self,
        docs: Dict[str, List[int]],
        val_queries: List[List[int]],
        kv_fn: Callable[[List[int], int], torch.Tensor],
        layer_idx: int = 0,
        codec: Optional[object] = None,
    ) -> None:
        """Full CEGAR loop: compile → evaluate → refine until convergence.

        Terminates when hit_rate >= target_hit_rate or no counterexamples remain
        or max_iterations is reached.
        """
        self.compile_corpus(docs, kv_fn, layer_idx, codec)
        for iteration in range(self.config.max_iterations):
            self._cegar_iteration = iteration
            hit_rate, counterexamples = self.evaluate(val_queries, layer_idx)
            self._hit_rate_history.append(hit_rate)
            if hit_rate >= self.config.target_hit_rate:
                break
            if not counterexamples:
                break
            self.refine(counterexamples, docs, kv_fn, layer_idx, codec)

    # ------------------------------------------------------------------ #
    # Segment-level helpers (runner.py compatibility)                      #
    # ------------------------------------------------------------------ #

    def get_segments(
        self,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> Tuple[List[Tuple[int, torch.Tensor]], List[int]]:
        """Delegate to SegmentedHashCache.get_segments()."""
        return self._store.get_segments(token_ids, layer_idx)

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Delegate to SegmentedHashCache.put_segment()."""
        self._store.put_segment(token_ids, chunk_idx, kv, layer_idx)

    def noncontiguous_hit_rate(self) -> float:
        """Non-contiguous hit fraction from the internal SegmentedHashCache."""
        return self._store.noncontiguous_hit_rate()

    def cegar_hit_rate_history(self) -> List[float]:
        """Per-iteration hit-rate list (for monotonicity verification in tests)."""
        return list(self._hit_rate_history)

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def save_artifacts(self, path: str) -> None:
        """Serialise KV store, chunk-size schedule, and CEGAR history."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "store_state": self._store._store,
                "chunk_sizes": self._chunk_sizes,
                "hit_rate_history": self._hit_rate_history,
                "cegar_iteration": self._cegar_iteration,
            },
            path,
        )

    def load_artifacts(self, path: str) -> None:
        """Restore artefacts from a previously saved file."""
        data = torch.load(path, weights_only=False)
        self._store._store = OrderedDict(data["store_state"])
        self._chunk_sizes = data["chunk_sizes"]
        self._hit_rate_history = data["hit_rate_history"]
        self._cegar_iteration = data["cegar_iteration"]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _put_segment_with_chunk_size(
        self,
        token_ids: List[int],
        chunk_idx: int,
        kv: torch.Tensor,
        layer_idx: int,
        chunk_size: int,
    ) -> None:
        """Store a segment using a specific chunk_size (may differ from store default)."""
        import hashlib
        import struct

        start = chunk_idx * chunk_size
        end = start + chunk_size
        chunk = token_ids[start:end]
        if not chunk:
            return
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        key = hashlib.sha256(layer_prefix + raw).hexdigest()
        self._store.put(key, kv)

    @staticmethod
    def _hash_prefix(token_ids: List[int]) -> str:
        """Stable short hash for a token sequence (used as counterexample key)."""
        import hashlib
        import struct

        raw = struct.pack(f"{len(token_ids)}I", *token_ids)
        return hashlib.sha256(raw).hexdigest()[:16]

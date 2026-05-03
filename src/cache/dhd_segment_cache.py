"""SemanticSegmentCache — DHD (Dual-Stage High Deviation) semantic similarity-based non-contiguous KV sharing.

Activity B+C: Integrates TurboQuantCodec (Activity C) for compressed storage with
cosine-similarity-based segment retrieval and DHD deviation-guided recompute decisions.

N_segments ≤ 10K assumption: brute-force cosine search is used. Beyond 10K segments
_cosine_search() may exceed latency budgets; consider FAISS for larger indices.
"""

import hashlib
import struct
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.cache.base import CacheStore
from src.cache.turbo_quant import TurboQuantCodec


class SemanticSegmentCache(CacheStore):
    """DHD semantic similarity-based non-contiguous KV sharing cache (Activity B+C).

    Storage layout:
      _exact_store: OrderedDict[str, torch.Tensor]  — raw tensors for CacheStore.get()
      _compressed_store: Dict[str, dict]             — key → TurboQuantCodec compressed dicts
      _semantic_index: List[Tuple[str, torch.Tensor]] — (key, embedding) for similarity search

    Hit classification:
      exact_hits:      token-hash-identical segment found
      semantic_hits:   cosine sim above threshold AND DHD deviation below threshold
      recompute_count: segments where deviation exceeded threshold (caller must recompute)
    """

    def __init__(
        self,
        codec: TurboQuantCodec,
        chunk_size: int = 128,
        max_entries: int = 1000,
        top_k: int = 5,
        similarity_threshold: float = 0.80,
        deviation_threshold: float = 0.20,
        recompute_budget: float = 0.20,
    ) -> None:
        self.codec = codec
        self.chunk_size = chunk_size
        self.max_entries = max_entries
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.deviation_threshold = deviation_threshold
        self.recompute_budget = recompute_budget

        self._exact_store: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._compressed_store: Dict[str, dict] = {}
        self._semantic_index: List[Tuple[str, torch.Tensor]] = []

        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._recompute_count = 0

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                  #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store raw tensor under key (CacheStore interface, no compression)."""
        if key in self._exact_store:
            self._exact_store.move_to_end(key)
        else:
            if len(self._exact_store) >= self.max_entries:
                self.evict()
            self._exact_store[key] = value.detach().clone()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve raw tensor by exact key (CacheStore interface, no semantic search)."""
        if key in self._exact_store:
            self._exact_store.move_to_end(key)
            self._exact_hits += 1
            return self._exact_store[key]
        self._misses += 1
        return None

    def evict(self) -> int:
        """LRU eviction: removes oldest entry from all storage structures.

        Returns bytes freed from _exact_store (compressed store may differ).
        """
        if not self._exact_store and not self._compressed_store:
            return 0

        freed = 0
        if self._exact_store:
            evict_key, evicted_val = self._exact_store.popitem(last=False)
            freed = evicted_val.nbytes
            self._compressed_store.pop(evict_key, None)
            self._semantic_index = [(k, e) for k, e in self._semantic_index if k != evict_key]
        elif self._compressed_store:
            # Compressed-only entry (put_segment without put)
            evict_key = next(iter(self._compressed_store))
            self._compressed_store.pop(evict_key)
            self._semantic_index = [(k, e) for k, e in self._semantic_index if k != evict_key]

        return freed

    def hit_rate(self) -> float:
        """Combined hit rate: (exact + semantic) / total accesses."""
        total = self._exact_hits + self._semantic_hits + self._misses
        if total == 0:
            return 0.0
        return (self._exact_hits + self._semantic_hits) / total

    def memory_bytes(self) -> int:
        """Estimate memory used by compressed store entries."""
        total = 0
        for entry in self._compressed_store.values():
            for tensor_key in ("k", "v"):
                if tensor_key in entry:
                    compressed = entry[tensor_key]
                    total += compressed["quantized"].nbytes
                    total += compressed["scale"].nbytes
                    total += compressed["qjl_packed"].nbytes
        return total

    def reset_stats(self) -> None:
        """Reset all hit/miss/recompute counters."""
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._recompute_count = 0

    # ------------------------------------------------------------------ #
    # Extended API (Activity B+C integration)                              #
    # ------------------------------------------------------------------ #

    def put_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Compress and store a KV segment with its semantic embedding.

        Args:
            token_ids: Full token sequence.
            chunk_idx: Index of the chunk within the sequence.
            keys: (n_tokens_in_chunk, d_head) Key tensor.
            values: (n_tokens_in_chunk, d_head) Value tensor.
            layer_idx: Transformer layer index.
        """
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)
        # Mean of Key vectors as segment embedding — no external model needed
        embedding = keys.float().mean(dim=0)

        k_compressed = self.codec.encode(keys, layer_idx, tensor_id=0)
        v_compressed = self.codec.encode(values, layer_idx, tensor_id=1)

        self._compressed_store[key] = {
            "k": k_compressed,
            "v": v_compressed,
            "layer_idx": layer_idx,
        }
        self._semantic_index.append((key, embedding))

        # Also store in exact_store for CacheStore.get() compatibility
        combined = torch.cat([keys.float(), values.float()], dim=-1)
        if key in self._exact_store:
            self._exact_store.move_to_end(key)
        else:
            if len(self._exact_store) >= self.max_entries:
                self.evict()
            self._exact_store[key] = combined

    def get_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        query_keys: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], str]:
        """Look up a KV segment by exact hash or semantic similarity.

        Args:
            token_ids: Full token sequence for hash key derivation.
            chunk_idx: Chunk index within the sequence.
            query_keys: (n_tokens_in_chunk, d_head) current Key tensor for DHD check.
            layer_idx: Transformer layer index.

        Returns:
            (kv_tensor, hit_type) where kv_tensor is concatenated [K, V] or None,
            and hit_type is "exact", "semantic", or "miss".
        """
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)

        # Fast path: exact hash match
        if key in self._compressed_store:
            entry = self._compressed_store[key]
            k = self.codec.decode(entry["k"], layer_idx)
            v = self.codec.decode(entry["v"], layer_idx)
            self._exact_hits += 1
            return torch.cat([k, v], dim=-1), "exact"

        # Semantic search path
        if not self._semantic_index:
            self._misses += 1
            return None, "miss"

        query_emb = query_keys.float().mean(dim=0)
        candidates = self._cosine_search(query_emb, self.top_k)

        for cand_key, _cand_emb, cos_sim in candidates:
            if cos_sim < self.similarity_threshold:
                continue
            if cand_key not in self._compressed_store:
                continue
            entry = self._compressed_store[cand_key]
            k_cand = self.codec.decode(entry["k"], layer_idx)
            v_cand = self.codec.decode(entry["v"], layer_idx)

            deviation = self._compute_dhd_deviation(query_keys, k_cand)
            if deviation <= self.deviation_threshold:
                self._semantic_hits += 1
                return torch.cat([k_cand, v_cand], dim=-1), "semantic"
            else:
                # DHD deviation too high: flag for caller recompute
                self._recompute_count += 1
                self._misses += 1
                return None, "miss"

        self._misses += 1
        return None, "miss"

    def _cosine_search(
        self,
        query_emb: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        """Brute-force cosine similarity search over semantic index.

        N_segments ≤ 10K assumption: this is O(N*d) which is acceptable up to ~10K entries.
        """
        if not self._semantic_index:
            return []

        keys_list = [k for k, _ in self._semantic_index]
        emb_matrix = torch.stack([emb for _, emb in self._semantic_index])

        q_norm = F.normalize(query_emb.unsqueeze(0), dim=-1)
        e_norm = F.normalize(emb_matrix, dim=-1)
        sims = (q_norm @ e_norm.T).squeeze(0)

        actual_k = min(top_k, len(self._semantic_index))
        top_indices = sims.argsort(descending=True)[:actual_k]

        return [
            (keys_list[i.item()], self._semantic_index[i.item()][1], sims[i.item()].item())
            for i in top_indices
        ]

    def _compute_dhd_deviation(
        self,
        query_keys: torch.Tensor,
        cached_keys: torch.Tensor,
    ) -> float:
        """Compute normalized L2 deviation between query and cached Key vectors.

        Truncates to minimum length when shapes differ.
        Returns deviation normalized by cached key norms (0 = identical, higher = more different).
        """
        min_len = min(query_keys.shape[0], cached_keys.shape[0])
        q = query_keys.float()[:min_len]
        c = cached_keys.float()[:min_len]
        deviation = (q - c).norm(dim=-1).mean().item() / (c.norm(dim=-1).mean().item() + 1e-8)
        return deviation

    def semantic_hit_rates(self) -> dict:
        """Return detailed hit rate breakdown including non-contiguous ratio."""
        total_hits = self._exact_hits + self._semantic_hits
        total = total_hits + self._misses
        return {
            "exact_hit_rate": self._exact_hits / total if total > 0 else 0.0,
            "semantic_hit_rate": self._semantic_hits / total if total > 0 else 0.0,
            "overall_hit_rate": total_hits / total if total > 0 else 0.0,
            "noncontiguous_ratio": self._semantic_hits / total_hits if total_hits > 0 else 0.0,
            "recompute_ratio": self._recompute_count / max(1, total_hits),
        }

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """Deterministic position-independent chunk key (SHA-256, same as segmented.py)."""
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start: start + self.chunk_size]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

"""block_manager_patch.py — Activity B: DHD semantic non-contiguous KV cache for vLLM 0.20.1.

2026-05-03: SemanticNonContiguousKVCacheManager — subclasses KVCacheManager and
            adds a SemanticSegmentIndex for DHD-based non-contiguous block mapping.

Port of src/cache/dhd_segment_cache.SemanticSegmentCache adapted for vLLM 0.20.1's
paged-block KV cache layout. Integration point: KVCacheManager subclass that adds
a parallel semantic segment lookup alongside vLLM's prefix cache.

vLLM version: 0.20.1
Activity: B — Non-Contiguous KV Cache Reuse (DHD semantic similarity)

Integration principles:
- KVCacheManager subclass — never breaks the public KVCacheManager interface.
- Block boundaries: segments are quantized at vLLM block_size granularity.
- Semantic index is read-only from scheduler (no stats pollution).
- N_segments <= 10K assumption for brute-force cosine search.
"""

import hashlib
import struct
from collections import OrderedDict
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from vllm.v1.core.kv_cache_manager import KVCacheManager


# ---------------------------------------------------------------------------
# SemanticSegmentIndex — DHD semantic similarity non-contiguous block index
# ---------------------------------------------------------------------------

class SemanticSegmentIndex:
    """Semantic similarity segment index for non-contiguous KV reuse.

    Stores compressed KV segments indexed by:
    - Exact SHA-256 chunk hash (fast path)
    - Semantic embedding (cosine similarity search) for non-contiguous reuse

    DHD (Dual-Stage High Deviation) checks cosine similarity of query
    embedding vs candidate embedding, then verifies per-token L2 deviation
    is below threshold before returning the cached KV.

    Block boundary contract: callers must ensure chunk_size divides evenly
    into vLLM block_size so that segment boundaries align with page boundaries.
    """

    def __init__(
        self,
        codec: Any,  # VllmTurboQuantCodec instance (or None for no compression)
        chunk_size: int = 16,  # Should align with vLLM block_size
        max_entries: int = 2000,
        top_k: int = 5,
        similarity_threshold: float = 0.80,
        deviation_threshold: float = 0.20,
    ) -> None:
        self.codec = codec
        self.chunk_size = chunk_size
        self.max_entries = max_entries
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.deviation_threshold = deviation_threshold

        # Exact hash store: key → compressed dict (or raw tensor if no codec)
        self._compressed_store: OrderedDict[str, dict] = OrderedDict()
        # Semantic index: list of (key, embedding_tensor)
        # N <= 10K assumption: brute-force cosine search is used.
        # For N > 10K consider FAISS IVF index.
        self._semantic_index: List[Tuple[str, torch.Tensor]] = []

        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._recompute_count = 0

    def chunk_key(
        self,
        token_ids: List[int],
        chunk_idx: int,
        layer_idx: int = 0,
    ) -> str:
        """Deterministic content-only chunk key (SHA-256, position-independent)."""
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start: start + self.chunk_size]
        if not chunk:
            chunk = [0]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        """Compress and store a KV segment.

        Args:
            token_ids: Full token sequence.
            chunk_idx: Chunk index within the sequence.
            keys:   (n_tokens_in_chunk, [num_kv_heads,] head_size) Key tensor.
            values: (n_tokens_in_chunk, [num_kv_heads,] head_size) Value tensor.
            layer_idx: Transformer layer index.

        Returns:
            The chunk key string.
        """
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)

        # Compute segment embedding: mean of Key vectors
        if keys.dim() == 3:
            # (n_tokens, num_heads, head_size) → (head_size,)
            embedding = keys.float().mean(dim=(0, 1))
        else:
            embedding = keys.float().mean(dim=0)

        # Compress K and V
        if self.codec is not None:
            k_compressed = self.codec.encode_tokens(keys, layer_idx, tensor_id=0)
            v_compressed = self.codec.encode_tokens(values, layer_idx, tensor_id=1)
        else:
            k_compressed = {"raw": keys.float()}
            v_compressed = {"raw": values.float()}

        # LRU eviction if at capacity
        if len(self._compressed_store) >= self.max_entries:
            evict_key, _ = self._compressed_store.popitem(last=False)
            self._semantic_index = [
                (k, e) for k, e in self._semantic_index if k != evict_key
            ]

        self._compressed_store[key] = {
            "k": k_compressed,
            "v": v_compressed,
            "layer_idx": layer_idx,
            "embedding": embedding,
        }
        self._semantic_index.append((key, embedding))
        return key

    def lookup_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        query_keys: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """Look up KV segment by exact hash or semantic similarity.

        Args:
            token_ids: Full token sequence.
            chunk_idx: Chunk index within the sequence.
            query_keys: (n_tokens_in_chunk, [num_kv_heads,] head_size) — for DHD check.
            layer_idx: Transformer layer index.

        Returns:
            (keys, values, hit_type) where keys/values are float32 tensors or None,
            and hit_type is "exact", "semantic", or "miss".
        """
        key = self.chunk_key(token_ids, chunk_idx, layer_idx)

        # Fast path: exact hash match
        if key in self._compressed_store:
            entry = self._compressed_store[key]
            k = self._decode_entry(entry["k"], layer_idx, 0)
            v = self._decode_entry(entry["v"], layer_idx, 1)
            self._exact_hits += 1
            # Move to end for LRU freshness
            self._compressed_store.move_to_end(key)
            return k, v, "exact"

        # Semantic search path
        if not self._semantic_index:
            self._misses += 1
            return None, None, "miss"

        query_emb = self._compute_embedding(query_keys)
        candidates = self._cosine_search(query_emb, self.top_k)

        for cand_key, _cand_emb, cos_sim in candidates:
            if cos_sim < self.similarity_threshold:
                continue
            if cand_key not in self._compressed_store:
                continue
            entry = self._compressed_store[cand_key]
            k_cand = self._decode_entry(entry["k"], layer_idx, 0)
            v_cand = self._decode_entry(entry["v"], layer_idx, 1)

            deviation = self._compute_dhd_deviation(query_keys, k_cand)
            if deviation <= self.deviation_threshold:
                self._semantic_hits += 1
                return k_cand, v_cand, "semantic"
            else:
                # DHD deviation too high: mark for recompute, return miss
                self._recompute_count += 1
                self._misses += 1
                return None, None, "miss"

        self._misses += 1
        return None, None, "miss"

    def _decode_entry(
        self,
        entry: dict,
        layer_idx: int,
        tensor_id: int,
    ) -> torch.Tensor:
        """Decode a compressed entry dict back to float32 tensor."""
        if "raw" in entry:
            return entry["raw"]
        if self.codec is not None:
            return self.codec.decode_tokens(entry, layer_idx, tensor_id)
        return entry.get("raw", torch.zeros(1))

    def _compute_embedding(self, keys: torch.Tensor) -> torch.Tensor:
        """Compute segment embedding as mean of Key vectors."""
        if keys.dim() == 3:
            return keys.float().mean(dim=(0, 1))
        return keys.float().mean(dim=0)

    def _cosine_search(
        self,
        query_emb: torch.Tensor,
        top_k: int,
    ) -> List[Tuple[str, torch.Tensor, float]]:
        """Brute-force cosine similarity search.

        N_segments <= 10K assumption. Beyond 10K this becomes a latency risk;
        consider FAISS IVF index for larger deployments.
        """
        if not self._semantic_index:
            return []

        keys_list = [k for k, _ in self._semantic_index]
        emb_matrix = torch.stack([emb for _, emb in self._semantic_index])

        # Align embedding dimensions (query may differ if heads were averaged)
        if query_emb.shape[0] != emb_matrix.shape[1]:
            min_d = min(query_emb.shape[0], emb_matrix.shape[1])
            query_emb = query_emb[:min_d]
            emb_matrix = emb_matrix[:, :min_d]

        q_norm = F.normalize(query_emb.unsqueeze(0).float(), dim=-1)
        e_norm = F.normalize(emb_matrix.float(), dim=-1)
        sims = (q_norm @ e_norm.T).squeeze(0)

        actual_k = min(top_k, len(self._semantic_index))
        top_indices = sims.argsort(descending=True)[:actual_k]

        return [
            (
                keys_list[i.item()],
                self._semantic_index[i.item()][1],
                sims[i.item()].item(),
            )
            for i in top_indices
        ]

    def _compute_dhd_deviation(
        self,
        query_keys: torch.Tensor,
        cached_keys: torch.Tensor,
    ) -> float:
        """Compute normalized L2 deviation between query and cached Key vectors."""
        # Flatten to 2D for comparison
        q = query_keys.float()
        c = cached_keys.float()
        if q.dim() == 3:
            q = q.reshape(q.shape[0], -1)
        if c.dim() == 3:
            c = c.reshape(c.shape[0], -1)
        min_len = min(q.shape[0], c.shape[0])
        q = q[:min_len]
        c = c[:min_len]
        deviation = (q - c).norm(dim=-1).mean().item() / (c.norm(dim=-1).mean().item() + 1e-8)
        return deviation

    def hit_rate(self) -> float:
        total = self._exact_hits + self._semantic_hits + self._misses
        return (self._exact_hits + self._semantic_hits) / total if total > 0 else 0.0

    def semantic_hit_rates(self) -> dict:
        """Detailed hit rate breakdown including non-contiguous ratio."""
        total_hits = self._exact_hits + self._semantic_hits
        total = total_hits + self._misses
        return {
            "exact_hit_rate": self._exact_hits / total if total > 0 else 0.0,
            "semantic_hit_rate": self._semantic_hits / total if total > 0 else 0.0,
            "overall_hit_rate": total_hits / total if total > 0 else 0.0,
            "noncontiguous_ratio": (
                self._semantic_hits / total_hits if total_hits > 0 else 0.0
            ),
            "recompute_ratio": self._recompute_count / max(1, total_hits),
        }

    def reset_stats(self) -> None:
        self._exact_hits = 0
        self._semantic_hits = 0
        self._misses = 0
        self._recompute_count = 0

    def memory_bytes(self) -> int:
        """Estimate memory used by compressed segment store."""
        total = 0
        for entry in self._compressed_store.values():
            for tensor_key in ("k", "v"):
                compressed = entry.get(tensor_key, {})
                if "quantized" in compressed:
                    total += compressed["quantized"].nbytes
                    total += compressed["scale"].nbytes
                    total += compressed["qjl_packed"].nbytes
                elif "raw" in compressed:
                    total += compressed["raw"].nbytes
                elif "fp16" in compressed:
                    total += compressed["fp16"].nbytes
        return total


# ---------------------------------------------------------------------------
# SemanticNonContiguousKVCacheManager — KVCacheManager subclass (Activity B)
# ---------------------------------------------------------------------------

class SemanticNonContiguousKVCacheManager(KVCacheManager):
    """KVCacheManager subclass adding DHD semantic non-contiguous KV reuse.

    Extends vLLM's KVCacheManager with a parallel SemanticSegmentIndex that
    stores compressed KV segments indexed by semantic embeddings. On each
    prefill, tokens are checked against the index for non-contiguous hits
    before falling through to normal computation.

    Activity B integration:
        - store_segment(): call after computing K/V for a chunk to populate index
        - lookup_segment(): call before computing K/V for a chunk to check for hits
        - lookup_all_segments(): batch lookup for all chunks in a sequence

    Activity C integration (via codec):
        - Pass a VllmTurboQuantCodec as `codec` to enable compressed storage.
        - Without codec, segments are stored as raw float32 tensors.

    Block boundary contract:
        - segment_chunk_size should equal vLLM's block_size (default 16) or be a
          divisor of it, so segment boundaries align with KV block page boundaries.
        - Crossing block boundaries within a segment is not supported.
    """

    def __init__(
        self,
        *args: Any,
        codec: Any = None,  # Optional VllmTurboQuantCodec
        segment_chunk_size: int = 16,
        segment_max_entries: int = 2000,
        segment_top_k: int = 5,
        similarity_threshold: float = 0.80,
        deviation_threshold: float = 0.20,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._segment_index = SemanticSegmentIndex(
            codec=codec,
            chunk_size=segment_chunk_size,
            max_entries=segment_max_entries,
            top_k=segment_top_k,
            similarity_threshold=similarity_threshold,
            deviation_threshold=deviation_threshold,
        )
        self._segment_codec = codec
        self._segment_chunk_size = segment_chunk_size

    # ------------------------------------------------------------------
    # Activity B public API
    # ------------------------------------------------------------------

    def store_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> str:
        """Compress and store a KV segment in the semantic index.

        Call this after computing K/V projections for a chunk during prefill.
        Block boundary contract: len(keys) must equal segment_chunk_size.

        Returns:
            The chunk key string.
        """
        return self._segment_index.store_segment(
            token_ids, chunk_idx, keys, values, layer_idx
        )

    def lookup_segment(
        self,
        token_ids: List[int],
        chunk_idx: int,
        query_keys: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """Look up cached KV for a single chunk.

        Args:
            token_ids: Full token sequence.
            chunk_idx: Chunk index within the sequence.
            query_keys: Current K projection for DHD deviation check.
            layer_idx: Transformer layer index.

        Returns:
            (keys, values, hit_type) — hit_type is "exact", "semantic", or "miss".
            On miss, keys and values are None.
        """
        return self._segment_index.lookup_segment(
            token_ids, chunk_idx, query_keys, layer_idx
        )

    def lookup_all_segments(
        self,
        token_ids: List[int],
        layer_idx: int,
        query_keys: torch.Tensor,
    ) -> List[Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor], str]]:
        """Batch lookup for all chunks in a token sequence.

        Args:
            token_ids: Full token sequence.
            layer_idx: Transformer layer index.
            query_keys: (n_tokens, [num_heads,] head_size) Key projection for
                        DHD deviation check. Sliced per chunk.

        Returns:
            List of (chunk_idx, keys, values, hit_type) tuples.
        """
        chunk_size = self._segment_chunk_size
        n_chunks = max(1, (len(token_ids) + chunk_size - 1) // chunk_size)
        results = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, query_keys.shape[0])
            chunk_query_keys = query_keys[start:end]

            k, v, hit_type = self.lookup_segment(
                token_ids, chunk_idx, chunk_query_keys, layer_idx
            )
            results.append((chunk_idx, k, v, hit_type))

        return results

    def segment_index_stats(self) -> dict:
        """Return semantic hit rate statistics from the segment index."""
        return self._segment_index.semantic_hit_rates()

    def segment_memory_bytes(self) -> int:
        """Return estimated memory used by the segment index."""
        return self._segment_index.memory_bytes()

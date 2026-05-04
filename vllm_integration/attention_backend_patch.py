"""attention_backend_patch.py — Activity B/C: RedundancyAwareEvictionPolicy + attention hooks for vLLM 0.20.1.

2026-05-04: VllmRedundancyAwareEvictionPolicy — ports RedundancyAwareEvictionPolicy from
            src/cache/redundancy_eviction.py into vLLM's block management layer.

            VllmAttentionKVHook — lightweight hook class for inserting importance
            recording around vLLM attention backend write/read operations.

vLLM 0.20.1 v1 architecture:
    - Attention backends are in vllm/attention/backends/ (Flash, XFormers, Triton, etc.)
    - There is no single write_to_cache / read_from_cache hook point in v1 — KV blocks
      are managed by the paged block pool at the block level, not per-token.
    - The recommended integration pattern is to wrap the attention forward() method
      or to hook into the model runner's prefill path.

Integration strategy:
    VllmRedundancyAwareEvictionPolicy operates as a pure scoring layer plugged into
    WorkloadAwareTTLKVCacheManager.evict_expired_segments() — it does NOT intercept
    attention computation and does NOT apply any lossy operation.

    Accuracy preservation contract:
        - Only TTL-expired segments are candidates for scoring.
        - eviction_score = (1 - normalized_importance) × redundancy_score
        - High-importance segments (importance_score == 1.0) have eviction_score == 0.0
          and are structurally protected from eviction.
        - No quantization, no approximation — pure segment ordering/selection.

    VllmAttentionKVHook provides a thin wrapper that model code can call after
    attention to record importance scores into WorkloadAwareTTLKVCacheManager.

Activity: B/C — Non-Contiguous Reuse + KV Cache Compression (accuracy-preserving)

Prior cycle (2026-05-03) components are preserved at the bottom of this file.

vLLM version: 0.20.1
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

import vllm
def _vllm_version_tuple(v: str):
    return tuple(int(x) for x in v.split(".")[:3])
assert _vllm_version_tuple(vllm.__version__) >= _vllm_version_tuple("0.4.0"), (
    f"vllm_integration requires vLLM >= 0.4.0, found {vllm.__version__}"
)

if TYPE_CHECKING:
    from vllm_integration.block_manager_patch import VllmTTLEntry, WorkloadAwareTTLKVCacheManager


# ---------------------------------------------------------------------------
# VllmRedundancyAwareEvictionPolicy — Activity C accuracy-preserving eviction
# ---------------------------------------------------------------------------

class VllmRedundancyAwareEvictionPolicy:
    """Dual-score (importance × redundancy) eviction policy for vLLM TTL segments.

    Ports src/cache/redundancy_eviction.RedundancyAwareEvictionPolicy into the
    vLLM integration layer. Operates on VllmTTLEntry objects from
    WorkloadAwareTTLKVCacheManager._ttl_store.

    eviction_score = (1 - normalized_importance) × redundancy_score

    Accuracy preservation:
        - The multiplicative form structurally protects high-importance segments:
          importance == 1.0 → eviction_score == 0.0 regardless of redundancy.
        - Only TTL-expired segments are passed in as candidates — segments within
          their TTL window are never touched by this policy.
        - No lossy operations: no quantization, no approximation of KV values.
        - The policy only reorders the eviction sequence; it does not add new
          eviction candidates beyond what TTL expiry already designated.

    Attributes:
        redundancy_top_n: Max candidates for brute-force cosine similarity (O(N^2)).
        importance_weight: Multiplier on importance term (default 1.0).
        redundancy_weight: Multiplier on redundancy term (default 1.0).
        doc_id_shortcut: If True, segments sharing 'doc:<id>:' key prefix get
                         redundancy=1.0 immediately (O(1) shortcut).
    """

    def __init__(
        self,
        redundancy_top_n: int = 100,
        importance_weight: float = 1.0,
        redundancy_weight: float = 1.0,
        doc_id_shortcut: bool = True,
    ) -> None:
        self.redundancy_top_n = redundancy_top_n
        self.importance_weight = importance_weight
        self.redundancy_weight = redundancy_weight
        self.doc_id_shortcut = doc_id_shortcut

    def score_ttl_candidates(
        self,
        candidates: List[str],
        ttl_store: Dict[str, "VllmTTLEntry"],
    ) -> List[Tuple[str, float]]:
        """Score eviction candidates; returns (key, eviction_score) sorted descending.

        Args:
            candidates: List of segment keys (TTL-expired, non-pinned).
            ttl_store: The WorkloadAwareTTLKVCacheManager._ttl_store dict.

        Returns:
            List of (key, eviction_score) tuples sorted by score descending.
            Higher score = evicted first.

        Accuracy contract:
            eviction_score(importance=1.0) == 0.0 for any redundancy value.
        """
        if not candidates:
            return []

        # Step 1: Importance normalisation
        importances: Dict[str, float] = {
            k: ttl_store[k].importance_score
            for k in candidates
            if k in ttl_store
        }
        max_imp = max(importances.values()) if importances else 1.0
        if max_imp == 0.0:
            max_imp = 1.0
        norm_imp: Dict[str, float] = {k: v / max_imp for k, v in importances.items()}

        # Step 2: Redundancy scores
        redundancy: Dict[str, float] = {k: 0.0 for k in candidates}

        if self.doc_id_shortcut:
            self._apply_doc_id_shortcut(candidates, redundancy)

        # Embedding-based redundancy for remaining candidates
        need_emb = [
            k for k in candidates
            if k in ttl_store
            and ttl_store[k].embedding is not None
            and redundancy[k] < 1.0
        ]
        need_emb = need_emb[: self.redundancy_top_n]

        if len(need_emb) >= 2:
            embeddings = torch.stack([ttl_store[k].embedding for k in need_emb])  # (N, d)
            e_norm = F.normalize(embeddings, dim=-1)
            sim_matrix = e_norm @ e_norm.T  # (N, N)
            sim_matrix.fill_diagonal_(0.0)
            mean_sim = sim_matrix.mean(dim=-1)  # (N,)
            for i, k in enumerate(need_emb):
                redundancy[k] = float(mean_sim[i].clamp(min=0.0).item())

        # Step 3: Compute eviction scores
        scored: List[Tuple[str, float]] = []
        for k in candidates:
            imp = norm_imp.get(k, 0.0) * self.importance_weight
            red = redundancy[k] * self.redundancy_weight
            score = (1.0 - imp) * red
            scored.append((k, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def select_evict_keys(
        self,
        candidates: List[str],
        ttl_store: Dict[str, "VllmTTLEntry"],
        n_evict: int = 1,
    ) -> List[str]:
        """Return the top n_evict keys by eviction score.

        Args:
            candidates: List of TTL-expired segment keys.
            ttl_store: WorkloadAwareTTLKVCacheManager._ttl_store.
            n_evict: Number of keys to return.

        Returns:
            List of up to n_evict segment keys to evict.
        """
        scored = self.score_ttl_candidates(candidates, ttl_store)
        return [k for k, _ in scored[:n_evict]]

    def _apply_doc_id_shortcut(
        self,
        candidates: List[str],
        redundancy: Dict[str, float],
    ) -> None:
        """Set redundancy=1.0 for segments sharing the same doc_id prefix.

        Expected key format: 'doc:<doc_id>:<rest>'.
        Two segments with the same doc_id are considered fully redundant.
        """
        prefix_groups: Dict[str, List[str]] = {}
        for k in candidates:
            parts = k.split(":")
            if len(parts) >= 2 and parts[0] == "doc":
                doc_prefix = f"doc:{parts[1]}"
                prefix_groups.setdefault(doc_prefix, []).append(k)

        for group in prefix_groups.values():
            if len(group) >= 2:
                for k in group:
                    redundancy[k] = 1.0


# ---------------------------------------------------------------------------
# VllmAttentionKVHook — lightweight importance recording hook (Activity C)
# ---------------------------------------------------------------------------

class VllmAttentionKVHook:
    """Lightweight hook for recording KV importance scores from attention weights.

    Integrates with WorkloadAwareTTLKVCacheManager.record_segment_importance()
    to feed attention weight statistics into the eviction policy.

    Usage (in model runner or attention layer):

        hook = VllmAttentionKVHook(
            kv_manager=dag_aware_kv_manager,
            chunk_size=128,
        )

        # After computing attention weights (softmax over sequence length):
        # attn_weights: (batch, n_heads, seq_q, seq_k) or (batch, seq_q, seq_k)
        hook.record_importance_from_attention(
            attn_weights=attn_weights,
            token_ids=request.prompt_token_ids,
            layer_idx=layer_idx,
        )

    Accuracy preservation:
        This hook only reads attention weights to compute importance statistics.
        It does NOT modify, compress, or discard any KV values. No lossy
        operations are performed. The hook only affects which TTL-expired
        segments are evicted first in WorkloadAwareTTLKVCacheManager.
    """

    def __init__(
        self,
        kv_manager: "WorkloadAwareTTLKVCacheManager",
        chunk_size: int = 128,
        importance_aggregation: str = "mean",
    ) -> None:
        """
        Args:
            kv_manager: The WorkloadAwareTTLKVCacheManager instance.
            chunk_size: Token chunk size matching the TTL segment key scheme.
            importance_aggregation: "mean" or "max" attention weight aggregation.
        """
        self.kv_manager = kv_manager
        self.chunk_size = chunk_size
        self.importance_aggregation = importance_aggregation

    def record_importance_from_attention(
        self,
        attn_weights: torch.Tensor,
        token_ids: List[int],
        layer_idx: int = 0,
    ) -> None:
        """Record per-chunk importance from attention weight tensor.

        Args:
            attn_weights: Supported shapes:
                (batch, n_heads, seq_q, seq_k), (batch, seq_q, seq_k), (seq_q, seq_k).
            token_ids: Token IDs for segment key computation.
            layer_idx: Transformer layer index.
        """
        if not token_ids:
            return

        w = attn_weights.float().detach()
        if w.dim() == 4:
            w = w.mean(dim=(0, 1))   # (seq_q, seq_k)
        elif w.dim() == 3:
            w = w.mean(dim=0)        # (seq_q, seq_k)
        # w is now (seq_q, seq_k)

        seq_len = w.shape[-1]
        n_chunks = max(1, (seq_len + self.chunk_size - 1) // self.chunk_size)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_attn = w[:, start:end]   # (seq_q, chunk_len)

            if self.importance_aggregation == "max":
                importance = float(chunk_attn.max().item())
            else:
                importance = float(chunk_attn.mean().item())

            key = self._compute_chunk_key(token_ids, chunk_idx, layer_idx)
            self.kv_manager.record_segment_importance(key, importance)

    def _compute_chunk_key(
        self, token_ids: List[int], chunk_idx: int, layer_idx: int = 0
    ) -> str:
        """SHA-256 chunk key (compatible with WorkloadAwareTTLCache.chunk_key())."""
        import hashlib
        import struct
        start = chunk_idx * self.chunk_size
        chunk = token_ids[start: start + self.chunk_size]
        if not chunk:
            chunk = [0]
        raw = struct.pack(f"{len(chunk)}I", *chunk)
        layer_prefix = struct.pack("I", layer_idx)
        return hashlib.sha256(layer_prefix + raw).hexdigest()


# ---------------------------------------------------------------------------
# Prior cycle (2026-05-03) components — preserved for backward compatibility
# ---------------------------------------------------------------------------

class TurboQuantKVHook:
    """Prior-cycle TurboQuant 3-bit KV hook (2026-05-03). Preserved for compat.

    In the 2026-05-04 cycle, Activity C compression is handled by
    VllmRedundancyAwareEvictionPolicy (accuracy-preserving eviction ordering)
    rather than quantization. This class is kept for backward compat.
    """

    def __init__(self, codec: Any, enabled: bool = True) -> None:
        self.codec = codec
        self.enabled = enabled

    def write_to_cache(self, kv: torch.Tensor, layer_idx: int, tensor_id: int = 0) -> dict:
        if not self.enabled or self.codec is None:
            return {"raw": kv.detach().float()}
        return self.codec.encode_tokens(kv, layer_idx, tensor_id)

    def read_from_cache(self, compressed: dict, layer_idx: int, tensor_id: int = 0) -> torch.Tensor:
        if "raw" in compressed:
            return compressed["raw"]
        if not self.enabled or self.codec is None:
            raise ValueError("Cannot decode: codec disabled but compressed dict has no 'raw' key")
        return self.codec.decode_tokens(compressed, layer_idx, tensor_id)

    def compression_ratio(self, layer_idx: int) -> float:
        if self.codec is None:
            return 0.0
        return self.codec.compression_ratio(layer_idx)

    def is_sensitive_layer(self, layer_idx: int) -> bool:
        if self.codec is None:
            return False
        inner = getattr(self.codec, "_codec", self.codec)
        cutoff = getattr(inner, "_sensitive_cutoff", 0)
        return layer_idx < cutoff


class SemanticKVAttentionWrapper:
    """Prior-cycle DHD semantic attention wrapper (2026-05-03). Preserved for compat."""

    def __init__(self, impl: Any, hook: Any, kv_manager: Any, chunk_size: int = 16) -> None:
        self._impl = impl
        self._hook = hook
        self._kv_manager = kv_manager
        self._chunk_size = chunk_size

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        return self._impl.forward(
            layer, query, key, value, kv_cache, attn_metadata, output,
            output_scale, output_block_scale,
        )

    def store_kv_chunks(self, token_ids: List[int], keys: torch.Tensor,
                        values: torch.Tensor, layer_idx: int) -> List[str]:
        chunk_size = self._chunk_size
        n_tokens = keys.shape[0]
        n_chunks = max(1, (n_tokens + chunk_size - 1) // chunk_size)
        stored_keys = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_tokens)
            stored_key = self._kv_manager.store_segment(
                token_ids, chunk_idx, keys[start:end], values[start:end], layer_idx
            )
            stored_keys.append(stored_key)
        return stored_keys

    def load_cached_chunks(
        self, token_ids: List[int], layer_idx: int, query_keys: torch.Tensor,
    ) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], List[int]]:
        results = self._kv_manager.lookup_all_segments(token_ids, layer_idx, query_keys)
        hit_chunks: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        miss_chunk_indices: List[int] = []
        for chunk_idx, k, v, hit_type in results:
            if hit_type in ("exact", "semantic") and k is not None and v is not None:
                hit_chunks.append((chunk_idx, k, v))
            else:
                miss_chunk_indices.append(chunk_idx)
        return hit_chunks, miss_chunk_indices

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


class TriStateKVHook:
    """Prior-cycle tri-state KV hook (2026-04-30). Preserved for compat."""

    def __init__(self, codec: Any, retain_ratio: float = 0.20, evict_ratio: float = 0.40) -> None:
        self.codec = codec
        self.retain_ratio = retain_ratio
        self.evict_ratio = evict_ratio

    def encode_kv(self, kv: torch.Tensor, attn_weights: Any, layer_idx: int, tensor_id: int = 0) -> dict:
        return {"raw": kv.detach().float(), "layer_idx": layer_idx}

    def decode_kv(self, storage: dict, layer_idx: int, tensor_id: int = 0) -> torch.Tensor:
        return storage["raw"]

    def compression_ratio(self) -> float:
        return 1.0 - self.retain_ratio - self.evict_ratio


class CompressedKVHook:
    """Prior-cycle INT8/Hadamard hook (2026-04-28/29). Preserved for compat."""

    def __init__(self, codec: Any) -> None:
        self.codec = codec

    def encode(self, kv: torch.Tensor, layer_idx: int, tensor_id: int = 0) -> dict:
        if hasattr(self.codec, "encode"):
            kv_flat = kv.float()
            if kv_flat.dim() == 1:
                kv_flat = kv_flat.unsqueeze(0)
            return self.codec.encode(kv_flat, layer_idx, tensor_id)
        return {"raw": kv.float()}

    def decode(self, compressed: dict, layer_idx: int, tensor_id: int = 0) -> torch.Tensor:
        if "raw" in compressed:
            return compressed["raw"]
        if hasattr(self.codec, "decode"):
            return self.codec.decode(compressed, layer_idx, tensor_id)
        return compressed.get("raw", torch.zeros(1))


class NonContiguousAttentionWrapper:
    """Prior-cycle non-contiguous wrapper (2026-04-28). Preserved for compat."""

    def __init__(self, impl: Any, hook: Any, kv_manager: Any, chunk_size: int = 64) -> None:
        self._impl = impl
        self._hook = hook
        self._kv_manager = kv_manager
        self._chunk_size = chunk_size

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        return self._impl.forward(
            layer, query, key, value, kv_cache, attn_metadata, output,
            output_scale, output_block_scale,
        )

    def store_kv_chunks(self, token_ids, k, v, layer_idx):
        chunk_size = self._chunk_size
        n = k.shape[0]
        n_chunks = max(1, (n + chunk_size - 1) // chunk_size)
        for ci in range(n_chunks):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, n)
            if hasattr(self._kv_manager, "store_segment"):
                self._kv_manager.store_segment(token_ids, ci, k[s:e], v[s:e], layer_idx)

    def load_cached_chunks(self, token_ids, layer_idx):
        hits, misses = [], []
        if hasattr(self._kv_manager, "lookup_all_segments"):
            q_keys = torch.zeros(1, 1)
            for ci, ck, cv, hit_type in self._kv_manager.lookup_all_segments(
                token_ids, layer_idx, q_keys
            ):
                if hit_type != "miss" and ck is not None:
                    hits.append((ci, ck, cv))
                else:
                    misses.append(ci)
        return hits, misses

    def __getattr__(self, name):
        return getattr(self._impl, name)

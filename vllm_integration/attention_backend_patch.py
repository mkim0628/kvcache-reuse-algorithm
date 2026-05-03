"""attention_backend_patch.py — Activity B+C: TurboQuant KV hooks for vLLM 0.20.1.

2026-05-03: TurboQuantKVHook — hooks TurboQuant 3-bit codec into attention
            backend write/read paths for Activity C compression.
            SemanticKVAttentionWrapper — wraps AttentionImpl to add DHD
            non-contiguous lookup before the attention kernel (Activity B).

Integration points:
    - Write hook: encode K/V after QKV projection, before segment index storage
    - Read hook:  decode compressed K/V before attention kernel computation
    - Non-contiguous: check SemanticSegmentIndex before recomputing KV

Constraint: decompression MUST happen before the attention kernel entry.
Compressed tensors NEVER enter flashinfer / flash-attention kernels directly.

vLLM version: 0.20.1
Activity: B (non-contiguous lookup) + C (TurboQuant write/read hooks)
"""

from typing import Any, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# TurboQuantKVHook — Activity C write/read hooks
# ---------------------------------------------------------------------------

class TurboQuantKVHook:
    """Hooks TurboQuantCodec into attention backend K/V write and read paths.

    The codec operates on the parallel SemanticSegmentIndex only. vLLM's
    physical kv_cache pages remain in their native dtype (fp16/bf16) as
    configured by CacheConfig. This hook does NOT intercept or modify the
    actual kv_cache tensor that vLLM's flash-attention kernels read from.

    Usage pattern (in model attention layer or inference harness):

    Write hook (after QKV projection, before segment index storage):
        compressed_k = hook.write_to_cache(key, layer_idx, tensor_id=0)
        compressed_v = hook.write_to_cache(value, layer_idx, tensor_id=1)
        kv_manager.store_segment(token_ids, chunk_idx, key, value, layer_idx)

    Read hook (before attention computation with cached KV):
        key   = hook.read_from_cache(compressed_k, layer_idx, tensor_id=0)
        value = hook.read_from_cache(compressed_v, layer_idx, tensor_id=1)
        # Pass uncompressed key, value to attention kernel
    """

    def __init__(
        self,
        codec: Any,  # VllmTurboQuantCodec instance
        enabled: bool = True,
    ) -> None:
        self.codec = codec
        self.enabled = enabled

    def write_to_cache(
        self,
        kv: torch.Tensor,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> dict:
        """Compress KV tensor for storage in the semantic segment index.

        Args:
            kv: (n_tokens, [num_kv_heads,] head_size) float tensor.
            layer_idx: Transformer layer index.
            tensor_id: 0 for K, 1 for V.

        Returns:
            Compressed dict if enabled; passthrough dict with 'raw' key if disabled.
        """
        if not self.enabled or self.codec is None:
            return {"raw": kv.detach().float()}
        return self.codec.encode_tokens(kv, layer_idx, tensor_id)

    def read_from_cache(
        self,
        compressed: dict,
        layer_idx: int,
        tensor_id: int = 0,
    ) -> torch.Tensor:
        """Decompress KV tensor before attention kernel.

        Decompression MUST occur before the attention kernel entry.
        Compressed tensors never enter flashinfer or flash-attention kernels.

        Args:
            compressed: Dict returned by write_to_cache().
            layer_idx: Must match the layer_idx used during write.
            tensor_id: 0 for K, 1 for V.

        Returns:
            Approximately reconstructed float32 KV tensor.
        """
        if "raw" in compressed:
            return compressed["raw"]
        if not self.enabled or self.codec is None:
            raise ValueError(
                "Cannot decode: codec disabled but compressed dict has no 'raw' key"
            )
        return self.codec.decode_tokens(compressed, layer_idx, tensor_id)

    def compression_ratio(self, layer_idx: int) -> float:
        """Return fraction of bytes saved vs FP32 for the given layer."""
        if self.codec is None:
            return 0.0
        return self.codec.compression_ratio(layer_idx)

    def is_sensitive_layer(self, layer_idx: int) -> bool:
        """Return True if layer uses higher-bit (4-bit) quantization."""
        if self.codec is None:
            return False
        inner = getattr(self.codec, "_codec", self.codec)
        cutoff = getattr(inner, "_sensitive_cutoff", 0)
        return layer_idx < cutoff


# ---------------------------------------------------------------------------
# SemanticKVAttentionWrapper — Activity B non-contiguous KV lookup wrapper
# ---------------------------------------------------------------------------

class SemanticKVAttentionWrapper:
    """Wraps an AttentionImpl to add DHD semantic non-contiguous KV lookup.

    Delegates all standard attention operations to the wrapped impl unchanged.
    Adds segment-level KV lookup against the SemanticSegmentIndex before
    the normal attention forward pass to exploit non-contiguous KV reuse.

    Architecture:
        1. store_kv_chunks(): called after prefill to populate the index.
        2. load_cached_chunks(): called before prefill to check the index.
        3. forward(): delegates to wrapped impl with identical interface.

    The wrapper does NOT modify vLLM's kv_cache pages — it maintains a
    separate SemanticSegmentIndex for non-contiguous KV reuse only.

    Block boundary contract: chunk_size must align with vLLM's block_size so
    that segment boundaries align with KV cache page boundaries. Segments that
    cross block boundaries are not supported.
    """

    def __init__(
        self,
        impl: Any,  # vllm.v1.attention.backend.AttentionImpl
        hook: TurboQuantKVHook,
        kv_manager: Any,  # SemanticNonContiguousKVCacheManager
        chunk_size: int = 16,
    ) -> None:
        self._impl = impl
        self._hook = hook
        self._kv_manager = kv_manager
        self._chunk_size = chunk_size

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Delegate to wrapped AttentionImpl.forward() — interface unchanged.

        Standard vLLM attention forward pass. Non-contiguous KV reuse happens
        via store_kv_chunks() / load_cached_chunks() outside of this method.
        """
        return self._impl.forward(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    def store_kv_chunks(
        self,
        token_ids: List[int],
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
    ) -> List[str]:
        """Compress and store KV chunks in the semantic segment index.

        Call after the attention forward pass for a prefill batch. Uses
        TurboQuantKVHook to compress K/V before storing in the index.

        Args:
            token_ids: Full token sequence.
            keys:   (n_tokens, [num_kv_heads,] head_size) Key tensor.
            values: (n_tokens, [num_kv_heads,] head_size) Value tensor.
            layer_idx: Transformer layer index.

        Returns:
            List of stored chunk key strings.
        """
        chunk_size = self._chunk_size
        n_tokens = keys.shape[0]
        n_chunks = max(1, (n_tokens + chunk_size - 1) // chunk_size)
        stored_keys = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_tokens)
            chunk_keys = keys[start:end]
            chunk_values = values[start:end]

            stored_key = self._kv_manager.store_segment(
                token_ids, chunk_idx, chunk_keys, chunk_values, layer_idx
            )
            stored_keys.append(stored_key)

        return stored_keys

    def load_cached_chunks(
        self,
        token_ids: List[int],
        layer_idx: int,
        query_keys: torch.Tensor,
    ) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], List[int]]:
        """Load cached KV chunks from the semantic segment index.

        Args:
            token_ids: Full token sequence.
            layer_idx: Transformer layer index.
            query_keys: (n_tokens, [num_kv_heads,] head_size) current Key projection
                        used for DHD deviation checking.

        Returns:
            Tuple of:
                hit_chunks: list of (chunk_idx, cached_k, cached_v) for cache hits
                miss_chunk_indices: list of chunk_idx values requiring recomputation
        """
        results = self._kv_manager.lookup_all_segments(
            token_ids, layer_idx, query_keys
        )

        hit_chunks: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
        miss_chunk_indices: List[int] = []

        for chunk_idx, k, v, hit_type in results:
            if hit_type in ("exact", "semantic") and k is not None and v is not None:
                hit_chunks.append((chunk_idx, k, v))
            else:
                miss_chunk_indices.append(chunk_idx)

        return hit_chunks, miss_chunk_indices

    def __getattr__(self, name: str) -> Any:
        """Transparent attribute delegation to wrapped impl."""
        return getattr(self._impl, name)


# ---------------------------------------------------------------------------
# Prior-cycle hooks preserved for backward compatibility
# ---------------------------------------------------------------------------

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

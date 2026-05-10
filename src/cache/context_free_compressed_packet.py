"""ContextFreeCompressedKVPacket — B+C integrated cache.

Combines KVPacketSoftAdapterCache (Activity B) with VQCodec (Activity C).
Stores (adapter_state_dict FP16, kv_vq_codes, kv_recent_fp16) 3-tuple per packet.
Reuse: VQ decode → concat recent FP16 → adapter.adapt() → context-independent attention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from src.cache.base import CacheStore
from src.cache.kv_packet_adapter import KVPacketSoftAdapterCache, SoftTokenAdapter
from src.compression.vq_codec import VQCodec


@dataclass
class CompressedPacket:
    """Compressed context-free KV packet."""

    doc_id: str
    adapter_state_dict: dict          # SoftTokenAdapter.state_dict() in FP16
    kv_vq_codes: Optional[dict]       # VQCodec.encode() result; None if fully FP16
    kv_recent_fp16: torch.Tensor      # [min(recent_window, n_tokens), 2, n_heads, d_head]
    layer_idx: int
    positions: torch.Tensor           # [n_tokens] int64 full positions
    n_tokens_total: int               # total token count (old + recent)
    embedding: Optional[torch.Tensor] = None  # [d_embed] for similarity search


class ContextFreeCompressedKVPacket(CacheStore):
    """B+C integrated: adapter-wrapped (B) + VQ-compressed (C) non-contiguous cache.

    Storage per packet: (adapter_state_dict FP16, kv_vq_codes, kv_recent_fp16).
    Reuse path: VQ decode → concat recent_fp16 → adapter.adapt() → context-free attention.
    Optionally uses TriangleInequalitySegmentIndex as a retrieval backend.
    """

    def __init__(
        self,
        vq_codec: VQCodec,
        n_heads: int,
        d_head: int,
        adapter_rank: int = 8,
        max_packets: int = 512,
        recent_window: int = 64,
        segment_index: Optional[object] = None,  # TriangleInequalitySegmentIndex
    ) -> None:
        self.vq_codec = vq_codec
        self.n_heads = n_heads
        self.d_head = d_head
        self.adapter_rank = adapter_rank
        self.max_packets = max_packets
        self.recent_window = recent_window
        self.segment_index = segment_index
        self._store: Dict[str, CompressedPacket] = {}
        self._lru: List[str] = []
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order: List[str] = []

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store kv_block using default positions [0..n_tokens-1] at layer 0."""
        positions = torch.arange(value.shape[0], dtype=torch.long)
        self.put_compressed(key, value, positions, layer_idx=0)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return VQ-decoded + adapter-adapted KV block, or None on miss."""
        return self.get_decompressed(key)

    def evict(self) -> int:
        """LRU eviction. Returns approximate bytes freed."""
        if not self._lru:
            return 0
        oldest = self._lru.pop(0)
        if oldest not in self._store:
            return 0
        packet = self._store.pop(oldest)
        # Estimate freed bytes: recent_fp16 + vq_codes (rough estimate)
        freed = packet.kv_recent_fp16.nbytes
        if packet.kv_vq_codes is not None:
            freed += (
                packet.kv_vq_codes["key_codes"].nbytes
                + packet.kv_vq_codes["val_codes"].nbytes
            )
        return freed

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        total = 0
        for packet in self._store.values():
            total += packet.kv_recent_fp16.nbytes
            if packet.kv_vq_codes is not None:
                total += packet.kv_vq_codes["key_codes"].nbytes
                total += packet.kv_vq_codes["val_codes"].nbytes
            # Adapter state dict bytes
            for v in packet.adapter_state_dict.values():
                if isinstance(v, torch.Tensor):
                    total += v.nbytes
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order.clear()

    # ------------------------------------------------------------------ #
    # B+C integrated API                                                  #
    # ------------------------------------------------------------------ #

    def put_compressed(
        self,
        doc_id: str,
        kv_block: torch.Tensor,      # [n_tokens, 2, n_heads, d_head]
        positions: torch.Tensor,     # [n_tokens] int64
        layer_idx: int,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        """Store packet with VQ compression on old tokens and FP16 on recent tokens.

        1. kv_block[-recent_window:] → FP16 (kv_recent_fp16)
        2. kv_block[:-recent_window] → VQCodec.encode() (if codebook fitted)
        3. Init SoftTokenAdapter; store state_dict in FP16
        4. Create and store CompressedPacket
        5. Register embedding in segment_index if provided
        """
        if doc_id in self._store:
            # Move to most-recent in LRU
            if doc_id in self._lru:
                self._lru.remove(doc_id)
            self._lru.append(doc_id)
            return

        if len(self._store) >= self.max_packets:
            self.evict()

        n_tokens = kv_block.shape[0]
        recent_w = min(self.recent_window, n_tokens)

        kv_recent_fp16 = kv_block[-recent_w:].to(torch.float16).detach().clone()

        kv_vq_codes: Optional[dict] = None
        n_old = n_tokens - recent_w
        if n_old > 0 and layer_idx in self.vq_codec.key_codebooks:
            kv_old = kv_block[:n_old]
            pos_old = positions[:n_old]
            kv_vq_codes = self.vq_codec.encode(kv_old.to(torch.float16), layer_idx, pos_old)
        elif n_old > 0:
            # Codec not fitted yet: keep old tokens as FP16 too
            extra = kv_block[:n_old].to(torch.float16).detach().clone()
            kv_recent_fp16 = torch.cat([extra, kv_recent_fp16], dim=0)

        # Create adapter and store its state dict in FP16
        adapter = SoftTokenAdapter(self.n_heads, self.d_head, self.adapter_rank)
        state_dict_fp16 = {
            k: v.to(torch.float16).detach().clone()
            for k, v in adapter.state_dict().items()
        }

        packet = CompressedPacket(
            doc_id=doc_id,
            adapter_state_dict=state_dict_fp16,
            kv_vq_codes=kv_vq_codes,
            kv_recent_fp16=kv_recent_fp16,
            layer_idx=layer_idx,
            positions=positions.clone(),
            n_tokens_total=n_tokens,
            embedding=embedding,
        )
        self._store[doc_id] = packet
        self._lru.append(doc_id)

        if self.segment_index is not None and embedding is not None:
            try:
                self.segment_index.add(doc_id, embedding)
            except Exception:
                pass

    def get_decompressed(
        self,
        doc_id: str,
    ) -> Optional[torch.Tensor]:
        """VQ decode → concat recent_fp16 → adapter.adapt() → [rank+n_tokens, 2, n_heads, d_head]."""
        if doc_id not in self._store:
            self._misses += 1
            return None

        self._hits += 1
        # Track non-contiguous access
        if self._access_order:
            prev = self._access_order[-1]
            if prev != doc_id:
                lru_prev = self._lru.index(prev) if prev in self._lru else -1
                lru_curr = self._lru.index(doc_id) if doc_id in self._lru else -1
                if abs(lru_curr - lru_prev) != 1:
                    self._noncontiguous_hits += 1
            else:
                # Same key accessed consecutively is contiguous
                pass
        self._access_order.append(doc_id)

        # Move to MRU position
        if doc_id in self._lru:
            self._lru.remove(doc_id)
        self._lru.append(doc_id)

        packet = self._store[doc_id]

        # Reconstruct full KV block
        if packet.kv_vq_codes is not None:
            kv_old = self.vq_codec.decode(packet.kv_vq_codes, packet.layer_idx)
            kv_full = torch.cat([kv_old, packet.kv_recent_fp16], dim=0)
        else:
            kv_full = packet.kv_recent_fp16

        # Restore adapter from state dict and apply
        adapter = SoftTokenAdapter(self.n_heads, self.d_head, self.adapter_rank)
        adapter.load_state_dict(
            {k: v.float() for k, v in packet.adapter_state_dict.items()}
        )
        with torch.no_grad():
            return adapter.adapt(kv_full)

    def search_similar(
        self,
        query_embedding: torch.Tensor,  # [d_embed]
        top_k: int = 5,
    ) -> List[str]:
        """Delegate to segment_index if available, else return empty list."""
        if self.segment_index is None:
            return []
        try:
            return self.segment_index.search(query_embedding, top_k)
        except Exception:
            return []

    def pack_packets(
        self,
        doc_ids: List[str],
    ) -> Optional[torch.Tensor]:
        """Concatenate multiple decompressed packets without recomputation.

        Returns [sum(rank + n_tokens_i), 2, n_heads, d_head] or None if any miss.
        """
        parts = []
        for doc_id in doc_ids:
            adapted = self.get_decompressed(doc_id)
            if adapted is None:
                return None
            parts.append(adapted)
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that came from non-contiguous access patterns."""
        if self._hits == 0:
            return 0.0
        return self._noncontiguous_hits / self._hits

    def compression_ratio(self) -> float:
        """Effective compression ratio across all stored packets.

        ratio = bytes_saved / total_original_bytes
        """
        if not self._store:
            return 0.0

        total_orig = 0
        total_stored = 0
        fp16_bytes_per_element = 2  # float16

        for packet in self._store.values():
            n = packet.n_tokens_total
            orig_bytes = n * 2 * self.n_heads * self.d_head * fp16_bytes_per_element
            total_orig += orig_bytes

            stored = packet.kv_recent_fp16.nbytes
            if packet.kv_vq_codes is not None:
                stored += packet.kv_vq_codes["key_codes"].nbytes
                stored += packet.kv_vq_codes["val_codes"].nbytes
            total_stored += stored

        if total_orig == 0:
            return 0.0
        return max(0.0, 1.0 - total_stored / total_orig)

"""KVPacketSoftAdapterCache — context-free non-contiguous KV packet cache.

Based on KV Packet (arXiv 2604.13226). Soft-token adapters prepended to each
KV block eliminate context dependency, enabling recomputation-free reuse.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.base import CacheStore


class SoftTokenAdapter(nn.Module):
    """Lightweight soft-token adapter prepended to a KV block.

    Parameters: 2 * n_heads * rank * d_head * 2 bytes (FP16).
    With rank=8, n_heads=8, d_head=128: ~32 KB vs 2 MB for 512-token block (1.6%).
    """

    def __init__(self, n_heads: int, d_head: int, rank: int = 8) -> None:
        super().__init__()
        self.rank = rank
        self.n_heads = n_heads
        self.d_head = d_head
        # [rank, n_heads, d_head]
        self.soft_key = nn.Parameter(torch.zeros(rank, n_heads, d_head))
        self.soft_val = nn.Parameter(torch.zeros(rank, n_heads, d_head))
        nn.init.normal_(self.soft_key, std=0.02)
        nn.init.normal_(self.soft_val, std=0.02)

    def adapt(
        self,
        kv_block: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """Prepend soft adapter tokens to the KV block.

        Returns [rank + n_tokens, 2, n_heads, d_head].
        """
        # soft_tokens: [rank, 2, n_heads, d_head]
        soft_tokens = torch.stack([self.soft_key, self.soft_val], dim=1)
        return torch.cat([soft_tokens, kv_block], dim=0)


@dataclass
class KVPacket:
    """Context-independent KV packet wrapping a document KV block."""

    doc_id: str
    kv_block: torch.Tensor          # [n_tokens, 2, n_heads, d_head] float16
    adapter: SoftTokenAdapter       # lightweight adapter
    embedding: Optional[torch.Tensor] = None  # [d_embed] for similarity search


class KVPacketSoftAdapterCache(CacheStore):
    """KV Packet (arXiv 2604.13226) based recomputation-free non-contiguous cache.

    put(key, value): key=doc_id, value=kv_block [n_tokens, 2, n_heads, d_head]
    get(key): returns adapter.adapt(kv_block) = [rank+n_tokens, 2, n_heads, d_head]
    """

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        adapter_rank: int = 8,
        max_packets: int = 512,
        embedding_dim: int = 64,
    ) -> None:
        self.n_heads = n_heads
        self.d_head = d_head
        self.adapter_rank = adapter_rank
        self.max_packets = max_packets
        self.embedding_dim = embedding_dim
        self._store: OrderedDict[str, KVPacket] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order: List[str] = []
        # Maintains doc insertion order (not mutated by LRU moves) for non-contiguous detection
        self._insertion_order: List[str] = []

    # ------------------------------------------------------------------ #
    # CacheStore interface                                                 #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Wrap kv_block in a packet and store. Evicts when at capacity."""
        if key in self._store:
            self._store.move_to_end(key)
            return
        if len(self._store) >= self.max_packets:
            self.evict()
        packet = self.create_packet(key, value)
        self._store[key] = packet
        self._insertion_order.append(key)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Return adapter-adapted KV block, or None on miss."""
        if key not in self._store:
            self._misses += 1
            return None

        self._store.move_to_end(key)
        self._hits += 1

        # Track non-contiguous access using stable insertion order (not LRU order).
        # A hit is non-contiguous when the current and previous accessed keys are
        # not adjacent in insertion order, signalling a "skip" across segments.
        if self._access_order:
            prev_key = self._access_order[-1]
            if (
                key in self._insertion_order
                and prev_key in self._insertion_order
            ):
                prev_pos = self._insertion_order.index(prev_key)
                curr_pos = self._insertion_order.index(key)
                if abs(curr_pos - prev_pos) != 1:
                    self._noncontiguous_hits += 1
            else:
                self._noncontiguous_hits += 1
        self._access_order.append(key)

        packet = self._store[key]
        with torch.no_grad():
            return packet.adapter.adapt(packet.kv_block)

    def evict(self) -> int:
        """LRU eviction. Returns bytes freed."""
        if not self._store:
            return 0
        key, packet = next(iter(self._store.items()))
        self._store.pop(key)
        if key in self._insertion_order:
            self._insertion_order.remove(key)
        freed = packet.kv_block.nbytes
        for p in packet.adapter.parameters():
            freed += p.nbytes
        return freed

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        total = 0
        for packet in self._store.values():
            total += packet.kv_block.nbytes
            for p in packet.adapter.parameters():
                total += p.nbytes
        return total

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._noncontiguous_hits = 0
        self._access_order.clear()
        # Note: _insertion_order is not reset — it tracks structural order, not hit stats

    # ------------------------------------------------------------------ #
    # Packet-level API                                                     #
    # ------------------------------------------------------------------ #

    def create_packet(
        self,
        doc_id: str,
        kv_block: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
        embedding: Optional[torch.Tensor] = None,
    ) -> KVPacket:
        """Create a KVPacket with a new SoftTokenAdapter. Does not store it."""
        adapter = SoftTokenAdapter(self.n_heads, self.d_head, self.adapter_rank)
        return KVPacket(
            doc_id=doc_id,
            kv_block=kv_block.detach().clone(),
            adapter=adapter,
            embedding=embedding,
        )

    def train_adapter(
        self,
        packet: KVPacket,
        context_kvs: List[torch.Tensor],  # list of [n_tokens, 2, n_heads, d_head]
        n_steps: int = 1000,
        lr: float = 1e-3,
    ) -> None:
        """Self-supervised distillation to adapt the packet to diverse contexts.

        Minimizes MSE between adapter output (excluding soft tokens) and target KV.
        Updates packet.adapter in-place.
        """
        if not context_kvs:
            return
        optimizer = torch.optim.Adam(packet.adapter.parameters(), lr=lr)
        rank = packet.adapter.rank

        for step in range(n_steps):
            target = random.choice(context_kvs)
            # Adapt and strip soft prefix tokens
            adapted = packet.adapter.adapt(packet.kv_block)
            pred = adapted[rank:]  # [n_tokens, 2, n_heads, d_head]
            # Align shapes in case target has different n_tokens
            min_t = min(pred.shape[0], target.shape[0])
            loss = F.mse_loss(pred[:min_t].float(), target[:min_t].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def pack(
        self,
        doc_ids: List[str],
    ) -> Optional[torch.Tensor]:
        """Concatenate multiple packets without recomputation.

        Returns [sum(rank + n_tokens_i), 2, n_heads, d_head] or None if any miss.
        """
        parts = []
        for doc_id in doc_ids:
            if doc_id not in self._store:
                return None
            packet = self._store[doc_id]
            with torch.no_grad():
                parts.append(packet.adapter.adapt(packet.kv_block))
        if not parts:
            return None
        return torch.cat(parts, dim=0)

    def noncontiguous_hit_rate(self) -> float:
        """Fraction of hits that came from non-contiguous access patterns."""
        if self._hits == 0:
            return 0.0
        return self._noncontiguous_hits / self._hits

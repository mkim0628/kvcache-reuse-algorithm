"""Activity C: LookaheadKVEvictionCodec — Draft-free future-aware KV eviction.

Based on LookaheadKV (arXiv 2603.10899, ICLR 2026, Samsung AI Research).
Learnable lookahead tokens + lightweight LoRA adapter predict future attention
patterns without draft generation. Kept KV is FP16 original (no quant distortion).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.cache.base import CacheStore


@dataclass
class LookaheadKVConfig:
    n_layers: int = 12
    n_heads: int = 8
    d_head: int = 64
    n_lookahead: int = 5
    lora_rank: int = 8
    eviction_ratio: float = 0.7
    blend_ratio: float = 0.0
    recent_window: int = 4
    seed: int = 42
    max_entries: int = 1000


class LookaheadModule(nn.Module):
    """Learnable lookahead tokens + LoRA adapter for future attention prediction.

    Each layer and head has independent lookahead query parameters.
    Only this module is trained; model weights remain frozen.
    """

    def __init__(self, config: LookaheadKVConfig) -> None:
        super().__init__()
        self.config = config
        # [n_layers, n_heads, n_la, d_head]
        self.lookahead_tokens = nn.Parameter(
            torch.randn(
                config.n_layers,
                config.n_heads,
                config.n_lookahead,
                config.d_head,
            ) * 0.02
        )
        # LoRA: A:[n_layers, d_head, lora_rank], B:[n_layers, lora_rank, d_head]
        self.lora_A = nn.Parameter(
            torch.randn(config.n_layers, config.d_head, config.lora_rank) * 0.02
        )
        self.lora_B = nn.Parameter(
            torch.zeros(config.n_layers, config.lora_rank, config.d_head)
        )

    def forward(
        self,
        key: torch.Tensor,  # [n_tokens, n_heads, d_head]
        layer_idx: int,
    ) -> torch.Tensor:
        """Compute per-token future-reference importance via lookahead attention.

        Algorithm:
          1. Apply LoRA correction: la_q = lookahead_tokens[layer] + delta(lora_A, lora_B)
          2. Compute attention scores [n_heads, n_la, n_tokens]
          3. Reduce max over (n_la, n_heads) → importance [n_tokens]

        Returns:
            importance: Tensor[n_tokens] — higher means future responses attend more
        """
        n_tokens, n_heads, d_head = key.shape
        la_q_base = self.lookahead_tokens[layer_idx]  # [n_heads, n_la, d_head]

        # LoRA delta applied to lookahead queries
        la_q_flat = la_q_base.reshape(n_heads * self.config.n_lookahead, d_head)
        lora_delta = la_q_flat @ self.lora_A[layer_idx] @ self.lora_B[layer_idx]
        la_q = la_q_base + lora_delta.reshape(n_heads, self.config.n_lookahead, d_head)

        scale = d_head ** -0.5
        k_t = key.permute(1, 2, 0)  # [n_heads, d_head, n_tokens]
        scores = torch.bmm(la_q, k_t) * scale  # [n_heads, n_la, n_tokens]

        # Token importance = max score across all heads and lookahead queries
        importance = scores.max(dim=1).values.max(dim=0).values  # [n_tokens]
        return importance


class LookaheadKVEvictionCodec(CacheStore):
    """Draft-free future-aware KV eviction using learnable lookahead tokens + LoRA.

    Activity C: KV Cache Compression via token eviction.
    Kept KV is FP16 original — zero quantization distortion.
    eviction_ratio fraction of low-importance tokens are removed.

    CacheStore interface fully implemented:
      put / get / evict / hit_rate / memory_bytes / reset_stats
    """

    def __init__(self, config: LookaheadKVConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self._lookahead = LookaheadModule(config)
        self._lookahead.eval()
        self._store: OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._total_tokens_original = 0
        self._total_tokens_kept = 0

    # ------------------------------------------------------------------ #
    # CacheStore abstract methods                                          #
    # ------------------------------------------------------------------ #

    def put(self, key: str, value: torch.Tensor) -> None:
        """Store KV after LookaheadKV eviction.

        value shape: [n_tokens, 2, n_heads, d_head]
        """
        compressed = self.compression_hook(key, value)
        if len(self._store) >= self.config.max_entries:
            self.evict()
        if key in self._store:
            self._store.move_to_end(key)
        keep_mask = torch.ones(compressed.shape[0], dtype=torch.bool)
        self._store[key] = (compressed, keep_mask)

    def get(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve evicted KV (only kept tokens are returned)."""
        if key not in self._store:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        kv, _ = self._store[key]
        return kv

    def evict(self) -> int:
        """LRU eviction. Returns freed bytes."""
        if not self._store:
            return 0
        _, (kv, _) = self._store.popitem(last=False)
        return kv.nbytes

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        return sum(kv.nbytes for kv, _ in self._store.values())

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._total_tokens_original = 0
        self._total_tokens_kept = 0

    # ------------------------------------------------------------------ #
    # Activity C core: compression_hook override                          #
    # ------------------------------------------------------------------ #

    def compression_hook(
        self,
        key: str,
        value: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
    ) -> torch.Tensor:
        """LookaheadKV eviction: remove low-importance eviction_ratio% tokens.

        Algorithm:
          1. Extract layer_idx from key format "layer{i}:..." (default 0)
          2. Compute K from value tensor for lookahead attention
          3. LookaheadModule.forward(K, layer_idx) → importance [n_tokens]
          4. Protect recent_window tokens (set importance = +inf)
          5. Mask bottom eviction_ratio fraction → keep_mask
          6. Return value[keep_mask] (kept_tokens, 2, n_heads, d_head)

        Returns:
            Tensor[kept_tokens, 2, n_heads, d_head] — eviction-filtered KV
        """
        if value.dim() < 4:
            return value

        n_tokens = value.shape[0]

        # Extract layer_idx from key if formatted as "layer{i}:..."
        layer_idx = 0
        if key.startswith("layer") and ":" in key:
            try:
                layer_idx = int(key.split(":")[0][5:]) % self.config.n_layers
            except (ValueError, IndexError):
                layer_idx = 0

        # Extract K: [n_tokens, n_heads, d_head]
        if value.dim() == 5:  # [n_tokens, n_layers, 2, n_heads, d_head]
            k = value[:, 0, 0, :, :]
        else:  # [n_tokens, 2, n_heads, d_head]
            k = value[:, 0, :, :]  # [n_tokens, n_heads, d_head]

        with torch.no_grad():
            importance = self._lookahead.forward(k, layer_idx=layer_idx)

        # Always preserve the most recent recent_window tokens
        if self.config.recent_window > 0 and n_tokens > 0:
            rw = min(self.config.recent_window, n_tokens)
            importance[-rw:] = float("inf")

        # Determine how many tokens to keep (at least 1, at most n_tokens)
        n_evict = int(n_tokens * self.config.eviction_ratio)
        # Clamp so we always keep at least recent_window or 1 token
        min_keep = max(1, min(self.config.recent_window, n_tokens))
        n_keep = max(min_keep, n_tokens - n_evict)

        # Keep the top-n_keep tokens by importance
        if n_keep >= n_tokens:
            kept = value.detach().clone()
        else:
            # torch.topk returns indices of top-k values
            _, keep_indices = torch.topk(importance, k=n_keep, sorted=False)
            keep_indices, _ = keep_indices.sort()
            kept = value[keep_indices].detach().clone()

        self._total_tokens_original += n_tokens
        self._total_tokens_kept += kept.shape[0]
        return kept

    # ------------------------------------------------------------------ #
    # Eviction metrics                                                     #
    # ------------------------------------------------------------------ #

    def eviction_rate(self) -> float:
        """Actual eviction rate = (original - kept) / original."""
        if self._total_tokens_original == 0:
            return 0.0
        return 1.0 - self._total_tokens_kept / self._total_tokens_original

    def memory_reduction_ratio(self) -> float:
        """Memory reduction vs FP16 baseline (0.7 eviction → 0.7 reduction)."""
        return self.eviction_rate()

    # ------------------------------------------------------------------ #
    # Training support                                                     #
    # ------------------------------------------------------------------ #

    def train_lookahead(
        self,
        calibration_data: List[torch.Tensor],  # List of [n_tokens, 2, n_heads, d_head]
        layer_idx: int,
        n_epochs: int = 5,
        lr: float = 1e-3,
    ) -> Dict[str, float]:
        """Fine-tune lookahead tokens + LoRA adapter on calibration data.

        Objective: align lookahead attention scores with actual future attention patterns.
        Loss: MSE(lookahead_score_i, normalized_future_attention_score_i)

        Returns: {"final_loss": float, "n_samples": int}
        """
        self._lookahead.train()
        optimizer = torch.optim.Adam(self._lookahead.parameters(), lr=lr)
        total_loss = 0.0
        n_samples = len(calibration_data)

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for kv in calibration_data:
                # K: [n_tokens, n_heads, d_head]
                if kv.dim() == 4:
                    k = kv[:, 0, :, :]
                    v = kv[:, 1, :, :]
                else:
                    continue

                n_tokens = k.shape[0]
                if n_tokens < 2:
                    continue

                # Predicted importance from lookahead
                pred_importance = self._lookahead.forward(k, layer_idx=layer_idx)

                # Target: actual attention score of each token against next token query
                # Use last token as a proxy for future query
                q_future = k[-1:, 0, :]  # [1, d_head] — proxy future query
                scale = k.shape[-1] ** -0.5
                with torch.no_grad():
                    target_scores = (q_future @ k[:, 0, :].T * scale).squeeze(0)
                    target_scores = F.softmax(target_scores, dim=-1)

                pred_norm = F.softmax(pred_importance, dim=-1)
                loss = F.mse_loss(pred_norm, target_scores)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            total_loss = epoch_loss / max(n_samples, 1)

        self._lookahead.eval()
        return {"final_loss": total_loss, "n_samples": n_samples}

    def save(self, path: str) -> None:
        torch.save(
            {
                "lookahead_state_dict": self._lookahead.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._lookahead.load_state_dict(data["lookahead_state_dict"])
        self.config = data["config"]

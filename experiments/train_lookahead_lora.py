"""LookaheadKVEvictionCodec — lookahead token + LoRA adapter training script.

Trains only the LookaheadModule (lookahead_tokens + lora_A + lora_B) on
synthetic calibration sequences. Model weights are never touched. Target
training time: ≤ 1 GPU-hour on 500–1000 samples.

Training objective:
    Align lookahead attention scores with actual "future" attention patterns.
    Loss: MSE(softmax(lookahead_scores), softmax(future_attention_scores))

Usage::

    python experiments/train_lookahead_lora.py \\
        --n_samples 500 --n_epochs 5 --lr 1e-3 \\
        --n_lookahead 5 --lora_rank 8 \\
        --output configs/lookahead_lora_weights.pt --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.cache.lookahead_kv_eviction import LookaheadKVConfig, LookaheadKVEvictionCodec


# ---------------------------------------------------------------------------
# Synthetic calibration data generation
# ---------------------------------------------------------------------------

def _generate_calibration_sample(
    n_tokens: int,
    n_heads: int,
    d_head: int,
    seed: int,
) -> torch.Tensor:
    """Generate one synthetic KV sample: [n_tokens, 2, n_heads, d_head]."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(n_tokens, 2, n_heads, d_head, generator=g)


def _generate_calibration_data(
    n_samples: int,
    n_tokens: int,
    n_heads: int,
    d_head: int,
    seed: int,
) -> List[torch.Tensor]:
    return [
        _generate_calibration_sample(n_tokens, n_heads, d_head, seed + i)
        for i in range(n_samples)
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    n_samples: int = 500,
    n_epochs: int = 5,
    lr: float = 1e-3,
    n_layers: int = 12,
    n_heads: int = 8,
    d_head: int = 64,
    n_lookahead: int = 5,
    lora_rank: int = 8,
    n_tokens: int = 64,
    output_path: str = "configs/lookahead_lora_weights.pt",
    seed: int = 42,
) -> Dict[str, float]:
    """Train lookahead tokens + LoRA adapter on synthetic calibration data.

    Returns:
        {"final_loss": float, "n_samples": int, "training_time_sec": float}
    """
    torch.manual_seed(seed)

    config = LookaheadKVConfig(
        n_layers=n_layers,
        n_heads=n_heads,
        d_head=d_head,
        n_lookahead=n_lookahead,
        lora_rank=lora_rank,
        seed=seed,
    )
    codec = LookaheadKVEvictionCodec(config)

    # Generate one synthetic calibration dataset (shared across all layers)
    calibration_data = _generate_calibration_data(
        n_samples, n_tokens, n_heads, d_head, seed
    )

    t0 = time.monotonic()
    final_loss = 0.0

    # Train each layer independently (weight-sharing between layers not assumed)
    for layer_idx in range(n_layers):
        result = codec.train_lookahead(
            calibration_data=calibration_data,
            layer_idx=layer_idx,
            n_epochs=n_epochs,
            lr=lr,
        )
        final_loss += result["final_loss"]

    final_loss /= max(n_layers, 1)
    training_time = time.monotonic() - t0

    # Save trained weights
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    codec.save(str(out))

    print(f"Training complete: final_loss={final_loss:.6f}, "
          f"n_samples={n_samples}, time={training_time:.1f}s")
    print(f"Weights saved to: {out}")

    return {
        "final_loss": final_loss,
        "n_samples": n_samples,
        "training_time_sec": training_time,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LookaheadKVEvictionCodec LoRA adapter")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--n_lookahead", type=int, default=5)
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--n_tokens", type=int, default=64)
    p.add_argument("--output", type=str, default="configs/lookahead_lora_weights.pt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        n_samples=args.n_samples,
        n_epochs=args.n_epochs,
        lr=args.lr,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        n_lookahead=args.n_lookahead,
        lora_rank=args.lora_rank,
        n_tokens=args.n_tokens,
        output_path=args.output,
        seed=args.seed,
    )

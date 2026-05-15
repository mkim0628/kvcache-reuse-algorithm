"""RelayUShapeLayerSelectiveSegmentCache layer-range profiler.

Generates 100–200 "similar but non-identical" segment pairs via token-ID
perturbation, computes per-layer KV cosine similarity using synthetic
attention tensors, and saves the U-shape layer reuse profile to YAML.

No real model API calls are required — KV tensors are synthesised from the
token IDs so that the profiler can run offline in < 1 min on a laptop CPU.

Usage::

    python experiments/run_relay_layer_calibration.py \\
        --n_pairs 100 --n_layers 12 --tau_layer 0.95 \\
        --output configs/relay_ulayer_profile.yaml --seed 42
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# Synthetic KV generation
# ---------------------------------------------------------------------------

def _token_ids_to_kv(
    token_ids: List[int],
    layer_idx: int,
    n_heads: int,
    d_head: int,
    chunk_size: int,
) -> torch.Tensor:
    """Deterministically generate a synthetic KV tensor from token IDs.

    Each token is embedded by seeding from (token_id XOR layer_idx * 1337),
    producing a reproducible [n_tokens, 2, n_heads, d_head] FP32 tensor that
    preserves relative structure across perturbed sequences.
    """
    n_tokens = min(len(token_ids), chunk_size)
    kv = torch.zeros(n_tokens, 2, n_heads, d_head)
    for t_idx, tid in enumerate(token_ids[:n_tokens]):
        seed = (tid ^ (layer_idx * 1337)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(seed)
        kv[t_idx] = torch.randn(2, n_heads, d_head, generator=g)
    return kv


# ---------------------------------------------------------------------------
# Cosine similarity aggregation
# ---------------------------------------------------------------------------

def _layer_cosine_similarity(
    kv_base: torch.Tensor,       # [n_tokens, 2, n_heads, d_head]
    kv_perturbed: torch.Tensor,  # [n_tokens, 2, n_heads, d_head]
) -> float:
    """Mean cosine similarity between base and perturbed KV across tokens and heads."""
    # Flatten to [n_tokens * n_heads, d_head] per K/V separately
    n_tokens, _, n_heads, d_head = kv_base.shape
    k_base = kv_base[:, 0, :, :].reshape(n_tokens * n_heads, d_head)
    k_pert = kv_perturbed[:, 0, :, :].reshape(n_tokens * n_heads, d_head)
    v_base = kv_base[:, 1, :, :].reshape(n_tokens * n_heads, d_head)
    v_pert = kv_perturbed[:, 1, :, :].reshape(n_tokens * n_heads, d_head)

    cos_k = F.cosine_similarity(k_base, k_pert, dim=-1).mean().item()
    cos_v = F.cosine_similarity(v_base, v_pert, dim=-1).mean().item()
    return (cos_k + cos_v) / 2.0


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def run_calibration(
    n_pairs: int = 100,
    n_layers: int = 12,
    n_heads: int = 8,
    d_head: int = 64,
    chunk_size: int = 128,
    tau_layer: float = 0.95,
    perturbation_ratio: float = 0.1,
    output_path: str = "configs/relay_ulayer_profile.yaml",
    seed: int = 42,
) -> None:
    """Run offline layer calibration and save profile to YAML.

    Args:
        n_pairs: number of (base, perturbed) segment pairs
        n_layers: total model layers
        n_heads: attention heads per layer
        d_head: head dimension
        chunk_size: tokens per segment
        tau_layer: cosine similarity threshold for marking a layer reusable
        perturbation_ratio: fraction of token IDs randomly replaced per pair
        output_path: destination YAML file path
        seed: random seed for reproducibility
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    # Vocabulary size for synthetic token IDs
    vocab_size = 32000
    n_perturb = max(1, int(chunk_size * perturbation_ratio))

    layer_sims: List[List[float]] = [[] for _ in range(n_layers)]

    for pair_idx in range(n_pairs):
        # Base segment: random token IDs
        base_ids = [rng.randint(0, vocab_size - 1) for _ in range(chunk_size)]
        # Perturbed segment: replace n_perturb positions
        perturbed_ids = list(base_ids)
        positions = rng.sample(range(chunk_size), n_perturb)
        for pos in positions:
            perturbed_ids[pos] = rng.randint(0, vocab_size - 1)

        for layer_idx in range(n_layers):
            kv_base = _token_ids_to_kv(base_ids, layer_idx, n_heads, d_head, chunk_size)
            kv_pert = _token_ids_to_kv(perturbed_ids, layer_idx, n_heads, d_head, chunk_size)
            sim = _layer_cosine_similarity(kv_base, kv_pert)
            layer_sims[layer_idx].append(sim)

    # Compute mean similarity per layer
    similarity_scores = [
        sum(sims) / len(sims) if sims else 0.0
        for sims in layer_sims
    ]

    reuse_layer_indices = [
        i for i, s in enumerate(similarity_scores) if s >= tau_layer
    ]
    boundary_layer_indices = [
        i for i in range(n_layers) if i not in set(reuse_layer_indices)
    ]

    profile: Dict = {
        "n_layers": n_layers,
        "reuse_layer_indices": reuse_layer_indices,
        "boundary_layer_indices": boundary_layer_indices,
        "similarity_scores": [round(s, 6) for s in similarity_scores],
        "tau_layer": tau_layer,
        "n_pairs": n_pairs,
        "perturbation_ratio": perturbation_ratio,
        "seed": seed,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(profile, f, default_flow_style=False, sort_keys=False)

    reuse_pct = 100.0 * len(reuse_layer_indices) / n_layers
    print(
        f"Calibration complete: {len(reuse_layer_indices)}/{n_layers} layers reusable "
        f"({reuse_pct:.1f}%) at tau_layer={tau_layer}"
    )
    print(f"Profile saved to: {out}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relay U-shape layer calibration")
    p.add_argument("--n_pairs", type=int, default=100)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--d_head", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--tau_layer", type=float, default=0.95)
    p.add_argument("--perturbation_ratio", type=float, default=0.1)
    p.add_argument("--output", type=str, default="configs/relay_ulayer_profile.yaml")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_calibration(
        n_pairs=args.n_pairs,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_head=args.d_head,
        chunk_size=args.chunk_size,
        tau_layer=args.tau_layer,
        perturbation_ratio=args.perturbation_ratio,
        output_path=args.output,
        seed=args.seed,
    )

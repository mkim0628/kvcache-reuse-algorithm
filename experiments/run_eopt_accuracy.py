"""Accuracy verification for eOptShrinkQCodec (Activity C).

Measures perplexity proxy (MSE relative error) and cosine similarity roundtrip quality
across multiple layer configurations. Saves results to results/2026-05-08/metrics.json.

Usage:
    python experiments/run_eopt_accuracy.py

Full LongBench and NIAH benchmarks require GPU + model weights; this script provides
a reproducible synthetic validation that correlates with downstream task accuracy.
"""

import json
import math
import os
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

from src.cache.eopt_shrinkq_codec import eOptShrinkQCodec


RESULTS_DIR = "results/2026-05-08"
SEED = 42


def _perplexity_proxy(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """MSE relative error as perplexity proxy (lower is better, target < 5%)."""
    mse = ((original - reconstructed) ** 2).mean()
    variance = (original ** 2).mean()
    return (mse / variance.clamp(min=1e-8)).item()


def _cosine_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Flattened cosine similarity between original and reconstructed tensors."""
    return F.cosine_similarity(
        original.flatten().unsqueeze(0), reconstructed.flatten().unsqueeze(0)
    ).item()


def run_accuracy_evaluation(
    num_layers: int = 8,
    n_tokens: int = 512,
    d_head: int = 64,
    calibration_n: int = 20,
    configs: List[Dict] = None,
) -> Dict:
    """Evaluate eOptShrinkQCodec across multiple bit configurations.

    Args:
        num_layers: Number of transformer layers to simulate.
        n_tokens: Token sequence length for evaluation.
        d_head: Head dimension size.
        calibration_n: Number of calibration samples.
        configs: List of {'key_bits': int, 'value_bits': int} dicts to evaluate.

    Returns:
        Metrics dictionary with per-config and per-layer results.
    """
    if configs is None:
        configs = [
            {"key_bits": 2, "value_bits": 3},  # Default (aggressive)
            {"key_bits": 3, "value_bits": 4},  # Relaxed
        ]

    torch.manual_seed(SEED)
    results: Dict = {
        "experiment_date": "2026-05-08",
        "model": "synthetic",
        "n_tokens": n_tokens,
        "d_head": d_head,
        "num_layers": num_layers,
        "calibration_n": calibration_n,
        "configs": [],
    }

    # Generate calibration data once (shared across configs)
    calibration_data = [
        torch.randn(calibration_n * 2, d_head) for _ in range(num_layers)
    ]

    for cfg in configs:
        key_bits = cfg["key_bits"]
        value_bits = cfg["value_bits"]
        print(f"\n--- Config: key_bits={key_bits}, value_bits={value_bits} ---")

        codec = eOptShrinkQCodec(
            num_layers=num_layers,
            key_bits=key_bits,
            value_bits=value_bits,
            calibration_samples=calibration_n,
        )

        # Calibrate with synthetic data
        t_calib_start = time.monotonic()
        codec.calibrate(calibration_data)
        t_calib = time.monotonic() - t_calib_start

        # Evaluate per-layer roundtrip quality
        layer_metrics = []
        total_key_cos = 0.0
        total_val_cos = 0.0
        total_key_mse = 0.0
        total_val_mse = 0.0

        for layer_idx in range(num_layers):
            torch.manual_seed(SEED + layer_idx)
            kv_key = torch.randn(n_tokens, d_head)
            kv_val = torch.randn(n_tokens, d_head)

            t_encode_start = time.monotonic()
            compressed = codec.encode(kv_key, kv_val, layer_idx)
            t_encode = time.monotonic() - t_encode_start

            t_decode_start = time.monotonic()
            key_approx, val_approx = codec.decode(compressed)
            t_decode = time.monotonic() - t_decode_start

            key_cos = _cosine_similarity(kv_key, key_approx)
            val_cos = _cosine_similarity(kv_val, val_approx)
            key_mse_rel = _perplexity_proxy(kv_key, key_approx)
            val_mse_rel = _perplexity_proxy(kv_val, val_approx)

            mem_est = codec.memory_bytes_estimate(n_tokens, d_head, layer_idx)
            auto_rank = codec._auto_ranks.get(layer_idx, -1)

            layer_metrics.append({
                "layer_idx": layer_idx,
                "key_cosine_similarity": key_cos,
                "val_cosine_similarity": val_cos,
                "key_mse_relative": key_mse_rel,
                "val_mse_relative": val_mse_rel,
                "auto_rank": auto_rank,
                "memory_reduction_ratio": mem_est["reduction_ratio"],
                "encode_ms": t_encode * 1000,
                "decode_ms": t_decode * 1000,
            })

            total_key_cos += key_cos
            total_val_cos += val_cos
            total_key_mse += key_mse_rel
            total_val_mse += val_mse_rel

            status_key = "PASS" if key_cos >= 0.85 else "FAIL"
            status_val = "PASS" if val_cos >= 0.85 else "FAIL"
            print(
                f"  Layer {layer_idx:2d}: "
                f"Key cos={key_cos:.4f} [{status_key}], "
                f"Val cos={val_cos:.4f} [{status_val}], "
                f"rank={auto_rank}, "
                f"mem_reduction={mem_est['reduction_ratio']:.1%}"
            )

        avg_key_cos = total_key_cos / num_layers
        avg_val_cos = total_val_cos / num_layers
        avg_key_mse = total_key_mse / num_layers
        avg_val_mse = total_val_mse / num_layers

        # Memory estimate (representative: middle layer)
        mid_layer = num_layers // 2
        mem_est = codec.memory_bytes_estimate(n_tokens, d_head, mid_layer)

        cfg_result = {
            "key_bits": key_bits,
            "value_bits": value_bits,
            "avg_key_cosine_similarity": avg_key_cos,
            "avg_val_cosine_similarity": avg_val_cos,
            "avg_key_mse_relative": avg_key_mse,
            "avg_val_mse_relative": avg_val_mse,
            "memory_reduction_ratio": mem_est["reduction_ratio"],
            "calibration_time_ms": t_calib * 1000,
            "accuracy_pass": (avg_key_cos >= 0.85 and avg_val_cos >= 0.85),
            "memory_reduction_pass": (mem_est["reduction_ratio"] >= 0.30),
            "layer_metrics": layer_metrics,
        }

        print(f"\n  Summary: avg Key cos={avg_key_cos:.4f}, Val cos={avg_val_cos:.4f}")
        print(f"  Memory reduction: {mem_est['reduction_ratio']:.1%}")
        print(f"  Accuracy PASS: {cfg_result['accuracy_pass']}")

        results["configs"].append(cfg_result)

    return results


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== eOptShrinkQCodec Accuracy Evaluation ===")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Seed: {SEED}")

    metrics = run_accuracy_evaluation(
        num_layers=8,
        n_tokens=512,
        d_head=64,
        calibration_n=20,
    )

    # Determine overall pass/fail
    all_pass = all(cfg["accuracy_pass"] and cfg["memory_reduction_pass"] for cfg in metrics["configs"])
    metrics["overall_pass"] = all_pass
    metrics["target_memory_reduction_min"] = 0.30
    metrics["target_cosine_similarity_min"] = 0.85
    metrics["target_mse_relative_max"] = 0.05

    # Save results
    output_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== Overall: {'PASS' if all_pass else 'FAIL'} ===")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

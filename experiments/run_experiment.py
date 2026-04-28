"""Run B+C experiment and save metrics to results/."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import torch

from src.cache.compression import CompressionCodec
from src.cache.compressed_segment import CompressedSegmentCache
from src.cache.contiguous import ContiguousCache
from src.engine.runner import InferenceRunner, InferenceRequest
from src.utils.prompt_gen import generate_requests


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: str = "configs/experiments/2026-04-28.yaml") -> None:
    cfg = load_config(config_path)
    seed = cfg["experiment"]["seed"]
    torch.manual_seed(seed)

    bench = cfg["benchmark"]
    token_sequences = generate_requests(
        n_requests=bench["num_requests"],
        seq_len=bench["sequence_length"],
        shared_prefix_len=int(bench["sequence_length"] * bench["shared_prefix_ratio"]),
        noncontiguous_ratio=bench["non_contiguous_ratio"],
        seed=seed,
    )

    # Baseline
    baseline_cache = ContiguousCache(max_entries=cfg["cache"]["max_entries"])
    baseline_runner = InferenceRunner(
        cache=baseline_cache,
        num_layers=cfg["compression"]["num_layers"],
        chunk_size=cfg["cache"]["chunk_size"],
        seed=seed,
    )
    reqs = [InferenceRequest(f"req_{i}", tok, seed=seed + i) for i, tok in enumerate(token_sequences)]
    baseline_runner.run_batch(reqs)

    # B+C
    codec = CompressionCodec(
        num_layers=cfg["compression"]["num_layers"],
        cutoff_ratio=cfg["compression"]["cutoff_ratio"],
    )
    bc_cache = CompressedSegmentCache(
        codec=codec,
        chunk_size=cfg["cache"]["chunk_size"],
        max_entries=cfg["cache"]["max_entries"],
    )
    bc_runner = InferenceRunner(
        cache=bc_cache,
        num_layers=cfg["compression"]["num_layers"],
        chunk_size=cfg["cache"]["chunk_size"],
        seed=seed,
    )
    bc_runner.run_batch(reqs)

    date = cfg["experiment"]["date"]
    out_dir = f"results/bc_{date}"
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "baseline": baseline_runner.metrics_summary(),
        "bc": bc_runner.metrics_summary(),
        "memory_reduction_percent": (
            1.0 - bc_cache.memory_bytes() / max(baseline_cache.memory_bytes(), 1)
        ) * 100,
        "noncontiguous_fraction": bc_runner.hit_metrics.noncontiguous_fraction(),
    }

    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_dir}/metrics.json")
    print(f"Non-contiguous hit fraction: {results['noncontiguous_fraction']:.1%}")
    print(f"Memory reduction: {results['memory_reduction_percent']:.1f}%")


if __name__ == "__main__":
    run()

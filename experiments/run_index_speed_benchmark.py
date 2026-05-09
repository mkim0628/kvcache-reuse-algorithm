"""TriangleInequalitySegmentIndex vs linear scan speed benchmark.

Activity B: Measures O(log N) index search vs O(N) brute-force linear scan
at N = 100, 1K, 10K segments.

Usage:
    python experiments/run_index_speed_benchmark.py
    python experiments/run_index_speed_benchmark.py --segment-counts 100 1000 10000
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from src.cache.segmented import SegmentedHashCache
from src.cache.triangle_index import TriangleInequalitySegmentIndex


def linear_search(
    embeddings: Dict[str, torch.Tensor],
    query: torch.Tensor,
    top_k: int = 5,
) -> List:
    """Brute-force linear scan: O(N) cosine distance computation."""
    dists = []
    for key, emb in embeddings.items():
        sim = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0), emb.unsqueeze(0)
        ).item()
        dists.append((key, 1.0 - sim))
    dists.sort(key=lambda x: x[1])
    return dists[:top_k]


def benchmark_n(
    n: int,
    embedding_dim: int = 64,
    trials: int = 10,
    top_k: int = 5,
    seed: int = 42,
) -> Dict:
    """Benchmark index search vs linear scan for N stored segments."""
    torch.manual_seed(seed)
    backend = SegmentedHashCache(max_entries=n + 10)
    index = TriangleInequalitySegmentIndex(
        backend_cache=backend,
        embedding_dim=embedding_dim,
        leaf_size=8,
        distance_fn="cosine",
    )

    # Populate index
    for i in range(n):
        index.put(f"seg_{i}", torch.randn(embedding_dim))

    # Force index build
    query = torch.randn(embedding_dim)
    index.search_nearest(query, top_k=top_k, max_distance=2.0)

    embeddings = dict(index._embeddings)

    # Warmup
    for _ in range(3):
        index.search_nearest(query, top_k=top_k, max_distance=2.0)
        linear_search(embeddings, query, top_k=top_k)

    # Benchmark index
    torch.manual_seed(seed + 1)
    t_start = time.perf_counter()
    for _ in range(trials):
        q = torch.randn(embedding_dim)
        index.search_nearest(q, top_k=top_k, max_distance=2.0)
    t_index = (time.perf_counter() - t_start) / trials * 1000  # ms per query

    # Benchmark linear scan
    torch.manual_seed(seed + 1)
    t_start = time.perf_counter()
    for _ in range(trials):
        q = torch.randn(embedding_dim)
        linear_search(embeddings, q, top_k=top_k)
    t_linear = (time.perf_counter() - t_start) / trials * 1000  # ms per query

    speedup = t_linear / t_index if t_index > 0 else float("inf")

    return {
        "n_segments": n,
        "index_ms": round(t_index, 4),
        "linear_ms": round(t_linear, 4),
        "speedup_x": round(speedup, 2),
        "target_speedup_x": 3.6,
        "meets_target": speedup >= 3.6 if n >= 10000 else True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Index speed benchmark")
    parser.add_argument(
        "--segment-counts", nargs="+", type=int,
        default=[100, 1000, 10000],
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir", type=str, default="results/2026-05-09",
    )
    args = parser.parse_args()

    results = []
    print(f"{'N':>8} | {'Index (ms)':>12} | {'Linear (ms)':>12} | {'Speedup':>8} | {'Target (3.6x)':>14}")
    print("-" * 70)

    for n in args.segment_counts:
        r = benchmark_n(
            n=n,
            embedding_dim=args.embedding_dim,
            trials=args.trials,
            seed=args.seed,
        )
        results.append(r)
        status = "PASS" if r["meets_target"] else "FAIL"
        print(
            f"{n:>8} | {r['index_ms']:>12.4f} | {r['linear_ms']:>12.4f} | "
            f"{r['speedup_x']:>8.2f} | {status:>14}"
        )

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "index_speed_benchmark.json")
    with open(out_path, "w") as f:
        json.dump({"benchmark": results, "args": vars(args)}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

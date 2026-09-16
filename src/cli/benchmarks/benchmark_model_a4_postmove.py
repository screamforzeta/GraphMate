"""CLI benchmark for MODEL A4 cached post-move reranking."""

from __future__ import annotations

import argparse

import torch

from src.benchmarks.model_a4_postmove import (
    benchmark_cached_a4_validation,
    write_benchmark,
)
from src.training.model_a.model_a4_postmove_reranker import ModelA4PostMoveConfig


def parse_args():
    """Parse A4 benchmark CLI options."""

    parser = argparse.ArgumentParser(description="Benchmark A4 cached validation throughput.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--cache-root", default="data/model_a4_postmove")
    parser.add_argument(
        "--output",
        default="artifacts/model_a4_postmove_gnn_reranker/benchmark_cached_validation.json",
    )
    return parser.parse_args()


def main():
    """Run the cached A4 benchmark."""

    args = parse_args()
    config = ModelA4PostMoveConfig(
        batch_size=args.batch_size,
        cache_root=args.cache_root,
        amp=args.amp,
    )
    summary = benchmark_cached_a4_validation(config, torch.device(args.device), args.max_batches)
    write_benchmark(args.output, summary)
    print(summary)
    print("TEST_SET_EVALUATED = NO")


if __name__ == "__main__":
    main()

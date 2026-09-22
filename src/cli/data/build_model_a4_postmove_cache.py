"""Build train/validation cache for MODEL A4 post-move reranking."""

from __future__ import annotations

import argparse

import torch

from src.graph.pyg_dataset import load_pyg_dataset
from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    build_a4_postmove_cache,
)


def parse_args():
    """Parse A4 cache-generation CLI options."""

    parser = argparse.ArgumentParser(description="Build A4 train/validation post-move cache.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-root", default="data/model_a4_postmove")
    parser.add_argument(
        "--a3-checkpoint",
        default="checkpoints/model_a3/best.pt",
    )
    return parser.parse_args()


def main():
    """Build A4 cache for train and validation only."""

    args = parse_args()
    config = ModelA4PostMoveConfig(
        batch_size=args.batch_size,
        cache_root=args.cache_root,
        a3_checkpoint=args.a3_checkpoint,
    )
    train_graphs = load_pyg_dataset(split="train")
    val_graphs = load_pyg_dataset(split="val")
    manifest = build_a4_postmove_cache(
        train_graphs,
        val_graphs,
        config,
        torch.device(args.device),
    )
    print(f"A4 cache built: {manifest}")
    print("TEST_SET_USED_FOR_A4_TRAINING = NO")
    print("TEST_SET_USED_FOR_CHECKPOINT_SELECTION = NO")
    print("TEST_SET_EVALUATED = NO")


if __name__ == "__main__":
    main()

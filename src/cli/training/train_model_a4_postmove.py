"""Command-line training entry point for MODEL A4 post-move reranker."""

from __future__ import annotations

import argparse

import torch

from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    train_model_a4,
)


def parse_args():
    """Parse A4 training CLI options."""

    parser = argparse.ArgumentParser(description="Train MODEL_A4_POSTMOVE_GNN_RERANKER.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--no-non-blocking", dest="non_blocking", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--cache-root", default="data/model_a4_postmove")
    parser.add_argument("--output-root", default="artifacts/model_a4_postmove_gnn_reranker")
    parser.add_argument(
        "--a3-checkpoint",
        default="checkpoints/model_a3/best.pt",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    """Run A4 training or a short smoke run."""

    args = parse_args()
    config = ModelA4PostMoveConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        non_blocking=args.non_blocking,
        amp=args.amp,
        max_epochs=args.max_epochs,
        cache_root=args.cache_root,
        output_root=args.output_root,
        a3_checkpoint=args.a3_checkpoint,
    )
    summary = train_model_a4(config, torch.device(args.device), smoke=args.smoke)
    print(summary)
    print("TEST_SET_USED_FOR_A4_TRAINING = NO")
    print("TEST_SET_USED_FOR_CHECKPOINT_SELECTION = NO")
    print("TEST_SET_EVALUATED = NO")


if __name__ == "__main__":
    main()

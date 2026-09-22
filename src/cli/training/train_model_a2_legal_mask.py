"""Command-line entry point for MODEL_A2_LEGAL_MASK_NO_TIMING.

Purpose:
    Train the unchanged ChessGATNoTiming backbone from scratch with legal-move
    masking applied to loss and official metrics.
Input:
    data/pyg sharded datasets and resources/move_encoder/move_to_idx.json.
Output:
    artifacts/model_a2_legal_mask_no_timing/ with A2 checkpoints and reports.
Run:
    python3 -m src.cli.training.train_model_a2_legal_mask --device cuda
"""

from __future__ import annotations

import argparse

import torch

from src.graph.pyg_dataset import load_pyg_dataset
from src.training.model_a.model_a2_legal_mask import (
    ModelA2LegalMaskConfig,
    train_model_a2,
)


def parse_args():
    """Parse MODEL_A2_LEGAL_MASK_NO_TIMING CLI options."""

    parser = argparse.ArgumentParser(
        description="Run MODEL_A2_LEGAL_MASK_NO_TIMING training."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--no-non-blocking", dest="non_blocking", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--max-epochs", type=int, default=300)
    return parser.parse_args()


def main():
    """Run A2 training from the command line."""

    args = parse_args()
    config = ModelA2LegalMaskConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        non_blocking=args.non_blocking,
        amp=args.amp,
    )
    train_graphs = load_pyg_dataset(split="train")
    val_graphs = load_pyg_dataset(split="val")
    test_graphs = load_pyg_dataset(split="test")
    train_model_a2(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        config=config,
        device=torch.device(args.device),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

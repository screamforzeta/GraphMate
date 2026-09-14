"""Command-line entry point for timing-aware Model B on the A3 baseline."""

from __future__ import annotations

import argparse

import torch

from src.graph.pyg_dataset import ShardedPyGDataset
from src.training.model_b.model_b_timing_legal_scorer import (
    ModelBTimingLegalScorerConfig,
    train_model_b,
)


def parse_args():
    """Parse Model B timing legal scorer CLI options."""

    parser = argparse.ArgumentParser(
        description="Run MODEL_B_TIMING_LEGAL_MOVE_SCORER training."
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset-root", default="data/pyg_games_timing")
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
    parser.add_argument("--timing-hidden-dim", type=int, default=16)
    return parser.parse_args()


def _load_timing_dataset(root, split):
    """Load legacy monolithic timing graphs through a list-like object."""

    import torch

    path = f"{root}/{split}_graphs.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        return ShardedPyGDataset(root=root, split=split)


def main():
    """Run Model B training from the command line."""

    args = parse_args()
    config = ModelBTimingLegalScorerConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        non_blocking=args.non_blocking,
        amp=args.amp,
        timing_hidden_dim=args.timing_hidden_dim,
    )
    train_model_b(
        train_graphs=_load_timing_dataset(args.dataset_root, "train"),
        val_graphs=_load_timing_dataset(args.dataset_root, "val"),
        test_graphs=_load_timing_dataset(args.dataset_root, "test"),
        config=config,
        device=torch.device(args.device),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

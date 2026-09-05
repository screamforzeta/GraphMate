"""Adaptive command-line training for ChessGATNoTiming.

Purpose:
    Run deterministic trial-based adaptive training for the no-timing chess GAT
    baseline without using the test split for trial selection.
Input:
    data/pyg sharded train/validation/test splits and artifacts/move_to_idx.json.
Output:
    Adaptive training artifacts under artifacts/adaptive_training/.
Role:
    Automates controlled hyperparameter adjustments for CURRENT_PYG_BASELINE.
"""

import argparse

import torch

from src.train_chess_gat import (
    limit_graphs,
    load_graphs,
    load_move_vocab_size,
    resolve_device,
)
from src.training import (
    AdaptiveTrainingConfig,
    run_adaptive_training,
)


def parse_args():
    """Parse adaptive controller CLI options."""

    parser = argparse.ArgumentParser(
        description="Run adaptive ChessGATNoTiming training."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--max-total-epochs", type=int, default=120)
    parser.add_argument("--epochs-per-trial", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--limit-train-graphs", type=int, default=None)
    parser.add_argument("--limit-val-graphs", type=int, default=None)
    parser.add_argument("--limit-test-graphs", type=int, default=None)
    return parser.parse_args()


def main():
    """Run adaptive training from the command line."""

    args = parse_args()
    device = resolve_device(args.device)

    if torch.cuda.is_available():
        print("cuda_available: True")
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda_available: False")

    num_classes = load_move_vocab_size()
    train_graphs = limit_graphs(
        load_graphs("train"),
        args.limit_train_graphs,
    )
    val_graphs = limit_graphs(
        load_graphs("val"),
        args.limit_val_graphs,
    )
    test_graphs = limit_graphs(
        load_graphs("test"),
        args.limit_test_graphs,
    )

    config = AdaptiveTrainingConfig(
        max_trials=args.max_trials,
        max_total_epochs=args.max_total_epochs,
        epochs_per_trial=args.epochs_per_trial,
        patience=args.patience,
        seed=args.seed,
        batch_size=args.batch_size,
        limit_train_graphs=args.limit_train_graphs,
        limit_val_graphs=args.limit_val_graphs,
        limit_test_graphs=args.limit_test_graphs,
    )

    run_adaptive_training(
        train_graphs,
        val_graphs,
        test_graphs,
        num_classes,
        config,
        device,
    )


if __name__ == "__main__":
    main()

"""Command-line entry point for progressive ChessGATNoTiming training.

Purpose:
    Run or resume the progressive pilot/confirmation/full/final-test pipeline.
Input:
    data/pyg sharded datasets and artifacts/move_to_idx.json.
Output:
    artifacts/progressive_training/chess_gat_no_timing/.
Run:
    python3 -m src.cli.training.train_model_a_progressive --device cuda --amp
"""

from pathlib import Path
import argparse
import json

import torch

from src.cli.training.train_model_a import (
    load_graphs,
    load_move_vocab_size,
    resolve_device,
)
from src.training.model_a.progressive_controller import (
    PROGRESSIVE_ROOT,
    ProgressiveTrainingConfig,
    run_progressive_training,
)


def parse_optional_float(value):
    """Parse a float or None-like CLI value."""

    if value is None or str(value).lower() in {"none", "null"}:
        return None
    return float(value)


def parse_args():
    """Parse progressive training CLI options."""

    parser = argparse.ArgumentParser(
        description="Run progressive ChessGATNoTiming training."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--blocking", action="store_false", dest="non_blocking")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--pilot-train-graphs", type=int, default=12000)
    parser.add_argument("--pilot-val-graphs", type=int, default=2000)
    parser.add_argument("--pilot-max-trials", type=int, default=4)
    parser.add_argument("--pilot-max-epochs", type=int, default=10)
    parser.add_argument("--pilot-patience", type=int, default=3)
    parser.add_argument("--confirmation-train-graphs", type=int, default=30000)
    parser.add_argument("--confirmation-val-graphs", type=int, default=4000)
    parser.add_argument("--confirmation-top-k", type=int, default=2)
    parser.add_argument("--confirmation-max-epochs", type=int, default=15)
    parser.add_argument("--confirmation-patience", type=int, default=4)
    parser.add_argument("--full-max-epochs", type=int, default=60)
    parser.add_argument("--full-patience", type=int, default=8)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-runtime-hours", type=parse_optional_float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-full-train-graphs", type=int, default=8000)
    parser.add_argument("--smoke-full-val-graphs", type=int, default=1500)
    parser.add_argument("--smoke-test-graphs", type=int, default=1500)
    parser.add_argument("--experiment-dir", type=Path, default=PROGRESSIVE_ROOT)
    return parser.parse_args()


def config_from_args(args):
    """Create ProgressiveTrainingConfig from CLI arguments."""

    config = ProgressiveTrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        non_blocking=args.non_blocking,
        amp=args.amp,
        pilot_train_graphs=args.pilot_train_graphs,
        pilot_val_graphs=args.pilot_val_graphs,
        pilot_max_trials=args.pilot_max_trials,
        pilot_max_epochs=args.pilot_max_epochs,
        pilot_patience=args.pilot_patience,
        confirmation_train_graphs=args.confirmation_train_graphs,
        confirmation_val_graphs=args.confirmation_val_graphs,
        confirmation_top_k=args.confirmation_top_k,
        confirmation_max_epochs=args.confirmation_max_epochs,
        confirmation_patience=args.confirmation_patience,
        full_max_epochs=args.full_max_epochs,
        full_patience=args.full_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        min_learning_rate=args.min_learning_rate,
        max_runtime_hours=args.max_runtime_hours,
        smoke=args.smoke,
        smoke_full_train_graphs=args.smoke_full_train_graphs,
        smoke_full_val_graphs=args.smoke_full_val_graphs,
        smoke_test_graphs=args.smoke_test_graphs,
        output_root=str(args.experiment_dir),
    )

    if args.smoke:
        config.pilot_train_graphs = min(config.pilot_train_graphs, 2000)
        config.pilot_val_graphs = min(config.pilot_val_graphs, 500)
        config.pilot_max_trials = min(config.pilot_max_trials, 2)
        config.pilot_max_epochs = min(config.pilot_max_epochs, 2)
        config.confirmation_train_graphs = min(config.confirmation_train_graphs, 4000)
        config.confirmation_val_graphs = min(config.confirmation_val_graphs, 1000)
        config.confirmation_top_k = min(config.confirmation_top_k, 1)
        config.confirmation_max_epochs = min(config.confirmation_max_epochs, 2)
        config.full_max_epochs = min(config.full_max_epochs, 3)
        config.full_patience = min(config.full_patience, 3)

    return config


def load_resume_config(experiment_dir):
    """Load the original experiment config when resuming."""

    config_path = Path(experiment_dir) / "experiment_config.json"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    payload.pop("device", None)
    return ProgressiveTrainingConfig(**payload)


def main():
    """Run progressive training from the command line."""

    args = parse_args()
    if args.resume:
        config = load_resume_config(args.experiment_dir) or config_from_args(args)
        config.output_root = str(args.experiment_dir)
        config.max_runtime_hours = args.max_runtime_hours
    else:
        config = config_from_args(args)

    device = resolve_device(args.device)
    if torch.cuda.is_available():
        print("cuda_available: True")
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda_available: False")

    train_dataset = load_graphs("train")
    val_dataset = load_graphs("val")
    test_dataset = load_graphs("test")
    num_classes = load_move_vocab_size()

    run_progressive_training(
        train_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        config,
        device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

"""Command-line entry point for the final Model A convergence run.

Purpose:
    Run or resume MODEL_A_CONVERGENCE_RUN_V1 for ChessGATNoTiming with fixed
    scientific config from MODEL_A_FULL_BASELINE_V1.
Input:
    data/pyg sharded datasets and artifacts/move_to_idx.json.
Output:
    artifacts/convergence_training/chess_gat_no_timing/.
Run:
    python3 -m src.train_chess_gat_convergence --device cuda --amp
"""

from pathlib import Path
import argparse

import torch

from src.train_chess_gat import (
    limit_graphs,
    load_graphs,
    load_move_vocab_size,
    resolve_device,
)
from src.training.convergence_run import (
    CONVERGENCE_ROOT,
    ConvergenceRunConfig,
    read_json,
    run_convergence_training,
)


def parse_optional_float(value):
    """Parse a float or None-like CLI value."""

    if value is None or str(value).lower() in {"none", "null"}:
        return None
    return float(value)


def parse_args():
    """Parse convergence run CLI options."""

    parser = argparse.ArgumentParser(
        description="Run MODEL_A_CONVERGENCE_RUN_V1 for ChessGATNoTiming."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--blocking", action="store_false", dest="non_blocking")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    parser.add_argument("--max-runtime-hours", type=parse_optional_float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-dir", type=Path, default=CONVERGENCE_ROOT)
    parser.add_argument("--limit-train-graphs", type=int, default=None)
    parser.add_argument("--limit-val-graphs", type=int, default=None)
    parser.add_argument("--limit-test-graphs", type=int, default=None)
    return parser.parse_args()


def config_from_args(args):
    """Create a convergence config from CLI arguments."""

    return ConvergenceRunConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_patience=args.lr_scheduler_patience,
        min_learning_rate=args.min_learning_rate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        non_blocking=args.non_blocking,
        amp=args.amp,
        max_runtime_hours=args.max_runtime_hours,
        output_root=str(args.experiment_dir),
    )


def load_resume_config(args):
    """Load saved config as source of truth for resume."""

    payload = read_json(args.experiment_dir / "experiment_config.json")
    payload.pop("device", None)
    payload.pop("num_classes", None)
    config = ConvergenceRunConfig(**payload)
    config.output_root = str(args.experiment_dir)
    config.max_runtime_hours = args.max_runtime_hours
    return config


def main():
    """Run or resume the convergence pipeline from the command line."""

    args = parse_args()
    config = load_resume_config(args) if args.resume else config_from_args(args)
    device = resolve_device(args.device)

    if torch.cuda.is_available():
        print("cuda_available: True")
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda_available: False")

    train_dataset = limit_graphs(load_graphs("train"), args.limit_train_graphs)
    val_dataset = limit_graphs(load_graphs("val"), args.limit_val_graphs)
    test_dataset = limit_graphs(load_graphs("test"), args.limit_test_graphs)
    num_classes = load_move_vocab_size()

    run_convergence_training(
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

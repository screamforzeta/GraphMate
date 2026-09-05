"""Command-line training for ChessGATNoTiming.

Purpose:
    Train the CURRENT_PYG_BASELINE no-timing chess GAT on generated PyG graph
    files, save the best validation checkpoint, reload it, and test once.
Input:
    data/pyg/train_graphs.pt, val_graphs.pt, test_graphs.pt, and
    artifacts/move_to_idx.json.
Output:
    artifacts/checkpoints/chess_gat_no_timing_best.pt and
    artifacts/training/chess_gat_no_timing_history.json.
Role:
    Main executable entry point for MODEL A training.
"""

from pathlib import Path
import argparse
import json

import torch

from src.training import (
    ChessGATTrainingConfig,
    train_model,
)


TRAIN_GRAPHS_PATH = Path("data/pyg/train_graphs.pt")
VAL_GRAPHS_PATH = Path("data/pyg/val_graphs.pt")
TEST_GRAPHS_PATH = Path("data/pyg/test_graphs.pt")
MOVE_ENCODER_PATH = Path("artifacts/move_to_idx.json")


def parse_args():
    """Parse command-line training options.

    Parameters:
        None.
    Returns:
        argparse.Namespace with training options.
    Side effects:
        Reads command-line arguments from sys.argv.
    """

    parser = argparse.ArgumentParser(
        description="Train the ChessGATNoTiming CURRENT_PYG_BASELINE."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--limit-train-graphs", type=int, default=None)
    parser.add_argument("--limit-val-graphs", type=int, default=None)
    parser.add_argument("--limit-test-graphs", type=int, default=None)

    return parser.parse_args()


def resolve_device(device_name):
    """Resolve a CLI device option to torch.device.

    Parameters:
        device_name: One of auto, cpu, or cuda.
    Returns:
        torch.device selected for training.
    Side effects:
        None.
    """

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available."
        )

    if device_name == "cuda":
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_move_vocab_size(path=MOVE_ENCODER_PATH):
    """Load the number of move classes from move_to_idx.json.

    Parameters:
        path: Move vocabulary JSON path.
    Returns:
        Integer vocabulary size.
    Side effects:
        Reads path from disk.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Move encoder not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        move_to_idx = json.load(file)

    return len(move_to_idx)


def load_graphs(path):
    """Load a PyG graph list from disk.

    Parameters:
        path: Torch .pt graph dataset path.
    Returns:
        List of PyG Data graphs.
    Side effects:
        Reads path from disk.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"PyG dataset not found: {path}"
        )

    return torch.load(
        path,
        weights_only=False,
    )


def limit_graphs(graphs, limit):
    """Take a deterministic first-N subset without modifying files.

    Parameters:
        graphs: Loaded graph list.
        limit: Optional maximum graph count.
    Returns:
        Original list or first limit graphs.
    Side effects:
        None.
    """

    if limit is None:
        return graphs

    return graphs[:limit]


def main():
    """Run CLI training.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Loads datasets and writes checkpoint/history artifacts.
    """

    args = parse_args()
    device = resolve_device(args.device)

    if torch.cuda.is_available():
        print("cuda_available: True")
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    else:
        print("cuda_available: False")

    num_classes = load_move_vocab_size()

    train_graphs = limit_graphs(
        load_graphs(TRAIN_GRAPHS_PATH),
        args.limit_train_graphs,
    )
    val_graphs = limit_graphs(
        load_graphs(VAL_GRAPHS_PATH),
        args.limit_val_graphs,
    )
    test_graphs = limit_graphs(
        load_graphs(TEST_GRAPHS_PATH),
        args.limit_test_graphs,
    )

    config = ChessGATTrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        limit_train_graphs=args.limit_train_graphs,
        limit_val_graphs=args.limit_val_graphs,
        limit_test_graphs=args.limit_test_graphs,
    )

    train_model(
        train_graphs,
        val_graphs,
        test_graphs,
        num_classes,
        config,
        device,
    )


if __name__ == "__main__":
    main()

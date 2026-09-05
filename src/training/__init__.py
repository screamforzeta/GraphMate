"""Training utilities for project models.

Purpose:
    Expose reusable training loops, metrics, and configuration for chess GNNs.
Input:
    PyTorch Geometric graph datasets and project model classes.
Output:
    Training helpers used by CLI scripts and tests.
Role:
    Keeps training code separate from preprocessing and graph generation.
"""

from src.training.chess_gat_trainer import (
    ChessGATTrainingConfig,
    EarlyStoppingState,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    train_model,
    train_one_epoch,
    validate_graph_splits,
)
from src.training.metrics import (
    compute_topk_accuracies,
)


__all__ = [
    "ChessGATTrainingConfig",
    "EarlyStoppingState",
    "compute_topk_accuracies",
    "evaluate",
    "load_checkpoint",
    "save_checkpoint",
    "set_seed",
    "train_model",
    "train_one_epoch",
    "validate_graph_splits",
]

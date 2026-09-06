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
    make_grad_scaler,
    save_checkpoint,
    set_seed,
    train_model,
    train_one_epoch,
    validate_graph_splits,
)
from src.training.adaptive_controller import (
    AdaptiveTrainingConfig,
    TrialConfig,
    TrialResult,
    choose_next_trial,
    run_adaptive_training,
    select_best_trial,
)
from src.training.convergence import (
    analyze_convergence,
)
from src.training.metrics import (
    compute_topk_accuracies,
)


__all__ = [
    "ChessGATTrainingConfig",
    "AdaptiveTrainingConfig",
    "EarlyStoppingState",
    "TrialConfig",
    "TrialResult",
    "analyze_convergence",
    "compute_topk_accuracies",
    "choose_next_trial",
    "evaluate",
    "load_checkpoint",
    "make_grad_scaler",
    "save_checkpoint",
    "set_seed",
    "train_model",
    "train_one_epoch",
    "run_adaptive_training",
    "select_best_trial",
    "validate_graph_splits",
]

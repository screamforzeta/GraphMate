"""Convergence diagnostics for adaptive chess GAT training.

Purpose:
    Classify train/validation histories into explicit optimization states.
Input:
    Per-epoch history dictionaries with train_loss, val_loss, train_top1,
    and val_top1.
Output:
    Structured diagnosis used by the adaptive controller.
Role:
    Keeps adaptive decisions transparent and testable.
"""

from dataclasses import dataclass
import math


IMPROVING = "IMPROVING"
PLATEAU = "PLATEAU"
OVERFITTING = "OVERFITTING"
UNDERFITTING_OR_CAPACITY_LIMIT = "UNDERFITTING_OR_CAPACITY_LIMIT"
UNSTABLE = "UNSTABLE"
DIVERGING = "DIVERGING"
CONVERGED = "CONVERGED"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


@dataclass
class ConvergenceDiagnosis:
    """Structured convergence diagnosis.

    Parameters:
        status: One of the exported diagnosis constants.
        reason: Human-readable rule explanation.
        warnings: Diagnostic warning codes.
    Returns:
        None.
    Side effects:
        None.
    """

    status: str
    reason: str
    warnings: list[str]


def _finite_losses(history):
    """Check whether train and validation losses are finite.

    Parameters:
        history: Epoch history list.
    Returns:
        True when all known losses are finite.
    Side effects:
        None.
    """

    for row in history:
        for key in ("train_loss", "val_loss"):
            value = row.get(key)
            if value is None or not math.isfinite(value):
                return False
    return True


def _relative_improvement(start, end):
    """Compute relative loss improvement.

    Parameters:
        start: Starting loss.
        end: Ending loss.
    Returns:
        Relative improvement, positive when loss decreased.
    Side effects:
        None.
    """

    if start <= 0:
        return 0.0
    return (start - end) / start


def analyze_convergence(
    history,
    plateau_window=4,
    overfit_window=3,
    min_relative_val_improvement=0.005,
):
    """Classify optimization behavior from train/validation history.

    Parameters:
        history: List of epoch metric dictionaries.
        plateau_window: Window used to detect flat validation loss.
        overfit_window: Window used to detect train/validation divergence.
        min_relative_val_improvement: Minimum relative validation improvement.
    Returns:
        ConvergenceDiagnosis.
    Side effects:
        None.
    """

    if not history:
        return ConvergenceDiagnosis(
            INSUFFICIENT_HISTORY,
            "No epoch history is available.",
            [],
        )

    if not _finite_losses(history):
        return ConvergenceDiagnosis(
            DIVERGING,
            "At least one train/validation loss is NaN or Inf.",
            ["WARNING_OPTIMIZATION_UNSTABLE"],
        )

    if len(history) >= 2:
        previous = history[-2]["val_loss"]
        current = history[-1]["val_loss"]
        if previous > 0 and current > previous * 1.5:
            return ConvergenceDiagnosis(
                UNSTABLE,
                "Validation loss increased by more than 50% in one epoch.",
                ["WARNING_OPTIMIZATION_UNSTABLE"],
            )

    if len(history) < max(plateau_window, overfit_window):
        return ConvergenceDiagnosis(
            INSUFFICIENT_HISTORY,
            "History is shorter than diagnostic windows.",
            [],
        )

    overfit_slice = history[-overfit_window:]
    train_improvement = _relative_improvement(
        overfit_slice[0]["train_loss"],
        overfit_slice[-1]["train_loss"],
    )
    val_improvement = _relative_improvement(
        overfit_slice[0]["val_loss"],
        overfit_slice[-1]["val_loss"],
    )

    if train_improvement > min_relative_val_improvement and val_improvement < 0:
        return ConvergenceDiagnosis(
            OVERFITTING,
            "Train loss decreased while validation loss increased over window.",
            ["WARNING_OVERFITTING"],
        )

    plateau_slice = history[-plateau_window:]
    window_val_improvement = _relative_improvement(
        plateau_slice[0]["val_loss"],
        min(row["val_loss"] for row in plateau_slice),
    )
    full_val_improvement = _relative_improvement(
        history[0]["val_loss"],
        min(row["val_loss"] for row in history),
    )

    if window_val_improvement < min_relative_val_improvement:
        train_top1 = history[-1].get("train_top1", 0.0)
        val_top1 = history[-1].get("val_top1", 0.0)
        if train_top1 < 0.02 and val_top1 < 0.02 and full_val_improvement < 0.02:
            return ConvergenceDiagnosis(
                UNDERFITTING_OR_CAPACITY_LIMIT,
                "Train and validation metrics are flat and low.",
                ["WARNING_MODEL_CAPACITY_LIMIT", "WARNING_TARGET_SPACE_TOO_LARGE"],
            )

        return ConvergenceDiagnosis(
            PLATEAU,
            "Validation loss improvement in the recent window is below threshold.",
            [],
        )

    return ConvergenceDiagnosis(
        IMPROVING,
        "Validation loss is still improving above threshold.",
        [],
    )

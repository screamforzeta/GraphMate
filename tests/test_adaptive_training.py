from dataclasses import asdict

import pytest

from src.training.adaptive_controller import (
    AdaptiveTrainingConfig,
    TrialConfig,
    TrialResult,
    choose_next_trial,
    select_best_trial,
)
from src.training.convergence import (
    DIVERGING,
    IMPROVING,
    OVERFITTING,
    PLATEAU,
    analyze_convergence,
)


def make_history(train_losses, val_losses, train_top1=0.1, val_top1=0.1):
    return [
        {
            "epoch": index + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_top1": train_top1,
            "val_top1": val_top1,
        }
        for index, (train_loss, val_loss)
        in enumerate(zip(train_losses, val_losses))
    ]


def make_state():
    return {
        "lr_reductions_used": 0,
        "dropout_increases_used": 0,
        "weight_decay_increases_used": 0,
    }


def test_plateau_detection():
    diagnosis = analyze_convergence(
        make_history(
            [5.0, 4.9, 4.85, 4.84],
            [5.0, 4.999, 4.998, 4.997],
            train_top1=0.1,
            val_top1=0.1,
        ),
        plateau_window=4,
        overfit_window=3,
    )

    assert diagnosis.status == PLATEAU


def test_improving_detection():
    diagnosis = analyze_convergence(
        make_history(
            [5.0, 4.8, 4.6, 4.4],
            [5.0, 4.7, 4.4, 4.0],
        ),
        plateau_window=4,
        overfit_window=3,
    )

    assert diagnosis.status == IMPROVING


def test_overfitting_detection():
    diagnosis = analyze_convergence(
        make_history(
            [5.0, 4.8, 4.5, 4.2],
            [5.0, 4.9, 5.1, 5.3],
        ),
        plateau_window=4,
        overfit_window=3,
    )

    assert diagnosis.status == OVERFITTING


def test_diverging_detection_for_non_finite_loss():
    diagnosis = analyze_convergence(
        make_history(
            [5.0, float("nan"), 4.0],
            [5.0, 4.8, 4.7],
        )
    )

    assert diagnosis.status == DIVERGING


def test_decision_engine_plateau_lowers_lr_one_change():
    current = TrialConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        dropout=0.10,
        batch_size=32,
    )
    diagnosis = analyze_convergence(
        make_history(
            [5.0, 4.9, 4.8, 4.7],
            [5.0, 4.999, 4.998, 4.997],
            train_top1=0.1,
            val_top1=0.1,
        )
    )

    next_config, action = choose_next_trial(
        current,
        diagnosis,
        make_state(),
        AdaptiveTrainingConfig(),
    )

    assert next_config.learning_rate == 5e-4
    assert next_config.dropout == current.dropout
    assert next_config.weight_decay == current.weight_decay
    assert "learning_rate" in action


def test_decision_engine_overfitting_increases_dropout_one_change():
    current = TrialConfig()
    diagnosis = analyze_convergence(
        make_history(
            [5.0, 4.8, 4.5, 4.2],
            [5.0, 4.9, 5.1, 5.3],
        )
    )

    next_config, action = choose_next_trial(
        current,
        diagnosis,
        make_state(),
        AdaptiveTrainingConfig(),
    )

    assert next_config.dropout == 0.20
    assert next_config.learning_rate == current.learning_rate
    assert next_config.weight_decay == current.weight_decay
    assert "dropout" in action


def test_decision_engine_diverging_lowers_lr():
    current = TrialConfig()
    diagnosis = analyze_convergence(
        make_history(
            [5.0, float("inf")],
            [5.0, 5.2],
        )
    )

    next_config, action = choose_next_trial(
        current,
        diagnosis,
        make_state(),
        AdaptiveTrainingConfig(),
    )

    assert next_config.learning_rate == 5e-4
    assert "learning_rate" in action


def test_decision_engine_stops_when_lr_budget_is_reached():
    state = make_state()
    state["lr_reductions_used"] = 3
    diagnosis = analyze_convergence(
        make_history(
            [5.0, 4.9, 4.8, 4.7],
            [5.0, 4.999, 4.998, 4.997],
            train_top1=0.1,
            val_top1=0.1,
        )
    )

    next_config, action = choose_next_trial(
        TrialConfig(),
        diagnosis,
        state,
        AdaptiveTrainingConfig(max_lr_reductions=3),
    )

    assert next_config is None
    assert action.startswith("STOP")


def test_select_best_trial_uses_validation_not_test_metrics():
    weak_test_good_val = TrialResult(
        trial_id=1,
        config=TrialConfig(),
        best_epoch=1,
        best_val_loss=2.0,
        best_val_top1=0.1,
        best_val_top3=0.2,
        best_val_top5=0.3,
        final_train_loss=2.0,
        final_val_loss=2.0,
        status="COMPLETED",
        diagnosis=PLATEAU,
        next_action="STOP",
        elapsed_time=1.0,
        epochs_completed=2,
        checkpoint_path="a.pt",
    )
    strong_test_bad_val = TrialResult(
        **{
            **asdict(weak_test_good_val),
            "trial_id": 2,
            "best_val_loss": 3.0,
            "checkpoint_path": "b.pt",
            "config": TrialConfig(learning_rate=5e-4),
        }
    )

    assert select_best_trial([strong_test_bad_val, weak_test_good_val]).trial_id == 1

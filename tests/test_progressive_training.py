from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from src.training.model_a.adaptive_controller import TrialConfig
from src.training.model_a.progressive_controller import (
    ProgressiveTrainingConfig,
    RunResult,
    deterministic_indices,
    make_progressive_subsets,
    rank_results,
    run_progressive_training,
)


def _graph(target=0):
    return Data(
        x=torch.zeros((64, 15), dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            dtype=torch.float,
        ),
        global_features=torch.zeros((1, 4), dtype=torch.float),
        y=torch.tensor(target, dtype=torch.long),
    )


def _dataset(size=12):
    return [
        _graph(index % 2)
        for index in range(size)
    ]


def _small_config(tmp_path):
    return ProgressiveTrainingConfig(
        seed=42,
        batch_size=4,
        learning_rate=1e-3,
        dropout=0.0,
        weight_decay=0.0,
        num_workers=0,
        pin_memory=False,
        non_blocking=False,
        amp=False,
        pilot_train_graphs=6,
        pilot_val_graphs=4,
        pilot_max_trials=1,
        pilot_max_epochs=1,
        pilot_patience=1,
        confirmation_train_graphs=8,
        confirmation_val_graphs=5,
        confirmation_top_k=1,
        confirmation_max_epochs=1,
        confirmation_patience=1,
        full_max_epochs=1,
        full_patience=1,
        smoke=True,
        smoke_test_graphs=4,
        output_root=str(tmp_path / "progressive"),
    )


def test_deterministic_subset_selection_and_nesting():
    train = _dataset(20)
    val = _dataset(10)
    config = ProgressiveTrainingConfig(
        seed=7,
        pilot_train_graphs=5,
        pilot_val_graphs=3,
        confirmation_train_graphs=12,
        confirmation_val_graphs=6,
    )

    first = deterministic_indices(20, 12, 7)
    second = deterministic_indices(20, 12, 7)
    subsets = make_progressive_subsets(train, val, config)

    assert first == second
    assert set(subsets["pilot_train"].indices).issubset(
        set(subsets["confirmation_train"].indices)
    )
    assert set(subsets["pilot_val"].indices).issubset(
        set(subsets["confirmation_val"].indices)
    )
    assert subsets["metadata"]["pilot_subset_of_confirmation"]


def test_promotion_ranking_uses_loss_top5_then_top1():
    cfg = TrialConfig()
    results = [
        RunResult("a", cfg, "COMPLETED", "MAX_EPOCHS_REACHED", 1, 1, 2.0, 0.1, 0.1, 0.5, 1.0, "a.pt", "a.json"),
        RunResult("b", cfg, "COMPLETED", "MAX_EPOCHS_REACHED", 1, 1, 1.0, 0.1, 0.1, 0.4, 1.0, "b.pt", "b.json"),
        RunResult("c", cfg, "COMPLETED", "MAX_EPOCHS_REACHED", 1, 1, 1.0, 0.2, 0.1, 0.4, 1.0, "c.pt", "c.json"),
        RunResult("d", cfg, "FAILED", "EXCEPTION", 0, None, None, None, None, None, 1.0, None, "d.json"),
    ]

    ranked = rank_results(results)

    assert [result.name for result in ranked] == ["c", "b", "a"]


def test_progressive_smoke_end_to_end_and_resume_skips_final_test(tmp_path):
    config = _small_config(tmp_path)

    report = run_progressive_training(
        _dataset(12),
        _dataset(8),
        _dataset(6),
        num_classes=2,
        config=config,
        device=torch.device("cpu"),
        resume=False,
    )
    resumed = run_progressive_training(
        _dataset(12),
        _dataset(8),
        _dataset(6),
        num_classes=2,
        config=config,
        device=torch.device("cpu"),
        resume=True,
    )

    assert report["experiment_scope"] == "SMOKE_TEST"
    assert report["controller_status"] == "COMPLETED"
    assert report["test_used_for_selection"] is False
    assert report["final_test"] is not None
    assert resumed["final_test"] == report["final_test"]
    assert (Path(config.output_root) / "full" / "best.pt").exists()
    assert (Path(config.output_root) / "full" / "last.pt").exists()


def test_time_limit_terminates_without_test(tmp_path):
    config = _small_config(tmp_path)
    config.max_runtime_hours = 0.0

    report = run_progressive_training(
        _dataset(12),
        _dataset(8),
        _dataset(6),
        num_classes=2,
        config=config,
        device=torch.device("cpu"),
        resume=False,
    )

    assert report["controller_status"] == "TIME_LIMIT_REACHED"
    assert report["final_test"] is None


def test_zero_valid_trial_raises_diagnostic(tmp_path, monkeypatch):
    config = _small_config(tmp_path)

    monkeypatch.setattr(
        "src.training.model_a.progressive_controller.run_training_run",
        lambda *args, **kwargs: RunResult(
            "pilot_trial_001",
            TrialConfig(),
            "FAILED",
            "EXCEPTION",
            0,
            None,
            None,
            None,
            None,
            None,
            0.0,
            None,
            "history.json",
        ),
    )

    with pytest.raises(RuntimeError, match="No valid pilot"):
        run_progressive_training(
            _dataset(12),
            _dataset(8),
            _dataset(6),
            num_classes=2,
            config=config,
            device=torch.device("cpu"),
            resume=False,
        )

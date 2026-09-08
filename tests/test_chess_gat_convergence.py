import json

import pytest
import torch
from torch_geometric.data import Data

from src.training.chess_gat_trainer import EarlyStoppingState
from src.training.convergence_run import (
    BASELINE_V1,
    ConvergenceRunConfig,
    apply_terminal_extension,
    assert_resume_config_compatible,
    compare_with_baseline,
    convergence_status,
    run_convergence_training,
)


def _graph(target=0):
    return Data(
        x=torch.zeros((64, 15), dtype=torch.float),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.zeros((2, 5), dtype=torch.float),
        global_features=torch.zeros((1, 4), dtype=torch.float),
        y=torch.tensor(target, dtype=torch.long),
    )


def _dataset(size=4):
    return [
        _graph(index % 2)
        for index in range(size)
    ]


def _config(tmp_path, max_epochs=3, patience=2):
    return ConvergenceRunConfig(
        output_root=str(tmp_path / "convergence"),
        batch_size=2,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        learning_rate=0.01,
        weight_decay=0.001,
        dropout=0.0,
        seed=42,
        num_workers=0,
        pin_memory=False,
        non_blocking=False,
        amp=False,
    )


def test_reduce_lr_on_plateau_uses_validation_loss_and_respects_min_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        min_lr=0.0025,
    )

    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)
    scheduler.step(1.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)
    state = scheduler.state_dict()
    restored = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        min_lr=0.0025,
    )
    restored.load_state_dict(state)
    assert restored.state_dict()["best"] == pytest.approx(1.0)


def test_early_stopping_counter_and_status_mapping():
    early = EarlyStoppingState(patience=2, min_delta=0.0)

    assert early.update(5.0) is True
    assert early.epochs_without_improvement == 0
    assert early.update(5.0) is False
    assert early.epochs_without_improvement == 1
    assert early.update(5.1) is False
    assert early.should_stop is True

    assert convergence_status("EARLY_STOPPING") == "CONVERGED_BY_EARLY_STOPPING"
    assert convergence_status("MAX_EPOCHS_REACHED") == "MAX_EPOCH_BUDGET_EXHAUSTED"


def test_resume_config_rejects_critical_mismatch(tmp_path):
    config = _config(tmp_path)
    saved = {**config.__dict__, "num_classes": 2}
    saved["batch_size"] = 4

    with pytest.raises(ValueError, match="batch_size"):
        assert_resume_config_compatible(saved, config, num_classes=2)


def test_convergence_run_smoke_final_test_once_and_reports(tmp_path, monkeypatch):
    config = _config(tmp_path, max_epochs=3, patience=3)
    train = _dataset(4)
    val = _dataset(3)
    test = _dataset(2)
    val_losses = iter([1.0, 0.9, 0.8, 0.7])
    calls = {"test_evaluations": 0, "train_epochs": []}

    def fake_train(*args, **kwargs):
        loader = args[1]
        calls["train_epochs"].append(getattr(loader.sampler, "epoch", None))
        return {
            "loss": 1.0,
            "top1": 0.1,
            "top3": 0.2,
            "top5": 0.3,
            "num_examples": len(loader.dataset),
        }

    def fake_evaluate(*args, **kwargs):
        loader = args[1]
        if loader.dataset is test:
            calls["test_evaluations"] += 1
            return {
                "loss": 0.75,
                "top1": 0.4,
                "top3": 0.5,
                "top5": 0.6,
                "num_examples": len(loader.dataset),
            }
        return {
            "loss": next(val_losses),
            "top1": 0.2,
            "top3": 0.3,
            "top5": 0.4,
            "num_examples": len(loader.dataset),
        }

    monkeypatch.setattr("src.training.convergence_run.train_one_epoch", fake_train)
    monkeypatch.setattr("src.training.convergence_run.evaluate", fake_evaluate)

    report = run_convergence_training(
        train,
        val,
        test,
        num_classes=2,
        config=config,
        device=torch.device("cpu"),
    )
    resumed = run_convergence_training(
        train,
        val,
        test,
        num_classes=2,
        config=config,
        device=torch.device("cpu"),
        resume=True,
    )

    root = tmp_path / "convergence"
    assert report["status"] == "COMPLETED"
    assert report["final_test"]["top1"] == pytest.approx(0.4)
    assert resumed["final_test"]["top1"] == pytest.approx(0.4)
    assert calls["test_evaluations"] == 1
    assert len(calls["train_epochs"]) == 3
    assert (root / "best.pt").exists()
    assert (root / "last.pt").exists()
    assert (root / "history.json").exists()
    assert (root / "controller_state.json").exists()
    assert (root / "final_report.json").exists()
    assert (root / "final_report.md").exists()


def test_resume_appends_history_from_next_epoch(tmp_path, monkeypatch):
    first_config = _config(tmp_path, max_epochs=2, patience=5)
    second_config = _config(tmp_path, max_epochs=2, patience=5)
    train = _dataset(4)
    val = _dataset(3)
    test = _dataset(2)
    val_losses = iter([1.0, 0.9, 0.8])

    monkeypatch.setattr(
        "src.training.convergence_run.train_one_epoch",
        lambda *args, **kwargs: {
            "loss": 1.0,
            "top1": 0.1,
            "top3": 0.2,
            "top5": 0.3,
            "num_examples": 4,
        },
    )
    monkeypatch.setattr(
        "src.training.convergence_run.evaluate",
        lambda *args, **kwargs: {
            "loss": next(val_losses),
            "top1": 0.2,
            "top3": 0.3,
            "top5": 0.4,
            "num_examples": 3,
        },
    )
    runtime_checks = iter([False, True, False])

    monkeypatch.setattr(
        "src.training.convergence_run.should_stop_for_runtime",
        lambda *args, **kwargs: next(runtime_checks),
    )

    run_convergence_training(
        train,
        val,
        test,
        2,
        first_config,
        torch.device("cpu"),
    )
    report = run_convergence_training(
        train,
        val,
        test,
        2,
        second_config,
        torch.device("cpu"),
        resume=True,
    )

    history = json.loads(
        (tmp_path / "convergence" / "history.json").read_text(encoding="utf-8")
    )["history"]
    assert [row["epoch"] for row in history] == [1, 2]
    assert report["training"]["epochs_completed"] == 2


def test_keyboard_interrupt_marks_state_without_final_test(tmp_path, monkeypatch):
    config = _config(tmp_path, max_epochs=2, patience=2)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.training.convergence_run.train_one_epoch", raise_interrupt)

    report = run_convergence_training(
        _dataset(4),
        _dataset(3),
        _dataset(2),
        2,
        config,
        torch.device("cpu"),
    )
    state = json.loads(
        (tmp_path / "convergence" / "controller_state.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "INTERRUPTED"
    assert state["stop_reason"] == "USER_INTERRUPT"
    assert state["final_test_completed"] is False


def test_compare_with_baseline_reports_deltas_and_methodology():
    metrics = {
        "loss": BASELINE_V1["test_loss"] - 0.1,
        "top1": BASELINE_V1["test_top1"] + 0.01,
        "top3": BASELINE_V1["test_top3"] + 0.02,
        "top5": BASELINE_V1["test_top5"] + 0.03,
    }

    delta = compare_with_baseline(metrics)

    assert delta["delta_test_loss"] == pytest.approx(-0.1)
    assert "not by using test V1" in delta["methodology_note"]


def test_terminal_extension_smoke_preserves_state_and_counts_tests(tmp_path, monkeypatch):
    config = _config(tmp_path, max_epochs=2, patience=12)
    train = _dataset(4)
    val = _dataset(3)
    test = _dataset(2)
    val_losses = iter([1.0, 0.9, 0.95, 0.96])
    calls = {
        "test_evaluations": 0,
        "test_calls_during_training": 0,
    }

    def fake_train(*args, **kwargs):
        optimizer = args[3]
        loader = args[1]
        if len(loader.dataset) == len(test) and loader.dataset is test:
            calls["test_calls_during_training"] += 1
        return {
            "loss": 1.0,
            "top1": 0.1,
            "top3": 0.2,
            "top5": 0.3,
            "num_examples": len(loader.dataset),
        }

    def fake_evaluate(*args, **kwargs):
        loader = args[1]
        if loader.dataset is test:
            calls["test_evaluations"] += 1
            return {
                "loss": 0.7 + calls["test_evaluations"] * 0.01,
                "top1": 0.4,
                "top3": 0.5,
                "top5": 0.6,
                "num_examples": len(loader.dataset),
            }
        return {
            "loss": next(val_losses),
            "top1": 0.2,
            "top3": 0.3,
            "top5": 0.4,
            "num_examples": len(loader.dataset),
        }

    monkeypatch.setattr("src.training.convergence_run.train_one_epoch", fake_train)
    monkeypatch.setattr("src.training.convergence_run.evaluate", fake_evaluate)

    first = run_convergence_training(
        train,
        val,
        test,
        2,
        config,
        torch.device("cpu"),
    )
    root = tmp_path / "convergence"
    checkpoint = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 0.005
    torch.save(checkpoint, root / "last.pt")

    extended = run_convergence_training(
        train,
        val,
        test,
        2,
        config,
        torch.device("cpu"),
        resume=True,
        extend_max_epochs=4,
    )

    history = json.loads((root / "history.json").read_text(encoding="utf-8"))[
        "history"
    ]
    state = json.loads(
        (root / "controller_state.json").read_text(encoding="utf-8")
    )

    assert first["training"]["best_epoch"] == 2
    assert [row["epoch"] for row in history] == [1, 2, 3, 4]
    assert extended["training"]["best_epoch"] == 2
    assert history[2]["previous_lr"] == pytest.approx(0.005)
    assert calls["test_evaluations"] == 2
    assert calls["test_calls_during_training"] == 0
    assert state["test_evaluation_count"] == 2
    assert state["intermediate_test_evaluations"][0]["after_epoch"] == 2
    assert (root / "snapshots" / "epoch_2_terminal" / "controller_state.json").exists()
    assert extended["continuation"]["current_max_epochs"] == 4


def test_extension_rejects_early_stopping_and_non_increasing_budget(tmp_path, monkeypatch):
    config = _config(tmp_path, max_epochs=2, patience=1)
    test = _dataset(2)
    val_losses = iter([1.0, 1.0])

    monkeypatch.setattr(
        "src.training.convergence_run.train_one_epoch",
        lambda *args, **kwargs: {
            "loss": 1.0,
            "top1": 0.1,
            "top3": 0.2,
            "top5": 0.3,
            "num_examples": 4,
        },
    )
    def fake_evaluate(*args, **kwargs):
        loader = args[1]
        if loader.dataset is test:
            return {
                "loss": 0.8,
                "top1": 0.2,
                "top3": 0.3,
                "top5": 0.4,
                "num_examples": 2,
            }
        return {
            "loss": next(val_losses),
            "top1": 0.2,
            "top3": 0.3,
            "top5": 0.4,
            "num_examples": 3,
        }

    monkeypatch.setattr("src.training.convergence_run.evaluate", fake_evaluate)

    run_convergence_training(
        _dataset(4),
        _dataset(3),
        test,
        2,
        config,
        torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="MAX_EPOCHS_REACHED"):
        apply_terminal_extension(tmp_path / "convergence", config, 2, 4)

    state_path = tmp_path / "convergence" / "controller_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "COMPLETED"
    state["stop_reason"] = "MAX_EPOCHS_REACHED"
    state["epochs_completed"] = 2
    state["best_epoch"] = 2
    state["early_stopping_counter"] = 0
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(ValueError, match="greater"):
        apply_terminal_extension(tmp_path / "convergence", config, 2, 2)


def test_extension_rejects_non_max_epoch_config_change(tmp_path, monkeypatch):
    config = _config(tmp_path, max_epochs=2, patience=12)
    test = _dataset(2)
    val_losses = iter([1.0, 0.9])

    monkeypatch.setattr(
        "src.training.convergence_run.train_one_epoch",
        lambda *args, **kwargs: {
            "loss": 1.0,
            "top1": 0.1,
            "top3": 0.2,
            "top5": 0.3,
            "num_examples": 4,
        },
    )
    def fake_evaluate(*args, **kwargs):
        loader = args[1]
        if loader.dataset is test:
            return {
                "loss": 0.8,
                "top1": 0.2,
                "top3": 0.3,
                "top5": 0.4,
                "num_examples": 2,
            }
        return {
            "loss": next(val_losses),
            "top1": 0.2,
            "top3": 0.3,
            "top5": 0.4,
            "num_examples": 3,
        }

    monkeypatch.setattr("src.training.convergence_run.evaluate", fake_evaluate)

    run_convergence_training(
        _dataset(4),
        _dataset(3),
        test,
        2,
        config,
        torch.device("cpu"),
    )
    changed = _config(tmp_path, max_epochs=2, patience=12)
    changed.batch_size = 4

    with pytest.raises(ValueError, match="Only max_epochs"):
        apply_terminal_extension(tmp_path / "convergence", changed, 2, 4)

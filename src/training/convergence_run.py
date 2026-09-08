"""Dedicated convergence run controller for ChessGATNoTiming.

Purpose:
    Train Model A from scratch on full train/validation data with the fixed
    baseline V1 config, larger epoch budget, scheduler, resume, and one-shot
    final test.
Input:
    Train/validation/test PyG datasets, move vocabulary size, and run config.
Output:
    best.pt, last.pt, controller_state.json, history.json, final reports.
Run:
    python3 -m src.train_chess_gat_convergence --device cuda --amp
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import time

import torch

from src.training.chess_gat_trainer import (
    ChessGATTrainingConfig,
    EarlyStoppingState,
    build_model,
    evaluate,
    load_checkpoint,
    make_grad_scaler,
    make_loaders,
    set_loader_epoch,
    set_seed,
    train_one_epoch,
    validate_graph_splits,
)


CONVERGENCE_ROOT = Path("artifacts/convergence_training/chess_gat_no_timing")
RUN_ID = "MODEL_A_CONVERGENCE_RUN_V1"
PARENT_RUN_ID = "MODEL_A_FULL_BASELINE_V1"
BASELINE_V1 = {
    "val_loss": 3.621614213882354,
    "val_top1": 0.31862517416172675,
    "val_top3": 0.46957733395262424,
    "val_top5": 0.5369252207331252,
    "test_loss": 3.590105460274372,
    "test_top1": 0.3186991869953313,
    "test_top3": 0.47723577246848714,
    "test_top5": 0.5428571429679064,
}


@dataclass
class ConvergenceRunConfig:
    """Configuration for the fixed Model A convergence run."""

    seed: int = 42
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    batch_size: int = 128
    max_epochs: int = 150
    early_stopping_patience: int = 12
    min_delta: float = 0.0
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    min_learning_rate: float = 1e-6
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    non_blocking: bool = True
    amp: bool = True
    max_runtime_hours: float | None = None
    output_root: str = str(CONVERGENCE_ROOT)
    run_id: str = RUN_ID
    parent_run_id: str = PARENT_RUN_ID
    model_class: str = "ChessGATNoTiming"
    config_source: str = "FIXED_FROM_MODEL_A_FULL_BASELINE_V1"


def atomic_write_json(path, payload):
    """Write JSON through a temporary file before replacing the destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def read_json(path):
    """Read a JSON file from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def convergence_status(stop_reason):
    """Map stop reason to a descriptive convergence status."""

    if stop_reason == "EARLY_STOPPING":
        return "CONVERGED_BY_EARLY_STOPPING"
    if stop_reason == "MAX_EPOCHS_REACHED":
        return "MAX_EPOCH_BUDGET_EXHAUSTED"
    if stop_reason == "RUNTIME_LIMIT_REACHED":
        return "RUNTIME_LIMIT_REACHED"
    if stop_reason == "USER_INTERRUPT":
        return "INTERRUPTED"
    return "FAILED"


def make_trainer_config(config, root):
    """Build a standard trainer config with convergence-run paths."""

    root = Path(root)
    return ChessGATTrainingConfig(
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        patience=config.early_stopping_patience,
        min_delta=config.min_delta,
        weight_decay=config.weight_decay,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        non_blocking=config.non_blocking,
        amp=config.amp,
        model_dropout=config.dropout,
        checkpoint_path=str(root / "best.pt"),
        history_path=str(root / "history.json"),
    )


def assert_resume_config_compatible(saved_config, config, num_classes):
    """Reject accidental changes to scientific config when resuming."""

    saved = dict(saved_config)
    checks = [
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "seed",
        "model_class",
    ]
    mismatches = [
        key
        for key in checks
        if saved.get(key) != getattr(config, key)
    ]
    if saved.get("num_classes") != num_classes:
        mismatches.append("num_classes")
    if mismatches:
        raise ValueError(
            "Resume config mismatch for critical fields: "
            + ", ".join(sorted(set(mismatches)))
        )


def assert_extension_config_compatible(saved_config, config, num_classes):
    """Allow only max_epochs to change during an explicit continuation."""

    saved = dict(saved_config)
    checks = [
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "seed",
        "model_class",
        "lr_scheduler_factor",
        "lr_scheduler_patience",
        "min_learning_rate",
        "early_stopping_patience",
        "min_delta",
        "amp",
    ]
    mismatches = [
        key
        for key in checks
        if saved.get(key) != getattr(config, key)
    ]
    if saved.get("num_classes") != num_classes:
        mismatches.append("num_classes")
    if mismatches:
        raise ValueError(
            "Only max_epochs can change during continuation. Mismatched fields: "
            + ", ".join(sorted(set(mismatches)))
        )


def snapshot_terminal_state(root, epoch):
    """Copy lightweight terminal metadata before mutating a completed run."""

    root = Path(root)
    snapshot_dir = root / "snapshots" / f"epoch_{epoch}_terminal"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "controller_state.json",
        "final_report.json",
        "history.json",
        "experiment_config.json",
    ):
        source = root / name
        destination = snapshot_dir / name
        if source.exists() and not destination.exists():
            shutil.copy2(source, destination)
    return snapshot_dir


def archived_test_evaluations(root, state, epoch):
    """Return previous test evaluations recorded before continuation."""

    existing = list(state.get("intermediate_test_evaluations") or [])
    if state.get("final_test_completed"):
        final_report_path = Path(root) / "final_report.json"
        final_test = None
        if final_report_path.exists():
            final_test = read_json(final_report_path).get("final_test")
        final_test = final_test or state.get("final_test")
        if final_test and not existing:
            existing.append(
                {
                    "after_epoch": epoch,
                    "reason": "PREVIOUS_TERMINAL_MAX_EPOCH_RUN",
                    "metrics": final_test,
                }
            )
    return existing


def apply_terminal_extension(root, config, num_classes, new_max_epochs):
    """Validate and persist an explicit max-epoch extension."""

    root = Path(root)
    experiment_config_path = root / "experiment_config.json"
    state_path = root / "controller_state.json"
    last_path = root / "last.pt"
    if not experiment_config_path.exists() or not state_path.exists():
        raise FileNotFoundError(
            "Continuation requires experiment_config.json and controller_state.json."
        )
    if not last_path.exists():
        raise FileNotFoundError("Continuation requires last.pt.")

    saved_config = read_json(experiment_config_path)
    state = read_json(state_path)
    assert_extension_config_compatible(saved_config, config, num_classes)

    if state.get("status") != "COMPLETED":
        raise ValueError("Continuation requires a terminal COMPLETED run.")
    if state.get("stop_reason") != "MAX_EPOCHS_REACHED":
        raise ValueError("Continuation is allowed only after MAX_EPOCHS_REACHED.")

    previous_max_epochs = int(saved_config["max_epochs"])
    epochs_completed = int(state.get("epochs_completed", -1))
    if epochs_completed != previous_max_epochs:
        raise ValueError(
            "Continuation requires epochs_completed == previous max_epochs."
        )
    if int(state.get("best_epoch", -1)) != epochs_completed:
        raise ValueError("Continuation requires best_epoch at the final epoch.")
    if int(state.get("early_stopping_counter", -1)) != 0:
        raise ValueError("Continuation requires early_stopping_counter == 0.")
    if new_max_epochs <= epochs_completed:
        raise ValueError("extend max epochs must be greater than completed epochs.")

    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != epochs_completed:
        raise ValueError("last.pt epoch does not match controller state.")

    snapshot_dir = snapshot_terminal_state(root, epochs_completed)
    evaluations = archived_test_evaluations(root, state, epochs_completed)
    continuation_count = int(state.get("continuation_count", 0)) + 1
    continued_at = time.time()

    saved_config["original_max_epochs"] = saved_config.get(
        "original_max_epochs",
        previous_max_epochs,
    )
    saved_config["max_epochs"] = new_max_epochs
    saved_config["current_max_epochs"] = new_max_epochs
    saved_config["continuation_count"] = continuation_count
    saved_config["continued_from_epoch"] = epochs_completed
    saved_config["continued_at"] = continued_at
    atomic_write_json(experiment_config_path, saved_config)

    state.update(
        {
            "status": "RUNNING",
            "stop_reason": None,
            "original_max_epochs": saved_config["original_max_epochs"],
            "current_max_epochs": new_max_epochs,
            "continuation_count": continuation_count,
            "continued_from_epoch": epochs_completed,
            "continued_at": continued_at,
            "continuation_reason": (
                "MAX_EPOCH_BUDGET_EXHAUSTED_WITH_BEST_AT_FINAL_EPOCH"
            ),
            "terminal_snapshot": str(snapshot_dir),
            "intermediate_test_evaluations": evaluations,
            "test_evaluation_count": len(evaluations),
            "final_test_completed": False,
        }
    )
    state.pop("final_test", None)
    atomic_write_json(state_path, state)

    config.max_epochs = new_max_epochs
    return state


def save_best_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_epoch,
    early_stopping,
    config,
    num_classes,
    history,
    train_metrics,
    val_metrics,
):
    """Save the best validation checkpoint with resumable metadata."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": early_stopping.best_val_loss,
            "current_learning_rate": optimizer.param_groups[0]["lr"],
            "config": {**asdict(config), "num_classes": num_classes},
            "seed": config.seed,
            "history": history,
            "early_stopping": asdict(early_stopping),
            "model_metadata": {
                "run_id": config.run_id,
                "parent_run_id": config.parent_run_id,
                "model_class": config.model_class,
                "config_source": config.config_source,
            },
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def save_last_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_epoch,
    early_stopping,
    config,
    num_classes,
    history,
):
    """Save enough state to resume after a completed epoch."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": early_stopping.best_val_loss,
            "config": {**asdict(config), "num_classes": num_classes},
            "history": history,
            "early_stopping": asdict(early_stopping),
        },
        path,
    )


def controller_state(
    status,
    stop_reason,
    started_at,
    epoch,
    best_epoch,
    early_stopping,
    optimizer,
    final_test_completed=False,
):
    """Build the persisted convergence controller state."""

    return {
        "status": status,
        "started_at": started_at,
        "last_updated_at": time.time(),
        "current_epoch": epoch,
        "epochs_completed": epoch,
        "best_epoch": best_epoch,
        "best_val_loss": (
            early_stopping.best_val_loss
            if best_epoch is not None
            else None
        ),
        "current_learning_rate": optimizer.param_groups[0]["lr"],
        "early_stopping_counter": early_stopping.epochs_without_improvement,
        "stop_reason": stop_reason,
        "final_test_completed": final_test_completed,
    }


def should_stop_for_runtime(started_at, max_runtime_hours):
    """Return True when the optional runtime limit has been reached."""

    return (
        max_runtime_hours is not None
        and time.time() - started_at >= max_runtime_hours * 3600
    )


def compare_with_baseline(test_metrics):
    """Compute test deltas against MODEL_A_FULL_BASELINE_V1."""

    if not test_metrics:
        return None
    return {
        "delta_test_loss": test_metrics["loss"] - BASELINE_V1["test_loss"],
        "delta_top1": test_metrics["top1"] - BASELINE_V1["test_top1"],
        "delta_top3": test_metrics["top3"] - BASELINE_V1["test_top3"],
        "delta_top5": test_metrics["top5"] - BASELINE_V1["test_top5"],
        "methodology_note": (
            "The convergence run was motivated by validation trend and "
            "best_epoch=max_epoch in baseline V1, not by using test V1 for "
            "checkpoint or stopping decisions."
        ),
    }


def write_final_reports(root, report):
    """Write final JSON and Markdown reports."""

    root = Path(root)
    atomic_write_json(root / "final_report.json", report)
    final_test = report.get("final_test") or {}
    lines = [
        f"# {report['run_id']}",
        "",
        f"- Parent: `{report['parent_run_id']}`",
        f"- Status: `{report['status']}`",
        f"- Stop reason: `{report['stop_reason']}`",
        f"- Convergence status: `{report['convergence_status']}`",
        f"- Epochs completed: `{report['training']['epochs_completed']}`",
        f"- Best epoch: `{report['training']['best_epoch']}`",
        f"- Best val loss: `{report['training']['best_val_loss']}`",
        f"- Final LR: `{report['training']['final_learning_rate']}`",
        f"- LR reductions: `{report['training']['lr_reduction_count']}`",
        "",
        "## Dataset",
        "",
        f"- Train: `{report['dataset_sizes']['train']}`",
        f"- Validation: `{report['dataset_sizes']['val']}`",
        f"- Test: `{report['dataset_sizes']['test']}`",
        "",
        "## Final Test",
        "",
        f"- Loss: `{final_test.get('loss')}`",
        f"- Top1: `{final_test.get('top1')}`",
        f"- Top3: `{final_test.get('top3')}`",
        f"- Top5: `{final_test.get('top5')}`",
        f"- Examples: `{final_test.get('num_examples')}`",
        "",
        "## Methodology",
        "",
        "Test is evaluated only after terminal training and best-checkpoint reload.",
        (
            "Previous terminal test evaluations are archived and reported when "
            "a continuation is approved."
        ),
        f"- Test evaluation count: `{report.get('test_evaluation_count', 0)}`",
    ]
    for evaluation in report.get("intermediate_test_evaluations", []):
        lines.append(
            "- Archived test evaluation: "
            f"after_epoch=`{evaluation.get('after_epoch')}`, "
            f"reason=`{evaluation.get('reason')}`"
        )
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_final_test_once(model, test_loader, criterion, device, config, root, state):
    """Evaluate test once from best.pt and persist the result."""

    if state.get("final_test_completed"):
        final_report = Path(root) / "final_report.json"
        if final_report.exists():
            return read_json(final_report).get("final_test")
        return state.get("final_test")

    load_checkpoint(Path(root) / "best.pt", model, device=device)
    metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        top_k=(1, 3, 5),
        non_blocking=config.non_blocking,
        amp=config.amp,
    )
    state["final_test_completed"] = True
    state["final_test"] = metrics
    atomic_write_json(Path(root) / "controller_state.json", state)
    return metrics


def run_convergence_training(
    train_dataset,
    val_dataset,
    test_dataset,
    num_classes,
    config,
    device,
    resume=False,
    extend_max_epochs=None,
):
    """Run or resume the dedicated Model A convergence pipeline."""

    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    experiment_config_path = root / "experiment_config.json"
    state_path = root / "controller_state.json"
    best_path = root / "best.pt"
    last_path = root / "last.pt"
    history_path = root / "history.json"

    if resume and extend_max_epochs is not None:
        existing_state = apply_terminal_extension(
            root,
            config,
            num_classes,
            int(extend_max_epochs),
        )
    elif resume and experiment_config_path.exists():
        saved_config = read_json(experiment_config_path)
        assert_resume_config_compatible(saved_config, config, num_classes)
    elif not resume:
        atomic_write_json(
            experiment_config_path,
            {**asdict(config), "device": str(device), "num_classes": num_classes},
        )

    validate_graph_splits(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        num_classes,
        sample_size=2,
    )

    trainer_config = make_trainer_config(config, root)
    train_loader, val_loader, test_loader = make_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        trainer_config,
    )

    set_seed(config.seed)
    model = build_model(num_classes, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    scaler = make_grad_scaler(device, amp=config.amp)
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStoppingState(
        patience=config.early_stopping_patience,
        min_delta=config.min_delta,
    )
    history = []
    best_epoch = None
    start_epoch = 1
    started_at = time.time()
    existing_state = read_json(state_path) if state_path.exists() else {}

    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        assert_resume_config_compatible(checkpoint["config"], config, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        stop_state = checkpoint["early_stopping"]
        early_stopping.best_val_loss = stop_state["best_val_loss"]
        early_stopping.epochs_without_improvement = stop_state[
            "epochs_without_improvement"
        ]
        early_stopping.should_stop = stop_state["should_stop"]
        history = checkpoint.get("history", [])
        best_epoch = checkpoint.get("best_epoch")
        start_epoch = int(checkpoint["epoch"]) + 1
        started_at = existing_state.get("started_at", started_at)
        print(f"{config.run_id}", flush=True)
        if extend_max_epochs is not None:
            print("Continuing terminal run", flush=True)
            print(
                f"Previous max epochs: {existing_state.get('continued_from_epoch')}",
                flush=True,
            )
            print(f"New max epochs: {config.max_epochs}", flush=True)
            print(f"Resume epoch: {start_epoch}", flush=True)
            print(f"Best epoch: {best_epoch}", flush=True)
            print(f"Best val loss: {early_stopping.best_val_loss:.8f}", flush=True)
            print(
                f"Current LR: {optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
            print(
                "Early stopping counter: "
                f"{early_stopping.epochs_without_improvement}",
                flush=True,
            )
            print(
                "Previous test evaluations: "
                f"{existing_state.get('test_evaluation_count', 0)}",
                flush=True,
            )
            print(
                "Continuation reason: "
                f"{existing_state.get('continuation_reason')}",
                flush=True,
            )

    status = "RUNNING"
    stop_reason = None
    lr_reductions = sum(
        1
        for row in history
        if row.get("scheduler_changed_lr")
    )
    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            if should_stop_for_runtime(started_at, config.max_runtime_hours):
                status = "RUNTIME_LIMIT_REACHED"
                stop_reason = "RUNTIME_LIMIT_REACHED"
                break

            set_loader_epoch(train_loader, epoch)
            epoch_start = time.perf_counter()
            train_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                top_k=(1, 3, 5),
                non_blocking=config.non_blocking,
                amp=config.amp,
                scaler=scaler,
            )
            train_seconds = time.perf_counter() - train_start

            val_start = time.perf_counter()
            val_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                top_k=(1, 3, 5),
                non_blocking=config.non_blocking,
                amp=config.amp,
            )
            val_seconds = time.perf_counter() - val_start

            previous_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(val_metrics["loss"])
            new_lr = optimizer.param_groups[0]["lr"]
            scheduler_changed_lr = new_lr < previous_lr
            if scheduler_changed_lr:
                lr_reductions += 1

            improved = early_stopping.update(val_metrics["loss"])
            row = {
                "epoch": epoch,
                "learning_rate": new_lr,
                "previous_lr": previous_lr,
                "new_lr": new_lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics.get("top1"),
                "train_top3": train_metrics.get("top3"),
                "train_top5": train_metrics.get("top5"),
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "val_top5": val_metrics["top5"],
                "train_runtime_seconds": train_seconds,
                "val_runtime_seconds": val_seconds,
                "elapsed_seconds": time.time() - started_at,
                "improved": improved,
                "early_stopping_counter": (
                    early_stopping.epochs_without_improvement
                ),
                "scheduler_changed_lr": scheduler_changed_lr,
            }
            history.append(row)
            if improved:
                best_epoch = epoch
                save_best_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_epoch,
                    early_stopping,
                    config,
                    num_classes,
                    history,
                    train_metrics,
                    val_metrics,
                )
            save_last_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_epoch,
                early_stopping,
                config,
                num_classes,
                history,
            )
            atomic_write_json(history_path, {"history": history})
            atomic_write_json(
                state_path,
                controller_state(
                    "RUNNING",
                    None,
                    started_at,
                    epoch,
                    best_epoch,
                    early_stopping,
                    optimizer,
                ),
            )

            print(
                f"Epoch {epoch:03d}/{config.max_epochs} "
                f"lr={new_lr:.6g} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_top1={val_metrics['top1'] * 100:.2f}% "
                f"val_top5={val_metrics['top5'] * 100:.2f}% "
                f"train_time={train_seconds:.2f}s val_time={val_seconds:.2f}s "
                f"best_epoch={best_epoch} "
                f"patience={early_stopping.epochs_without_improvement}/"
                f"{config.early_stopping_patience}",
                flush=True,
            )
            if scheduler_changed_lr:
                print(f"LR REDUCED {previous_lr:.6g} -> {new_lr:.6g}", flush=True)
            if improved:
                print("NEW BEST CHECKPOINT", flush=True)
            if early_stopping.should_stop:
                status = "COMPLETED"
                stop_reason = "EARLY_STOPPING"
                print(f"EARLY STOPPING best epoch={best_epoch}", flush=True)
                break

        if stop_reason is None:
            status = "COMPLETED"
            stop_reason = "MAX_EPOCHS_REACHED"
    except KeyboardInterrupt:
        status = "INTERRUPTED"
        stop_reason = "USER_INTERRUPT"
        epoch = start_epoch - 1 + len(history)
        atomic_write_json(
            state_path,
            controller_state(
                status,
                stop_reason,
                started_at,
                epoch,
                best_epoch,
                early_stopping,
                optimizer,
            ),
        )
        print(
            "Interrupted. Resume with: "
            "python3 -m src.train_chess_gat_convergence --resume --device cuda",
            flush=True,
        )
        return {"status": status, "stop_reason": stop_reason, "history": history}
    except Exception as error:
        status = "FAILED"
        stop_reason = "ERROR"
        atomic_write_json(root / "error.json", {"error": str(error)})
        raise

    final_state = controller_state(
        status,
        stop_reason,
        started_at,
        len(history),
        best_epoch,
        early_stopping,
        optimizer,
    )
    if existing_state.get("intermediate_test_evaluations"):
        final_state["intermediate_test_evaluations"] = existing_state[
            "intermediate_test_evaluations"
        ]
        final_state["test_evaluation_count"] = existing_state.get(
            "test_evaluation_count",
            len(existing_state["intermediate_test_evaluations"]),
        )
    for key in (
        "original_max_epochs",
        "current_max_epochs",
        "continuation_count",
        "continued_from_epoch",
        "continued_at",
        "continuation_reason",
        "terminal_snapshot",
    ):
        if key in existing_state:
            final_state[key] = existing_state[key]
    atomic_write_json(state_path, final_state)

    final_test = None
    if status == "COMPLETED" and best_path.exists():
        best_model = build_model(num_classes, dropout=config.dropout).to(device)
        final_state.update(existing_state)
        final_state.update(
            controller_state(
                status,
                stop_reason,
                started_at,
                len(history),
                best_epoch,
                early_stopping,
                optimizer,
                final_test_completed=existing_state.get(
                    "final_test_completed",
                    False,
                ),
            )
        )
        final_test = run_final_test_once(
            best_model,
            test_loader,
            criterion,
            device,
            config,
            root,
            final_state,
        )
        final_state["final_test_completed"] = True
        final_state["test_evaluation_count"] = (
            int(final_state.get("test_evaluation_count", 0)) + 1
        )
        atomic_write_json(state_path, final_state)

    best_val_metrics = None
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best_val_metrics = best_checkpoint.get("val_metrics")

    report = {
        "run_id": config.run_id,
        "parent_run_id": config.parent_run_id,
        "status": status,
        "stop_reason": stop_reason,
        "convergence_status": convergence_status(stop_reason),
        "dataset_sizes": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "model_config": {
            "model_class": config.model_class,
            "dropout": config.dropout,
            "config_source": config.config_source,
        },
        "runtime_config": {
            "device": str(device),
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers,
            "prefetch_factor": config.prefetch_factor,
            "non_blocking": config.non_blocking,
            "amp": config.amp,
        },
        "training": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": early_stopping.best_val_loss,
            "best_val_top1": (best_val_metrics or {}).get("top1"),
            "best_val_top3": (best_val_metrics or {}).get("top3"),
            "best_val_top5": (best_val_metrics or {}).get("top5"),
            "final_learning_rate": optimizer.param_groups[0]["lr"],
            "lr_reduction_count": lr_reductions,
            "total_runtime_seconds": time.time() - started_at,
        },
        "final_test": final_test,
        "intermediate_test_evaluations": final_state.get(
            "intermediate_test_evaluations",
            [],
        ),
        "test_evaluation_count": final_state.get(
            "test_evaluation_count",
            1 if final_test else 0,
        ),
        "continuation": {
            key: final_state.get(key)
            for key in (
                "original_max_epochs",
                "current_max_epochs",
                "continuation_count",
                "continued_from_epoch",
                "continued_at",
                "continuation_reason",
                "terminal_snapshot",
            )
            if final_state.get(key) is not None
        },
        "baseline_v1_reference": BASELINE_V1,
        "baseline_v1_delta": compare_with_baseline(final_test),
        "test_used_for_training_or_selection": False,
        "history_path": str(history_path),
        "best_checkpoint_path": str(best_path),
        "last_checkpoint_path": str(last_path),
    }
    write_final_reports(root, report)
    return report

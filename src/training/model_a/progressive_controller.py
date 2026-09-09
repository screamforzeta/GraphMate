"""Progressive multi-stage training controller for ChessGATNoTiming.

Purpose:
    Run pilot, confirmation, full training, and one final test using increasing
    dataset fidelity while keeping test isolated until the end.
Input:
    Train/validation/test PyG datasets, move vocabulary size, and controller
    configuration.
Output:
    Stage checkpoints, histories, controller state, and final reports under
    artifacts/progressive_training/chess_gat_no_timing/.
Role:
    Provides unattended/resumable orchestration without changing the model.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import shutil
import time

import torch
from torch.utils.data import Subset

from src.training.model_a.adaptive_controller import TrialConfig
from src.training.model_a.chess_gat_trainer import (
    ChessGATTrainingConfig,
    EarlyStoppingState,
    build_model,
    evaluate,
    load_checkpoint,
    make_grad_scaler,
    make_loaders,
    save_checkpoint,
    set_seed,
    set_loader_epoch,
    train_one_epoch,
    validate_graph_splits,
)


PROGRESSIVE_ROOT = Path("artifacts/progressive_training/chess_gat_no_timing")
OLD_5K_BASELINE = {
    "train_graphs_approx": 5000,
    "test_top1": 0.0527,
    "test_top3": 0.1019,
    "test_top5": 0.1378,
    "best_val_loss": 6.245944,
}


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive no-timing GAT training."""

    seed: int = 42
    batch_size: int = 128
    learning_rate: float = 5e-4
    dropout: float = 0.30
    weight_decay: float = 1e-4
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    non_blocking: bool = True
    amp: bool = True
    pilot_train_graphs: int = 12000
    pilot_val_graphs: int = 2000
    pilot_max_trials: int = 4
    pilot_max_epochs: int = 10
    pilot_patience: int = 3
    confirmation_train_graphs: int = 30000
    confirmation_val_graphs: int = 4000
    confirmation_top_k: int = 2
    confirmation_max_epochs: int = 15
    confirmation_patience: int = 4
    full_max_epochs: int = 60
    full_patience: int = 8
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 3
    min_learning_rate: float = 1e-6
    max_runtime_hours: float | None = None
    smoke: bool = False
    smoke_full_train_graphs: int = 8000
    smoke_full_val_graphs: int = 1500
    smoke_test_graphs: int = 1500
    output_root: str = str(PROGRESSIVE_ROOT)


@dataclass
class RunResult:
    """Serializable summary for one pilot/confirmation/full run."""

    name: str
    config: TrialConfig
    status: str
    stop_reason: str
    epochs_completed: int
    best_epoch: int | None
    best_val_loss: float | None
    best_val_top1: float | None
    best_val_top3: float | None
    best_val_top5: float | None
    runtime_seconds: float
    checkpoint_path: str | None
    history_path: str


def atomic_write_json(path, payload):
    """Write JSON through a temporary file to keep resume state consistent."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    tmp_path.replace(path)


def read_json(path):
    """Read a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def deterministic_indices(total, count, seed):
    """Return a deterministic shuffled prefix of dataset indices."""

    count = min(total, max(0, count))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.randperm(total, generator=generator).tolist()[:count]


def indices_hash(indices):
    """Return a short reproducibility hash for a subset index list."""

    digest = hashlib.sha256()
    for index in indices:
        digest.update(f"{index},".encode("utf-8"))
    return digest.hexdigest()[:16]


def make_progressive_subsets(train_dataset, val_dataset, config):
    """Create deterministic nested pilot and confirmation subsets."""

    train_indices = deterministic_indices(
        len(train_dataset),
        min(config.confirmation_train_graphs, len(train_dataset)),
        config.seed,
    )
    val_indices = deterministic_indices(
        len(val_dataset),
        min(config.confirmation_val_graphs, len(val_dataset)),
        config.seed + 10_000,
    )
    pilot_train = train_indices[: min(config.pilot_train_graphs, len(train_indices))]
    pilot_val = val_indices[: min(config.pilot_val_graphs, len(val_indices))]

    return {
        "pilot_train": Subset(train_dataset, pilot_train),
        "pilot_val": Subset(val_dataset, pilot_val),
        "confirmation_train": Subset(train_dataset, train_indices),
        "confirmation_val": Subset(val_dataset, val_indices),
        "metadata": {
            "seed": config.seed,
            "pilot_train_size": len(pilot_train),
            "pilot_val_size": len(pilot_val),
            "confirmation_train_size": len(train_indices),
            "confirmation_val_size": len(val_indices),
            "pilot_train_indices_hash": indices_hash(pilot_train),
            "pilot_val_indices_hash": indices_hash(pilot_val),
            "confirmation_train_indices_hash": indices_hash(train_indices),
            "confirmation_val_indices_hash": indices_hash(val_indices),
            "pilot_subset_of_confirmation": (
                set(pilot_train).issubset(set(train_indices))
                and set(pilot_val).issubset(set(val_indices))
            ),
        },
    }


def initial_pilot_configs(config):
    """Return controlled hyperparameter configs around the known 5k result."""

    candidates = [
        TrialConfig(config.learning_rate, config.weight_decay, config.dropout, config.batch_size),
        TrialConfig(config.learning_rate / 2, config.weight_decay, config.dropout, config.batch_size),
        TrialConfig(config.learning_rate, config.weight_decay * 10, config.dropout, config.batch_size),
        TrialConfig(config.learning_rate, config.weight_decay, min(0.5, config.dropout + 0.10), config.batch_size),
    ]
    unique = []
    seen = set()
    for candidate in candidates:
        key = asdict(candidate)
        frozen = tuple(sorted(key.items()))
        if frozen not in seen:
            unique.append(candidate)
            seen.add(frozen)
    return unique[: config.pilot_max_trials]


def rank_results(results):
    """Rank successful runs by validation loss, Top5, then Top1."""

    valid = [
        result
        for result in results
        if result.status == "COMPLETED" and result.best_val_loss is not None
    ]
    return sorted(
        valid,
        key=lambda result: (
            result.best_val_loss,
            -(result.best_val_top5 or 0.0),
            -(result.best_val_top1 or 0.0),
        ),
    )


class RuntimeBudget:
    """Track optional wall-clock limit at safe stage/epoch boundaries."""

    def __init__(self, max_runtime_hours=None, started_at=None):
        self.max_runtime_seconds = (
            max_runtime_hours * 3600
            if max_runtime_hours is not None
            else None
        )
        self.started_at = started_at or time.time()

    def elapsed(self):
        """Return elapsed wall-clock seconds."""

        return time.time() - self.started_at

    def exceeded(self):
        """Return True when the optional runtime limit has been reached."""

        return (
            self.max_runtime_seconds is not None
            and self.elapsed() >= self.max_runtime_seconds
        )


def _trial_config_from_dict(payload):
    """Rebuild TrialConfig from serialized dict."""

    return TrialConfig(
        learning_rate=payload["learning_rate"],
        weight_decay=payload["weight_decay"],
        dropout=payload["dropout"],
        batch_size=payload["batch_size"],
    )


def _result_from_dict(payload):
    """Rebuild RunResult from serialized dict."""

    return RunResult(
        name=payload["name"],
        config=_trial_config_from_dict(payload["config"]),
        status=payload["status"],
        stop_reason=payload["stop_reason"],
        epochs_completed=payload["epochs_completed"],
        best_epoch=payload["best_epoch"],
        best_val_loss=payload["best_val_loss"],
        best_val_top1=payload["best_val_top1"],
        best_val_top3=payload["best_val_top3"],
        best_val_top5=payload["best_val_top5"],
        runtime_seconds=payload["runtime_seconds"],
        checkpoint_path=payload["checkpoint_path"],
        history_path=payload["history_path"],
    )


def _runtime_trainer_config(config, trial_config, max_epochs, patience, checkpoint_path, history_path):
    """Build a training config carrying runtime options for DataLoaders/AMP."""

    return ChessGATTrainingConfig(
        batch_size=trial_config.batch_size,
        learning_rate=trial_config.learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        weight_decay=trial_config.weight_decay,
        seed=config.seed,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
        non_blocking=config.non_blocking,
        amp=config.amp,
        model_dropout=trial_config.dropout,
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
    )


def save_last_checkpoint(path, model, optimizer, scheduler, scaler, epoch, early_stopping, history, config, num_classes):
    """Save resumable full-stage state after a completed epoch."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "early_stopping": {
            "best_val_loss": early_stopping.best_val_loss,
            "epochs_without_improvement": early_stopping.epochs_without_improvement,
            "should_stop": early_stopping.should_stop,
            "patience": early_stopping.patience,
            "min_delta": early_stopping.min_delta,
        },
        "history": history,
        "config": asdict(config),
        "num_classes": num_classes,
    }
    torch.save(payload, path)


def enrich_best_checkpoint(path, scheduler, scaler, subset_metadata, runtime_config):
    """Add progressive-only resume metadata to a standard best checkpoint."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["scheduler_state_dict"] = scheduler.state_dict() if scheduler else None
    checkpoint["scaler_state_dict"] = scaler.state_dict() if scaler else None
    checkpoint["subset_metadata"] = subset_metadata
    checkpoint["runtime_config"] = runtime_config
    torch.save(checkpoint, path)


def run_training_run(
    name,
    trial_config,
    train_dataset,
    val_dataset,
    num_classes,
    config,
    max_epochs,
    patience,
    run_dir,
    budget,
    use_scheduler=False,
    resume=False,
):
    """Train one run from scratch or resume the full stage from last.pt."""

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history_path = run_dir / "history.json"
    atomic_write_json(run_dir / "config.json", asdict(trial_config))

    trainer_config = _runtime_trainer_config(
        config,
        trial_config,
        max_epochs,
        patience,
        checkpoint_path,
        history_path,
    )
    train_loader, val_loader, _ = make_loaders(
        train_dataset,
        val_dataset,
        val_dataset,
        trainer_config,
    )

    set_seed(config.seed)
    model = build_model(num_classes, dropout=trial_config.dropout).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trial_config.learning_rate,
        weight_decay=trial_config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            threshold=1e-4,
            min_lr=config.min_learning_rate,
        )
        if use_scheduler
        else None
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = make_grad_scaler(config.device, amp=config.amp)
    early_stopping = EarlyStoppingState(patience=patience)
    history = []
    start_epoch = 1
    best_metrics = None
    best_epoch = None
    start = time.perf_counter()

    if resume and use_scheduler and last_path.exists():
        checkpoint = torch.load(last_path, map_location=config.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        saved_stop = checkpoint["early_stopping"]
        early_stopping.best_val_loss = saved_stop["best_val_loss"]
        early_stopping.epochs_without_improvement = saved_stop["epochs_without_improvement"]
        early_stopping.should_stop = saved_stop["should_stop"]
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        if checkpoint_path.exists():
            best_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            best_epoch = best_checkpoint.get("epoch")
            best_metrics = best_checkpoint.get("val_metrics")

    status = "COMPLETED"
    stop_reason = "MAX_EPOCHS_REACHED"
    try:
        for epoch in range(start_epoch, max_epochs + 1):
            if budget.exceeded():
                status = "INTERRUPTED_RESUMABLE"
                stop_reason = "TIME_LIMIT_REACHED"
                break

            set_loader_epoch(train_loader, epoch)
            epoch_start = time.perf_counter()
            train_start = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                config.device,
                top_k=(1,),
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
                config.device,
                top_k=(1, 3, 5),
                non_blocking=config.non_blocking,
                amp=config.amp,
            )
            validation_seconds = time.perf_counter() - val_start
            old_lr = optimizer.param_groups[0]["lr"]
            if scheduler:
                scheduler.step(val_metrics["loss"])
            new_lr = optimizer.param_groups[0]["lr"]
            scheduler_reduced_lr = new_lr < old_lr

            improved = early_stopping.update(val_metrics["loss"])
            checkpoint_start = time.perf_counter()
            if improved:
                best_epoch = epoch
                best_metrics = val_metrics
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch,
                    early_stopping.best_val_loss,
                    trainer_config,
                    num_classes,
                    train_metrics,
                    val_metrics,
                )
            checkpoint_seconds = time.perf_counter() - checkpoint_start

            epoch_total_seconds = time.perf_counter() - epoch_start
            row = {
                "epoch": epoch,
                "learning_rate": new_lr,
                "train_loss": train_metrics["loss"],
                "train_top1": train_metrics["top1"],
                "val_loss": val_metrics["loss"],
                "val_top1": val_metrics["top1"],
                "val_top3": val_metrics["top3"],
                "val_top5": val_metrics["top5"],
                "improved": improved,
                "scheduler_reduced_lr": scheduler_reduced_lr,
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "epoch_total_seconds": epoch_total_seconds,
                "train_graphs_per_second": len(train_dataset) / train_seconds,
                "validation_graphs_per_second": len(val_dataset) / validation_seconds,
            }
            history.append(row)
            if improved and use_scheduler:
                enrich_best_checkpoint(
                    checkpoint_path,
                    scheduler,
                    scaler,
                    {
                        "train_size": len(train_dataset),
                        "val_size": len(val_dataset),
                    },
                    {
                        "batch_size": config.batch_size,
                        "num_workers": config.num_workers,
                        "pin_memory": config.pin_memory,
                        "persistent_workers": config.persistent_workers,
                        "prefetch_factor": config.prefetch_factor,
                        "non_blocking": config.non_blocking,
                        "amp": config.amp,
                    },
                )
            if use_scheduler:
                save_last_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    early_stopping,
                    history,
                    config,
                    num_classes,
                )
            atomic_write_json(history_path, {"history": history})

            print(
                f"{name} epoch {epoch:02d}/{max_epochs} "
                f"lr={new_lr:.6g} train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_top1={val_metrics['top1'] * 100:.2f}% "
                f"val_top5={val_metrics['top5'] * 100:.2f}% "
                f"train_time={train_seconds:.2f}s val_time={validation_seconds:.2f}s "
                f"elapsed={budget.elapsed():.1f}s"
            )
            if scheduler_reduced_lr:
                print(f"{name} scheduler: lr {old_lr:.6g} -> {new_lr:.6g}")

            if early_stopping.should_stop:
                stop_reason = "EARLY_STOPPING"
                break

        if best_epoch is None and status == "COMPLETED":
            status = "FAILED"
            stop_reason = "NO_VALID_CONFIGURATION"
    except RuntimeError as error:
        message = str(error)
        status = "FAILED"
        stop_reason = "CUDA_OOM" if "out of memory" in message.lower() else "NUMERICAL_FAILURE"
        atomic_write_json(run_dir / "error.json", {"error": message, "stop_reason": stop_reason})
    except Exception as error:
        status = "FAILED"
        stop_reason = "EXCEPTION"
        atomic_write_json(run_dir / "error.json", {"error": str(error), "stop_reason": stop_reason})

    result = RunResult(
        name=name,
        config=trial_config,
        status=status,
        stop_reason=stop_reason,
        epochs_completed=len(history),
        best_epoch=best_epoch,
        best_val_loss=early_stopping.best_val_loss if best_epoch is not None else None,
        best_val_top1=best_metrics.get("top1") if best_metrics else None,
        best_val_top3=best_metrics.get("top3") if best_metrics else None,
        best_val_top5=best_metrics.get("top5") if best_metrics else None,
        runtime_seconds=time.perf_counter() - start,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        history_path=str(history_path),
    )
    atomic_write_json(run_dir / "summary.json", {**asdict(result), "config": asdict(result.config)})
    return result


def model_performance_status(test_metrics):
    """Classify final test performance descriptively against old 5k metadata."""

    if not test_metrics:
        return "INCONCLUSIVE"
    delta = test_metrics.get("top1", 0.0) - OLD_5K_BASELINE["test_top1"]
    if delta > 0.005:
        return "IMPROVED_OVER_5K_BASELINE"
    if delta < -0.005:
        return "WORSE_THAN_5K_BASELINE"
    return "SIMILAR_TO_5K_BASELINE"


def write_final_report(root, report):
    """Write final progressive JSON and Markdown reports."""

    root = Path(root)
    atomic_write_json(root / "final_report.json", report)
    lines = [
        "# Progressive Chess GAT Training Report",
        "",
        f"- Scope: `{report['experiment_scope']}`",
        f"- Controller status: `{report['controller_status']}`",
        f"- Model performance status: `{report['model_performance_status']}`",
        f"- Seed: `{report['seed']}`",
        f"- Device: `{report['runtime_config']['device']}`",
        f"- Batch size: `{report['runtime_config']['batch_size']}`",
        f"- AMP: `{report['runtime_config']['amp']}`",
        "",
        "## Dataset",
        "",
        f"- Train: `{report['dataset']['full_train_size']}`",
        f"- Val: `{report['dataset']['full_val_size']}`",
        f"- Test: `{report['dataset']['full_test_size']}`",
        "",
        "## Pilot",
        "",
    ]
    for run in report["pilot"]:
        lines.append(
            f"- `{run['name']}` {run['status']} loss={run['best_val_loss']} top5={run['best_val_top5']}"
        )
    lines.extend(["", "## Confirmation", ""])
    for run in report["confirmation"]:
        lines.append(
            f"- `{run['name']}` {run['status']} loss={run['best_val_loss']} top5={run['best_val_top5']}"
        )
    lines.extend(
        [
            "",
            "## Full",
            "",
            f"- Stop reason: `{report['full'].get('stop_reason') if report['full'] else None}`",
            f"- Best epoch: `{report['full'].get('best_epoch') if report['full'] else None}`",
            f"- Best val loss: `{report['full'].get('best_val_loss') if report['full'] else None}`",
            "",
            "## Final Test",
            "",
            f"```json\n{json.dumps(report['final_test'], indent=2)}\n```",
        ]
    )
    (root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


class ProgressiveTrainingConfigRuntime(ProgressiveTrainingConfig):
    """Progressive config with resolved device attached for internal use."""

    device: torch.device


def _with_device(config, device):
    """Attach the resolved device without changing serialized user config."""

    config.device = device
    return config


def _load_existing_results(root, stage):
    """Load completed stage summaries from disk for resume."""

    stage_dir = Path(root) / stage
    if not stage_dir.exists():
        return []
    results = []
    for summary_path in sorted(stage_dir.glob("*/summary.json")):
        payload = read_json(summary_path)
        results.append(_result_from_dict(payload))
    return results


def run_final_test(test_dataset, full_result, num_classes, config, root, state):
    """Evaluate the final full best checkpoint on test exactly once."""

    if state.get("final_test_completed"):
        return state.get("final_test")

    model = build_model(num_classes, dropout=full_result.config.dropout).to(config.device)
    load_checkpoint(full_result.checkpoint_path, model, device=config.device)
    trainer_config = _runtime_trainer_config(
        config,
        full_result.config,
        1,
        1,
        Path(full_result.checkpoint_path),
        Path(root) / "test_history.json",
    )
    _, _, test_loader = make_loaders(test_dataset, test_dataset, test_dataset, trainer_config)
    start = time.perf_counter()
    metrics = evaluate(
        model,
        test_loader,
        torch.nn.CrossEntropyLoss(),
        config.device,
        top_k=(1, 3, 5),
        non_blocking=config.non_blocking,
        amp=config.amp,
    )
    metrics["runtime_seconds"] = time.perf_counter() - start
    state["final_test_completed"] = True
    state["final_test"] = metrics
    atomic_write_json(Path(root) / "controller_state.json", state)
    return metrics


def run_progressive_training(train_dataset, val_dataset, test_dataset, num_classes, config, device, resume=False):
    """Run or resume the full progressive training pipeline."""

    config = _with_device(config, device)
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "controller_state.json"
    started_at = time.time()
    state = {
        "controller_status": "RUNNING",
        "current_stage": "PILOT",
        "started_at": started_at,
        "final_test_completed": False,
    }
    if resume and state_path.exists():
        state = read_json(state_path)
        started_at = state.get("started_at", started_at)

    budget = RuntimeBudget(config.max_runtime_hours, started_at=started_at)
    subsets = make_progressive_subsets(train_dataset, val_dataset, config)
    test_for_run = (
        Subset(test_dataset, range(min(config.smoke_test_graphs, len(test_dataset))))
        if config.smoke
        else test_dataset
    )
    full_train_for_run = (
        Subset(
            train_dataset,
            deterministic_indices(
                len(train_dataset),
                min(config.smoke_full_train_graphs, len(train_dataset)),
                config.seed + 20_000,
            ),
        )
        if config.smoke
        else train_dataset
    )
    full_val_for_run = (
        Subset(
            val_dataset,
            deterministic_indices(
                len(val_dataset),
                min(config.smoke_full_val_graphs, len(val_dataset)),
                config.seed + 30_000,
            ),
        )
        if config.smoke
        else val_dataset
    )
    experiment_scope = "SMOKE_TEST" if config.smoke else "FULL_DATASET"
    atomic_write_json(root / "experiment_config.json", {**asdict(config), "device": str(device)})

    validate_graph_splits(
        {
            "train": subsets["pilot_train"],
            "val": subsets["pilot_val"],
            "test": test_for_run,
        },
        num_classes,
        sample_size=2,
    )

    print("=" * 50)
    print("PROGRESSIVE CHESS GAT TRAINING")
    print("=" * 50)
    print(f"Scope: {experiment_scope}")
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print(f"AMP: {config.amp}")

    if budget.exceeded():
        state["controller_status"] = "TIME_LIMIT_REACHED"
        state["stop_reason"] = "TIME_LIMIT_REACHED"
        atomic_write_json(state_path, state)
        report = {
            "experiment_scope": experiment_scope,
            "controller_status": "TIME_LIMIT_REACHED",
            "model_performance_status": "INCONCLUSIVE",
            "seed": config.seed,
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
            "dataset": {
                "full_train_size": len(train_dataset),
                "full_val_size": len(val_dataset),
                "full_test_size": len(test_dataset),
                "full_train_size_used": len(full_train_for_run),
                "full_val_size_used": len(full_val_for_run),
                "test_size_used": len(test_for_run),
                **subsets["metadata"],
            },
            "pilot": [],
            "promoted_to_confirmation": [],
            "confirmation": [],
            "selected_full_config": None,
            "full": None,
            "final_test": None,
            "old_5k_baseline_reference": OLD_5K_BASELINE,
            "total_runtime_seconds": budget.elapsed(),
            "test_used_for_selection": False,
        }
        write_final_report(root, report)
        return report

    pilot_results = _load_existing_results(root, "pilot") if resume else []
    completed_pilot_names = {result.name for result in pilot_results if result.status == "COMPLETED"}
    if not budget.exceeded():
        for trial_id, trial_config in enumerate(initial_pilot_configs(config), start=1):
            name = f"pilot_trial_{trial_id:03d}"
            if name in completed_pilot_names:
                continue
            print(f"\n[PILOT {trial_id}/{config.pilot_max_trials}]")
            result = run_training_run(
                name,
                trial_config,
                subsets["pilot_train"],
                subsets["pilot_val"],
                num_classes,
                config,
                config.pilot_max_epochs,
                config.pilot_patience,
                root / "pilot" / f"trial_{trial_id:03d}",
                budget,
            )
            pilot_results.append(result)
            state["completed_pilot_trials"] = [asdict(result) for result in pilot_results]
            atomic_write_json(state_path, state)
            if budget.exceeded():
                break

    ranked_pilot = rank_results(pilot_results)
    if not ranked_pilot:
        state["controller_status"] = "FAILED_PILOT"
        state["stop_reason"] = "NO_VALID_CONFIGURATION"
        atomic_write_json(state_path, state)
        raise RuntimeError("No valid pilot configuration completed.")
    promoted = ranked_pilot[: config.confirmation_top_k]
    state["promoted_pilot_configs"] = [asdict(result.config) for result in promoted]

    confirmation_results = _load_existing_results(root, "confirmation") if resume else []
    completed_confirmation_names = {result.name for result in confirmation_results if result.status == "COMPLETED"}
    state["current_stage"] = "CONFIRMATION"
    atomic_write_json(state_path, state)
    if not budget.exceeded():
        for run_id, pilot_result in enumerate(promoted, start=1):
            name = f"confirmation_run_{run_id:03d}"
            if name in completed_confirmation_names:
                continue
            print(f"\n[CONFIRMATION {run_id}/{len(promoted)}]")
            result = run_training_run(
                name,
                pilot_result.config,
                subsets["confirmation_train"],
                subsets["confirmation_val"],
                num_classes,
                config,
                config.confirmation_max_epochs,
                config.confirmation_patience,
                root / "confirmation" / f"run_{run_id:03d}",
                budget,
            )
            confirmation_results.append(result)
            state["completed_confirmation_runs"] = [asdict(result) for result in confirmation_results]
            atomic_write_json(state_path, state)
            if budget.exceeded():
                break

    ranked_confirmation = rank_results(confirmation_results)
    if not ranked_confirmation:
        state["controller_status"] = "FAILED_CONFIRMATION"
        state["stop_reason"] = "NO_VALID_CONFIGURATION"
        atomic_write_json(state_path, state)
        raise RuntimeError("No valid confirmation configuration completed.")

    selected = ranked_confirmation[0].config
    state["selected_full_config"] = asdict(selected)
    state["current_stage"] = "FULL"
    atomic_write_json(state_path, state)

    full_summary_path = root / "full" / "summary.json"
    full_result = _result_from_dict(read_json(full_summary_path)) if resume and full_summary_path.exists() else None
    if full_result is None or full_result.status != "COMPLETED":
        print("\n[FULL]")
        full_result = run_training_run(
            "full",
            selected,
            full_train_for_run,
            full_val_for_run,
            num_classes,
            config,
            config.full_max_epochs,
            config.full_patience,
            root / "full",
            budget,
            use_scheduler=True,
            resume=resume,
        )
        state["full"] = asdict(full_result)
        atomic_write_json(state_path, state)

    final_test = None
    if full_result.status == "COMPLETED" and not budget.exceeded():
        state["current_stage"] = "FINAL_TEST"
        atomic_write_json(state_path, state)
        print("\nFINAL TEST" if not state.get("final_test_completed") else "\nFINAL TEST already completed; reusing stored result")
        final_test = run_final_test(test_for_run, full_result, num_classes, config, root, state)
        state["controller_status"] = "COMPLETED"
        state["stop_reason"] = full_result.stop_reason
    elif budget.exceeded():
        state["controller_status"] = "TIME_LIMIT_REACHED"
        state["stop_reason"] = "TIME_LIMIT_REACHED"
    else:
        state["controller_status"] = "FAILED_FULL"
        state["stop_reason"] = full_result.stop_reason
    atomic_write_json(state_path, state)

    report = {
        "experiment_scope": experiment_scope,
        "controller_status": state["controller_status"],
        "model_performance_status": model_performance_status(final_test),
        "seed": config.seed,
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
        "dataset": {
            "full_train_size": len(train_dataset),
            "full_val_size": len(val_dataset),
            "full_test_size": len(test_dataset),
            "full_train_size_used": len(full_train_for_run),
            "full_val_size_used": len(full_val_for_run),
            "test_size_used": len(test_for_run),
            **subsets["metadata"],
        },
        "pilot": [{**asdict(result), "config": asdict(result.config)} for result in pilot_results],
        "promoted_to_confirmation": [asdict(result.config) for result in promoted],
        "confirmation": [{**asdict(result), "config": asdict(result.config)} for result in confirmation_results],
        "selected_full_config": asdict(selected),
        "full": {**asdict(full_result), "config": asdict(full_result.config)} if full_result else None,
        "final_test": final_test,
        "old_5k_baseline_reference": OLD_5K_BASELINE,
        "total_runtime_seconds": budget.elapsed(),
        "test_used_for_selection": False,
    }
    write_final_report(root, report)
    return report

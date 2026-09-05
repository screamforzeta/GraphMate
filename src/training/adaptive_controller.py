"""Adaptive controller for ChessGATNoTiming training.

Purpose:
    Run controlled no-timing GAT training trials, diagnose convergence, choose
    one hyperparameter change at a time, and select the best validation trial.
Input:
    Train/validation/test PyG graph lists, move vocabulary size, and controller
    configuration.
Output:
    Per-trial checkpoints/history plus final adaptive JSON/Markdown reports.
Role:
    Automates CURRENT_PYG_BASELINE optimization without changing architecture.
"""

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import time

import torch
from torch_geometric.loader import DataLoader

from src.graph.pyg_dataset import (
    ShardAwareShuffleSampler,
    is_sharded_dataset,
)
from src.training.chess_gat_trainer import (
    ChessGATTrainingConfig,
    EarlyStoppingState,
    build_model,
    evaluate,
    save_checkpoint,
    set_seed,
    train_one_epoch,
    validate_graph_splits,
)
from src.training.convergence import (
    DIVERGING,
    IMPROVING,
    OVERFITTING,
    PLATEAU,
    UNDERFITTING_OR_CAPACITY_LIMIT,
    UNSTABLE,
    analyze_convergence,
)


ADAPTIVE_ROOT = Path(
    "artifacts/adaptive_training/chess_gat_no_timing"
)
LR_VALUES = [1e-3, 5e-4, 2.5e-4, 1e-4]
WEIGHT_DECAY_VALUES = [0.0, 1e-5, 1e-4, 1e-3]
DROPOUT_VALUES = [0.10, 0.20, 0.30]


@dataclass(frozen=True)
class TrialConfig:
    """Immutable hyperparameters for one adaptive trial."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.10
    batch_size: int = 32


@dataclass
class AdaptiveTrainingConfig:
    """Configuration for the adaptive training controller."""

    max_trials: int = 6
    epochs_per_trial: int = 30
    patience: int = 5
    seed: int = 42
    min_delta: float = 0.0
    min_relative_val_improvement: float = 0.005
    plateau_window: int = 4
    overfit_window: int = 3
    max_lr_reductions: int = 3
    max_dropout_increases: int = 2
    max_weight_decay_increases: int = 2
    max_total_epochs: int = 120
    batch_size: int = 32
    num_workers: int = 0
    limit_train_graphs: int | None = None
    limit_val_graphs: int | None = None
    limit_test_graphs: int | None = None
    output_root: str = str(ADAPTIVE_ROOT)


@dataclass
class TrialResult:
    """Summary of one completed adaptive trial."""

    trial_id: int
    config: TrialConfig
    best_epoch: int
    best_val_loss: float
    best_val_top1: float
    best_val_top3: float
    best_val_top5: float
    final_train_loss: float
    final_val_loss: float
    status: str
    diagnosis: str
    next_action: str
    elapsed_time: float
    epochs_completed: int
    checkpoint_path: str


def _write_json(path, payload):
    """Write JSON payload to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _trial_dir(output_root, trial_id):
    """Return the directory for one trial."""

    return Path(output_root) / f"trial_{trial_id:03d}"


def _make_loader(graphs, batch_size, shuffle, num_workers):
    """Create a PyG DataLoader for adaptive trials."""

    sampler = (
        ShardAwareShuffleSampler(graphs)
        if shuffle and is_sharded_dataset(graphs)
        else None
    )
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
    )


def _next_value(values, current):
    """Return the next larger-index value from an ordered list."""

    index = values.index(current)
    if index + 1 >= len(values):
        return None
    return values[index + 1]


def choose_next_trial(trial_config, diagnosis, controller_state, adaptive_config):
    """Choose the next immutable trial config from a diagnosis.

    Parameters:
        trial_config: Current TrialConfig.
        diagnosis: ConvergenceDiagnosis from the current trial.
        controller_state: Mutable controller counters.
        adaptive_config: Adaptive controller limits.
    Returns:
        Tuple of next TrialConfig or None, plus action string.
    Side effects:
        Mutates controller_state counters only when an action is selected.
    """

    status = diagnosis.status

    if status in {DIVERGING, UNSTABLE, PLATEAU}:
        next_lr = _next_value(LR_VALUES, trial_config.learning_rate)
        if (
            next_lr is not None
            and controller_state["lr_reductions_used"]
            < adaptive_config.max_lr_reductions
        ):
            controller_state["lr_reductions_used"] += 1
            return (
                TrialConfig(
                    learning_rate=next_lr,
                    weight_decay=trial_config.weight_decay,
                    dropout=trial_config.dropout,
                    batch_size=trial_config.batch_size,
                ),
                f"learning_rate {trial_config.learning_rate} -> {next_lr}",
            )
        return None, "STOP_NO_LOWER_LEARNING_RATE_AVAILABLE"

    if status == OVERFITTING:
        next_dropout = _next_value(DROPOUT_VALUES, trial_config.dropout)
        if (
            next_dropout is not None
            and controller_state["dropout_increases_used"]
            < adaptive_config.max_dropout_increases
        ):
            controller_state["dropout_increases_used"] += 1
            return (
                TrialConfig(
                    learning_rate=trial_config.learning_rate,
                    weight_decay=trial_config.weight_decay,
                    dropout=next_dropout,
                    batch_size=trial_config.batch_size,
                ),
                f"dropout {trial_config.dropout} -> {next_dropout}",
            )

        next_wd = _next_value(WEIGHT_DECAY_VALUES, trial_config.weight_decay)
        if (
            next_wd is not None
            and controller_state["weight_decay_increases_used"]
            < adaptive_config.max_weight_decay_increases
        ):
            controller_state["weight_decay_increases_used"] += 1
            return (
                TrialConfig(
                    learning_rate=trial_config.learning_rate,
                    weight_decay=next_wd,
                    dropout=trial_config.dropout,
                    batch_size=trial_config.batch_size,
                ),
                f"weight_decay {trial_config.weight_decay} -> {next_wd}",
            )
        return None, "STOP_NO_REGULARIZATION_CHANGE_AVAILABLE"

    if status == UNDERFITTING_OR_CAPACITY_LIMIT:
        next_lr = _next_value(LR_VALUES, trial_config.learning_rate)
        if next_lr is not None and controller_state["lr_reductions_used"] < adaptive_config.max_lr_reductions:
            controller_state["lr_reductions_used"] += 1
            return (
                TrialConfig(
                    learning_rate=next_lr,
                    weight_decay=trial_config.weight_decay,
                    dropout=trial_config.dropout,
                    batch_size=trial_config.batch_size,
                ),
                f"learning_rate {trial_config.learning_rate} -> {next_lr}",
            )
        return None, "STOP_POSSIBLE_CAPACITY_OR_REPRESENTATION_LIMIT"

    if status == IMPROVING:
        return None, "STOP_TRIAL_STILL_IMPROVING_WITHIN_BUDGET"

    return None, "STOP_NO_ACTION_FOR_DIAGNOSIS"


def select_best_trial(trial_results):
    """Select the best trial by validation loss and validation top-k tie-breakers."""

    return min(
        trial_results,
        key=lambda result: (
            result.best_val_loss,
            -result.best_val_top5,
            -result.best_val_top1,
        ),
    )


def summarize_target_space(train_graphs, num_classes):
    """Compute descriptive target sparsity diagnostics from train graphs."""

    counts = {}
    for graph in train_graphs:
        target = int(graph.y.item())
        counts[target] = counts.get(target, 0) + 1

    frequencies = sorted(counts.values())
    midpoint = len(frequencies) // 2
    median = (
        frequencies[midpoint]
        if len(frequencies) % 2 == 1
        else (frequencies[midpoint - 1] + frequencies[midpoint]) / 2
    )

    return {
        "num_classes": num_classes,
        "classes_present": len(counts),
        "classes_absent_from_train_pyg": num_classes - len(counts),
        "examples_per_class_min": min(frequencies),
        "examples_per_class_max": max(frequencies),
        "examples_per_class_median": median,
        "singleton_classes": sum(1 for value in frequencies if value == 1),
    }


def run_trial(
    trial_id,
    trial_config,
    train_graphs,
    val_graphs,
    num_classes,
    adaptive_config,
    device,
):
    """Run one train/validation trial from scratch without test access."""

    set_seed(adaptive_config.seed)

    trial_path = _trial_dir(adaptive_config.output_root, trial_id)
    checkpoint_path = trial_path / "best.pt"
    start = time.perf_counter()

    trainer_config = ChessGATTrainingConfig(
        batch_size=trial_config.batch_size,
        learning_rate=trial_config.learning_rate,
        max_epochs=adaptive_config.epochs_per_trial,
        patience=adaptive_config.patience,
        min_delta=adaptive_config.min_delta,
        weight_decay=trial_config.weight_decay,
        seed=adaptive_config.seed,
        num_workers=adaptive_config.num_workers,
        model_dropout=trial_config.dropout,
        checkpoint_path=str(checkpoint_path),
        history_path=str(trial_path / "history.json"),
    )

    _write_json(trial_path / "config.json", asdict(trial_config))

    train_loader = _make_loader(
        train_graphs,
        trial_config.batch_size,
        True,
        adaptive_config.num_workers,
    )
    val_loader = _make_loader(
        val_graphs,
        trial_config.batch_size,
        False,
        adaptive_config.num_workers,
    )

    model = build_model(
        num_classes,
        dropout=trial_config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trial_config.learning_rate,
        weight_decay=trial_config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    early_stopping = EarlyStoppingState(
        patience=adaptive_config.patience,
        min_delta=adaptive_config.min_delta,
    )

    history = []
    best_metrics = None
    best_epoch = None

    for epoch in range(1, adaptive_config.epochs_per_trial + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            top_k=(1,),
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            top_k=(1, 3, 5),
        )
        improved = early_stopping.update(val_metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_top5": val_metrics["top5"],
            "improved": improved,
        }
        history.append(row)

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

        print(
            f"Epoch {epoch:02d}/{adaptive_config.epochs_per_trial} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_top1={train_metrics['top1'] * 100:.2f}% "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_top1={val_metrics['top1'] * 100:.2f}% "
            f"val_top3={val_metrics['top3'] * 100:.2f}% "
            f"val_top5={val_metrics['top5'] * 100:.2f}%"
        )

        if early_stopping.should_stop:
            break

    diagnosis = analyze_convergence(
        history,
        plateau_window=adaptive_config.plateau_window,
        overfit_window=adaptive_config.overfit_window,
        min_relative_val_improvement=adaptive_config.min_relative_val_improvement,
    )
    elapsed = time.perf_counter() - start

    result = TrialResult(
        trial_id=trial_id,
        config=trial_config,
        best_epoch=best_epoch,
        best_val_loss=early_stopping.best_val_loss,
        best_val_top1=best_metrics["top1"],
        best_val_top3=best_metrics["top3"],
        best_val_top5=best_metrics["top5"],
        final_train_loss=history[-1]["train_loss"],
        final_val_loss=history[-1]["val_loss"],
        status="COMPLETED",
        diagnosis=diagnosis.status,
        next_action="PENDING",
        elapsed_time=elapsed,
        epochs_completed=len(history),
        checkpoint_path=str(checkpoint_path),
    )

    _write_json(trial_path / "history.json", {"history": history})
    _write_json(
        trial_path / "summary.json",
        {
            **asdict(result),
            "config": asdict(trial_config),
            "diagnosis_reason": diagnosis.reason,
            "warnings": diagnosis.warnings,
        },
    )

    return result, diagnosis


def evaluate_best_trial_on_test(best_trial, test_graphs, num_classes, device):
    """Evaluate the selected validation trial on test exactly once."""

    model = build_model(
        num_classes,
        dropout=best_trial.config.dropout,
    ).to(device)
    checkpoint = torch.load(
        best_trial.checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = _make_loader(
        test_graphs,
        best_trial.config.batch_size,
        False,
        0,
    )
    return evaluate(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        device,
        top_k=(1, 3, 5),
    )


def _build_warnings(trial_results, target_space, termination_reason):
    """Create structured diagnostic warnings for the final report."""

    warnings = []
    if target_space["classes_absent_from_train_pyg"] > 0 or target_space["singleton_classes"] > 0:
        warnings.append(
            {
                "code": "WARNING_TARGET_SPACE_TOO_LARGE",
                "possible_causes": [
                    "CURRENT_PYG_BASELINE has few graphs relative to move classes",
                    "move target distribution is sparse",
                ],
            }
        )
    if termination_reason == "TRAINING_BUDGET_EXHAUSTED":
        warnings.append(
            {
                "code": "WARNING_NO_CONVERGENCE_WITHIN_BUDGET",
                "possible_causes": [
                    "insufficient number of training graphs",
                    "model capacity too low",
                    "target space too sparse",
                    "missing legal-move masking",
                    "representation not sufficiently informative",
                ],
            }
        )
    if any(result.diagnosis == OVERFITTING for result in trial_results):
        warnings.append({"code": "WARNING_OVERFITTING", "possible_causes": ["regularization may be insufficient"]})
    if any(result.diagnosis in {DIVERGING, UNSTABLE} for result in trial_results):
        warnings.append({"code": "WARNING_OPTIMIZATION_UNSTABLE", "possible_causes": ["learning rate may be too high"]})
    return warnings


def _write_markdown_report(path, final_report):
    """Write a readable Markdown adaptive training report."""

    lines = [
        "# Adaptive Chess GAT Training Report",
        "",
        f"Dataset scope: `{final_report['dataset_scope']}`",
        f"Termination reason: `{final_report['termination_reason']}`",
        f"Optimization converged: `{final_report['optimization_converged']}`",
        f"Model performance status: `{final_report['model_performance_status']}`",
        "",
        "## Trial Summary",
        "",
        "| Trial | LR | Dropout | WD | Best Epoch | Val Loss | Top1 | Top5 | Diagnosis |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for trial in final_report["trials"]:
        cfg = trial["config"]
        lines.append(
            f"| {trial['trial_id']} | {cfg['learning_rate']} | {cfg['dropout']} | "
            f"{cfg['weight_decay']} | {trial['best_epoch']} | "
            f"{trial['best_val_loss']:.6f} | {trial['best_val_top1']:.4f} | "
            f"{trial['best_val_top5']:.4f} | {trial['diagnosis']} |"
        )

    lines.extend(
        [
            "",
            "## Best Trial",
            "",
            f"Trial: `{final_report['best_trial']['trial_id']}`",
            f"Best validation loss: `{final_report['best_trial']['best_val_loss']}`",
            "",
            "## Final Test",
            "",
            f"Loss: `{final_report['test_metrics']['loss']}`",
            f"Top1: `{final_report['test_metrics']['top1']}`",
            f"Top3: `{final_report['test_metrics']['top3']}`",
            f"Top5: `{final_report['test_metrics']['top5']}`",
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in final_report["warnings"]:
        lines.append(f"- `{warning['code']}`")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_adaptive_training(train_graphs, val_graphs, test_graphs, num_classes, config, device):
    """Run adaptive trial-based training and final test evaluation."""

    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    validate_graph_splits({"train": train_graphs, "val": val_graphs, "test": test_graphs}, num_classes)

    print("=" * 50)
    print("Adaptive Chess GAT Training")
    print("=" * 50)
    print("Dataset scope: CURRENT_PYG_BASELINE")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(f"Classes: {num_classes}")

    target_space = summarize_target_space(train_graphs, num_classes)
    controller_state = {
        "completed_trials": 0,
        "current_trial": 0,
        "best_trial": None,
        "best_val_loss": None,
        "lr_reductions_used": 0,
        "dropout_increases_used": 0,
        "weight_decay_increases_used": 0,
        "termination_reason": None,
        "epochs_used": 0,
    }

    current_config = TrialConfig(batch_size=config.batch_size)
    trial_results = []
    next_action = "INITIAL_TRIAL"

    while len(trial_results) < config.max_trials and controller_state["epochs_used"] < config.max_total_epochs:
        trial_id = len(trial_results) + 1
        print("\n" + "-" * 50)
        print(f"Trial {trial_id}")
        print("-" * 50)
        print(
            f"lr={current_config.learning_rate} "
            f"dropout={current_config.dropout:.2f} "
            f"weight_decay={current_config.weight_decay} "
            f"batch_size={current_config.batch_size}"
        )

        result, diagnosis = run_trial(
            trial_id,
            current_config,
            train_graphs,
            val_graphs,
            num_classes,
            config,
            device,
        )
        controller_state["completed_trials"] += 1
        controller_state["current_trial"] = trial_id
        controller_state["epochs_used"] += result.epochs_completed
        trial_results.append(result)

        next_config, next_action = choose_next_trial(
            current_config,
            diagnosis,
            controller_state,
            config,
        )
        result.next_action = next_action
        print(f"Diagnosis: {diagnosis.status}")
        print(f"Action: {next_action}")

        best_so_far = select_best_trial(trial_results)
        controller_state["best_trial"] = best_so_far.trial_id
        controller_state["best_val_loss"] = best_so_far.best_val_loss
        _write_json(output_root / "controller_state.json", controller_state)

        if next_config is None:
            controller_state["termination_reason"] = next_action
            break
        if len(trial_results) >= config.max_trials:
            controller_state["termination_reason"] = "MAX_TRIALS_REACHED"
            break
        if controller_state["epochs_used"] >= config.max_total_epochs:
            controller_state["termination_reason"] = "TRAINING_BUDGET_EXHAUSTED"
            break

        current_config = next_config

    if controller_state["termination_reason"] is None:
        controller_state["termination_reason"] = "TRAINING_BUDGET_EXHAUSTED"

    best_trial = select_best_trial(trial_results)
    best_overall = output_root / "best_overall.pt"
    shutil.copyfile(best_trial.checkpoint_path, best_overall)
    test_metrics = evaluate_best_trial_on_test(best_trial, test_graphs, num_classes, device)

    optimization_converged = controller_state["termination_reason"] not in {
        "TRAINING_BUDGET_EXHAUSTED",
        "MAX_TRIALS_REACHED",
    }
    final_report = {
        "adaptive_training_pipeline_ready": True,
        "dataset_scope": "CURRENT_PYG_BASELINE",
        "initial_config": asdict(TrialConfig(batch_size=config.batch_size)),
        "controller_config": asdict(config),
        "number_of_trials": len(trial_results),
        "trials": [{**asdict(result), "config": asdict(result.config)} for result in trial_results],
        "best_trial": {**asdict(best_trial), "config": asdict(best_trial.config)},
        "best_overall_checkpoint": str(best_overall),
        "test_metrics": test_metrics,
        "termination_reason": controller_state["termination_reason"],
        "optimization_converged": optimization_converged,
        "model_performance_status": "INCONCLUSIVE",
        "target_space": target_space,
        "random_baselines": {
            "random_top1": 1 / num_classes,
            "random_top3": min(3, num_classes) / num_classes,
            "random_top5": min(5, num_classes) / num_classes,
        },
        "warnings": _build_warnings(trial_results, target_space, controller_state["termination_reason"]),
        "recommended_next_actions": [
            "Run full adaptive baseline on CURRENT_PYG_BASELINE",
            "Consider full PyG dataset generation",
            "Consider legal move masking if performance remains weak",
        ],
        "test_set_used_for_tuning": False,
    }

    _write_json(output_root / "final_report.json", final_report)
    _write_markdown_report(output_root / "final_report.md", final_report)
    _write_json(output_root / "controller_state.json", controller_state)

    print("\nTRIAL SUMMARY")
    print("Trial | LR | Dropout | WD | Best Epoch | Val Loss | Top1 | Top5 | Diagnosis")
    for result in trial_results:
        print(
            f"{result.trial_id} | {result.config.learning_rate} | "
            f"{result.config.dropout:.2f} | {result.config.weight_decay} | "
            f"{result.best_epoch} | {result.best_val_loss:.6f} | "
            f"{result.best_val_top1:.4f} | {result.best_val_top5:.4f} | "
            f"{result.diagnosis}"
        )

    print("\nBEST TRIAL")
    print(f"trial: {best_trial.trial_id}")
    print(f"config: {asdict(best_trial.config)}")
    print(f"best_epoch: {best_trial.best_epoch}")
    print(f"best_val_loss: {best_trial.best_val_loss:.6f}")
    print(f"best_val_top1: {best_trial.best_val_top1:.4f}")
    print(f"best_val_top3: {best_trial.best_val_top3:.4f}")
    print(f"best_val_top5: {best_trial.best_val_top5:.4f}")

    print("\nFINAL TEST")
    print(f"test_loss: {test_metrics['loss']:.6f}")
    print(f"test_top1: {test_metrics['top1']:.4f}")
    print(f"test_top3: {test_metrics['top3']:.4f}")
    print(f"test_top5: {test_metrics['top5']:.4f}")
    print(f"termination_reason: {controller_state['termination_reason']}")
    print(f"final_report_json: {output_root / 'final_report.json'}")
    print(f"final_report_md: {output_root / 'final_report.md'}")

    return final_report

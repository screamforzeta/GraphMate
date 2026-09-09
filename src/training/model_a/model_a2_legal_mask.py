"""Training controller for MODEL_A2_LEGAL_MASK_NO_TIMING.

Purpose:
    Train the unchanged ChessGATNoTiming backbone from scratch with per-sample
    legal-move masking applied to loss and official A2 metrics.
Input:
    Sharded PyG train/validation/test datasets, final puzzle CSV metadata, and
    train-derived move vocabulary.
Output:
    Separate A2 artifacts: best.pt, last.pt, history.json,
    controller_state.json, final_report.json, and final_report.md.
Run:
    python3 -m src.cli.training.train_model_a2_legal_mask --device cuda
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch

from src.graph.pyg_dataset import load_move_encoder
from src.models import count_trainable_parameters
from src.training.model_a.chess_gat_trainer import (
    EarlyStoppingState,
    build_model,
    load_checkpoint,
    make_grad_scaler,
    make_loaders,
    save_checkpoint,
    set_loader_epoch,
    set_seed,
    validate_graph_splits,
)
from src.training.model_a.legal_mask import (
    build_legal_mask_for_batch,
    masked_cross_entropy,
)
from src.training.common.metrics import compute_topk_accuracies


RUN_ID = "MODEL_A2_LEGAL_MASK_NO_TIMING"
OUTPUT_ROOT = Path("artifacts/model_a2_legal_mask_no_timing")


@dataclass
class ModelA2LegalMaskConfig:
    """Fixed scientific config for Model A2 legal-mask ablation."""

    seed: int = 42
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    dropout: float = 0.30
    batch_size: int = 128
    max_epochs: int = 300
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
    output_root: str = str(OUTPUT_ROOT)
    run_id: str = RUN_ID
    model_class: str = "ChessGATNoTiming"
    config_source: str = "FIXED_FROM_MODEL_A_NO_TIMING_FROZEN_BASELINE"


def _write_json(path, payload):
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trainer_config(config):
    """Build a lightweight object accepted by the existing loader factory."""

    from src.training.model_a.chess_gat_trainer import ChessGATTrainingConfig

    root = Path(config.output_root)
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


def _move_batch_to_device(batch, device, non_blocking=False):
    """Move a PyG batch to device with optional non-blocking transfer."""

    return batch.to(
        device,
        non_blocking=bool(non_blocking and torch.device(device).type == "cuda"),
    )


def _amp_enabled(device, amp):
    """Return True only when CUDA AMP should run."""

    return bool(amp and torch.device(device).type == "cuda")


def _metric_totals():
    """Create accumulation counters for raw and masked metrics."""

    return {
        "loss_sum": 0.0,
        "num_examples": 0,
        "masked_top1_correct": 0.0,
        "masked_top3_correct": 0.0,
        "masked_top5_correct": 0.0,
        "raw_top1_correct": 0.0,
        "raw_top3_correct": 0.0,
        "raw_top5_correct": 0.0,
        "raw_illegal_top1": 0,
        "masked_illegal_top1": 0,
    }


def _update_totals(totals, loss, raw_logits, masked_logits, targets, legal_mask):
    """Accumulate official masked metrics and raw diagnostics."""

    batch_size = targets.numel()
    totals["loss_sum"] += loss.item() * batch_size
    totals["num_examples"] += batch_size
    masked_acc = compute_topk_accuracies(masked_logits, targets, (1, 3, 5))
    raw_acc = compute_topk_accuracies(raw_logits, targets, (1, 3, 5))
    for k in (1, 3, 5):
        totals[f"masked_top{k}_correct"] += masked_acc[k] * batch_size
        totals[f"raw_top{k}_correct"] += raw_acc[k] * batch_size
    raw_top1 = raw_logits.argmax(dim=1)
    masked_top1 = masked_logits.argmax(dim=1)
    rows = torch.arange(batch_size, device=targets.device)
    totals["raw_illegal_top1"] += int((~legal_mask[rows, raw_top1]).sum().item())
    totals["masked_illegal_top1"] += int((~legal_mask[rows, masked_top1]).sum().item())


def _finalize(totals):
    """Finalize accumulated A2 metrics."""

    n = totals["num_examples"]
    if n == 0:
        raise ValueError("Cannot finalize metrics with zero examples.")
    metrics = {
        "loss": totals["loss_sum"] / n,
        "num_examples": n,
        "raw_illegal_top1_rate": totals["raw_illegal_top1"] / n,
        "masked_illegal_top1_rate": totals["masked_illegal_top1"] / n,
    }
    for prefix in ("masked", "raw"):
        for k in (1, 3, 5):
            metrics[f"{prefix}_top{k}"] = totals[f"{prefix}_top{k}_correct"] / n
    return metrics


def run_masked_epoch(model, loader, optimizer, device, move_to_idx, num_classes, config, scaler=None):
    """Run one A2 training epoch with masked CE."""

    model.train()
    totals = _metric_totals()
    use_amp = _amp_enabled(device, config.amp)
    scaler = scaler or make_grad_scaler(device, amp=config.amp)
    criterion = torch.nn.CrossEntropyLoss()
    for batch in loader:
        batch = _move_batch_to_device(batch, device, config.non_blocking)
        optimizer.zero_grad()
        legal_mask = build_legal_mask_for_batch(batch, move_to_idx, num_classes)
        with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
            raw_logits = model(batch)
            loss, masked_logits = masked_cross_entropy(
                raw_logits,
                batch.y,
                legal_mask,
                criterion,
            )
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        _update_totals(totals, loss, raw_logits.detach(), masked_logits.detach(), batch.y, legal_mask)
    return _finalize(totals)


def evaluate_masked(model, loader, device, move_to_idx, num_classes, config):
    """Evaluate A2 using masked official metrics and raw diagnostics."""

    model.eval()
    totals = _metric_totals()
    use_amp = _amp_enabled(device, config.amp)
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device, config.non_blocking)
            legal_mask = build_legal_mask_for_batch(batch, move_to_idx, num_classes)
            with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
                raw_logits = model(batch)
                loss, masked_logits = masked_cross_entropy(
                    raw_logits,
                    batch.y,
                    legal_mask,
                    criterion,
                )
            _update_totals(totals, loss, raw_logits, masked_logits, batch.y, legal_mask)
    return _finalize(totals)


def save_last(path, model, optimizer, scheduler, scaler, epoch, best_val_loss, best_epoch, config, history, num_classes):
    """Save resumable A2 state."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "config": asdict(config),
            "history": history,
            "num_classes": num_classes,
        },
        path,
    )


def train_model_a2(train_graphs, val_graphs, test_graphs, config, device, resume=False):
    """Train or resume MODEL_A2_LEGAL_MASK_NO_TIMING."""

    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    move_to_idx = load_move_encoder()
    num_classes = len(move_to_idx)
    set_seed(config.seed)
    validate_graph_splits({"train": train_graphs, "val": val_graphs, "test": test_graphs}, num_classes)
    train_loader, val_loader, test_loader = make_loaders(
        train_graphs,
        val_graphs,
        test_graphs,
        _trainer_config(config),
    )
    model = build_model(num_classes, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    scaler = make_grad_scaler(device, amp=config.amp)
    early_stopping = EarlyStoppingState(config.early_stopping_patience, config.min_delta)
    history = []
    start_epoch = 1
    best_epoch = None
    if resume:
        checkpoint = load_checkpoint(root / "last.pt", model, optimizer, device)
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        early_stopping.best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = checkpoint.get("best_epoch")
        history = list(checkpoint.get("history", []))

    _write_json(root / "experiment_config.json", {**asdict(config), "num_classes": num_classes})
    start_time = time.perf_counter()
    stop_reason = "MAX_EPOCHS_REACHED"
    for epoch in range(start_epoch, config.max_epochs + 1):
        set_loader_epoch(train_loader, epoch)
        train_metrics = run_masked_epoch(model, train_loader, optimizer, device, move_to_idx, num_classes, config, scaler)
        val_metrics = evaluate_masked(model, val_loader, device, move_to_idx, num_classes, config)
        scheduler.step(val_metrics["loss"])
        improved = early_stopping.update(val_metrics["loss"])
        if improved:
            best_epoch = epoch
            save_checkpoint(
                root / "best.pt",
                model,
                optimizer,
                epoch,
                early_stopping.best_val_loss,
                _trainer_config(config),
                num_classes,
                train_metrics,
                val_metrics,
            )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"], "improved": improved}
        history.append(record)
        save_last(root / "last.pt", model, optimizer, scheduler, scaler, epoch, early_stopping.best_val_loss, best_epoch, config, history, num_classes)
        _write_json(root / "history.json", history)
        _write_json(root / "controller_state.json", {"run_id": RUN_ID, "status": "RUNNING", "epoch": epoch, "best_epoch": best_epoch})
        print(f"Epoch {epoch}/{config.max_epochs} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} masked_val_top1={val_metrics['masked_top1']*100:.2f}% raw_illegal_top1={val_metrics['raw_illegal_top1_rate']*100:.2f}%")
        if early_stopping.should_stop:
            stop_reason = "EARLY_STOPPING"
            break

    best_model = build_model(num_classes, dropout=config.dropout).to(device)
    checkpoint = load_checkpoint(root / "best.pt", best_model, device=device)
    test_metrics = evaluate_masked(best_model, test_loader, device, move_to_idx, num_classes, config)
    report = {
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "stop_reason": stop_reason,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "test_metrics": test_metrics,
        "history": history,
        "config": asdict(config),
        "dataset_sizes": {"train": len(train_graphs), "val": len(val_graphs), "test": len(test_graphs)},
        "training_seconds": time.perf_counter() - start_time,
        "comparison_reference": {
            "model_a_raw_top1": 0.3961672474213732,
            "model_a_raw_top3": 0.5580720094022851,
            "model_a_raw_top5": 0.6275261325010993,
            "model_a_best_legal_top1": 0.4983,
            "model_a_mate_in1_raw_top1": 0.47218,
            "model_a_mate_in1_best_legal_top1": 0.56140,
        },
    }
    _write_json(root / "final_report.json", report)
    _write_json(root / "controller_state.json", {"run_id": RUN_ID, "status": "COMPLETED", "stop_reason": stop_reason, "best_epoch": checkpoint["epoch"]})
    (root / "final_report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report):
    """Render A2 final report markdown."""

    test = report["test_metrics"]
    return "\n".join([
        "# MODEL_A2_LEGAL_MASK_NO_TIMING",
        "",
        "Status: diagnostic ablation result.",
        "",
        "## Official Masked Test Metrics",
        "",
        f"- Loss: `{test['loss']}`",
        f"- Masked Top1: `{test['masked_top1']}`",
        f"- Masked Top3: `{test['masked_top3']}`",
        f"- Masked Top5: `{test['masked_top5']}`",
        f"- Raw Top1 diagnostic: `{test['raw_top1']}`",
        f"- Raw illegal Top1 rate: `{test['raw_illegal_top1_rate']}`",
        f"- Masked illegal Top1 rate: `{test['masked_illegal_top1_rate']}`",
        "",
        "## Comparison Frame",
        "",
        "| Metric | Model A Raw | Model A Best-Legal Diagnostic | Model A2 Legal-Masked |",
        "|---|---:|---:|---:|",
        f"| Global Top1 | {report['comparison_reference']['model_a_raw_top1']} | {report['comparison_reference']['model_a_best_legal_top1']} | {test['masked_top1']} |",
        f"| Global Top3 | {report['comparison_reference']['model_a_raw_top3']} | N/A | {test['masked_top3']} |",
        f"| Global Top5 | {report['comparison_reference']['model_a_raw_top5']} | N/A | {test['masked_top5']} |",
        "",
        "Test remains post-terminal only and does not guide stopping.",
    ])

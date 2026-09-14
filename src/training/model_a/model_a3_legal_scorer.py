"""Training utilities for MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING.

Purpose:
    Train a no-timing GAT that scores legal move candidates directly instead
    of predicting over the fixed move vocabulary.
Input:
    Sharded PyG graph splits with FEN and target_move metadata.
Output:
    A3 checkpoints, history, controller state, and final reports under
    artifacts/model_a3_legal_move_scorer_no_timing/.
Run:
    python3 -m src.cli.training.train_model_a3_legal_scorer --device cuda
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import time

import chess
import torch

from src.graph.pyg_dataset import load_move_encoder
from src.models import count_trainable_parameters
from src.models.model_a.chess_legal_scorer import ChessGATLegalMoveScorer
from src.training.model_a.chess_gat_trainer import (
    EarlyStoppingState,
    make_grad_scaler,
    make_loaders,
    set_loader_epoch,
    set_seed,
    validate_graph_splits,
)


RUN_ID = "MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING"
OUTPUT_ROOT = Path("artifacts/model_a3_legal_move_scorer_no_timing")
COMPARISON_BASELINES = {
    "model_a_raw": {
        "top1": 0.3961672474213732,
        "top3": 0.5580720094022851,
        "top5": 0.6275261325010993,
    },
    "model_a_best_legal": {
        "top1": 0.49814169570959405,
        "top3": 0.704994192743689,
        "top5": 0.7955865272107717,
    },
    "model_a2_masked": {
        "top1": 0.4925667828106852,
        "top3": 0.7149825784802852,
        "top5": 0.8117305459461146,
    },
}


@dataclass
class ModelA3LegalScorerConfig:
    """Fixed scientific config for A3 legal-candidate scoring."""

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
    scorer_hidden_dim: int = 128
    promotion_embedding_dim: int = 8
    output_root: str = str(OUTPUT_ROOT)
    run_id: str = RUN_ID
    model_class: str = "ChessGATLegalMoveScorer"
    config_source: str = "FIXED_FROM_MODEL_A_CONVERGENCE_PROTOCOL"


def _write_json(path, payload):
    """Write JSON with stable formatting.

    Parameters:
        path: Destination file path.
        payload: JSON-serializable object.
    Returns:
        None.
    Side effects:
        Creates parent directories and writes path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _trainer_config(config):
    """Build a loader-compatible config object.

    Parameters:
        config: A3 config.
    Returns:
        ChessGATTrainingConfig with matching runtime loader settings.
    Side effects:
        None.
    """

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


def _amp_enabled(device, amp):
    """Return True only for requested CUDA AMP."""

    return bool(amp and torch.device(device).type == "cuda")


def _move_batch_to_device(batch, device, non_blocking=False):
    """Move a PyG batch to device with optional non-blocking transfer."""

    return batch.to(
        device,
        non_blocking=bool(non_blocking and torch.device(device).type == "cuda"),
    )


def legal_candidates_from_fen(fen):
    """Enumerate legal moves for a FEN with python-chess.

    Parameters:
        fen: Position FEN observed by the solver.
    Returns:
        List of python-chess Move objects in python-chess legal order.
    Side effects:
        Raises ValueError for invalid FEN or positions with no legal moves.
    """

    try:
        board = chess.Board(str(fen))
    except ValueError as exc:
        raise ValueError(f"Invalid FEN for legal candidates: {fen}") from exc
    moves = list(board.legal_moves)
    if not moves:
        raise ValueError(f"No legal candidates for FEN: {fen}")
    return moves


def _as_list(value, batch_size=None):
    """Normalize PyG-batched metadata into a Python list."""

    if isinstance(value, (list, tuple)):
        return list(value)
    if torch.is_tensor(value):
        flat = value.detach().cpu().view(-1).tolist()
        return flat
    if batch_size == 1:
        return [value]
    raise ValueError("Cannot normalize batched metadata to a list.")


def target_moves_from_batch(batch):
    """Return target UCI strings stored in a PyG batch.

    Parameters:
        batch: Batched PyG object with target_move metadata.
    Returns:
        List of target UCI strings.
    Side effects:
        Raises ValueError when target_move is missing.
    """

    if not hasattr(batch, "target_move") or batch.target_move is None:
        raise ValueError(
            "A3 requires Data.target_move metadata; class index y is not enough."
        )
    batch_size = int(batch.global_features.shape[0])
    return [str(move) for move in _as_list(batch.target_move, batch_size)]


def fens_from_batch(batch):
    """Return FEN strings stored in a PyG batch."""

    if not hasattr(batch, "fen") or batch.fen is None:
        raise ValueError("A3 requires Data.fen metadata for legal candidates.")
    batch_size = int(batch.global_features.shape[0])
    return [str(fen) for fen in _as_list(batch.fen, batch_size)]


def build_candidate_batch(batch):
    """Build legal candidates and local target indices for a PyG batch.

    Parameters:
        batch: Batched PyG object containing fen and target_move metadata.
    Returns:
        Dict with candidate_moves, target_indices, candidate_counts, and
        target_uci.
    Side effects:
        Raises ValueError when the target is not legal in its graph position.
    """

    fens = fens_from_batch(batch)
    targets = target_moves_from_batch(batch)
    candidate_moves = []
    target_indices = []
    candidate_counts = []
    for graph_index, (fen, target_uci) in enumerate(zip(fens, targets)):
        moves = legal_candidates_from_fen(fen)
        legal_uci = [move.uci() for move in moves]
        if target_uci not in legal_uci:
            raise ValueError(
                f"Target move {target_uci} is not legal for graph {graph_index}."
            )
        candidate_moves.append(moves)
        target_indices.append(legal_uci.index(target_uci))
        candidate_counts.append(len(moves))
    return {
        "candidate_moves": candidate_moves,
        "target_indices": torch.tensor(target_indices, dtype=torch.long),
        "candidate_counts": candidate_counts,
        "target_uci": targets,
    }


def grouped_cross_entropy(scores, candidate_ptr, target_indices):
    """Compute listwise CE inside each graph's legal candidate group.

    Parameters:
        scores: Flat candidate scores [num_candidates].
        candidate_ptr: Prefix offsets [batch_size + 1].
        target_indices: Correct local candidate index per graph.
    Returns:
        Scalar mean loss across graphs.
    Side effects:
        Raises ValueError for empty groups or invalid target indices.
    """

    losses = []
    for graph_index in range(target_indices.numel()):
        start = int(candidate_ptr[graph_index].item())
        end = int(candidate_ptr[graph_index + 1].item())
        target = int(target_indices[graph_index].item())
        if end <= start:
            raise ValueError(f"Graph {graph_index} has no candidate scores.")
        if target < 0 or target >= end - start:
            raise ValueError(
                f"Target index {target} outside candidate group for graph {graph_index}."
            )
        group_scores = scores[start:end]
        losses.append(torch.logsumexp(group_scores, dim=0) - group_scores[target])
    return torch.stack(losses).mean()


def _empty_totals():
    """Create A3 metric accumulation counters."""

    return {
        "loss_sum": 0.0,
        "num_examples": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "top5_correct": 0,
        "rank_sum": 0.0,
        "ranks": [],
        "candidate_counts": [],
    }


def _update_totals(totals, loss, scores, candidate_ptr, target_indices, candidate_counts):
    """Accumulate candidate Top-k and rank metrics."""

    batch_size = target_indices.numel()
    totals["loss_sum"] += loss.item() * batch_size
    totals["num_examples"] += batch_size
    totals["candidate_counts"].extend(int(count) for count in candidate_counts)
    for graph_index in range(batch_size):
        start = int(candidate_ptr[graph_index].item())
        end = int(candidate_ptr[graph_index + 1].item())
        target = int(target_indices[graph_index].item())
        group_scores = scores[start:end]
        descending = torch.argsort(group_scores, descending=True)
        rank = int((descending == target).nonzero(as_tuple=False).item()) + 1
        totals["rank_sum"] += rank
        totals["ranks"].append(rank)
        for k in (1, 3, 5):
            effective_k = min(k, group_scores.numel())
            if target in descending[:effective_k].tolist():
                totals[f"top{k}_correct"] += 1


def _median(values):
    """Return the median of a non-empty numeric list."""

    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2)


def _finalize(totals):
    """Finalize accumulated A3 metrics."""

    n = totals["num_examples"]
    if n == 0:
        raise ValueError("Cannot finalize metrics with zero examples.")
    counts = totals["candidate_counts"]
    return {
        "loss": totals["loss_sum"] / n,
        "num_examples": n,
        "top1": totals["top1_correct"] / n,
        "top3": totals["top3_correct"] / n,
        "top5": totals["top5_correct"] / n,
        "mean_legal_target_rank": totals["rank_sum"] / n,
        "median_legal_target_rank": _median(totals["ranks"]),
        "illegal_top1_rate": 0.0,
        "candidate_count_mean": sum(counts) / len(counts),
        "candidate_count_min": min(counts),
        "candidate_count_max": max(counts),
        "candidate_count_median": _median(counts),
    }


def run_a3_epoch(model, loader, optimizer, device, config, scaler=None):
    """Run one A3 training epoch.

    Parameters:
        model: ChessGATLegalMoveScorer instance.
        loader: PyG train DataLoader.
        optimizer: Adam optimizer.
        device: Torch device.
        config: A3 runtime config.
        scaler: Optional AMP GradScaler.
    Returns:
        Dict of training metrics.
    Side effects:
        Updates model parameters.
    """

    model.train()
    totals = _empty_totals()
    use_amp = _amp_enabled(device, config.amp)
    scaler = scaler or make_grad_scaler(device, amp=config.amp)
    for batch in loader:
        batch = _move_batch_to_device(batch, device, config.non_blocking)
        candidate = build_candidate_batch(batch)
        target_indices = candidate["target_indices"].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
            output = model(batch, candidate["candidate_moves"])
            loss = grouped_cross_entropy(
                output["scores"],
                output["candidate_ptr"],
                target_indices,
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite A3 training loss encountered.")
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        _update_totals(
            totals,
            loss.detach(),
            output["scores"].detach(),
            output["candidate_ptr"],
            target_indices.detach(),
            candidate["candidate_counts"],
        )
    return _finalize(totals)


def evaluate_a3(model, loader, device, config):
    """Evaluate A3 without updating model parameters."""

    model.eval()
    totals = _empty_totals()
    use_amp = _amp_enabled(device, config.amp)
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device, config.non_blocking)
            candidate = build_candidate_batch(batch)
            target_indices = candidate["target_indices"].to(device)
            with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
                output = model(batch, candidate["candidate_moves"])
                loss = grouped_cross_entropy(
                    output["scores"],
                    output["candidate_ptr"],
                    target_indices,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite A3 evaluation loss encountered.")
            _update_totals(
                totals,
                loss,
                output["scores"],
                output["candidate_ptr"],
                target_indices,
                candidate["candidate_counts"],
            )
    return _finalize(totals)


def build_model_a3(config):
    """Instantiate the A3 no-timing legal-candidate scorer."""

    return ChessGATLegalMoveScorer(
        input_dim=15,
        edge_dim=5,
        hidden_per_head=32,
        heads=4,
        num_layers=2,
        global_feature_dim=4,
        scorer_hidden_dim=config.scorer_hidden_dim,
        promotion_embedding_dim=config.promotion_embedding_dim,
        dropout=config.dropout,
    )


def save_a3_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_loss, best_epoch, early_stopping_counter, config, history):
    """Save a resumable A3 checkpoint."""

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
            "early_stopping_counter": early_stopping_counter,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "config": asdict(config),
            "history": history,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path,
    )


def load_a3_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cpu"):
    """Load an A3 checkpoint into model and optional training state."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint


def _parameter_report(model):
    """Return encoder/scorer/total trainable parameter counts."""

    encoder = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(("gat1", "gat2"))
    )
    scorer = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(("gat1", "gat2"))
    )
    return {
        "encoder_params": encoder,
        "scorer_params": scorer,
        "total_params": count_trainable_parameters(model),
    }


def train_model_a3(train_graphs, val_graphs, test_graphs, config, device, resume=False):
    """Train or resume MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING."""

    run_id = getattr(config, "run_id", RUN_ID)
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    move_to_idx = load_move_encoder()
    validate_graph_splits(
        {"train": train_graphs, "val": val_graphs, "test": test_graphs},
        len(move_to_idx),
    )
    set_seed(config.seed)
    train_loader, val_loader, test_loader = make_loaders(
        train_graphs,
        val_graphs,
        test_graphs,
        _trainer_config(config),
    )
    model = build_model_a3(config).to(device)
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
    early_stopping = EarlyStoppingState(
        config.early_stopping_patience,
        config.min_delta,
    )
    history = []
    start_epoch = 1
    best_epoch = None

    if resume:
        checkpoint = load_a3_checkpoint(
            root / "last.pt",
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        early_stopping.best_val_loss = float(checkpoint["best_val_loss"])
        early_stopping.epochs_without_improvement = int(
            checkpoint.get("early_stopping_counter", 0)
        )
        best_epoch = checkpoint.get("best_epoch")
        history = list(checkpoint.get("history", []))

    parameter_counts = _parameter_report(model)
    experiment_config = {
        **asdict(config),
        "vocabulary_size_for_compatibility": len(move_to_idx),
        "output_space": "variable_legal_candidates",
        "parameter_counts": parameter_counts,
    }
    _write_json(root / "experiment_config.json", experiment_config)

    start_time = time.perf_counter()
    stop_reason = "MAX_EPOCHS_REACHED"
    completed = False
    try:
        for epoch in range(start_epoch, config.max_epochs + 1):
            set_loader_epoch(train_loader, epoch)
            train_metrics = run_a3_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config,
                scaler,
            )
            val_metrics = evaluate_a3(model, val_loader, device, config)
            scheduler.step(val_metrics["loss"])
            improved = early_stopping.update(val_metrics["loss"])
            if improved:
                best_epoch = epoch
                save_a3_checkpoint(
                    root / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    early_stopping.best_val_loss,
                    best_epoch,
                    early_stopping.epochs_without_improvement,
                    config,
                    history,
                )
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
                "lr": optimizer.param_groups[0]["lr"],
                "improved": improved,
            }
            history.append(record)
            save_a3_checkpoint(
                root / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                early_stopping.best_val_loss,
                best_epoch,
                early_stopping.epochs_without_improvement,
                config,
                history,
            )
            _write_json(root / "history.json", history)
            _write_json(
                root / "controller_state.json",
                {
                    "run_id": run_id,
                    "status": "RUNNING",
                    "epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": early_stopping.best_val_loss,
                },
            )
            print(
                f"Epoch {epoch}/{config.max_epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"val_loss={val_metrics['loss']:.6f} "
                f"val_top1={val_metrics['top1']*100:.2f}% "
                f"val_top3={val_metrics['top3']*100:.2f}%",
                flush=True,
            )
            if early_stopping.should_stop:
                stop_reason = "EARLY_STOPPING"
                break
        completed = True
    except KeyboardInterrupt:
        _write_json(
            root / "controller_state.json",
            {
                "run_id": run_id,
                "status": "INTERRUPTED",
                "epoch": history[-1]["epoch"] if history else start_epoch - 1,
                "best_epoch": best_epoch,
                "resume_command": "./venv/bin/python -m src.cli.training.train_model_a3_legal_scorer --resume --device cuda",
            },
        )
        raise

    if not completed:
        return None

    best_model = build_model_a3(config).to(device)
    checkpoint = load_a3_checkpoint(root / "best.pt", best_model, device=device)
    test_metrics = evaluate_a3(best_model, test_loader, device, config)
    report = {
        "run_id": run_id,
        "status": "COMPLETED",
        "stop_reason": stop_reason,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "test_metrics": test_metrics,
        "history": history,
        "config": experiment_config,
        "dataset_sizes": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        },
        "training_seconds": time.perf_counter() - start_time,
        "comparison_reference": COMPARISON_BASELINES,
    }
    _write_json(root / "final_report.json", report)
    _write_json(
        root / "controller_state.json",
        {
            "run_id": run_id,
            "status": "COMPLETED",
            "stop_reason": stop_reason,
            "best_epoch": checkpoint["epoch"],
        },
    )
    (root / "final_report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report):
    """Render A3 final report markdown."""

    test = report["test_metrics"]
    baselines = report["comparison_reference"]
    run_id = report.get("run_id", RUN_ID)
    return "\n".join(
        [
            f"# {run_id}",
            "",
            "Status: terminal test report for the legal-candidate scorer.",
            "",
            "## Official Test Metrics",
            "",
            f"- Candidate CE / NLL loss: `{test['loss']}`",
            f"- Top1: `{test['top1']}`",
            f"- Top3: `{test['top3']}`",
            f"- Top5: `{test['top5']}`",
            f"- Mean legal target rank: `{test['mean_legal_target_rank']}`",
            f"- Median legal target rank: `{test['median_legal_target_rank']}`",
            f"- Illegal Top1 rate: `{test['illegal_top1_rate']}`",
            f"- Mean legal candidates: `{test['candidate_count_mean']}`",
            "",
            "Top-k is capped by the number of legal moves in each position.",
            "",
            "## Shared Comparison Frame",
            "",
            "| Metric | Model A Raw | Model A Best-Legal | Model A2 Masked | Model A3 |",
            "|---|---:|---:|---:|---:|",
            f"| Top1 | {baselines['model_a_raw']['top1']} | {baselines['model_a_best_legal']['top1']} | {baselines['model_a2_masked']['top1']} | {test['top1']} |",
            f"| Top3 | {baselines['model_a_raw']['top3']} | {baselines['model_a_best_legal']['top3']} | {baselines['model_a2_masked']['top3']} | {test['top3']} |",
            f"| Top5 | {baselines['model_a_raw']['top5']} | {baselines['model_a_best_legal']['top5']} | {baselines['model_a2_masked']['top5']} | {test['top5']} |",
            "",
            "Test is evaluated only after terminal training and never drives checkpoint selection.",
        ]
    )

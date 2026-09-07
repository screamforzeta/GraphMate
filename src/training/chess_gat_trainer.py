"""Training loop for ChessGATNoTiming.

Purpose:
    Train, validate, checkpoint, reload, and test the no-timing chess GAT
    baseline on generated PyTorch Geometric graph datasets.
    PyG datasets, move vocabulary size, and training config.
Output:
    Best checkpoint, JSON training history, and final test metrics.
Role:
    Implements CURRENT_PYG_BASELINE training without changing datasets.
"""

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import json
import random
import time

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.graph.pyg_dataset import (
    is_sharded_dataset,
    make_shard_aware_sampler,
)
from src.models import (
    ChessGATNoTiming,
    count_trainable_parameters,
)
from src.training.metrics import (
    compute_topk_accuracies,
)


CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/chess_gat_no_timing_best.pt"
)
HISTORY_PATH = Path(
    "artifacts/training/chess_gat_no_timing_history.json"
)


@dataclass
class ChessGATTrainingConfig:
    """Configuration for the no-timing chess GAT baseline.

    Parameters:
        batch_size: Number of graphs per batch.
        learning_rate: Adam learning rate.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience on validation loss.
        min_delta: Minimum validation loss improvement.
        weight_decay: Adam weight decay.
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker count.
        pin_memory: Whether DataLoader should allocate pinned host memory.
        persistent_workers: Keep DataLoader workers alive across epochs.
        prefetch_factor: Number of batches prefetched by each worker.
        non_blocking: Use non-blocking device transfers when possible.
        amp: Enable CUDA automatic mixed precision.
        top_k: Top-k accuracy values.
        limit_train_graphs: Optional deterministic train subset size.
        limit_val_graphs: Optional deterministic validation subset size.
        limit_test_graphs: Optional deterministic test subset size.
        model_dropout: Dropout used by ChessGATNoTiming.
        checkpoint_path: Best checkpoint path.
        history_path: JSON history path.
    Returns:
        None.
    Side effects:
        None.
    """

    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 30
    patience: int = 5
    min_delta: float = 0.0
    weight_decay: float = 0.0
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    non_blocking: bool = False
    amp: bool = False
    top_k: tuple[int, ...] = (1, 3, 5)
    limit_train_graphs: int | None = None
    limit_val_graphs: int | None = None
    limit_test_graphs: int | None = None
    model_dropout: float = 0.10
    checkpoint_path: str = str(CHECKPOINT_PATH)
    history_path: str = str(HISTORY_PATH)


@dataclass
class EarlyStoppingState:
    """Track validation-loss early stopping state.

    Parameters:
        patience: Number of non-improving epochs allowed.
        min_delta: Minimum loss decrease required for improvement.
    Returns:
        None.
    Side effects:
        Mutates counters when update() is called.
    """

    patience: int
    min_delta: float = 0.0
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0
    should_stop: bool = False

    def update(self, val_loss):
        """Update state from one validation loss.

        Parameters:
            val_loss: Current validation loss.
        Returns:
            True when validation loss improved, otherwise False.
        Side effects:
            Updates best loss, patience counter, and stop flag.
        """

        improved = val_loss < self.best_val_loss - self.min_delta

        if improved:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            self.should_stop = False
        else:
            self.epochs_without_improvement += 1
            self.should_stop = (
                self.epochs_without_improvement >= self.patience
            )

        return improved


def set_seed(seed):
    """Set random seeds for reproducible training runs.

    Parameters:
        seed: Integer random seed.
    Returns:
        None.
    Side effects:
        Sets random, NumPy, PyTorch, and CUDA seeds. PyG/CUDA operations may
        still have environment-specific nondeterminism.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _accuracy_key(k):
    """Return the metric key used for a top-k value.

    Parameters:
        k: Top-k value.
    Returns:
        Metric dictionary key.
    Side effects:
        None.
    """

    return f"top{k}"


def _empty_metric_totals(top_k):
    """Create metric accumulation containers.

    Parameters:
        top_k: Top-k values.
    Returns:
        Dict with loss/correct/example counters.
    Side effects:
        None.
    """

    totals = {
        "loss_sum": 0.0,
        "num_examples": 0,
    }
    for k in top_k:
        totals[f"{_accuracy_key(k)}_correct"] = 0.0

    return totals


def _update_metric_totals(totals, loss, logits, targets, top_k):
    """Accumulate weighted loss and top-k correct counts.

    Parameters:
        totals: Mutable totals dictionary.
        loss: Batch loss tensor.
        logits: Raw logits [batch_size, num_classes].
        targets: Integer labels [batch_size].
        top_k: Top-k values.
    Returns:
        None.
    Side effects:
        Mutates totals.
    """

    batch_size = targets.numel()
    totals["loss_sum"] += loss.item() * batch_size
    totals["num_examples"] += batch_size

    accuracies = compute_topk_accuracies(
        logits,
        targets,
        top_k,
    )
    for k, accuracy in accuracies.items():
        totals[f"{_accuracy_key(k)}_correct"] += (
            accuracy * batch_size
        )


def _finalize_metrics(totals, top_k):
    """Convert accumulated totals into average metrics.

    Parameters:
        totals: Accumulated loss and correct counts.
        top_k: Top-k values.
    Returns:
        Dict with loss, top-k accuracies, and example count.
    Side effects:
        None.
    """

    num_examples = totals["num_examples"]
    if num_examples == 0:
        raise ValueError(
            "Cannot finalize metrics with zero examples."
        )

    metrics = {
        "loss": totals["loss_sum"] / num_examples,
        "num_examples": num_examples,
    }
    for k in top_k:
        metrics[_accuracy_key(k)] = (
            totals[f"{_accuracy_key(k)}_correct"] / num_examples
        )

    return metrics


def _effective_non_blocking(device, non_blocking):
    """Return True only when async transfer can be useful."""

    return bool(non_blocking and torch.device(device).type == "cuda")


def _amp_enabled(device, amp):
    """Return True only for requested CUDA AMP."""

    return bool(amp and torch.device(device).type == "cuda")


def _move_batch_to_device(batch, device, non_blocking=False):
    """Move a PyG batch to device with optional non-blocking transfer."""

    return batch.to(
        device,
        non_blocking=_effective_non_blocking(device, non_blocking),
    )


def make_grad_scaler(device, amp=False):
    """Create a GradScaler enabled only for CUDA AMP."""

    return torch.amp.GradScaler(
        "cuda",
        enabled=_amp_enabled(device, amp),
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    top_k=(1,),
    non_blocking=False,
    amp=False,
    scaler=None,
):
    """Run one training epoch.

    Parameters:
        model: ChessGATNoTiming model.
        loader: PyG DataLoader for train graphs.
        criterion: CrossEntropyLoss.
        optimizer: Adam optimizer.
        device: Torch device.
        top_k: Top-k values to accumulate.
        non_blocking: Use non-blocking transfer when CUDA pinned memory exists.
        amp: Enable CUDA autocast for forward/loss.
        scaler: Optional GradScaler used for AMP backward/update.
    Returns:
        Dict with loss, top-k accuracies, and example count.
    Side effects:
        Updates model parameters.
    """

    model.train()
    totals = _empty_metric_totals(top_k)
    use_amp = _amp_enabled(device, amp)
    if scaler is None:
        scaler = make_grad_scaler(
            device,
            amp=use_amp,
        )

    for batch in loader:
        batch = _move_batch_to_device(
            batch,
            device,
            non_blocking=non_blocking,
        )

        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=torch.device(device).type,
            enabled=use_amp,
        ):
            logits = model(batch)
            loss = criterion(
                logits,
                batch.y,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss encountered."
            )

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        _update_metric_totals(
            totals,
            loss,
            logits.detach(),
            batch.y,
            top_k,
        )

    return _finalize_metrics(
        totals,
        top_k,
    )


def evaluate(
    model,
    loader,
    criterion,
    device,
    top_k=(1, 3, 5),
    non_blocking=False,
    amp=False,
):
    """Evaluate a model without updating parameters.

    Parameters:
        model: ChessGATNoTiming model.
        loader: PyG DataLoader for validation or test graphs.
        criterion: CrossEntropyLoss.
        device: Torch device.
        top_k: Top-k values to report.
        non_blocking: Use non-blocking transfer when CUDA pinned memory exists.
        amp: Enable CUDA autocast for forward/loss.
    Returns:
        Dict with loss, top-k accuracies, and example count.
    Side effects:
        Temporarily switches model to eval mode.
    """

    model.eval()
    totals = _empty_metric_totals(top_k)
    use_amp = _amp_enabled(device, amp)

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(
                batch,
                device,
                non_blocking=non_blocking,
            )
            with torch.amp.autocast(
                device_type=torch.device(device).type,
                enabled=use_amp,
            ):
                logits = model(batch)
                loss = criterion(
                    logits,
                    batch.y,
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite evaluation loss encountered."
                )

            _update_metric_totals(
                totals,
                loss,
                logits,
                batch.y,
                top_k,
            )

    return _finalize_metrics(
        totals,
        top_k,
    )


def set_loader_epoch(loader, epoch):
    """Forward the current epoch to a sampler when it supports epoch seeding.

    Parameters:
        loader: DataLoader whose sampler may implement set_epoch().
        epoch: One-based epoch number.
    Returns:
        None.
    Side effects:
        Updates sampler epoch state for deterministic per-epoch shuffling.
    """

    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def validate_graph_splits(splits, num_classes, sample_size=8):
    """Run lightweight checks on loaded PyG graph splits.

    Parameters:
        splits: Dict mapping split name to graph datasets.
        num_classes: Move vocabulary size.
        sample_size: Number of initial graphs checked per split.
    Returns:
        None.
    Side effects:
        Raises ValueError when a required invariant fails.
    """

    for split_name, graphs in splits.items():
        if len(graphs) == 0:
            raise ValueError(
                f"{split_name} split is empty."
            )

        checked = min(
            len(graphs),
            sample_size,
        )
        for graph_index in range(checked):
            graph = graphs[graph_index]
            prefix = f"{split_name}[{graph_index}]"

            for attribute in ("x", "edge_index", "edge_attr", "global_features", "y"):
                if not hasattr(graph, attribute) or getattr(graph, attribute) is None:
                    raise ValueError(
                        f"{prefix} missing required attribute {attribute}."
                    )

            if graph.x.ndim != 2 or graph.x.shape != (64, 15):
                raise ValueError(
                    f"{prefix} x must have shape [64,15]."
                )
            if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
                raise ValueError(
                    f"{prefix} edge_index must have shape [2,E]."
                )
            if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != 5:
                raise ValueError(
                    f"{prefix} edge_attr must have shape [E,5]."
                )
            if graph.edge_attr.shape[0] != graph.edge_index.shape[1]:
                raise ValueError(
                    f"{prefix} edge_attr rows must match edge count."
                )
            if graph.global_features.shape != (1, 4):
                raise ValueError(
                    f"{prefix} global_features must have shape [1,4]."
                )
            if graph.y.dtype != torch.long:
                raise ValueError(
                    f"{prefix} y must be torch.long."
                )

            target = int(graph.y.item())
            if target < 0 or target >= num_classes:
                raise ValueError(
                    f"{prefix} target {target} outside [0,{num_classes})."
                )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_loss,
    config,
    num_classes,
    train_metrics,
    val_metrics,
):
    """Save the current best model checkpoint.

    Parameters:
        path: Destination checkpoint path.
        model: Trained PyTorch module.
        optimizer: Optimizer with state to save.
        epoch: Epoch number.
        best_val_loss: Best validation loss.
        config: Training config.
        num_classes: Move vocabulary size.
        train_metrics: Metrics from the epoch.
        val_metrics: Validation metrics from the epoch.
    Returns:
        None.
    Side effects:
        Writes a torch checkpoint file.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "num_classes": num_classes,
            "move_vocab_size": num_classes,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a saved checkpoint into a model and optional optimizer.

    Parameters:
        path: Checkpoint path.
        model: Model instance receiving the state dict.
        optimizer: Optional optimizer receiving its state.
        device: Torch device or map_location.
    Returns:
        Parsed checkpoint dictionary.
    Side effects:
        Mutates model and optionally optimizer state.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint


def save_history(path, payload):
    """Save training history as JSON.

    Parameters:
        path: Destination JSON path.
        payload: JSON-serializable training summary.
    Returns:
        None.
    Side effects:
        Writes path to disk.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def make_loaders(train_graphs, val_graphs, test_graphs, config):
    """Create PyG DataLoaders for train, validation, and test splits.

    Parameters:
        train_graphs: Training graph dataset.
        val_graphs: Validation graph dataset.
        test_graphs: Test graph dataset.
        config: Training config.
    Returns:
        Tuple of train, validation, and test DataLoaders.
    Side effects:
        None.
    """

    if config.persistent_workers and config.num_workers <= 0:
        raise ValueError(
            "persistent_workers=True requires num_workers > 0."
        )
    if config.prefetch_factor is not None and config.num_workers <= 0:
        raise ValueError(
            "prefetch_factor requires num_workers > 0."
        )

    loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": (
            config.persistent_workers
            if config.num_workers > 0
            else False
        ),
    }
    if config.num_workers > 0:
        loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_sampler = (
        make_shard_aware_sampler(
            train_graphs,
            shuffle=True,
            seed=config.seed,
        )
        if is_sharded_dataset(train_graphs)
        else None
    )
    val_sampler = (
        make_shard_aware_sampler(
            val_graphs,
            shuffle=False,
            seed=config.seed,
        )
        if is_sharded_dataset(val_graphs)
        else None
    )
    test_sampler = (
        make_shard_aware_sampler(
            test_graphs,
            shuffle=False,
            seed=config.seed,
        )
        if is_sharded_dataset(test_graphs)
        else None
    )
    train_loader = DataLoader(
        train_graphs,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_graphs,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_graphs,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=test_sampler,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def build_model(num_classes, dropout=0.10):
    """Instantiate the approved no-timing chess GAT architecture.

    Parameters:
        num_classes: Number of target move classes.
        dropout: Dropout probability for the approved architecture.
    Returns:
        ChessGATNoTiming instance.
    Side effects:
        Initializes model parameters.
    """

    return ChessGATNoTiming(
        input_dim=15,
        edge_dim=5,
        hidden_per_head=32,
        heads=4,
        num_layers=2,
        global_feature_dim=4,
        classifier_hidden_dim=128,
        num_classes=num_classes,
        dropout=dropout,
    )


def train_model(train_graphs, val_graphs, test_graphs, num_classes, config, device):
    """Train ChessGATNoTiming and evaluate the best checkpoint on test.

    Parameters:
        train_graphs: Training PyG graph dataset.
        val_graphs: Validation PyG graph dataset. Used for early stopping only.
        test_graphs: Test PyG graph dataset. Used once after best checkpoint reload.
        num_classes: Move vocabulary size.
        config: Training configuration.
        device: Torch device.
    Returns:
        Dict with model, history, checkpoint path, and test metrics.
    Side effects:
        Writes the best checkpoint and JSON training history.
    """

    set_seed(config.seed)

    validate_graph_splits(
        {
            "train": train_graphs,
            "val": val_graphs,
            "test": test_graphs,
        },
        num_classes,
    )

    train_loader, val_loader, test_loader = make_loaders(
        train_graphs,
        val_graphs,
        test_graphs,
        config,
    )

    model = build_model(
        num_classes,
        dropout=config.model_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()
    scaler = make_grad_scaler(
        device,
        amp=config.amp,
    )

    print("ChessGATNoTiming training")
    print("baseline: CURRENT_PYG_BASELINE")
    print(f"device: {device}")
    print(f"train graphs: {len(train_graphs):,}")
    print(f"val graphs: {len(val_graphs):,}")
    print(f"test graphs: {len(test_graphs):,}")
    print(f"num_classes: {num_classes:,}")
    print(f"trainable_parameters: {count_trainable_parameters(model):,}")
    print(f"num_workers: {config.num_workers}")
    print(f"pin_memory: {config.pin_memory}")
    print(f"persistent_workers: {config.persistent_workers}")
    print(f"prefetch_factor: {config.prefetch_factor}")
    print(f"non_blocking: {config.non_blocking}")
    print(f"amp: {config.amp} (enabled={_amp_enabled(device, config.amp)})")
    print(f"random_top1 ~= {1 / num_classes:.6f}")
    print("test set discipline: test is used only after best checkpoint reload")

    early_stopping = EarlyStoppingState(
        patience=config.patience,
        min_delta=config.min_delta,
    )
    history = []
    best_epoch = None
    total_start = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        set_loader_epoch(train_loader, epoch)
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            top_k=(1,),
            non_blocking=config.non_blocking,
            amp=config.amp,
            scaler=scaler,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            top_k=config.top_k,
            non_blocking=config.non_blocking,
            amp=config.amp,
        )

        epoch_duration = time.perf_counter() - epoch_start
        improved = early_stopping.update(
            val_metrics["loss"]
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics.get("top3"),
            "val_top5": val_metrics.get("top5"),
            "epoch_duration": epoch_duration,
            "improved": improved,
        }
        history.append(epoch_record)

        if improved:
            best_epoch = epoch
            save_checkpoint(
                config.checkpoint_path,
                model,
                optimizer,
                epoch,
                early_stopping.best_val_loss,
                config,
                num_classes,
                train_metrics,
                val_metrics,
            )

        print(f"Epoch {epoch:02d}/{config.max_epochs}")
        print(
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_top1={train_metrics['top1'] * 100:.2f}% "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_top1={val_metrics['top1'] * 100:.2f}% "
            f"val_top3={val_metrics.get('top3', 0.0) * 100:.2f}% "
            f"val_top5={val_metrics.get('top5', 0.0) * 100:.2f}% "
            f"best_val_loss={early_stopping.best_val_loss:.6f} "
            f"elapsed={epoch_duration:.2f}s"
        )

        if early_stopping.should_stop:
            print(
                f"[INFO] Early stopping after epoch {epoch}."
            )
            break

    if best_epoch is None:
        raise RuntimeError(
            "No checkpoint was saved because validation never improved."
        )

    # Final test uses the selected validation checkpoint, not last in-memory weights.
    best_model = build_model(
        num_classes,
        dropout=config.model_dropout,
    ).to(device)
    checkpoint = load_checkpoint(
        config.checkpoint_path,
        best_model,
        device=device,
    )
    test_metrics = evaluate(
        best_model,
        test_loader,
        criterion,
        device,
        top_k=config.top_k,
        non_blocking=config.non_blocking,
        amp=config.amp,
    )

    total_training_time = time.perf_counter() - total_start
    stopped_early = len(history) < config.max_epochs

    summary = {
        "baseline": "CURRENT_PYG_BASELINE",
        "history": history,
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "total_training_time": total_training_time,
        "test_metrics": test_metrics,
        "config": asdict(config),
        "num_classes": num_classes,
        "random_top1": 1 / num_classes,
        "dataset_sizes": {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        },
        "checkpoint_path": str(config.checkpoint_path),
        "test_set_used_for_early_stopping": False,
    }
    save_history(
        config.history_path,
        summary,
    )

    print("\nFinal test from best checkpoint")
    print(f"best_epoch: {checkpoint['epoch']}")
    print(f"checkpoint: {config.checkpoint_path}")
    print(
        f"test_loss={test_metrics['loss']:.6f} "
        f"test_top1={test_metrics['top1'] * 100:.2f}% "
        f"test_top3={test_metrics.get('top3', 0.0) * 100:.2f}% "
        f"test_top5={test_metrics.get('top5', 0.0) * 100:.2f}%"
    )
    print(f"history: {config.history_path}")
    print(f"total_training_time={total_training_time:.2f}s")

    if device.type == "cuda":
        print(
            "cuda_max_memory_allocated="
            f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f} MB"
        )

    return {
        "model": best_model,
        "history": summary,
        "checkpoint_path": str(config.checkpoint_path),
        "test_metrics": test_metrics,
    }

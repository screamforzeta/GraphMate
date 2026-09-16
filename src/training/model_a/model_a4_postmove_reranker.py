"""Training and cache utilities for MODEL A4 post-move reranking.

Purpose:
    Train a no-timing A4 reranker over the frozen A3 Top-5 candidate set.
Input:
    Sharded PyG train/validation splits and the official frozen A3 checkpoint.
Output:
    A4 cache files, checkpoints, metrics, and training history.
Run:
    python3 -m src.cli.training.train_model_a4_postmove --device cuda
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

import chess
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.evaluation.model_a.model_a_vs_a2_vs_a3 import MODEL_A3_CHECKPOINT
from src.evaluation.model_a.model_a_vs_a2_vs_a3 import load_eval_model_a3
from src.graph.graph_builder import build_graph
from src.graph.pyg_dataset import load_move_encoder
from src.models import count_trainable_parameters
from src.models.model_a.chess_postmove_reranker import (
    OFFICIAL_A4_K,
    ChessA4PostMoveReranker,
)
from src.training.model_a.chess_gat_trainer import (
    EarlyStoppingState,
    make_grad_scaler,
    set_seed,
)
from src.training.model_a.model_a3_legal_scorer import (
    build_candidate_batch,
    legal_candidates_from_fen,
)
from src.models.model_a.chess_legal_scorer import promotion_id


RUN_ID = "MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING"
OUTPUT_ROOT = Path("artifacts/model_a4_postmove_gnn_reranker")
CACHE_ROOT = Path("data/model_a4_postmove")
GRAPH_REPRESENTATION_VERSION = "pyg_chess_graph_v1_node15_edge5_global4"


@dataclass
class ModelA4PostMoveConfig:
    """Fixed scientific config for A4 post-move reranking."""

    seed: int = 42
    top_k: int = OFFICIAL_A4_K
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
    non_blocking: bool = True
    amp: bool = True
    scorer_hidden_dim: int = 128
    output_root: str = str(OUTPUT_ROOT)
    cache_root: str = str(CACHE_ROOT)
    a3_checkpoint: str = str(MODEL_A3_CHECKPOINT)
    run_id: str = RUN_ID


def _write_json(path, payload):
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def checkpoint_sha256(path):
    """Return a SHA256 fingerprint for a checkpoint file."""

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_a3(a3):
    """Freeze A3 parameters and switch it to eval mode."""

    for parameter in a3.parameters():
        parameter.requires_grad = False
    a3.eval()
    return a3


def assert_a3_frozen(a3):
    """Raise when any A3 parameter remains trainable."""

    trainable = [name for name, parameter in a3.named_parameters() if parameter.requires_grad]
    if trainable:
        raise ValueError(f"A3 must be frozen for A4; trainable params: {trainable[:5]}")


def build_model_a4(config=None):
    """Instantiate the A4 post-move GNN reranker."""

    config = config or ModelA4PostMoveConfig()
    return ChessA4PostMoveReranker(
        scorer_hidden_dim=config.scorer_hidden_dim,
        dropout=config.dropout,
    )


def load_frozen_a3(checkpoint_path, device):
    """Load the official A3 checkpoint and freeze all parameters."""

    model, checkpoint = load_eval_model_a3(Path(checkpoint_path), device)
    freeze_a3(model)
    assert_a3_frozen(model)
    return model, checkpoint


def _batch_ptr(batch, batch_size):
    """Return graph node offsets for a PyG batch."""

    ptr = getattr(batch, "ptr", None)
    if ptr is not None:
        return ptr
    counts = torch.bincount(batch.batch, minlength=batch_size)
    return torch.cat(
        [torch.zeros(1, device=batch.batch.device, dtype=torch.long), counts.cumsum(0)]
    )


def _a3_candidate_features(a3, batch, candidate_moves):
    """Return A3 candidate features before A3 scorer for all legal moves.

    Parameters:
        a3: Frozen ChessGATLegalMoveScorer.
        batch: Batched original-position graphs.
        candidate_moves: Per-graph legal move lists.
    Returns:
        Tensor [num_legal_candidates,396] matching A3 candidate representation.
    Side effects:
        None.
    """

    node_embeddings, graph_context = a3.encode_board(batch)
    batch_size = graph_context.shape[0]
    ptr = _batch_ptr(batch, batch_size)
    source_indices = []
    target_indices = []
    graph_indices = []
    promotion_ids = []
    for graph_index, moves in enumerate(candidate_moves):
        base = int(ptr[graph_index].item())
        for move in moves:
            source_indices.append(base + int(move.from_square))
            target_indices.append(base + int(move.to_square))
            graph_indices.append(graph_index)
            promotion_ids.append(promotion_id(move))
    device = node_embeddings.device
    source = torch.tensor(source_indices, dtype=torch.long, device=device)
    target = torch.tensor(target_indices, dtype=torch.long, device=device)
    graph_index_tensor = torch.tensor(graph_indices, dtype=torch.long, device=device)
    promotion_tensor = torch.tensor(promotion_ids, dtype=torch.long, device=device)
    return torch.cat(
        [
            node_embeddings[source],
            node_embeddings[target],
            graph_context[graph_index_tensor],
            a3.promotion_embedding(promotion_tensor),
        ],
        dim=1,
    )


def select_topk_a3(output, target_indices, top_k=OFFICIAL_A4_K):
    """Select A3 Top-K candidates with official device-side ordering.

    Parameters:
        output: A3 output dict with scores, candidate_ptr, and candidate_uci.
        target_indices: Local target indices for original candidate groups.
        top_k: Maximum number of candidates to keep per graph.
    Returns:
        List of per-graph Top-K metadata dictionaries.
    Side effects:
        None.
    """

    rows = []
    ptr = output["candidate_ptr"]
    scores = output["scores"]
    for graph_index in range(target_indices.numel()):
        start = int(ptr[graph_index].item())
        end = int(ptr[graph_index + 1].item())
        group_scores = scores[start:end]
        ordered = torch.argsort(group_scores, descending=True)
        chosen = ordered[: min(int(top_k), ordered.numel())]
        target = int(target_indices[graph_index].item())
        chosen_local = [int(index.item()) for index in chosen]
        target_candidate_index = chosen_local.index(target) if target in chosen_local else None
        top_scores = [float(group_scores[index].detach().cpu().item()) for index in chosen]
        mean_score = sum(top_scores) / len(top_scores)
        rows.append(
            {
                "local_indices": chosen_local,
                "uci": [output["candidate_uci"][start + local] for local in chosen_local],
                "raw_scores": top_scores,
                "centered_scores": [score - mean_score for score in top_scores],
                "target_candidate_index": target_candidate_index,
                "rerankable": target_candidate_index is not None,
            }
        )
    return rows


def apply_candidate_to_fen(fen, move_uci):
    """Return the resulting FEN after applying one legal candidate."""

    board = chess.Board(str(fen))
    move = chess.Move.from_uci(str(move_uci))
    if move not in board.legal_moves:
        raise ValueError(f"Illegal candidate {move_uci} for FEN: {fen}")
    board.push(move)
    return board.fen()


def build_postmove_graph(fen, move_uci, target_move, move_to_idx):
    """Build a PyG graph for the resulting position after one candidate.

    Parameters:
        fen: Solver-position FEN.
        move_uci: Candidate UCI to apply.
        target_move: Original puzzle target move, retained only for y metadata.
        move_to_idx: Train-only move vocabulary.
    Returns:
        PyG Data for the resulting board.
    Side effects:
        None.
    """

    resulting_fen = apply_candidate_to_fen(fen, move_uci)
    graph = build_graph(resulting_fen, target_move, move_to_idx)
    graph.original_fen = str(fen)
    graph.candidate_move = str(move_uci)
    graph.resulting_fen = resulting_fen
    return graph


class A4PostMoveCacheDataset(torch.utils.data.Dataset):
    """Dataset backed by precomputed A4 rerankable examples."""

    def __init__(self, cache_root=CACHE_ROOT, split="train"):
        self.cache_root = Path(cache_root)
        self.split = split
        self.path = self.cache_root / split / "examples.pt"
        if not self.path.exists():
            raise FileNotFoundError(f"A4 cache not found for split {split}: {self.path}")
        self.examples = torch.load(self.path, weights_only=False)

    def __len__(self):
        """Return number of rerankable examples in the cache."""

        return len(self.examples)

    def __getitem__(self, index):
        """Return one cached rerankable example."""

        return self.examples[index]


def collate_a4_examples(examples):
    """Collate cached A4 examples into one training batch."""

    graphs = []
    features = []
    scores = []
    ptr = [0]
    targets = []
    a3_top1 = []
    target_uci = []
    candidate_uci = []
    for example in examples:
        n = len(example["candidate_uci"])
        graphs.extend(example["postmove_graphs"])
        features.append(example["a3_candidate_features"])
        scores.append(example["score_features"])
        ptr.append(ptr[-1] + n)
        targets.append(int(example["target_candidate_index"]))
        a3_top1.append(example["candidate_uci"][0])
        target_uci.append(example["target_move"])
        candidate_uci.extend(example["candidate_uci"])
    return {
        "postmove_batch": Batch.from_data_list(graphs),
        "a3_candidate_features": torch.cat(features, dim=0),
        "score_features": torch.cat(scores, dim=0),
        "candidate_ptr": torch.tensor(ptr, dtype=torch.long),
        "target_indices": torch.tensor(targets, dtype=torch.long),
        "a3_top1": a3_top1,
        "target_uci": target_uci,
        "candidate_uci": candidate_uci,
    }


def move_a4_batch_to_device(batch, device, non_blocking=False):
    """Move an A4 collated batch to device."""

    use_non_blocking = bool(non_blocking and torch.device(device).type == "cuda")
    return {
        **batch,
        "postmove_batch": batch["postmove_batch"].to(device, non_blocking=use_non_blocking),
        "a3_candidate_features": batch["a3_candidate_features"].to(device, non_blocking=use_non_blocking),
        "score_features": batch["score_features"].to(device, non_blocking=use_non_blocking),
        "candidate_ptr": batch["candidate_ptr"].to(device, non_blocking=use_non_blocking),
        "target_indices": batch["target_indices"].to(device, non_blocking=use_non_blocking),
    }


def grouped_cross_entropy_from_topk(scores, candidate_ptr, target_indices):
    """Compute CE across each A4 Top-K candidate group."""

    losses = []
    for graph_index in range(target_indices.numel()):
        start = int(candidate_ptr[graph_index].item())
        end = int(candidate_ptr[graph_index + 1].item())
        target = int(target_indices[graph_index].item())
        if end <= start:
            raise ValueError(f"A4 graph {graph_index} has no candidates.")
        if target < 0 or target >= end - start:
            raise ValueError(f"A4 target {target} outside candidate group.")
        group_scores = scores[start:end]
        losses.append(torch.logsumexp(group_scores, dim=0) - group_scores[target])
    return torch.stack(losses).mean()


def top1_metrics_from_scores(scores, candidate_ptr, target_indices, a3_top1, target_uci):
    """Return conditional and paired Top1 counters for one A4 batch."""

    totals = {
        "num_examples": int(target_indices.numel()),
        "a4_top1_correct": 0,
        "a3_top1_correct": 0,
        "a3_correct_a4_correct": 0,
        "a3_correct_a4_wrong": 0,
        "a3_wrong_a4_correct": 0,
        "a3_wrong_a4_wrong": 0,
    }
    for graph_index in range(target_indices.numel()):
        start = int(candidate_ptr[graph_index].item())
        end = int(candidate_ptr[graph_index + 1].item())
        target = int(target_indices[graph_index].item())
        ordered = torch.argsort(scores[start:end], descending=True)
        a4_correct = int(ordered[0].item()) == target
        a3_correct = str(a3_top1[graph_index]) == str(target_uci[graph_index])
        totals["a4_top1_correct"] += int(a4_correct)
        totals["a3_top1_correct"] += int(a3_correct)
        if a3_correct and a4_correct:
            totals["a3_correct_a4_correct"] += 1
        elif a3_correct and not a4_correct:
            totals["a3_correct_a4_wrong"] += 1
        elif not a3_correct and a4_correct:
            totals["a3_wrong_a4_correct"] += 1
        else:
            totals["a3_wrong_a4_wrong"] += 1
    return totals


def _amp_enabled(device, amp):
    """Return True only when CUDA AMP is requested and available."""

    return bool(amp and torch.device(device).type == "cuda")


def _empty_epoch_totals():
    """Create A4 epoch metric accumulators."""

    return {
        "loss_sum": 0.0,
        "num_examples": 0,
        "a4_top1_correct": 0,
        "a3_top1_correct": 0,
        "a3_correct_a4_correct": 0,
        "a3_correct_a4_wrong": 0,
        "a3_wrong_a4_correct": 0,
        "a3_wrong_a4_wrong": 0,
    }


def _update_epoch_totals(totals, loss, metrics):
    """Accumulate one batch of A4 metrics."""

    n = metrics["num_examples"]
    totals["loss_sum"] += float(loss.item()) * n
    for key, value in metrics.items():
        totals[key] += int(value)


def _finalize_epoch_totals(totals, all_examples):
    """Finalize A4 epoch metrics."""

    n = totals["num_examples"]
    if n == 0:
        raise ValueError("Cannot finalize A4 metrics with zero rerankable examples.")
    a3_preserved_den = totals["a3_correct_a4_correct"] + totals["a3_correct_a4_wrong"]
    a3_recovered_den = totals["a3_wrong_a4_correct"] + totals["a3_wrong_a4_wrong"]
    return {
        "loss": totals["loss_sum"] / n,
        "rerankable_examples": n,
        "all_examples": int(all_examples),
        "a3_recall_at_5": n / all_examples if all_examples else 0.0,
        "a4_conditional_top1": totals["a4_top1_correct"] / n,
        "a4_end_to_end_top1": totals["a4_top1_correct"] / all_examples if all_examples else 0.0,
        "a3_top1_on_rerankable": totals["a3_top1_correct"] / n,
        "paired_transitions": {
            "a3_correct_a4_correct": totals["a3_correct_a4_correct"],
            "a3_correct_a4_wrong": totals["a3_correct_a4_wrong"],
            "a3_wrong_a4_correct": totals["a3_wrong_a4_correct"],
            "a3_wrong_a4_wrong": totals["a3_wrong_a4_wrong"],
        },
        "a3_correct_preserved_rate": (
            totals["a3_correct_a4_correct"] / a3_preserved_den
            if a3_preserved_den else None
        ),
        "a3_errors_recovered_rate": (
            totals["a3_wrong_a4_correct"] / a3_recovered_den
            if a3_recovered_den else None
        ),
    }


def run_a4_epoch(model, loader, optimizer, device, config, scaler=None):
    """Run one A4 training epoch over rerankable cached examples."""

    model.train()
    totals = _empty_epoch_totals()
    scaler = scaler or make_grad_scaler(device, amp=config.amp)
    use_amp = _amp_enabled(device, config.amp)
    for batch in loader:
        batch = move_a4_batch_to_device(batch, device, config.non_blocking)
        optimizer.zero_grad()
        with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
            scores = model(
                batch["a3_candidate_features"],
                batch["postmove_batch"],
                batch["score_features"],
            )
            loss = grouped_cross_entropy_from_topk(
                scores,
                batch["candidate_ptr"],
                batch["target_indices"],
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite A4 training loss encountered.")
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        metrics = top1_metrics_from_scores(
            scores.detach(),
            batch["candidate_ptr"],
            batch["target_indices"],
            batch["a3_top1"],
            batch["target_uci"],
        )
        _update_epoch_totals(totals, loss.detach(), metrics)
    return _finalize_epoch_totals(totals, all_examples=loader.dataset.total_examples)


def evaluate_a4(model, loader, device, config):
    """Evaluate A4 on rerankable cached validation examples."""

    model.eval()
    totals = _empty_epoch_totals()
    use_amp = _amp_enabled(device, config.amp)
    with torch.no_grad():
        for batch in loader:
            batch = move_a4_batch_to_device(batch, device, config.non_blocking)
            with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
                scores = model(
                    batch["a3_candidate_features"],
                    batch["postmove_batch"],
                    batch["score_features"],
                )
                loss = grouped_cross_entropy_from_topk(
                    scores,
                    batch["candidate_ptr"],
                    batch["target_indices"],
                )
            metrics = top1_metrics_from_scores(
                scores,
                batch["candidate_ptr"],
                batch["target_indices"],
                batch["a3_top1"],
                batch["target_uci"],
            )
            _update_epoch_totals(totals, loss, metrics)
    return _finalize_epoch_totals(totals, all_examples=loader.dataset.total_examples)


def cache_manifest_path(cache_root=CACHE_ROOT):
    """Return the A4 cache manifest path."""

    return Path(cache_root) / "manifest.json"


def _cache_provenance(config, split, dataset_len, checkpoint_hash):
    """Return cache provenance metadata."""

    return {
        "run_id": RUN_ID,
        "split": split,
        "dataset_examples": int(dataset_len),
        "a3_checkpoint": str(config.a3_checkpoint),
        "a3_checkpoint_sha256": checkpoint_hash,
        "top_k": int(config.top_k),
        "graph_representation_version": GRAPH_REPRESENTATION_VERSION,
        "generated_at_unix": time.time(),
    }


def build_a4_postmove_cache_for_split(dataset, split, config, device):
    """Build and save A4 rerankable cache for one non-test split.

    Parameters:
        dataset: PyG dataset for train or validation.
        split: Split name. Test is rejected by design.
        config: A4 config.
        device: Device used for frozen A3 retrieval.
    Returns:
        Summary dictionary.
    Side effects:
        Writes data/model_a4_postmove/<split>/examples.pt and manifest data.
    """

    if split not in {"train", "val", "validation"}:
        raise ValueError("A4 cache building is restricted to train and validation.")
    normalized_split = "val" if split == "validation" else split
    checkpoint_hash = checkpoint_sha256(config.a3_checkpoint)
    a3, _ = load_frozen_a3(config.a3_checkpoint, device)
    move_to_idx = load_move_encoder()
    examples = []
    total = 0
    rerankable = 0
    unrerankable = 0
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            original_graphs = batch.to_data_list()
            batch_device = batch.to(device, non_blocking=False)
            candidate = build_candidate_batch(batch_device)
            target_indices = candidate["target_indices"].to(device)
            output = a3(batch_device, candidate["candidate_moves"])
            a3_features = _a3_candidate_features(
                a3,
                batch_device,
                candidate["candidate_moves"],
            )
            topk_rows = select_topk_a3(output, target_indices, config.top_k)
            for graph_index, row in enumerate(topk_rows):
                total += 1
                if not row["rerankable"]:
                    unrerankable += 1
                    continue
                rerankable += 1
                start = int(output["candidate_ptr"][graph_index].item())
                selected_flat = [start + local for local in row["local_indices"]]
                graph = original_graphs[graph_index]
                postmove_graphs = [
                    build_postmove_graph(
                        graph.fen,
                        move_uci,
                        str(graph.target_move),
                        move_to_idx,
                    )
                    for move_uci in row["uci"]
                ]
                score_features = torch.tensor(
                    list(zip(row["raw_scores"], row["centered_scores"])),
                    dtype=torch.float,
                )
                examples.append(
                    {
                        "puzzle_id": str(getattr(graph, "puzzle_id", "")),
                        "source_row_index": int(getattr(graph, "source_row_index", torch.tensor(-1)).item()),
                        "fen": str(graph.fen),
                        "target_move": str(graph.target_move),
                        "candidate_uci": row["uci"],
                        "a3_raw_scores": row["raw_scores"],
                        "a3_centered_scores": row["centered_scores"],
                        "target_candidate_index": int(row["target_candidate_index"]),
                        "a3_candidate_features": a3_features[selected_flat].detach().cpu(),
                        "score_features": score_features,
                        "postmove_graphs": postmove_graphs,
                    }
                )
    split_dir = Path(config.cache_root) / normalized_split
    split_dir.mkdir(parents=True, exist_ok=True)
    torch.save(examples, split_dir / "examples.pt")
    summary = {
        **_cache_provenance(config, normalized_split, len(dataset), checkpoint_hash),
        "total": total,
        "rerankable": rerankable,
        "unrerankable": unrerankable,
        "a3_recall_at_5": rerankable / total if total else 0.0,
    }
    _write_json(split_dir / "summary.json", summary)
    return summary


def build_a4_postmove_cache(train_graphs, val_graphs, config, device):
    """Build A4 post-move caches for train and validation only."""

    train = build_a4_postmove_cache_for_split(train_graphs, "train", config, device)
    val = build_a4_postmove_cache_for_split(val_graphs, "val", config, device)
    manifest = {
        "run_id": RUN_ID,
        "test_split_cached": False,
        "splits": {"train": train, "val": val},
    }
    _write_json(cache_manifest_path(config.cache_root), manifest)
    return manifest


def validate_a4_cache(config):
    """Load and validate the A4 cache manifest."""

    manifest_path = cache_manifest_path(config.cache_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"A4 cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = checkpoint_sha256(config.a3_checkpoint)
    for split in ("train", "val"):
        info = manifest["splits"][split]
        if info["a3_checkpoint_sha256"] != expected_hash:
            raise ValueError(f"A4 cache for {split} is stale: A3 checkpoint hash mismatch.")
        if int(info["top_k"]) != int(config.top_k):
            raise ValueError(f"A4 cache for {split} is stale: top_k mismatch.")
        if info["graph_representation_version"] != GRAPH_REPRESENTATION_VERSION:
            raise ValueError(f"A4 cache for {split} is stale: graph representation mismatch.")
    return manifest


def make_a4_loaders(config):
    """Create train/validation loaders from the A4 cache."""

    manifest = validate_a4_cache(config)
    train = A4PostMoveCacheDataset(config.cache_root, "train")
    val = A4PostMoveCacheDataset(config.cache_root, "val")
    train.total_examples = int(manifest["splits"]["train"]["total"])
    val.total_examples = int(manifest["splits"]["val"]["total"])
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_a4_examples,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_a4_examples,
    )
    return train_loader, val_loader, manifest


def _parameter_report(a3, a4):
    """Return A3 frozen and A4 trainable parameter counts."""

    postmove = count_trainable_parameters(a4.postmove_encoder)
    scorer = sum(parameter.numel() for parameter in a4.scorer.parameters() if parameter.requires_grad)
    a4_trainable = count_trainable_parameters(a4)
    a3_total = sum(parameter.numel() for parameter in a3.parameters())
    return {
        "a3_frozen_params": a3_total,
        "a4_postmove_encoder_params": postmove,
        "a4_scorer_params": scorer,
        "a4_trainable_params": a4_trainable,
        "total_inference_params": a3_total + a4_trainable,
        "a4_candidate_dim": a4.candidate_dim,
    }


def save_a4_checkpoint(path, model, optimizer, scheduler, epoch, metrics, config, manifest, parameter_report):
    """Save the best A4 checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "validation_metrics": metrics,
            "cache_manifest": manifest,
            "parameter_report": parameter_report,
            "requires_frozen_a3_checkpoint": str(config.a3_checkpoint),
        },
        path,
    )


def train_model_a4(config, device, smoke=False):
    """Train A4 from the precomputed train/validation cache."""

    set_seed(config.seed)
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)
    a3, _ = load_frozen_a3(config.a3_checkpoint, device)
    before_state = {key: value.detach().cpu().clone() for key, value in a3.state_dict().items()}
    model = build_model_a4(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )
    train_loader, val_loader, manifest = make_a4_loaders(config)
    scaler = make_grad_scaler(device, amp=config.amp)
    parameters = _parameter_report(a3, model)
    best_score = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    early = EarlyStoppingState(config.early_stopping_patience, config.min_delta)
    history = []
    max_epochs = min(config.max_epochs, 3) if smoke else config.max_epochs
    for epoch in range(1, max_epochs + 1):
        train_metrics = run_a4_epoch(model, train_loader, optimizer, device, config, scaler)
        val_metrics = evaluate_a4(model, val_loader, device, config)
        scheduler.step(val_metrics["a4_end_to_end_top1"])
        improved = (
            val_metrics["a4_end_to_end_top1"] > best_score
            or (
                val_metrics["a4_end_to_end_top1"] == best_score
                and val_metrics["loss"] < best_val_loss
            )
        )
        early.update(-val_metrics["a4_end_to_end_top1"])
        if improved:
            best_score = val_metrics["a4_end_to_end_top1"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            save_a4_checkpoint(
                root / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                val_metrics,
                config,
                manifest,
                parameters,
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if early.should_stop and not smoke:
            break
    for key, value in a3.state_dict().items():
        if not torch.equal(before_state[key], value.detach().cpu()):
            raise RuntimeError("Frozen A3 state changed during A4 training.")
    summary = {
        "run_id": RUN_ID,
        "smoke": bool(smoke),
        "best_epoch": best_epoch,
        "best_validation_end_to_end_top1": best_score,
        "best_validation_loss": best_val_loss,
        "history": history,
        "parameter_report": parameters,
        "test_set_used_for_training": False,
        "test_set_used_for_checkpoint_selection": False,
        "test_set_evaluated": False,
    }
    _write_json(root / ("smoke_summary.json" if smoke else "training_summary.json"), summary)
    return summary

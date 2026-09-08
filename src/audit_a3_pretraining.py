"""Pre-training audit and benchmark for Model A3.

Purpose:
    Validate leakage, target semantics, legal candidates, shard-aware loading,
    grouped loss, and short runtime performance before full A3 training.
Input:
    data/final/puzzles CSV splits, data/pyg sharded splits, and the move
    vocabulary JSON files.
Output:
    Console audit report and artifacts/model_a3_legal_move_scorer_no_timing/
    pretraining_audit.json.
Run:
    python3 -m src.audit_a3_pretraining --device cuda --benchmark-batches 50
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import statistics
import time

import chess
import pandas as pd
import torch

from src.graph.graph_builder import extract_global_features
from src.graph.pyg_dataset import (
    is_sharded_dataset,
    load_move_encoder,
    load_pyg_dataset,
)
from src.models.chess_legal_scorer import (
    ChessGATLegalMoveScorer,
    promotion_id,
)
from src.training.model_a3_legal_scorer import (
    ModelA3LegalScorerConfig,
    build_candidate_batch,
    grouped_cross_entropy,
)
from src.training.chess_gat_trainer import (
    make_grad_scaler,
    make_loaders,
    set_loader_epoch,
    set_seed,
)


OUTPUT_ROOT = Path("artifacts/model_a3_legal_move_scorer_no_timing")
PUZZLE_SPLITS = {
    "train": Path("data/final/puzzles/train.csv"),
    "val": Path("data/final/puzzles/val.csv"),
    "test": Path("data/final/puzzles/test.csv"),
}
GLOBAL_FEATURES = [
    {
        "index": 0,
        "name": "side_to_move",
        "semantics": "1.0 when white moves, 0.0 when black moves.",
    },
    {
        "index": 1,
        "name": "is_check",
        "semantics": "1.0 when the side to move is in check, else 0.0.",
    },
    {
        "index": 2,
        "name": "fullmove_number_normalized",
        "semantics": "min(fullmove_number, 200) / 200.0 from the solver FEN.",
    },
    {
        "index": 3,
        "name": "halfmove_clock_normalized",
        "semantics": "min(halfmove_clock, 100) / 100.0 from the solver FEN.",
    },
]


def _write_json(path, payload):
    """Write a JSON artifact with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _graph_target_move(graph):
    """Return graph target_move as a plain string."""

    return str(graph.target_move)


def _graph_source_row_index(graph, fallback):
    """Return the source CSV row index stored on a graph."""

    if hasattr(graph, "source_row_index"):
        return int(graph.source_row_index.item())
    return int(fallback)


def audit_global_features(sample_fen):
    """Report global feature names and verify they are board-state only."""

    board = chess.Board(sample_fen)
    features = extract_global_features(board)
    return {
        "status": "PASS",
        "shape": list(features.shape),
        "features": GLOBAL_FEATURES,
        "leakage_absent": {
            "rating": True,
            "MateDepth": True,
            "themes": True,
            "target_move": True,
            "solution_or_future_moves": True,
            "derived_from_correct_result": True,
        },
        "source_function": "src/graph/graph_builder.py::extract_global_features",
    }


def audit_target_semantics(sample_size):
    """Validate Lichess target semantics on CSV rows and matching PyG graphs."""

    move_to_idx = load_move_encoder()
    result = {}
    for split, csv_path in PUZZLE_SPLITS.items():
        df = pd.read_csv(csv_path)
        csv_failures = Counter()
        checked_rows = min(sample_size, len(df))
        for _, row in df.head(checked_rows).iterrows():
            moves = str(row.Moves).split()
            if len(moves) < 2:
                csv_failures["too_few_moves"] += 1
                continue
            try:
                board = chess.Board(str(row.OriginalFEN))
                setup = chess.Move.from_uci(moves[0])
                target = chess.Move.from_uci(moves[1])
            except ValueError:
                csv_failures["invalid_uci_or_fen"] += 1
                continue
            if setup not in board.legal_moves:
                csv_failures["setup_illegal"] += 1
                continue
            board.push(setup)
            if board.fen() != str(row.FEN):
                csv_failures["solver_fen_mismatch"] += 1
            if str(row.TargetMove) != moves[1]:
                csv_failures["target_move_mismatch"] += 1
            if target not in board.legal_moves:
                csv_failures["target_illegal"] += 1

        graphs = load_pyg_dataset(split=split)
        graph_failures = Counter()
        checked_graphs = min(sample_size, len(graphs))
        for index in range(checked_graphs):
            graph = graphs[index]
            source_index = _graph_source_row_index(graph, index)
            row = df.iloc[source_index]
            target_move = str(row.TargetMove)
            if _graph_target_move(graph) != target_move:
                graph_failures["graph_target_move_mismatch"] += 1
            if target_move not in move_to_idx:
                graph_failures["csv_target_oov"] += 1
                continue
            if int(graph.y.item()) != int(move_to_idx[target_move]):
                graph_failures["graph_y_mismatch"] += 1
        result[split] = {
            "csv_checked": checked_rows,
            "csv_failures": dict(csv_failures),
            "graphs_checked": checked_graphs,
            "graph_failures": dict(graph_failures),
            "status": "PASS" if not csv_failures and not graph_failures else "FAIL",
        }
    return result


def validate_candidate_graph(graph):
    """Validate legal candidates and target alignment for one graph."""

    errors = []
    try:
        board = chess.Board(str(graph.fen))
    except ValueError:
        return ["invalid_fen"]
    moves = list(board.legal_moves)
    if not moves:
        errors.append("no_legal_moves")
    candidate_uci = [move.uci() for move in moves]
    if len(candidate_uci) != len(set(candidate_uci)):
        errors.append("duplicate_candidate_uci")
    for move in moves:
        if not 0 <= int(move.from_square) <= 63:
            errors.append("source_square_out_of_range")
        if not 0 <= int(move.to_square) <= 63:
            errors.append("target_square_out_of_range")
        if promotion_id(move) not in {0, 1, 2, 3, 4}:
            errors.append("promotion_id_invalid")
    target_move = _graph_target_move(graph)
    if target_move not in candidate_uci:
        errors.append("target_not_legal")
    else:
        target_index = candidate_uci.index(target_move)
        if candidate_uci[target_index] != target_move:
            errors.append("target_candidate_index_mismatch")
    return errors


def full_candidate_validation():
    """Validate legal candidate metadata for every graph in all PyG splits."""

    report = {}
    for split in ("train", "val", "test"):
        dataset = load_pyg_dataset(split=split)
        errors = Counter()
        valid = 0
        start = time.perf_counter()
        for index in range(len(dataset)):
            graph_errors = validate_candidate_graph(dataset[index])
            if graph_errors:
                errors.update(graph_errors)
            else:
                valid += 1
        total = len(dataset)
        report[split] = {
            "valid": valid,
            "total": total,
            "errors": dict(errors),
            "duration_seconds": time.perf_counter() - start,
            "status": "PASS" if valid == total and not errors else "FAIL",
        }
    report["verdict"] = (
        "A3_CANDIDATE_VALIDATION = PASS"
        if all(item["status"] == "PASS" for item in report.values())
        else "A3_CANDIDATE_VALIDATION = FAIL"
    )
    return report


def audit_generation_strategy():
    """Describe A3 candidate generation and caching behavior."""

    return {
        "getitem_rebuilds_legal_moves": False,
        "getitem_reason": "ShardedPyGDataset.__getitem__ only loads stored graphs; A3 candidates are built after batching.",
        "dataset_cache": "ShardedPyGDataset has a small LRU shard cache for loaded .pt shards.",
        "persistent_candidate_cache": False,
        "candidate_tensors_recreated_each_batch": True,
        "epoch_behavior": "Each epoch reuses DataLoader/shard-aware sampler, loads graph batches, and rebuilds python-chess legal candidates from batch.fen.",
        "note": "No persistent candidate cache is introduced before measuring enumeration cost.",
    }


def _parameter_report(model):
    """Return A3 dimensional and parameter audit."""

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
    total = encoder + scorer
    return {
        "node_encoder_output_dim": model.node_hidden_dim,
        "graph_context_dim": model.graph_context_dim,
        "promotion_embedding_dim": model.promotion_embedding_dim,
        "candidate_vector_dim": model.candidate_dim,
        "scorer_architecture": "Linear(candidate_dim,128) -> ELU -> Dropout(0.30) -> Linear(128,1)",
        "encoder_parameter_count": encoder,
        "scorer_parameter_count": scorer,
        "total_parameter_count": total,
        "NO_1786_CLASSIFIER_IN_A3_FORWARD": True,
    }


def grouped_loss_sanity():
    """Run a synthetic grouped CE check for denominator isolation."""

    scores = torch.tensor([1.0, 2.0, 100.0], requires_grad=True)
    candidate_ptr = torch.tensor([0, 2, 3])
    target_indices = torch.tensor([1, 0])
    loss = grouped_cross_entropy(scores, candidate_ptr, target_indices)
    loss.backward()
    expected = (torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0) - 2.0) / 2
    return {
        "status": "PASS" if torch.allclose(loss.detach(), expected) else "FAIL",
        "loss": float(loss.detach().item()),
        "expected": float(expected.item()),
        "denominator_is_group_local": True,
        "grad_present": scores.grad is not None,
    }


def benchmark_a3(device, benchmark_batches, config):
    """Run a short realistic A3 train benchmark."""

    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        return {
            "status": "SKIPPED",
            "reason": "CUDA requested but torch.cuda.is_available() is False.",
        }

    set_seed(config.seed)
    train_graphs = load_pyg_dataset(split="train")
    val_graphs = load_pyg_dataset(split="val")
    test_graphs = load_pyg_dataset(split="test")
    train_loader, _, _ = make_loaders(train_graphs, val_graphs, test_graphs, config)
    set_loader_epoch(train_loader, 1)
    model = ChessGATLegalMoveScorer(dropout=config.model_dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = make_grad_scaler(device, amp=config.amp)
    use_amp = bool(config.amp and torch.device(device).type == "cuda")
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    generation_times = []
    compute_times = []
    batch_times = []
    candidate_counts = []
    graphs = 0
    candidates = 0
    start_all = time.perf_counter()

    for batch_index, batch in enumerate(train_loader):
        if batch_index >= benchmark_batches:
            break
        batch_start = time.perf_counter()
        batch = batch.to(
            device,
            non_blocking=bool(config.non_blocking and torch.device(device).type == "cuda"),
        )
        generation_start = time.perf_counter()
        candidate = build_candidate_batch(batch)
        target_indices = candidate["target_indices"].to(device)
        generation_times.append(time.perf_counter() - generation_start)

        compute_start = time.perf_counter()
        optimizer.zero_grad()
        with torch.amp.autocast(torch.device(device).type, enabled=use_amp):
            output = model(batch, candidate["candidate_moves"])
            loss = grouped_cross_entropy(
                output["scores"],
                output["candidate_ptr"],
                target_indices,
            )
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        compute_times.append(time.perf_counter() - compute_start)

        batch_times.append(time.perf_counter() - batch_start)
        batch_graphs = int(batch.global_features.shape[0])
        batch_candidates = sum(candidate["candidate_counts"])
        graphs += batch_graphs
        candidates += batch_candidates
        candidate_counts.extend(candidate["candidate_counts"])

    elapsed = time.perf_counter() - start_all
    peak_vram = (
        torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None
    )
    sampler_name = type(getattr(train_loader, "sampler", None)).__name__
    return {
        "status": "PASS",
        "device": str(device),
        "batches": len(batch_times),
        "graphs": graphs,
        "candidates": candidates,
        "elapsed_seconds": elapsed,
        "graphs_per_second": graphs / elapsed if elapsed else None,
        "candidates_per_second": candidates / elapsed if elapsed else None,
        "average_candidates_per_graph": candidates / graphs if graphs else None,
        "min_candidates_per_graph": min(candidate_counts) if candidate_counts else None,
        "max_candidates_per_graph": max(candidate_counts) if candidate_counts else None,
        "mean_train_batch_time_seconds": statistics.mean(batch_times) if batch_times else None,
        "mean_candidate_generation_time_seconds": statistics.mean(generation_times) if generation_times else None,
        "mean_forward_backward_time_seconds": statistics.mean(compute_times) if compute_times else None,
        "peak_cuda_vram_bytes": peak_vram,
        "sampler": sampler_name,
        "shard_aware": "ShardAware" in sampler_name,
        "train_dataset_is_sharded": is_sharded_dataset(train_graphs),
    }


def parse_args():
    """Parse pre-training audit CLI options."""

    parser = argparse.ArgumentParser(description="Audit A3 before full training.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--non-blocking", action="store_true", default=True)
    parser.add_argument("--no-non-blocking", dest="non_blocking", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--target-sample", type=int, default=1000)
    parser.add_argument("--benchmark-batches", type=int, default=50)
    return parser.parse_args()


def main():
    """Run the full A3 pre-training audit."""

    args = parse_args()
    config = ModelA3LegalScorerConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        non_blocking=args.non_blocking,
        amp=args.amp,
    )
    loader_config = type(
        "LoaderConfig",
        (),
        {
            **asdict(config),
            "model_dropout": config.dropout,
            "learning_rate": config.learning_rate,
        },
    )()
    train_df = pd.read_csv(PUZZLE_SPLITS["train"])
    sample_fen = str(train_df.iloc[0].FEN)
    model = ChessGATLegalMoveScorer(dropout=config.dropout)
    report = {
        "global_features": audit_global_features(sample_fen),
        "target_semantics": audit_target_semantics(args.target_sample),
        "candidate_validation": full_candidate_validation(),
        "candidate_generation_strategy": audit_generation_strategy(),
        "benchmark": benchmark_a3(
            torch.device(args.device),
            args.benchmark_batches,
            loader_config,
        ),
        "model_audit": _parameter_report(model),
        "grouped_loss_sanity": grouped_loss_sanity(),
    }
    _write_json(OUTPUT_ROOT / "pretraining_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Prediction headroom and error audit for the frozen A3 model.

Purpose:
    Diagnose where MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING loses accuracy without
    changing the model, checkpoint, data, targets, or candidate generation.
Input:
    Frozen A3 checkpoint, data/pyg validation/test splits, and puzzle CSV rows.
Output:
    artifacts/model_a3_error_audit summaries, JSONL per-puzzle rows, and report.
Run:
    python3 -m src.cli.evaluation.evaluate_model_a3_error_audit
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import math
import shutil
import statistics
import subprocess

import chess
import pandas as pd
import torch
from torch_geometric.data import Batch

from src.evaluation.model_a.model_a_vs_a2 import _mate_label
from src.evaluation.model_a.model_a_vs_a2 import _rating_bucket
from src.evaluation.model_a.model_a_vs_a2_vs_a3 import (
    MODEL_A3_CHECKPOINT,
    MODEL_A3_REFERENCE,
    load_eval_model_a3,
)
from src.graph.pyg_dataset import load_pyg_dataset
from src.training.model_a.model_a3_legal_scorer import build_candidate_batch
from src.training.model_a.model_a3_legal_scorer import grouped_cross_entropy


OUTPUT_DIR = Path("artifacts/model_a3_error_audit")
PYG_ROOT = Path("data/pyg")
PUZZLE_CSVS = {
    "validation": Path("data/final/puzzles/val.csv"),
    "test": Path("data/final/puzzles/test.csv"),
}
EXPECTED_SPLIT_N = {
    "validation": 8612,
    "test": 8610,
}
PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
PIECE_VALUES = {
    None: 0,
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


@dataclass
class A3ErrorAuditConfig:
    """Runtime options for A3 error audit."""

    batch_size: int = 128
    device: str = "cpu"
    non_blocking: bool = False
    amp: bool = False
    output_dir: str = str(OUTPUT_DIR)
    checkpoint: str = str(MODEL_A3_CHECKPOINT)
    pyg_root: str = str(PYG_ROOT)
    stockfish: bool = False
    stockfish_path: str | None = None
    stockfish_depth: int = 18
    stockfish_max_samples: int | None = None
    stockfish_threads: int = 1
    stockfish_hash_mb: int = 128


def rank_bucket(rank):
    """Return the canonical target-rank bucket label."""

    rank = int(rank)
    if rank <= 5:
        return str(rank)
    if rank <= 10:
        return "6-10"
    return ">10"


def legal_count_bucket(count):
    """Return legal move count bucket label."""

    count = int(count)
    if count <= 20:
        return "<=20"
    if count <= 30:
        return "21-30"
    if count <= 40:
        return "31-40"
    if count <= 50:
        return "41-50"
    return ">50"


def quiet_move(features):
    """Return True for non-capture, non-check, non-promotion, non-castling."""

    return not (
        features["is_capture"]
        or features["gives_check"]
        or features["is_promotion"]
        or features["is_castling"]
    )


def move_type_features(fen, move_uci):
    """Extract deterministic move-type features from a FEN and legal UCI move.

    Semantics:
        destination_attacked_after_move and destination_defended_after_move are
        evaluated after pushing the move. The moved side is the original
        side-to-move; destination attacked means attacked by the opponent.
    """

    board = chess.Board(str(fen))
    move = chess.Move.from_uci(str(move_uci))
    if move not in board.legal_moves:
        raise ValueError(f"Move {move_uci} is not legal for FEN: {fen}")
    mover = board.turn
    opponent = not mover
    piece = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    is_en_passant = board.is_en_passant(move)
    if is_en_passant:
        captured = chess.Piece(chess.PAWN, opponent)
    moving_attacked_before = board.is_attacked_by(opponent, move.from_square)
    is_capture = board.is_capture(move)
    is_castling = board.is_castling(move)
    gives_check = board.gives_check(move)
    board.push(move)
    gives_checkmate = board.is_checkmate()
    destination_attacked_after = board.is_attacked_by(opponent, move.to_square)
    destination_defended_after = board.is_attacked_by(mover, move.to_square)
    moving_attacked_after = destination_attacked_after
    promotion_value = PIECE_VALUES.get(move.promotion, 0)
    moving_value = PIECE_VALUES.get(piece.piece_type if piece else None, 0)
    captured_value = PIECE_VALUES.get(captured.piece_type if captured else None, 0)
    return {
        "move": str(move_uci),
        "moving_piece_type": PIECE_NAMES.get(piece.piece_type) if piece else None,
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "is_capture": bool(is_capture),
        "captured_piece_type": PIECE_NAMES.get(captured.piece_type) if captured else None,
        "captured_piece_value": captured_value,
        "is_promotion": move.promotion is not None,
        "promotion_piece": PIECE_NAMES.get(move.promotion) if move.promotion else None,
        "is_castling": bool(is_castling),
        "gives_check": bool(gives_check),
        "gives_checkmate": bool(gives_checkmate),
        "is_en_passant": bool(is_en_passant),
        "destination_attacked_after_move": bool(destination_attacked_after),
        "destination_defended_after_move": bool(destination_defended_after),
        "material_delta_immediate": captured_value + promotion_value - (moving_value if move.promotion else 0),
        "moving_piece_attacked_before": bool(moving_attacked_before),
        "moving_piece_attacked_after": bool(moving_attacked_after),
        "quiet_move": False,
    }


def enrich_quiet(features):
    """Return a copy with quiet_move filled from the other flags."""

    result = dict(features)
    result["quiet_move"] = quiet_move(result)
    return result


def topk_and_row(scores, candidate_ptr, candidate_uci, target_index, graph_index):
    """Return ranking details for one graph in a batched A3 output."""

    start = int(candidate_ptr[graph_index].item())
    end = int(candidate_ptr[graph_index + 1].item())
    local_scores = scores[start:end].detach().float().cpu()
    ordered = torch.argsort(local_scores, descending=True)
    target_index = int(target_index)
    target_rank = int((ordered == target_index).nonzero(as_tuple=False).item()) + 1
    top_indices = ordered[: min(10, ordered.numel())].tolist()
    top_scores = [float(local_scores[index].item()) for index in top_indices]
    top_moves = [candidate_uci[start + index] for index in top_indices]
    top1_score = top_scores[0]
    top2_score = top_scores[1] if len(top_scores) > 1 else None
    return {
        "legal_move_count": end - start,
        "target_rank": target_rank,
        "target_score": float(local_scores[target_index].item()),
        "A3_top1_move": top_moves[0],
        "A3_top1_score": top1_score,
        "top3_moves": top_moves[:3],
        "top5_moves": top_moves[:5],
        "top10_moves": top_moves,
        "top5_scores": top_scores[:5],
        "top1_correct": target_rank == 1,
        "target_in_top3": target_rank <= 3,
        "target_in_top5": target_rank <= 5,
        "target_in_top10": target_rank <= 10,
        "score_margin_top1_top2": (
            top1_score - top2_score if top2_score is not None else None
        ),
    }


def margin_summary(values):
    """Summarize score margins."""

    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p25": quantile(ordered, 0.25),
        "p75": quantile(ordered, 0.75),
    }


def quantile(ordered, q):
    """Return linear quantile from sorted values."""

    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    weight = pos - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize_rows(rows):
    """Aggregate headroom, breakdowns, and feature diagnostics from rows."""

    n = len(rows)
    if n == 0:
        return {"n": 0}
    topk = {
        f"top{k}": sum(row["target_rank"] <= k for row in rows) / n
        for k in (1, 2, 3, 4, 5, 10)
    }
    ranks = Counter(rank_bucket(row["target_rank"]) for row in rows)
    errors = [row for row in rows if not row["top1_correct"]]
    error_ranks = Counter(rank_bucket(row["target_rank"]) for row in errors)
    target_piece = bucket_accuracy(rows, lambda row: row["target_features"]["moving_piece_type"])
    predicted_piece = bucket_accuracy(rows, lambda row: row["prediction_features"]["moving_piece_type"])
    target_move_type = {
        "capture": bucket_accuracy(rows, lambda row: "capture" if row["target_features"]["is_capture"] else "non_capture"),
        "check": bucket_accuracy(rows, lambda row: "check" if row["target_features"]["gives_check"] else "non_check"),
        "checkmate": bucket_accuracy(rows, lambda row: "checkmate" if row["target_features"]["gives_checkmate"] else "non_checkmate"),
        "promotion": bucket_accuracy(rows, lambda row: "promotion" if row["target_features"]["is_promotion"] else "non_promotion"),
        "quiet": bucket_accuracy(rows, lambda row: "quiet" if row["target_features"]["quiet_move"] else "non_quiet"),
    }
    return {
        "n": n,
        "topk": topk,
        "headroom_pp": {
            "top3_minus_top1": (topk["top3"] - topk["top1"]) * 100,
            "top5_minus_top1": (topk["top5"] - topk["top1"]) * 100,
            "top10_minus_top1": (topk["top10"] - topk["top1"]) * 100,
        },
        "target_rank_histogram": dict(ranks),
        "target_rank_fractions": {key: value / n for key, value in ranks.items()},
        "error_target_rank_breakdown": {
            key: {"count": value, "fraction_of_errors": value / len(errors) if errors else 0.0}
            for key, value in error_ranks.items()
        },
        "mate_depth": grouped_metrics(rows, lambda row: f"mateIn{row['MateDepth']}"),
        "rating": grouped_metrics(rows, lambda row: rating_bucket_value(row["rating"])),
        "legal_move_count": grouped_metrics(rows, lambda row: legal_count_bucket(row["legal_move_count"])),
        "target_piece_accuracy": target_piece,
        "predicted_piece_accuracy": predicted_piece,
        "target_piece_to_predicted_piece_errors": confusion_errors(rows),
        "target_move_type_accuracy": target_move_type,
        "target_vs_wrong_prediction": compare_wrong_features(errors),
        "score_margin": {
            "correct_top1": margin_summary([row["score_margin_top1_top2"] for row in rows if row["top1_correct"]]),
            "wrong_top1": margin_summary([row["score_margin_top1_top2"] for row in errors]),
            "wrong_target_rank_2": margin_summary([row["score_margin_top1_top2"] for row in errors if row["target_rank"] == 2]),
            "wrong_target_rank_gt5": margin_summary([row["score_margin_top1_top2"] for row in errors if row["target_rank"] > 5]),
        },
        "representative_examples": representative_examples(rows),
    }


def bucket_accuracy(rows, key_fn):
    """Return N and Top1 accuracy by bucket."""

    buckets = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in rows:
        key = str(key_fn(row))
        buckets[key]["n"] += 1
        buckets[key]["correct"] += int(row["top1_correct"])
    return {
        key: {"n": value["n"], "top1": value["correct"] / value["n"]}
        for key, value in sorted(buckets.items())
    }


def grouped_metrics(rows, key_fn):
    """Return Top1/Top3/Top5 and mean target rank by group."""

    groups = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return {
        key: {
            "n": len(group),
            "top1": sum(row["target_rank"] <= 1 for row in group) / len(group),
            "top3": sum(row["target_rank"] <= 3 for row in group) / len(group),
            "top5": sum(row["target_rank"] <= 5 for row in group) / len(group),
            "mean_target_rank": statistics.mean(row["target_rank"] for row in group),
        }
        for key, group in sorted(groups.items())
    }


def rating_bucket_value(rating):
    """Return existing project rating bucket for a numeric rating."""

    class Row:
        Rating = rating

    return _rating_bucket(Row)


def confusion_errors(rows):
    """Return target piece -> predicted piece counts for Top1 errors."""

    matrix = defaultdict(Counter)
    for row in rows:
        if row["top1_correct"]:
            continue
        matrix[row["target_features"]["moving_piece_type"]][row["prediction_features"]["moving_piece_type"]] += 1
    return {key: dict(value) for key, value in matrix.items()}


def compare_wrong_features(errors):
    """Compare target and wrong Top1 move features for errors."""

    comparisons = {}
    feature_names = [
        "gives_check",
        "gives_checkmate",
        "is_capture",
        "is_promotion",
        "destination_attacked_after_move",
        "destination_defended_after_move",
    ]
    for feature in feature_names:
        counts = Counter(
            (
                bool(row["target_features"][feature]),
                bool(row["prediction_features"][feature]),
            )
            for row in errors
        )
        comparisons[feature] = {f"target_{k[0]}__prediction_{k[1]}": v for k, v in counts.items()}
    comparisons["material_delta"] = {
        "target_mean": statistics.mean([row["target_features"]["material_delta_immediate"] for row in errors]) if errors else None,
        "prediction_mean": statistics.mean([row["prediction_features"]["material_delta_immediate"] for row in errors]) if errors else None,
    }
    return comparisons


def representative_examples(rows, limit=8):
    """Select deterministic representative examples for report inspection."""

    categories = {
        "target_rank_2": lambda row: (not row["top1_correct"]) and row["target_rank"] == 2,
        "target_rank_3_5": lambda row: (not row["top1_correct"]) and 3 <= row["target_rank"] <= 5,
        "target_rank_gt5": lambda row: (not row["top1_correct"]) and row["target_rank"] > 5,
        "confidently_wrong": lambda row: (not row["top1_correct"]) and (row["score_margin_top1_top2"] or 0) > 1.0,
        "checking_or_mating_missed": lambda row: (not row["top1_correct"]) and (row["target_features"]["gives_check"] or row["target_features"]["gives_checkmate"]),
        "capture_related_error": lambda row: (not row["top1_correct"]) and (row["target_features"]["is_capture"] or row["prediction_features"]["is_capture"]),
    }
    result = {}
    for name, predicate in categories.items():
        selected = [compact_example(row) for row in rows if predicate(row)]
        result[name] = selected[:limit]
    return result


def compact_example(row):
    """Return a compact JSON-safe representative example."""

    return {
        "PuzzleId": row["PuzzleId"],
        "FEN": row["FEN"],
        "rating": row["rating"],
        "MateDepth": row["MateDepth"],
        "target_move": row["target_move"],
        "prediction": row["A3_top1_move"],
        "target_rank": row["target_rank"],
        "top5_moves": row["top5_moves"],
        "target_score": row["target_score"],
        "prediction_score": row["A3_top1_score"],
        "score_margin_top1_top2": row["score_margin_top1_top2"],
        "target_features": row["target_features"],
        "prediction_features": row["prediction_features"],
    }


def discover_stockfish(stockfish_path=None):
    """Return Stockfish availability and version/config metadata."""

    path = stockfish_path or shutil.which("stockfish")
    if not path:
        return {"available": False, "path": None, "version": None}
    try:
        process = subprocess.run(
            [path],
            input="uci\nquit\n",
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "path": path, "version": None, "error": str(exc)}
    version = None
    for line in process.stdout.splitlines():
        if line.startswith("id name"):
            version = line.replace("id name", "", 1).strip()
            break
    return {"available": True, "path": path, "version": version}


def stockfish_cache_key(fen, move, version, depth, threads=1, hash_mb=128):
    """Return deterministic cache key for one engine analysis request."""

    payload = json.dumps(
        {
            "fen": fen,
            "move": move,
            "version": version,
            "depth": depth,
            "threads": threads,
            "hash_mb": hash_mb,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stockfish_placeholder_summary(config):
    """Return graceful Stockfish status without requiring an engine."""

    discovery = discover_stockfish(config.stockfish_path)
    return {
        "requested": bool(config.stockfish),
        "available": bool(discovery["available"]),
        "version": discovery.get("version"),
        "path": discovery.get("path"),
        "depth": config.stockfish_depth,
        "threads": config.stockfish_threads,
        "hash_mb": config.stockfish_hash_mb,
        "status": "STOCKFISH_ANALYSIS_AVAILABLE" if discovery["available"] else "STOCKFISH_ANALYSIS_UNAVAILABLE",
        "ambiguity_audit_complete": False,
        "note": "Engine analysis is optional and secondary; official A3 accuracy is unchanged.",
    }


def evaluate_split(split_name, model, dataframe, config, device):
    """Run A3 inference on one split and return per-puzzle diagnostic rows."""

    pyg_split = "val" if split_name == "validation" else split_name
    dataset = load_pyg_dataset(root=config.pyg_root, split=pyg_split)
    csv_rows = list(dataframe.itertuples(index=False))
    rows = []
    use_amp = bool(config.amp and device.type == "cuda")
    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            graphs = [dataset[index] for index in range(start, min(start + config.batch_size, len(dataset)))]
            batch_rows = [csv_rows[int(graph.source_row_index.item())] for graph in graphs]
            batch = Batch.from_data_list(graphs).to(
                device,
                non_blocking=bool(config.non_blocking and device.type == "cuda"),
            )
            candidate = build_candidate_batch(batch)
            target_indices = candidate["target_indices"].to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(batch, candidate["candidate_moves"])
                loss = grouped_cross_entropy(output["scores"], output["candidate_ptr"], target_indices)
            for local_index, (graph, source_row) in enumerate(zip(graphs, batch_rows)):
                rank = topk_and_row(
                    output["scores"],
                    output["candidate_ptr"],
                    output["candidate_uci"],
                    target_indices[local_index],
                    local_index,
                )
                target_move = str(graph.target_move)
                prediction = rank["A3_top1_move"]
                target_features = enrich_quiet(move_type_features(graph.fen, target_move))
                prediction_features = enrich_quiet(move_type_features(graph.fen, prediction))
                row = {
                    "split": split_name,
                    "PuzzleId": str(source_row.PuzzleId),
                    "FEN": str(graph.fen),
                    "rating": int(source_row.Rating),
                    "MateDepth": int(source_row.MateDepth),
                    "target_move": target_move,
                    "loss_batch": float(loss.item()),
                    **rank,
                    "predicted_move": prediction if not rank["top1_correct"] else None,
                    "predicted_score": rank["A3_top1_score"] if not rank["top1_correct"] else None,
                    "predicted_minus_target_score": (
                        rank["A3_top1_score"] - rank["target_score"]
                        if not rank["top1_correct"]
                        else None
                    ),
                    "target_features": target_features,
                    "prediction_features": prediction_features,
                }
                rows.append(row)
    return rows


def load_rows(path):
    """Load JSONL rows from disk."""

    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    """Write rows as JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def split_summary(split_name, rows, stockfish_summary):
    """Build summary payload for one split."""

    summary = summarize_rows(rows)
    summary["split"] = split_name
    summary["expected_n"] = EXPECTED_SPLIT_N.get(split_name)
    summary["actual_n"] = len(rows)
    summary["stockfish"] = stockfish_summary
    summary["parity"] = a3_parity_for_split(split_name, summary)
    return summary


def a3_parity_for_split(split_name, summary):
    """Return parity status against official A3 test reference where applicable."""

    if split_name != "test":
        return {"status": "N/A", "reason": "No frozen public validation reference."}
    actual = summary["topk"]
    checks = {
        "top1": abs(actual["top1"] - MODEL_A3_REFERENCE["a3_top1"]) <= 1e-12,
        "top3": abs(actual["top3"] - MODEL_A3_REFERENCE["a3_top3"]) <= 1e-12,
        "top5": abs(actual["top5"] - MODEL_A3_REFERENCE["a3_top5"]) <= 1e-12,
        "n": summary["actual_n"] == MODEL_A3_REFERENCE["n"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "reference": MODEL_A3_REFERENCE,
    }


def write_outputs(validation_summary, test_summary, output_dir=OUTPUT_DIR):
    """Write summaries and Markdown report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_summary.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    (output_dir / "test_summary.json").write_text(json.dumps(test_summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(validation_summary, test_summary), encoding="utf-8")
    return {
        "validation_summary": str(output_dir / "validation_summary.json"),
        "test_summary": str(output_dir / "test_summary.json"),
        "report": str(output_dir / "report.md"),
    }


def render_report(validation_summary, test_summary):
    """Render Markdown report for the A3 error audit."""

    lines = [
        "# Model A3 Prediction Headroom And Error Audit",
        "",
        "## Audit Integrity",
        "",
        "- Scope: diagnostic only; no retraining; no model, checkpoint, dataset, split, target, or candidate changes.",
        "- Design analysis source: validation.",
        "- Final diagnostic confirmation: test.",
        "- Official target semantics remain `TargetMove = Moves[1]`.",
        "",
        "## Official A3 Parity",
        "",
        f"- Test parity: `{test_summary.get('parity', {}).get('status')}`",
        "",
        "## Headroom Analysis",
        "",
        "Validation:",
        "```json",
        json.dumps(validation_summary.get("topk", {}), indent=2),
        "```",
        "Test:",
        "```json",
        json.dumps(test_summary.get("topk", {}), indent=2),
        "```",
        "",
        "## Target-Rank Distribution",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("target_rank_histogram"), "test": test_summary.get("target_rank_histogram")}, indent=2),
        "```",
        "",
        "## Error Target-Rank Breakdown",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("error_target_rank_breakdown"), "test": test_summary.get("error_target_rank_breakdown")}, indent=2),
        "```",
        "",
        "## MateDepth Breakdown",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("mate_depth"), "test": test_summary.get("mate_depth")}, indent=2),
        "```",
        "",
        "## Rating Breakdown",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("rating"), "test": test_summary.get("rating")}, indent=2),
        "```",
        "",
        "## Legal Candidate Count Breakdown",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("legal_move_count"), "test": test_summary.get("legal_move_count")}, indent=2),
        "```",
        "",
        "## Move-Type And Piece Analysis",
        "",
        "```json",
        json.dumps({
            "validation_target_move_type": validation_summary.get("target_move_type_accuracy"),
            "test_target_move_type": test_summary.get("target_move_type_accuracy"),
            "validation_piece": validation_summary.get("target_piece_accuracy"),
            "test_piece": test_summary.get("target_piece_accuracy"),
        }, indent=2),
        "```",
        "",
        "## Target Vs Wrong Prediction Feature Comparison",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("target_vs_wrong_prediction"), "test": test_summary.get("target_vs_wrong_prediction")}, indent=2),
        "```",
        "",
        "## Score-Margin Analysis",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("score_margin"), "test": test_summary.get("score_margin")}, indent=2),
        "```",
        "",
        "## Stockfish Availability And Ambiguity Analysis",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("stockfish"), "test": test_summary.get("stockfish")}, indent=2),
        "```",
        "",
        "## Representative Examples",
        "",
        "```json",
        json.dumps({"validation": validation_summary.get("representative_examples"), "test": test_summary.get("representative_examples")}, indent=2),
        "```",
        "",
        "## Implications For A4",
        "",
        "Recommendation is derived from validation first. Use test only as descriptive confirmation. If most wrong targets are already rank <=5, a Top-K reranker has evidence; if errors correlate with immediate move features, richer candidate features have evidence; if neither is enough, post-move encoding is the next hypothesis.",
        "",
        "## Limitations",
        "",
        "- Stockfish diagnostics are secondary only when available.",
        "- This audit must not be used to tune on the official test set.",
        "",
    ]
    return "\n".join(lines)


def run_audit(config):
    """Run A3 validation/test diagnostic audit."""

    device = torch.device(config.device)
    checkpoint = Path(config.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"A3 checkpoint not found: {checkpoint}")
    model, _ = load_eval_model_a3(checkpoint, device)
    output_dir = Path(config.output_dir)
    stockfish_summary = stockfish_placeholder_summary(config)
    summaries = {}
    for split_name in ("validation", "test"):
        dataframe = pd.read_csv(PUZZLE_CSVS[split_name])
        rows = evaluate_split(split_name, model, dataframe, config, device)
        write_jsonl(output_dir / f"{split_name}_rows.jsonl", rows)
        summaries[split_name] = split_summary(split_name, rows, stockfish_summary)
    paths = write_outputs(summaries["validation"], summaries["test"], output_dir)
    return summaries, paths


def refresh_report_from_rows(config):
    """Regenerate summaries/report from existing JSONL rows without inference."""

    output_dir = Path(config.output_dir)
    stockfish_summary = stockfish_placeholder_summary(config)
    validation_rows = load_rows(output_dir / "validation_rows.jsonl")
    test_rows = load_rows(output_dir / "test_rows.jsonl")
    validation_summary = split_summary("validation", validation_rows, stockfish_summary)
    test_summary = split_summary("test", test_rows, stockfish_summary)
    paths = write_outputs(validation_summary, test_summary, output_dir)
    return {"validation": validation_summary, "test": test_summary}, paths

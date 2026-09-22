"""Inference helpers for the frozen ChessGATNoTiming baseline.

Purpose:
    Load the train-derived move vocabulary, discover a local Model A
    checkpoint, run raw graph-level inference, and compute UI diagnostics.
Input:
    Final puzzle rows, official PyG graph construction, move vocabulary JSON,
    and a local checkpoint when available.
Output:
    Raw top-k predictions, legal-move diagnostics, target-rank diagnostics, and
    Mate-in-1 subgroup metrics.
Run:
    Imported by the Streamlit debugger and unit tests; this module has no CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import statistics

import chess
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from src.graph.graph_builder import build_graph
from src.graph.pyg_dataset import load_pyg_dataset
from src.models.model_a.chess_gat import ChessGATNoTiming


MODEL_A_NAME = "MODEL_A_NO_TIMING_FROZEN_BASELINE"
REFERENCE_TEST_METRICS = {
    "examples": 8610,
    "top1": 0.3961672474213732,
    "top3": 0.5580720094022851,
    "top5": 0.6275261325010993,
}
PARITY_TOLERANCE = 1e-6
EXPECTED_CHECKPOINT_PATH = Path(
    "checkpoints/model_a/best.pt"
)
FALLBACK_CHECKPOINT_PATHS = [
    EXPECTED_CHECKPOINT_PATH,
]
MOVE_TO_IDX_PATH = Path("resources/move_encoder/move_to_idx.json")
IDX_TO_MOVE_PATH = Path("resources/move_encoder/idx_to_move.json")


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model and metadata used by read-only inference."""

    model: ChessGATNoTiming
    device: torch.device
    checkpoint_path: Path
    move_to_idx: dict[str, int]
    idx_to_move: dict[int, str]
    num_classes: int
    is_official_checkpoint: bool
    checkpoint_metadata: dict


def load_json(path):
    """Read a JSON file from disk.

    Parameters:
        path: JSON path.
    Returns:
        Parsed JSON object.
    Side effects:
        Reads path from disk.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_move_vocabulary(
    move_to_idx_path=MOVE_TO_IDX_PATH,
    idx_to_move_path=IDX_TO_MOVE_PATH,
):
    """Load the train-only move vocabulary and reverse mapping.

    Parameters:
        move_to_idx_path: Path to move_to_idx.json.
        idx_to_move_path: Path to idx_to_move.json.
    Returns:
        Tuple (move_to_idx, idx_to_move) with integer class ids.
    Side effects:
        Reads both JSON files from disk.
    """

    move_to_idx = load_json(move_to_idx_path)
    raw_idx_to_move = load_json(idx_to_move_path)
    idx_to_move = {
        int(index): move
        for index, move in raw_idx_to_move.items()
    }
    return move_to_idx, idx_to_move


def discover_checkpoint(paths=None):
    """Return the first available Model A checkpoint path.

    Parameters:
        paths: Optional ordered candidate paths.
    Returns:
        Path to the first existing checkpoint, or None.
    Side effects:
        Checks file existence only.
    """

    for path in paths or FALLBACK_CHECKPOINT_PATHS:
        candidate = Path(path)
        if candidate.exists():
            return candidate
    return None


def _state_dict_from_checkpoint(checkpoint):
    """Extract a model state dict from common project checkpoint formats."""

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model_bundle(
    checkpoint_path=None,
    device=None,
    move_to_idx_path=MOVE_TO_IDX_PATH,
    idx_to_move_path=IDX_TO_MOVE_PATH,
):
    """Load ChessGATNoTiming and its vocabulary for read-only inference.

    Parameters:
        checkpoint_path: Optional explicit checkpoint. When None, discovery is
            attempted using the project artifact order.
        device: Optional torch device string/object. Defaults to CUDA when
            available, otherwise CPU.
        move_to_idx_path: Path to move_to_idx.json.
        idx_to_move_path: Path to idx_to_move.json.
    Returns:
        ModelBundle with model.eval() on the selected device.
    Side effects:
        Reads JSON vocabulary and checkpoint from disk.
    """

    move_to_idx, idx_to_move = load_move_vocabulary(
        move_to_idx_path,
        idx_to_move_path,
    )
    resolved_checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else discover_checkpoint()
    )
    if resolved_checkpoint is None:
        raise FileNotFoundError(
            f"Model A checkpoint unavailable. Expected: {EXPECTED_CHECKPOINT_PATH}"
        )

    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = ChessGATNoTiming(
        num_classes=len(move_to_idx),
        dropout=0.30,
    )
    checkpoint = torch.load(
        resolved_checkpoint,
        map_location=resolved_device,
        weights_only=False,
    )
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
    model.to(resolved_device)
    model.eval()
    return ModelBundle(
        model=model,
        device=resolved_device,
        checkpoint_path=resolved_checkpoint,
        move_to_idx=move_to_idx,
        idx_to_move=idx_to_move,
        num_classes=len(move_to_idx),
        is_official_checkpoint=resolved_checkpoint == EXPECTED_CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint if isinstance(checkpoint, dict) else {},
    )


def puzzle_row_to_graph(row, move_to_idx):
    """Build the official PyG graph for a final puzzle CSV row.

    Parameters:
        row: Pandas row-like object containing FEN and TargetMove.
        move_to_idx: Train-only move vocabulary.
    Returns:
        PyTorch Geometric Data object built by src.graph.graph_builder.
    Side effects:
        None.
    """

    original_target_move = str(row.TargetMove)
    encoded_target_move = original_target_move
    target_oov = False
    if encoded_target_move not in move_to_idx:
        encoded_target_move = next(iter(move_to_idx))
        target_oov = True
    graph = build_graph(
        fen=row.FEN,
        target_move=encoded_target_move,
        move_to_idx=move_to_idx,
    )
    graph.target_move = original_target_move
    graph.target_oov = target_oov
    return graph


def move_to_san(board, move_uci):
    """Convert a UCI move to SAN when the move is syntactically legal."""

    try:
        move = chess.Move.from_uci(str(move_uci))
    except ValueError:
        return None
    if move not in board.legal_moves:
        return None
    return board.san(move)


def is_legal_uci(board, move_uci):
    """Return whether a UCI move is legal on the given board."""

    try:
        move = chess.Move.from_uci(str(move_uci))
    except ValueError:
        return False
    return move in board.legal_moves


def check_user_move(board, user_move, target_move):
    """Validate a human move against syntax, legality, and target.

    Parameters:
        board: Solver-position chess.Board.
        user_move: User-entered move in UCI format.
        target_move: Expected target move in UCI format.
    Returns:
        Dictionary with status, normalized move, SAN, and message.
    Side effects:
        None.
    """

    normalized = str(user_move).strip().lower()
    try:
        move = chess.Move.from_uci(normalized)
    except ValueError:
        return {
            "status": "INVALID_SYNTAX",
            "move": normalized,
            "san": None,
            "message": "Invalid UCI syntax.",
        }
    if move not in board.legal_moves:
        return {
            "status": "ILLEGAL_MOVE",
            "move": normalized,
            "san": None,
            "message": "Move is syntactically valid but illegal here.",
        }
    return {
        "status": "CORRECT" if normalized == str(target_move) else "LEGAL_BUT_WRONG",
        "move": normalized,
        "san": board.san(move),
        "message": "Move matches the puzzle target."
        if normalized == str(target_move)
        else "Move is legal but does not match the puzzle target.",
    }


def decode_topk(logits, idx_to_move, board, target_move, k=5):
    """Decode raw top-k logits without legal-move masking.

    Parameters:
        logits: Raw logits for one graph with shape [num_classes].
        idx_to_move: Mapping from class index to UCI move.
        board: Solver-position chess.Board.
        target_move: Ground-truth UCI target.
        k: Number of rows to return.
    Returns:
        List of dictionaries describing raw top-k predictions.
    Side effects:
        None.
    """

    probabilities = torch.softmax(logits.detach().cpu(), dim=0)
    values, indices = torch.topk(
        probabilities,
        k=min(k, probabilities.numel()),
    )
    rows = []
    for rank, (score, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
        move = idx_to_move[int(index)]
        rows.append(
            {
                "rank": rank,
                "class_index": int(index),
                "move": move,
                "san": move_to_san(board, move),
                "softmax_score": float(score),
                "legal": is_legal_uci(board, move),
                "ground_truth": move == str(target_move),
            }
        )
    return rows


def target_rank_from_logits(logits, target_move, move_to_idx):
    """Compute raw target rank across the full vocabulary.

    Parameters:
        logits: Raw logits for one graph with shape [num_classes].
        target_move: Ground-truth UCI move.
        move_to_idx: Train-only move vocabulary.
    Returns:
        One-based rank, or None when the target is OOV.
    Side effects:
        None.
    """

    if str(target_move) not in move_to_idx:
        return None
    target_idx = int(move_to_idx[str(target_move)])
    ordered = torch.argsort(
        logits.detach().cpu(),
        descending=True,
    )
    matches = (ordered == target_idx).nonzero(as_tuple=False)
    return int(matches[0].item()) + 1


def summarize_prediction(logits, board, target_move, move_to_idx, idx_to_move, k=5):
    """Build a complete raw Top-K and target-rank inference summary."""

    topk = decode_topk(
        logits=logits,
        idx_to_move=idx_to_move,
        board=board,
        target_move=target_move,
        k=k,
    )
    rank = target_rank_from_logits(
        logits=logits,
        target_move=target_move,
        move_to_idx=move_to_idx,
    )
    return {
        "inference_status": "SUCCESS",
        "topk": topk,
        "target_rank": rank,
        "target_oov": rank is None,
        "top1_hit": bool(topk and topk[0]["ground_truth"]),
        "top3_hit": any(row["ground_truth"] for row in topk[:3]),
        "top5_hit": any(row["ground_truth"] for row in topk[:5]),
        "top1_legal": bool(topk and topk[0]["legal"]),
        "top1_label": hit_label(rank, 1),
        "top3_label": hit_label(rank, 3),
        "top5_label": hit_label(rank, 5),
        "rank_bucket": rank_bucket(rank),
        "error_category": error_category(rank, bool(topk and topk[0]["legal"])),
        "best_legal": best_legal_prediction(logits, idx_to_move, board, target_move),
    }


def hit_label(target_rank, k):
    """Return HIT/MISS/OOV for a target rank and Top-K cutoff."""

    if target_rank is None:
        return "OOV"
    return "HIT" if target_rank <= k else "MISS"


def rank_bucket(target_rank):
    """Map a one-based target rank to the analysis bucket label."""

    if target_rank is None:
        return "TARGET_OOV"
    if target_rank == 1:
        return "rank_1"
    if target_rank <= 3:
        return "rank_2_3"
    if target_rank <= 5:
        return "rank_4_5"
    if target_rank <= 10:
        return "rank_6_10"
    if target_rank <= 20:
        return "rank_11_20"
    if target_rank <= 50:
        return "rank_21_50"
    return "rank_gt_50"


def error_category(target_rank, top1_legal):
    """Return the mutually interpretable global error category."""

    if target_rank is None:
        return "TARGET_OOV"
    if target_rank == 1:
        return "TOP1_CORRECT"
    if top1_legal:
        return "TOP1_WRONG_LEGAL"
    return "TOP1_ILLEGAL"


def target_row_from_logits(logits, target_move, move_to_idx, board):
    """Return details for the ground-truth move even when it is outside Top-K."""

    rank = target_rank_from_logits(logits, target_move, move_to_idx)
    if rank is None:
        return None
    probabilities = torch.softmax(logits.detach().cpu(), dim=0)
    target_idx = int(move_to_idx[str(target_move)])
    return {
        "rank": rank,
        "class_index": target_idx,
        "move": str(target_move),
        "san": move_to_san(board, target_move),
        "softmax_score": float(probabilities[target_idx].item()),
        "legal": is_legal_uci(board, target_move),
        "ground_truth": True,
    }


def best_legal_prediction(logits, idx_to_move, board, target_move):
    """Return the first legal move in the raw ranking for diagnostics only."""

    probabilities = torch.softmax(logits.detach().cpu(), dim=0)
    ordered = torch.argsort(logits.detach().cpu(), descending=True)
    for rank, class_index in enumerate(ordered.tolist(), 1):
        move = idx_to_move[int(class_index)]
        if is_legal_uci(board, move):
            return {
                "rank": rank,
                "class_index": int(class_index),
                "move": move,
                "san": move_to_san(board, move),
                "softmax_score": float(probabilities[int(class_index)].item()),
                "ground_truth": move == str(target_move),
            }
    return None


def target_rank_explanation(target_rank):
    """Return plain text explaining HIT/MISS for the current target rank."""

    if target_rank is None:
        return "Target rank is unavailable because the target move is OOV."
    if target_rank == 1:
        return "The correct move is the model's first-ranked raw prediction."
    if target_rank <= 3:
        return f"The correct move is ranked #{target_rank}, so Top-3 and Top-5 are HIT."
    if target_rank <= 5:
        return f"The correct move is ranked #{target_rank}, so Top-5 is HIT."
    return f"The target is ranked #{target_rank}, therefore it is outside Top-5."


def empty_error_analysis_metrics():
    """Create mutable counters for Model A diagnostic error analysis."""

    return {
        "total": 0,
        "evaluable": 0,
        "oov": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "best_legal_top1": 0,
        "legal_top1": 0,
        "illegal_top1": 0,
        "correct_top1": 0,
        "wrong_legal_top1": 0,
        "rank_buckets": {
            "rank_1": 0,
            "rank_2_3": 0,
            "rank_4_5": 0,
            "rank_6_10": 0,
            "rank_11_20": 0,
            "rank_21_50": 0,
            "rank_gt_50": 0,
            "TARGET_OOV": 0,
        },
        "error_categories": {
            "TOP1_CORRECT": 0,
            "TOP1_WRONG_LEGAL": 0,
            "TOP1_ILLEGAL": 0,
            "TARGET_OOV": 0,
        },
        "target_ranks": [],
        "illegal_examples": [],
        "mistake_examples": [],
        "mate_in_one_inconsistencies": 0,
    }


def update_error_analysis_metrics(metrics, row, summary, max_examples=25):
    """Update aggregate analysis counters from one prediction summary."""

    metrics["total"] += 1
    metrics["rank_buckets"][summary["rank_bucket"]] += 1
    metrics["error_categories"][summary["error_category"]] += 1
    if summary["target_oov"]:
        metrics["oov"] += 1
        return

    metrics["evaluable"] += 1
    metrics["target_ranks"].append(summary["target_rank"])
    metrics["top1"] += int(summary["top1_hit"])
    metrics["top3"] += int(summary["top3_hit"])
    metrics["top5"] += int(summary["top5_hit"])
    best_legal = summary.get("best_legal")
    metrics["best_legal_top1"] += int(bool(best_legal and best_legal["ground_truth"]))
    metrics["legal_top1"] += int(summary["top1_legal"])
    metrics["illegal_top1"] += int(not summary["top1_legal"])
    metrics["correct_top1"] += int(summary["top1_hit"])
    metrics["wrong_legal_top1"] += int(
        summary["top1_legal"] and not summary["top1_hit"]
    )

    top1 = summary["topk"][0] if summary["topk"] else {}
    example = {
        "PuzzleId": getattr(row, "PuzzleId", None),
        "TargetMove": str(row.TargetMove),
        "PredictedMove": top1.get("move"),
        "TargetRank": summary["target_rank"],
        "Themes": getattr(row, "Themes", ""),
        "Rating": getattr(row, "Rating", None),
        "ErrorCategory": summary["error_category"],
    }
    if not summary["top1_hit"] and len(metrics["mistake_examples"]) < max_examples:
        metrics["mistake_examples"].append(example)
    if not summary["top1_legal"] and len(metrics["illegal_examples"]) < max_examples:
        metrics["illegal_examples"].append(example)


def finalize_error_analysis_metrics(metrics):
    """Add percentages, rank summaries, and parity fields to metrics."""

    evaluable = max(1, metrics["evaluable"])
    total = max(1, metrics["total"])
    top1_misses = max(1, metrics["evaluable"] - metrics["top1"])
    top3_misses = max(1, metrics["evaluable"] - metrics["top3"])
    ranks = metrics["target_ranks"]
    metrics["rates"] = {
        "top1": metrics["top1"] / evaluable,
        "top3": metrics["top3"] / evaluable,
        "top5": metrics["top5"] / evaluable,
        "best_legal_top1": metrics["best_legal_top1"] / evaluable,
        "illegal_top1": metrics["illegal_top1"] / total,
        "legal_top1": metrics["legal_top1"] / total,
        "oov": metrics["oov"] / total,
    }
    metrics["rank_recovery"] = {
        "top3_minus_top1_pp": (
            metrics["rates"]["top3"] - metrics["rates"]["top1"]
        ) * 100,
        "top5_minus_top3_pp": (
            metrics["rates"]["top5"] - metrics["rates"]["top3"]
        ) * 100,
        "rank_2_3_over_top1_misses": (
            metrics["rank_buckets"]["rank_2_3"] / top1_misses
        ),
        "rank_4_5_over_top3_misses": (
            metrics["rank_buckets"]["rank_4_5"] / top3_misses
        ),
    }
    metrics["target_rank_summary"] = {
        "average": statistics.mean(ranks) if ranks else None,
        "median": statistics.median(ranks) if ranks else None,
    }
    metrics["parity"] = compute_metric_parity(metrics)
    return metrics


def compute_metric_parity(metrics, reference=REFERENCE_TEST_METRICS):
    """Compare computed global metrics against frozen reference metrics."""

    actual = {
        "examples": metrics["evaluable"],
        "top1": metrics.get("rates", {}).get("top1"),
        "top3": metrics.get("rates", {}).get("top3"),
        "top5": metrics.get("rates", {}).get("top5"),
    }
    deltas = {
        key: None if actual[key] is None else actual[key] - reference[key]
        for key in actual
    }
    passed = (
        actual["examples"] == reference["examples"]
        and all(
            abs(deltas[key]) < PARITY_TOLERANCE
            for key in ("top1", "top3", "top5")
            if deltas[key] is not None
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "reference": dict(reference),
        "actual": actual,
        "deltas": deltas,
        "tolerance": PARITY_TOLERANCE,
    }


def run_single_inference(bundle, row, top_k=5):
    """Run Model A on one final puzzle row.

    Parameters:
        bundle: Loaded ModelBundle.
        row: Pandas row-like puzzle record with transformed FEN.
        top_k: Number of raw top predictions to return.
    Returns:
        Prediction summary dictionary.
    Side effects:
        Performs a no-grad model forward pass.
    """

    graph = puzzle_row_to_graph(row, bundle.move_to_idx)
    batch = Batch.from_data_list([graph]).to(bundle.device)
    with torch.no_grad():
        logits = bundle.model(batch)[0]
    board = chess.Board(row.FEN)
    return summarize_prediction(
        logits=logits,
        board=board,
        target_move=str(row.TargetMove),
        move_to_idx=bundle.move_to_idx,
        idx_to_move=bundle.idx_to_move,
        k=top_k,
    )


def is_mate_in_one_row(row):
    """Return True when dataset metadata marks a puzzle as mateIn1."""

    themes = str(getattr(row, "Themes", ""))
    mate_depth = getattr(row, "MateDepth", None)
    return "mateIn1" in themes.split() or int(mate_depth) == 1


def verify_target_checkmate(fen, target_move):
    """Verify that the target move checkmates from the solver position."""

    board = chess.Board(fen)
    move = chess.Move.from_uci(str(target_move))
    if move not in board.legal_moves:
        return False
    board.push(move)
    return board.is_checkmate()


def evaluate_rows_with_logits(rows, logits_list, move_to_idx, idx_to_move, k=5):
    """Evaluate precomputed logits for a small set of puzzle rows.

    Parameters:
        rows: Iterable of row-like objects with FEN and TargetMove.
        logits_list: Iterable of raw logits aligned with rows.
        move_to_idx: Train-only move vocabulary.
        idx_to_move: Reverse class mapping.
        k: Top-K decoding size.
    Returns:
        Diagnostic subgroup metrics.
    Side effects:
        None.
    """

    metrics = {
        "total": 0,
        "evaluable": 0,
        "oov": 0,
        "top1": 0,
        "top3": 0,
        "top5": 0,
        "illegal_top1": 0,
        "legal_top1": 0,
        "correct_top1": 0,
        "wrong_legal_top1": 0,
        "target_ranks": [],
    }
    for row, logits in zip(rows, logits_list):
        metrics["total"] += 1
        board = chess.Board(row.FEN)
        target = str(row.TargetMove)
        summary = summarize_prediction(
            logits=logits,
            board=board,
            target_move=target,
            move_to_idx=move_to_idx,
            idx_to_move=idx_to_move,
            k=k,
        )
        if summary["target_oov"]:
            metrics["oov"] += 1
        else:
            metrics["evaluable"] += 1
            metrics["target_ranks"].append(summary["target_rank"])
            metrics["top1"] += int(summary["top1_hit"])
            metrics["top3"] += int(summary["top3_hit"])
            metrics["top5"] += int(summary["top5_hit"])
        if summary["top1_legal"]:
            metrics["legal_top1"] += 1
            if summary["top1_hit"]:
                metrics["correct_top1"] += 1
            else:
                metrics["wrong_legal_top1"] += 1
        else:
            metrics["illegal_top1"] += 1

    ranks = metrics["target_ranks"]
    metrics["average_target_rank"] = statistics.mean(ranks) if ranks else None
    metrics["median_target_rank"] = statistics.median(ranks) if ranks else None
    return metrics


def evaluate_mate_in_one_dataframe(bundle, dataframe, batch_size=128, limit=None):
    """Run batched diagnostic evaluation on Mate-in-1 rows.

    Parameters:
        bundle: Loaded ModelBundle.
        dataframe: Final puzzle CSV DataFrame.
        batch_size: PyG DataLoader batch size.
        limit: Optional maximum number of mateIn1 rows to evaluate.
    Returns:
        Diagnostic subgroup metrics; not an official test score.
    Side effects:
        Performs no-grad model forward passes.
    """

    mate_rows = [
        row
        for row in dataframe.itertuples(index=False)
        if is_mate_in_one_row(row)
    ]
    if limit is not None:
        mate_rows = mate_rows[: int(limit)]

    graphs = []
    rows_for_graphs = []
    oov_rows = []
    for row in mate_rows:
        if str(row.TargetMove) not in bundle.move_to_idx:
            oov_rows.append(row)
            continue
        graphs.append(puzzle_row_to_graph(row, bundle.move_to_idx))
        rows_for_graphs.append(row)

    logits_list = []
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(bundle.device)
            logits = bundle.model(batch).detach().cpu()
            logits_list.extend(logits)

    metrics = evaluate_rows_with_logits(
        rows=rows_for_graphs,
        logits_list=logits_list,
        move_to_idx=bundle.move_to_idx,
        idx_to_move=bundle.idx_to_move,
    )
    metrics["total"] = len(mate_rows)
    metrics["oov"] += len(oov_rows)
    metrics["evaluable"] = len(rows_for_graphs)
    return metrics


def _rate(numerator, denominator):
    """Return a safe rate or None when denominator is zero."""

    return numerator / denominator if denominator else None


def _theme_tokens(row):
    """Return Lichess theme tokens for a row-like object."""

    return str(getattr(row, "Themes", "")).split()


def _mate_depth_label(row):
    """Return the concrete mateInN label when present."""

    for theme in _theme_tokens(row):
        if theme.startswith("mateIn"):
            return theme
    mate_depth = getattr(row, "MateDepth", None)
    if mate_depth is None:
        return "unknown"
    return f"mateIn{int(mate_depth)}"


def _rating_bucket(row):
    """Return the fixed human-rating bucket label for a puzzle row."""

    rating = getattr(row, "Rating", None)
    if rating is None:
        return "unknown"
    rating = int(rating)
    if rating < 1200:
        return "<1200"
    if rating < 1600:
        return "1200-1599"
    if rating < 2000:
        return "1600-1999"
    if rating < 2400:
        return "2000-2399"
    return "2400+"


def _subgroup_row(name, metrics):
    """Build a serializable row from finalized metrics."""

    rates = metrics["rates"]
    summary = metrics["target_rank_summary"]
    return {
        "name": name,
        "N": metrics["total"],
        "evaluable": metrics["evaluable"],
        "oov": metrics["oov"],
        "top1": rates["top1"],
        "top3": rates["top3"],
        "top5": rates["top5"],
        "illegal_top1_rate": rates["illegal_top1"],
        "best_legal_top1": rates["best_legal_top1"],
        "average_target_rank": summary["average"],
        "median_target_rank": summary["median"],
    }


def run_error_analysis(
    bundle,
    dataframe,
    split_name="test",
    batch_size=128,
    min_theme_samples=30,
    progress_callback=None,
):
    """Run post-hoc Model A error analysis with batched inference.

    Parameters:
        bundle: Official frozen ModelBundle.
        dataframe: Final split CSV DataFrame.
        split_name: Split label used in the report.
        batch_size: PyG DataLoader batch size.
        min_theme_samples: Minimum samples required for theme rows.
        progress_callback: Optional callable receiving processed/total counts.
    Returns:
        Dictionary with overall metrics, subgroups, examples, and parity status.
    Side effects:
        Performs no-grad model inference; does not update model or data.
    """

    rows = list(dataframe.itertuples(index=False))
    dataset = None
    try:
        dataset = load_pyg_dataset(split=split_name)
    except Exception:
        dataset = None
    overall = empty_error_analysis_metrics()
    theme_metrics = {}
    mate_metrics = {}
    rating_metrics = {}

    for row in rows:
        if str(row.TargetMove) in bundle.move_to_idx:
            continue
        update_error_analysis_metrics(
            overall,
            row,
            {
                "target_oov": True,
                "rank_bucket": "TARGET_OOV",
                "error_category": "TARGET_OOV",
            },
        )

    if dataset is None:
        graph_source = "csv_rebuild_fallback"
        graph_rows = [
            row for row in rows
            if str(row.TargetMove) in bundle.move_to_idx
        ]
        total_graphs = len(graph_rows)
    else:
        graph_source = "official_pyg_shards"
        graph_rows = None
        total_graphs = len(dataset)

    processed = 0
    with torch.no_grad():
        for start in range(0, total_graphs, batch_size):
            stop = min(start + batch_size, total_graphs)
            if dataset is None:
                batch_rows = graph_rows[start:stop]
                batch_graphs = [
                    puzzle_row_to_graph(row, bundle.move_to_idx)
                    for row in batch_rows
                ]
            else:
                batch_graphs = [
                    dataset[index]
                    for index in range(start, stop)
                ]
                batch_rows = [
                    rows[int(graph.source_row_index.item())]
                    for graph in batch_graphs
                ]
            batch = Batch.from_data_list(batch_graphs).to(bundle.device)
            logits_batch = bundle.model(batch).detach().cpu()
            for row, logits in zip(batch_rows, logits_batch):
                board = chess.Board(row.FEN)
                summary = summarize_prediction(
                    logits=logits,
                    board=board,
                    target_move=str(row.TargetMove),
                    move_to_idx=bundle.move_to_idx,
                    idx_to_move=bundle.idx_to_move,
                    k=5,
                )
                update_error_analysis_metrics(overall, row, summary)

                mate_label = _mate_depth_label(row)
                mate_metrics.setdefault(mate_label, empty_error_analysis_metrics())
                update_error_analysis_metrics(mate_metrics[mate_label], row, summary)
                if mate_label == "mateIn1" and not verify_target_checkmate(
                    row.FEN,
                    row.TargetMove,
                ):
                    overall["mate_in_one_inconsistencies"] += 1
                    mate_metrics[mate_label]["mate_in_one_inconsistencies"] += 1

                rating_label = _rating_bucket(row)
                rating_metrics.setdefault(rating_label, empty_error_analysis_metrics())
                update_error_analysis_metrics(rating_metrics[rating_label], row, summary)

                for theme in _theme_tokens(row):
                    theme_metrics.setdefault(theme, empty_error_analysis_metrics())
                    update_error_analysis_metrics(theme_metrics[theme], row, summary)

            processed += len(batch_rows)
            if progress_callback is not None:
                progress_callback(processed, total_graphs)

    finalized_overall = finalize_error_analysis_metrics(overall)
    finalized_mate = {
        name: finalize_error_analysis_metrics(metrics)
        for name, metrics in sorted(mate_metrics.items())
    }
    finalized_rating = {
        name: finalize_error_analysis_metrics(metrics)
        for name, metrics in sorted(rating_metrics.items())
    }
    finalized_themes = {
        name: finalize_error_analysis_metrics(metrics)
        for name, metrics in sorted(theme_metrics.items())
        if metrics["total"] >= min_theme_samples
    }
    return {
        "status": "DIAGNOSTIC_POST_HOC_ANALYSIS",
        "model": MODEL_A_NAME,
        "split": split_name,
        "checkpoint": str(bundle.checkpoint_path),
        "checkpoint_status": (
            "OFFICIAL FROZEN CHECKPOINT"
            if bundle.is_official_checkpoint
            else "FALLBACK / NON-OFFICIAL CHECKPOINT"
        ),
        "dataset_source": graph_source,
        "overall": finalized_overall,
        "mate_depth": {
            name: _subgroup_row(name, metrics)
            for name, metrics in finalized_mate.items()
        },
        "theme": {
            name: _subgroup_row(name, metrics)
            for name, metrics in finalized_themes.items()
        },
        "rating": {
            name: _subgroup_row(name, metrics)
            for name, metrics in finalized_rating.items()
        },
    }


def write_error_analysis_report(analysis, output_dir="artifacts/model_a_error_analysis"):
    """Write post-hoc analysis artifacts without touching scientific outputs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(render_error_analysis_markdown(analysis))
    return {
        "summary_json": str(summary_path),
        "report_md": str(report_path),
    }


def render_error_analysis_markdown(analysis):
    """Render a compact Markdown report for Model A error analysis."""

    overall = analysis["overall"]
    rates = overall["rates"]
    parity = overall["parity"]
    lines = [
        "# Model A Error Analysis",
        "",
        "## Frozen Baseline",
        "",
        f"- Model: `{analysis['model']}`",
        f"- Split: `{analysis['split']}`",
        f"- Checkpoint: `{analysis['checkpoint']}`",
        f"- Checkpoint status: `{analysis['checkpoint_status']}`",
        "- Status: `DIAGNOSTIC POST-HOC ANALYSIS`",
        "",
        "## Metric Parity",
        "",
        f"- OFFICIAL METRIC PARITY: `{parity['status']}`",
        f"- Expected examples: `{parity['reference']['examples']}`",
        f"- Actual examples: `{parity['actual']['examples']}`",
        "",
        "## Overall Performance",
        "",
        f"- Top1: `{rates['top1']:.6f}`",
        f"- Top3: `{rates['top3']:.6f}`",
        f"- Top5: `{rates['top5']:.6f}`",
        "",
        "## Error Categories",
        "",
        json.dumps(overall["error_categories"], indent=2),
        "",
        "## Target Rank Distribution",
        "",
        json.dumps(overall["rank_buckets"], indent=2),
        "",
        "## Raw vs Legal Diagnostic",
        "",
        f"- Raw Top1: `{rates['top1']:.6f}`",
        f"- Diagnostic best-legal Top1: `{rates['best_legal_top1']:.6f}`",
        f"- Delta pp: `{(rates['best_legal_top1'] - rates['top1']) * 100:.4f}`",
        "",
        "## Mate Puzzle Analysis",
        "",
        json.dumps(analysis["mate_depth"], indent=2),
        "",
        "## Mate-in-1 Analysis",
        "",
        json.dumps(analysis["mate_depth"].get("mateIn1", {}), indent=2),
        "",
        "## Theme Analysis",
        "",
        "Theme metrics are overlapping subgroup analyses and do not sum to total.",
        "",
        "## Rating Analysis",
        "",
        json.dumps(analysis["rating"], indent=2),
        "",
        "## Example Errors",
        "",
        json.dumps(overall["mistake_examples"][:10], indent=2),
        "",
        "## Interpretation",
        "",
        "This report separates ranking misses, illegal raw Top-1 predictions, "
        "wrong-but-legal predictions, and train-vocabulary OOV targets.",
        "",
        "## Limitations",
        "",
        "This is post-hoc test-set analysis. It must not be used to modify or "
        "select Model A. Future architecture changes require a separate protocol.",
        "",
        "## Conclusions",
        "",
        "Use these diagnostics to design future experiments, not to redefine the "
        "frozen Model A score.",
        "",
    ]
    return "\n".join(lines)

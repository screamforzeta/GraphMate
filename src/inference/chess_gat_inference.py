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
from src.models.chess_gat import ChessGATNoTiming


MODEL_A_NAME = "MODEL_A_NO_TIMING_FROZEN_BASELINE"
EXPECTED_CHECKPOINT_PATH = Path(
    "artifacts/convergence_training/chess_gat_no_timing/best.pt"
)
FALLBACK_CHECKPOINT_PATHS = [
    EXPECTED_CHECKPOINT_PATH,
    Path("artifacts/checkpoints/chess_gat_no_timing_best.pt"),
    Path("artifacts/adaptive_training/chess_gat_no_timing/best_overall.pt"),
]
MOVE_TO_IDX_PATH = Path("artifacts/move_to_idx.json")
IDX_TO_MOVE_PATH = Path("artifacts/idx_to_move.json")


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model and metadata used by read-only inference."""

    model: ChessGATNoTiming
    device: torch.device
    checkpoint_path: Path
    move_to_idx: dict[str, int]
    idx_to_move: dict[int, str]
    num_classes: int


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
        "topk": topk,
        "target_rank": rank,
        "target_oov": rank is None,
        "top1_hit": bool(topk and topk[0]["ground_truth"]),
        "top3_hit": any(row["ground_truth"] for row in topk[:3]),
        "top5_hit": any(row["ground_truth"] for row in topk[:5]),
        "top1_legal": bool(topk and topk[0]["legal"]),
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

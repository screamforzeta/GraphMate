"""No-timing multi-model inference helpers.

Purpose:
    Provide shared inference and diagnostic helpers for Model A raw, Model A
    best-legal, Model A2 masked, and Model A3 legal-candidate scorer.
Input:
    Final puzzle rows, move vocabulary, official checkpoint bundles, and PyG
    graph construction helpers.
Output:
    Normalized prediction summaries and subgroup metrics for Streamlit.
Run:
    Imported by the Streamlit debugger and unit tests; no standalone CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics

import chess
import torch
from torch_geometric.data import Batch

from src.inference.chess_gat_inference import (
    ModelBundle,
    best_legal_prediction,
    load_model_bundle,
    move_to_san,
    puzzle_row_to_graph,
    summarize_prediction,
)
from src.models.chess_legal_scorer import ChessGATLegalMoveScorer
from src.training.legal_mask import apply_legal_mask, build_legal_mask_for_batch
from src.training.model_a3_legal_scorer import (
    build_candidate_batch,
    build_model_a3,
    ModelA3LegalScorerConfig,
)


MODEL_A2_CHECKPOINT_PATH = Path("artifacts/model_a2_legal_mask_no_timing/best.pt")
MODEL_A3_CHECKPOINT_PATH = Path("artifacts/model_a3_legal_move_scorer_no_timing/best.pt")

MODEL_OPTIONS = [
    "Model A Raw",
    "Model A Best-Legal",
    "Model A2 Masked",
    "Model A3 Legal Scorer",
]


@dataclass(frozen=True)
class A3Bundle:
    """Loaded A3 model and metadata for read-only inference."""

    model: ChessGATLegalMoveScorer
    device: torch.device
    checkpoint_path: Path
    move_to_idx: dict[str, int]
    idx_to_move: dict[int, str]
    is_official_checkpoint: bool
    checkpoint_metadata: dict


def _state_dict_from_checkpoint(checkpoint):
    """Extract a model state dict from common checkpoint formats."""

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model_a2_bundle(checkpoint_path=MODEL_A2_CHECKPOINT_PATH, device=None):
    """Load the official A2 masked model using the Model A architecture."""

    return load_model_bundle(checkpoint_path=checkpoint_path, device=device)


def load_model_a3_bundle(
    checkpoint_path=MODEL_A3_CHECKPOINT_PATH,
    device=None,
    model_config=None,
):
    """Load the official A3 legal-candidate scorer."""

    from src.inference.chess_gat_inference import load_move_vocabulary

    move_to_idx, idx_to_move = load_move_vocabulary()
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = model_config or ModelA3LegalScorerConfig()
    model = build_model_a3(config)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model A3 checkpoint unavailable: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model.load_state_dict(_state_dict_from_checkpoint(checkpoint))
    model.to(resolved_device)
    model.eval()
    return A3Bundle(
        model=model,
        device=resolved_device,
        checkpoint_path=checkpoint_path,
        move_to_idx=move_to_idx,
        idx_to_move=idx_to_move,
        is_official_checkpoint=checkpoint_path == MODEL_A3_CHECKPOINT_PATH,
        checkpoint_metadata=checkpoint if isinstance(checkpoint, dict) else {},
    )


def legal_candidate_summary(scores, candidate_uci, board, target_move, k=5):
    """Build a native legal-candidate Top-K summary from A3 scores."""

    ordered = torch.argsort(scores, descending=True)
    target_move = str(target_move)
    target_index = candidate_uci.index(target_move)
    target_rank = int((ordered == target_index).nonzero(as_tuple=False)[0].item()) + 1
    top_rows = []
    for rank, candidate_index in enumerate(ordered[: min(k, len(candidate_uci))].tolist(), 1):
        move = candidate_uci[int(candidate_index)]
        top_rows.append(
            {
                "rank": rank,
                "move": move,
                "san": move_to_san(board, move),
                "score": float(scores[int(candidate_index)].detach().cpu().item()),
                "legal": True,
                "ground_truth": move == target_move,
            }
        )
    return {
        "inference_status": "SUCCESS",
        "topk": top_rows,
        "target_rank": target_rank,
        "target_oov": False,
        "top1_hit": target_rank == 1,
        "top3_hit": target_rank <= min(3, len(candidate_uci)),
        "top5_hit": target_rank <= min(5, len(candidate_uci)),
        "top1_legal": True,
        "top1_label": "HIT" if target_rank == 1 else "MISS",
        "top3_label": "HIT" if target_rank <= min(3, len(candidate_uci)) else "MISS",
        "top5_label": "HIT" if target_rank <= min(5, len(candidate_uci)) else "MISS",
        "best_legal": top_rows[0] if top_rows else None,
        "legal_candidate_count": len(candidate_uci),
        "illegal_top1_rate": 0.0,
        "model_mode": "Model A3 Legal Scorer",
    }


def best_legal_summary(logits, bundle, row, k=5):
    """Summarize Model A logits after canonical legal filtering."""

    board = chess.Board(row.FEN)
    legal_rows = []
    probabilities = torch.softmax(logits.detach().cpu(), dim=0)
    ordered = torch.argsort(logits.detach().cpu(), descending=True)
    target = str(row.TargetMove)
    target_rank = None
    legal_rank = 0
    for raw_rank, class_index in enumerate(ordered.tolist(), 1):
        move = bundle.idx_to_move[int(class_index)]
        if chess.Move.from_uci(move) not in board.legal_moves:
            continue
        legal_rank += 1
        if move == target and target_rank is None:
            target_rank = legal_rank
        if len(legal_rows) < k:
            legal_rows.append(
                {
                    "rank": legal_rank,
                    "raw_rank": raw_rank,
                    "move": move,
                    "san": move_to_san(board, move),
                    "softmax_score": float(probabilities[int(class_index)].item()),
                    "legal": True,
                    "ground_truth": move == target,
                }
            )
    return {
        "inference_status": "SUCCESS",
        "topk": legal_rows,
        "target_rank": target_rank,
        "target_oov": target_rank is None,
        "top1_hit": target_rank == 1,
        "top3_hit": target_rank is not None and target_rank <= min(3, legal_rank),
        "top5_hit": target_rank is not None and target_rank <= min(5, legal_rank),
        "top1_legal": True,
        "top1_label": "HIT" if target_rank == 1 else "MISS",
        "top3_label": "HIT" if target_rank is not None and target_rank <= 3 else "MISS",
        "top5_label": "HIT" if target_rank is not None and target_rank <= 5 else "MISS",
        "best_legal": best_legal_prediction(logits, bundle.idx_to_move, board, target),
        "legal_candidate_count": legal_rank,
        "illegal_top1_rate": 0.0,
        "model_mode": "Model A Best-Legal",
    }


def run_a3_single_inference(bundle, row, top_k=5):
    """Run A3 native legal-candidate inference on one final puzzle row."""

    graph = puzzle_row_to_graph(row, bundle.move_to_idx)
    batch = Batch.from_data_list([graph]).to(bundle.device)
    candidate = build_candidate_batch(batch)
    with torch.no_grad():
        output = bundle.model(batch, candidate["candidate_moves"])
    return legal_candidate_summary(
        scores=output["scores"],
        candidate_uci=output["candidate_uci"],
        board=chess.Board(row.FEN),
        target_move=str(row.TargetMove),
        k=top_k,
    )


def run_model_mode_inference(mode, row, bundle_a=None, bundle_a2=None, bundle_a3=None, top_k=5):
    """Run one selected no-timing inference mode."""

    if mode == "Model A3 Legal Scorer":
        if bundle_a3 is None:
            raise ValueError("Model A3 bundle is required.")
        return run_a3_single_inference(bundle_a3, row, top_k=top_k)

    if mode in {"Model A Raw", "Model A Best-Legal"}:
        if bundle_a is None:
            raise ValueError("Model A bundle is required.")
        graph = puzzle_row_to_graph(row, bundle_a.move_to_idx)
        batch = Batch.from_data_list([graph]).to(bundle_a.device)
        with torch.no_grad():
            logits = bundle_a.model(batch)[0]
        if mode == "Model A Raw":
            result = summarize_prediction(
                logits=logits,
                board=chess.Board(row.FEN),
                target_move=str(row.TargetMove),
                move_to_idx=bundle_a.move_to_idx,
                idx_to_move=bundle_a.idx_to_move,
                k=top_k,
            )
            result["model_mode"] = mode
            return result
        return best_legal_summary(logits, bundle_a, row, k=top_k)

    if mode == "Model A2 Masked":
        if bundle_a2 is None:
            raise ValueError("Model A2 bundle is required.")
        graph = puzzle_row_to_graph(row, bundle_a2.move_to_idx)
        batch = Batch.from_data_list([graph]).to(bundle_a2.device)
        with torch.no_grad():
            logits = bundle_a2.model(batch)
            legal_mask = build_legal_mask_for_batch(
                batch,
                bundle_a2.move_to_idx,
                bundle_a2.num_classes,
            )
            masked = apply_legal_mask(logits, legal_mask)[0]
        result = summarize_prediction(
            logits=masked,
            board=chess.Board(row.FEN),
            target_move=str(row.TargetMove),
            move_to_idx=bundle_a2.move_to_idx,
            idx_to_move=bundle_a2.idx_to_move,
            k=top_k,
        )
        result["model_mode"] = mode
        result["illegal_top1_rate"] = 0.0
        return result

    raise ValueError(f"Unknown model mode: {mode}")


def finalize_prediction_metrics(records):
    """Compute Top-K and rank metrics from normalized prediction records."""

    if not records:
        return {
            "N": 0,
            "top1": None,
            "top3": None,
            "top5": None,
            "mean_legal_target_rank": None,
            "median_legal_target_rank": None,
            "illegal_top1_rate": None,
        }
    ranks = [
        record["target_rank"]
        for record in records
        if record.get("target_rank") is not None
    ]
    n = len(records)
    return {
        "N": n,
        "top1": sum(record["top1_hit"] for record in records) / n,
        "top3": sum(record["top3_hit"] for record in records) / n,
        "top5": sum(record["top5_hit"] for record in records) / n,
        "mean_legal_target_rank": statistics.mean(ranks) if ranks else None,
        "median_legal_target_rank": statistics.median(ranks) if ranks else None,
        "illegal_top1_rate": sum(not record.get("top1_legal", True) for record in records) / n,
    }


def evaluate_mate_in_one_modes(dataframe, modes, bundles, limit=None, top_k=5):
    """Evaluate selected no-timing modes on Mate-in-1 rows."""

    rows = [
        row
        for row in dataframe.itertuples(index=False)
        if "mateIn1" in str(getattr(row, "Themes", "")).split() or int(getattr(row, "MateDepth")) == 1
    ]
    if limit is not None:
        rows = rows[: int(limit)]
    result = {}
    for mode in modes:
        records = [
            run_model_mode_inference(
                mode,
                row,
                bundle_a=bundles.get("Model A Raw") or bundles.get("Model A Best-Legal"),
                bundle_a2=bundles.get("Model A2 Masked"),
                bundle_a3=bundles.get("Model A3 Legal Scorer"),
                top_k=top_k,
            )
            for row in rows
        ]
        result[mode] = finalize_prediction_metrics(records)
    return result


def error_analysis_record(row, prediction):
    """Return one normalized error-analysis row for the Streamlit UI."""

    top1 = prediction["topk"][0] if prediction.get("topk") else {}
    return {
        "PuzzleId": getattr(row, "PuzzleId", None),
        "FEN": str(row.FEN),
        "TargetMove": str(row.TargetMove),
        "PredictedTop1": top1.get("move"),
        "TopK": prediction.get("topk", []),
        "TargetRank": prediction.get("target_rank"),
        "Top1Hit": bool(prediction.get("top1_hit")),
        "Top3Hit": bool(prediction.get("top3_hit")),
        "Top5Hit": bool(prediction.get("top5_hit")),
        "MateDepth": getattr(row, "MateDepth", None),
        "Rating": getattr(row, "Rating", None),
        "Themes": getattr(row, "Themes", ""),
        "Top1Legal": prediction.get("top1_legal"),
        "LegalCandidates": prediction.get("legal_candidate_count"),
        "ModelMode": prediction.get("model_mode"),
    }

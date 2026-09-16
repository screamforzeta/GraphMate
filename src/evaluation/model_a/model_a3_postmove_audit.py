"""Post-move and opponent-response diagnostic audit for frozen A3.

Purpose:
    Measure whether A3 Top5 candidate errors expose deterministic signals
    after applying a candidate move, and whether one-ply opponent-response
    aggregates add diagnostic information beyond immediate post-move features.
Input:
    Frozen A3 checkpoint, official data/pyg test split, and puzzle test CSV.
Output:
    artifacts/model_a3_postmove_audit/*.json and report.md.
Run:
    python3 -m src.cli.evaluation.evaluate_model_a3_postmove_audit
"""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import statistics
import time

import chess
import pandas as pd
import torch
from torch_geometric.data import Batch

from src.evaluation.model_a.model_a3_error_audit import (
    EXPECTED_SPLIT_N,
    PIECE_NAMES,
    PIECE_VALUES,
    move_type_features,
    topk_and_row,
)
from src.evaluation.model_a.model_a_vs_a2_vs_a3 import (
    MODEL_A3_CHECKPOINT,
    MODEL_A3_REFERENCE,
    load_eval_model_a3,
)
from src.graph.pyg_dataset import load_pyg_dataset
from src.training.model_a.model_a3_legal_scorer import build_candidate_batch


OUTPUT_DIR = Path("artifacts/model_a3_postmove_audit")
TEST_CSV = Path("data/final/puzzles/test.csv")
PYG_ROOT = Path("data/pyg")
TOP_K = 5


@dataclass
class A3PostMoveAuditConfig:
    """Runtime options for A3 post-move diagnostic audit."""

    batch_size: int = 128
    device: str = "cpu"
    non_blocking: bool = False
    amp: bool = False
    output_dir: str = str(OUTPUT_DIR)
    checkpoint: str = str(MODEL_A3_CHECKPOINT)
    pyg_root: str = str(PYG_ROOT)
    top_k: int = TOP_K


def material_for_color(board, color):
    """Return simple material count for one side, excluding kings."""

    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if piece_type is None or piece_type == chess.KING:
            continue
        total += len(board.pieces(piece_type, color)) * value
    return total


def attacked_piece_count(board, color):
    """Count pieces of color currently attacked by the opponent."""

    opponent = not color
    count = 0
    for square, piece in board.piece_map().items():
        if piece.color == color and board.is_attacked_by(opponent, square):
            count += 1
    return count


def legal_mobility(board, color):
    """Return legal move count for color in board's current position."""

    if board.turn == color:
        return board.legal_moves.count()
    copy_board = board.copy(stack=False)
    copy_board.turn = color
    return copy_board.legal_moves.count()


def candidate_level1_features(fen, move_uci):
    """Compute deterministic Level-1 features after applying one candidate."""

    board = chess.Board(str(fen))
    move = chess.Move.from_uci(str(move_uci))
    if move not in board.legal_moves:
        raise ValueError(f"Illegal candidate {move_uci} for FEN: {fen}")
    mover = board.turn
    opponent = not mover
    base = move_type_features(fen, move_uci)
    board.push(move)
    moved_piece = board.piece_at(move.to_square)
    attackers = board.attackers(opponent, move.to_square)
    defenders = board.attackers(mover, move.to_square)
    own_material = material_for_color(board, mover)
    opponent_material = material_for_color(board, opponent)
    gives_check = base["gives_check"]
    return {
        **base,
        "opponent_in_check": board.is_check(),
        "opponent_legal_move_count": board.legal_moves.count(),
        "opponent_has_legal_moves": bool(board.legal_moves.count()),
        "resulting_checkmate": board.is_checkmate(),
        "resulting_stalemate": board.is_stalemate(),
        "moved_piece_attacked_after": bool(attackers),
        "moved_piece_defended_after": bool(defenders),
        "attackers_on_destination_count": len(attackers),
        "defenders_on_destination_count": len(defenders),
        "own_king_in_check_after": board.is_check() if board.turn == mover else False,
        "opponent_king_mobility_after": legal_mobility(board, opponent),
        "checking_piece_attacked_after": bool(attackers) if gives_check else None,
        "checking_piece_defended_after": bool(defenders) if gives_check else None,
        "own_material_after": own_material,
        "opponent_material_after": opponent_material,
        "material_balance_after": own_material - opponent_material,
        "own_attacked_piece_count_after": attacked_piece_count(board, mover),
        "opponent_attacked_piece_count_after": attacked_piece_count(board, opponent),
        "own_legal_mobility_after": legal_mobility(board, mover),
        "opponent_legal_mobility_after": legal_mobility(board, opponent),
        "moved_piece_type_after": PIECE_NAMES.get(moved_piece.piece_type) if moved_piece else None,
    }


def response_features_for_move(board, response, candidate_to_square, mover):
    """Return one legal opponent response's shallow tactical features."""

    opponent = board.turn
    captured_piece = board.piece_at(response.to_square)
    captures_candidate = response.to_square == candidate_to_square and board.is_capture(response)
    response_capture = board.is_capture(response)
    if board.is_en_passant(response):
        captured_piece = chess.Piece(chess.PAWN, mover)
    copy_board = board.copy(stack=False)
    gives_check = copy_board.gives_check(response)
    piece = copy_board.piece_at(response.from_square)
    copy_board.push(response)
    mover_material = material_for_color(copy_board, mover)
    opponent_material = material_for_color(copy_board, opponent)
    return {
        "response_is_capture": bool(response_capture),
        "response_gives_check": bool(gives_check),
        "response_gives_checkmate": copy_board.is_checkmate(),
        "captures_candidate_piece": bool(captures_candidate),
        "captured_material_value": PIECE_VALUES.get(captured_piece.piece_type if captured_piece else None, 0),
        "material_balance_after_response": mover_material - opponent_material,
        "responder_piece_type": PIECE_NAMES.get(piece.piece_type) if piece else None,
    }


def aggregate_response_features(fen, move_uci):
    """Compute Level-2 aggregate features over all legal opponent responses."""

    board = chess.Board(str(fen))
    move = chess.Move.from_uci(str(move_uci))
    mover = board.turn
    board.push(move)
    responses = list(board.legal_moves)
    if not responses:
        return {
            "opponent_response_count": 0,
            "fraction_responses_capture_candidate_piece": 0.0,
            "fraction_responses_give_check": 0.0,
            "fraction_responses_give_checkmate": 0.0,
            "max_material_loss_after_one_response": 0.0,
            "mean_material_loss_after_one_response": 0.0,
            "min_material_balance_after_response": None,
            "mean_material_balance_after_response": None,
            "fraction_responses_leaving_original_side_in_check": 0.0,
            "number_of_forcing_responses": 0,
        }
    before_balance = material_for_color(board, mover) - material_for_color(board, not mover)
    response_rows = [
        response_features_for_move(board, response, move.to_square, mover)
        for response in responses
    ]
    balances = [row["material_balance_after_response"] for row in response_rows]
    losses = [before_balance - balance for balance in balances]
    forcing = [
        row for row in response_rows
        if row["response_gives_check"] or row["response_gives_checkmate"] or row["captures_candidate_piece"]
    ]
    n = len(response_rows)
    return {
        "opponent_response_count": n,
        "fraction_responses_capture_candidate_piece": sum(row["captures_candidate_piece"] for row in response_rows) / n,
        "fraction_responses_give_check": sum(row["response_gives_check"] for row in response_rows) / n,
        "fraction_responses_give_checkmate": sum(row["response_gives_checkmate"] for row in response_rows) / n,
        "max_material_loss_after_one_response": max(losses),
        "mean_material_loss_after_one_response": statistics.mean(losses),
        "min_material_balance_after_response": min(balances),
        "mean_material_balance_after_response": statistics.mean(balances),
        "fraction_responses_leaving_original_side_in_check": sum(row["response_gives_check"] for row in response_rows) / n,
        "number_of_forcing_responses": len(forcing),
    }


def candidate_diagnostic_features(fen, move_uci):
    """Return Level-1 and Level-2 features for a candidate move."""

    level1 = candidate_level1_features(fen, move_uci)
    level2 = aggregate_response_features(fen, move_uci)
    return {"level1": level1, "level2": level2}


def feature_value(features, path):
    """Read a nested candidate feature by dotted path."""

    current = features
    for part in path.split("."):
        current = current.get(part)
        if current is None:
            return None
    return current


def summarize_numeric(values):
    """Return compact numeric summary."""

    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def paired_feature_analysis(pairs, feature_paths):
    """Compare target-vs-wrong candidate features for recoverable A3 errors."""

    result = {}
    for path in feature_paths:
        target_values = [feature_value(pair["target_features"], path) for pair in pairs]
        wrong_values = [feature_value(pair["wrong_features"], path) for pair in pairs]
        numeric_rows = [
            (float(t), float(w))
            for t, w in zip(target_values, wrong_values)
            if isinstance(t, (int, float, bool)) and isinstance(w, (int, float, bool))
        ]
        if not numeric_rows:
            continue
        diffs = [t - w for t, w in numeric_rows]
        entry = {
            "n": len(numeric_rows),
            "target": summarize_numeric([t for t, _ in numeric_rows]),
            "wrong": summarize_numeric([w for _, w in numeric_rows]),
            "paired_difference": summarize_numeric(diffs),
            "fraction_target_gt_wrong": sum(t > w for t, w in numeric_rows) / len(numeric_rows),
            "fraction_target_eq_wrong": sum(t == w for t, w in numeric_rows) / len(numeric_rows),
            "fraction_target_lt_wrong": sum(t < w for t, w in numeric_rows) / len(numeric_rows),
        }
        if all(isinstance(value, bool) for value in target_values + wrong_values if value is not None):
            counts = Counter((bool(t), bool(w)) for t, w in zip(target_values, wrong_values))
            entry["transition_counts"] = {
                f"target_{key[0]}__wrong_{key[1]}": value
                for key, value in counts.items()
            }
        result[path] = entry
    return result


def rank_group(rank):
    """Return requested target-rank group label."""

    rank = int(rank)
    if rank == 1:
        return "rank1"
    if rank == 2:
        return "rank2"
    if 3 <= rank <= 5:
        return "rank3_5"
    return "rank_gt5"


def bucket_label(row):
    """Return bucket labels requested by the audit."""

    target_move_features = row.get("target_move_features") or row.get("target_features")
    labels = [
        rank_group(row["target_rank"]),
        f"mateIn{row['MateDepth']}",
        f"target_piece_{target_move_features['moving_piece_type']}",
    ]
    if target_move_features["quiet_move"]:
        labels.append("quiet_target")
    if target_move_features["is_capture"]:
        labels.append("capture_target")
    if target_move_features["gives_check"]:
        labels.append("checking_target")
    if target_move_features["gives_checkmate"]:
        labels.append("mating_target")
    return labels


def runtime_summary(started, candidates, responses=0):
    """Return runtime counters and throughput."""

    elapsed = max(time.perf_counter() - started, 1e-12)
    return {
        "wall_time_seconds": elapsed,
        "candidates_processed": candidates,
        "responses_enumerated": responses,
        "candidates_per_second": candidates / elapsed,
        "responses_per_second": responses / elapsed if responses else None,
    }


def parity_from_rows(rows):
    """Validate A3 test parity before interpreting diagnostics."""

    n = len(rows)
    ranks = [row["target_rank"] for row in rows]
    metrics = {
        "n": n,
        "top1": sum(rank <= 1 for rank in ranks) / n,
        "top3": sum(rank <= 3 for rank in ranks) / n,
        "top5": sum(rank <= 5 for rank in ranks) / n,
        "top10": sum(rank <= 10 for rank in ranks) / n,
        "mean_rank": statistics.mean(ranks),
        "median_rank": statistics.median(ranks),
        "illegal_top1": 0.0,
    }
    checks = {
        "n": metrics["n"] == MODEL_A3_REFERENCE["n"],
        "top1": abs(metrics["top1"] - MODEL_A3_REFERENCE["a3_top1"]) <= 1e-12,
        "top3": abs(metrics["top3"] - MODEL_A3_REFERENCE["a3_top3"]) <= 1e-12,
        "top5": abs(metrics["top5"] - MODEL_A3_REFERENCE["a3_top5"]) <= 1e-12,
        "mean_rank": abs(metrics["mean_rank"] - MODEL_A3_REFERENCE["a3_mean_legal_rank"]) <= 1e-12,
        "median_rank": abs(metrics["median_rank"] - MODEL_A3_REFERENCE["a3_median_legal_rank"]) <= 1e-12,
        "illegal_top1": metrics["illegal_top1"] == MODEL_A3_REFERENCE["a3_illegal_top1_rate"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "metrics": metrics, "checks": checks}


LEVEL1_FEATURES = [
    "level1.gives_check",
    "level1.gives_checkmate",
    "level1.is_capture",
    "level1.material_delta_immediate",
    "level1.opponent_legal_move_count",
    "level1.moved_piece_attacked_after",
    "level1.moved_piece_defended_after",
    "level1.attackers_on_destination_count",
    "level1.defenders_on_destination_count",
    "level1.material_balance_after",
    "level1.own_attacked_piece_count_after",
    "level1.opponent_attacked_piece_count_after",
]
LEVEL2_FEATURES = [
    "level2.opponent_response_count",
    "level2.fraction_responses_capture_candidate_piece",
    "level2.fraction_responses_give_check",
    "level2.fraction_responses_give_checkmate",
    "level2.max_material_loss_after_one_response",
    "level2.mean_material_loss_after_one_response",
    "level2.min_material_balance_after_response",
    "level2.mean_material_balance_after_response",
    "level2.number_of_forcing_responses",
]


def analyze_pairs(rows):
    """Build pairwise target-vs-wrong analysis for recoverable Top5 errors."""

    pairs = []
    bucket_pairs = defaultdict(list)
    unrecoverable = 0
    recoverable = 0
    level1_started = time.perf_counter()
    level1_candidates = 0
    level2_started = time.perf_counter()
    level2_candidates = 0
    level2_responses = []
    for row in rows:
        if row["top1_correct"]:
            continue
        if row["target_rank"] > TOP_K:
            unrecoverable += 1
            continue
        recoverable += 1
        target_features = candidate_diagnostic_features(row["FEN"], row["target_move"])
        wrong_features = candidate_diagnostic_features(row["FEN"], row["A3_top1_move"])
        level1_candidates += 2
        level2_candidates += 2
        level2_responses.extend([
            target_features["level2"]["opponent_response_count"],
            wrong_features["level2"]["opponent_response_count"],
        ])
        pair = {
            "PuzzleId": row["PuzzleId"],
            "FEN": row["FEN"],
            "MateDepth": row["MateDepth"],
            "rating": row["rating"],
            "target_rank": row["target_rank"],
            "target_move": row["target_move"],
            "wrong_move": row["A3_top1_move"],
            "target_move_features": row["target_features"],
            "target_features": target_features,
            "wrong_features": wrong_features,
        }
        pairs.append(pair)
        for label in bucket_label(row):
            bucket_pairs[label].append(pair)
    all_analysis = paired_feature_analysis(pairs, LEVEL1_FEATURES + LEVEL2_FEATURES)
    level1_analysis = {key: all_analysis[key] for key in LEVEL1_FEATURES if key in all_analysis}
    level2_analysis = {key: all_analysis[key] for key in LEVEL2_FEATURES if key in all_analysis}
    bucket_analysis = {
        label: {
            "n": len(items),
            "level1": paired_feature_analysis(items, LEVEL1_FEATURES),
            "level2": paired_feature_analysis(items, LEVEL2_FEATURES),
        }
        for label, items in sorted(bucket_pairs.items())
    }
    level1_runtime = runtime_summary(level1_started, level1_candidates)
    level2_runtime = runtime_summary(level2_started, level2_candidates, sum(level2_responses))
    if level2_responses:
        level2_runtime.update({
            "mean_responses_per_candidate": statistics.mean(level2_responses),
            "p95_responses_per_candidate": sorted(level2_responses)[int(0.95 * (len(level2_responses) - 1))],
            "max_responses_per_candidate": max(level2_responses),
        })
    return {
        "pairs": pairs,
        "recoverable_top5_errors": recoverable,
        "unrecoverable_top5_errors": unrecoverable,
        "level1_feature_analysis": level1_analysis,
        "level2_feature_analysis": level2_analysis,
        "pairwise_target_vs_wrong": all_analysis,
        "bucket_analysis": bucket_analysis,
        "runtime": {"level1": level1_runtime, "level2": level2_runtime},
    }


def evidence_label(level1, level2):
    """Return documented architecture evidence label from descriptive signals."""

    l1 = max((abs(v["fraction_target_gt_wrong"] - v["fraction_target_lt_wrong"]) for v in level1.values()), default=0.0)
    l2 = max((abs(v["fraction_target_gt_wrong"] - v["fraction_target_lt_wrong"]) for v in level2.values()), default=0.0)
    if l2 >= l1 + 0.10:
        return "OPPONENT_RESPONSE_RERANKER_SUPPORTED"
    if l1 >= 0.15:
        return "POST_MOVE_RERANKER_SUPPORTED"
    return "EVIDENCE_INCONCLUSIVE"


def run_rows_inference(config):
    """Run frozen A3 inference on the official test split and return rows."""

    device = torch.device(config.device)
    model, _ = load_eval_model_a3(Path(config.checkpoint), device)
    dataset = load_pyg_dataset(root=config.pyg_root, split="test")
    dataframe = pd.read_csv(TEST_CSV)
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
            for local_index, (graph, source_row) in enumerate(zip(graphs, batch_rows)):
                rank = topk_and_row(
                    output["scores"],
                    output["candidate_ptr"],
                    output["candidate_uci"],
                    target_indices[local_index],
                    local_index,
                )
                target_features = move_type_features(graph.fen, str(graph.target_move))
                rows.append({
                    "PuzzleId": str(source_row.PuzzleId),
                    "FEN": str(graph.fen),
                    "rating": int(source_row.Rating),
                    "MateDepth": int(source_row.MateDepth),
                    "target_move": str(graph.target_move),
                    "target_features": target_features,
                    **rank,
                })
    return rows


def write_json(path, payload):
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_report(summary):
    """Render Markdown report for post-move audit."""

    return "\n".join([
        "# Model A3 Post-Move Error Audit",
        "",
        "POST_HOC_TEST_SET_ANALYSIS = YES",
        "",
        "This is a diagnostic-only audit. It does not implement A4 and must not be used to tune on the official test set.",
        "",
        "## A3 Parity",
        "",
        f"`{summary['a3_parity']['status']}`",
        "",
        "## Top5 Reranker Ceiling",
        "",
        f"- Top5 recall / ceiling: `{summary['top5_reranker_ceiling']}`",
        f"- Recoverable Top5 errors: `{summary['recoverable_top5_errors']}`",
        f"- Unrecoverable Top5 errors: `{summary['unrecoverable_top5_errors']}`",
        "",
        "## Level 1 Feature Analysis",
        "",
        "```json",
        json.dumps(summary["top_level1_signals"], indent=2),
        "```",
        "",
        "## Level 2 Feature Analysis",
        "",
        "```json",
        json.dumps(summary["top_level2_signals"], indent=2),
        "```",
        "",
        "## Runtime",
        "",
        "```json",
        json.dumps(summary["runtime"], indent=2),
        "```",
        "",
        "## A4 Architecture Evidence",
        "",
        f"`{summary['a4_architecture_evidence']}`",
        "",
    ])


def top_signals(analysis, limit=5):
    """Return feature entries ordered by absolute paired directional gap."""

    scored = []
    for key, value in analysis.items():
        gap = abs(value["fraction_target_gt_wrong"] - value["fraction_target_lt_wrong"])
        scored.append((gap, key, value))
    return [
        {"feature": key, "directional_gap": gap, **value}
        for gap, key, value in sorted(scored, reverse=True)[:limit]
    ]


def run_postmove_audit(config):
    """Run full A3 post-move diagnostic audit."""

    rows = run_rows_inference(config)
    parity = parity_from_rows(rows)
    if parity["status"] != "PASS":
        raise RuntimeError(f"A3 parity failed; refusing diagnostics: {parity}")
    analysis = analyze_pairs(rows)
    top5_recall = parity["metrics"]["top5"]
    level1 = analysis["level1_feature_analysis"]
    level2 = analysis["level2_feature_analysis"]
    summary = {
        "post_hoc_test_set_analysis": True,
        "a3_parity": parity,
        "test_n": len(rows),
        "a3_top1": parity["metrics"]["top1"],
        "a3_top5": parity["metrics"]["top5"],
        "top5_reranker_ceiling": top5_recall,
        "recoverable_top5_errors": analysis["recoverable_top5_errors"],
        "unrecoverable_top5_errors": analysis["unrecoverable_top5_errors"],
        "level2_adds_measurable_signal": "INCONCLUSIVE",
        "top_level1_signals": top_signals(level1),
        "top_level2_signals": top_signals(level2),
        "runtime": analysis["runtime"],
        "a4_architecture_evidence": evidence_label(level1, level2),
        "a4_implemented": False,
    }
    output_dir = Path(config.output_dir)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "level1_feature_analysis.json", level1)
    write_json(output_dir / "level2_feature_analysis.json", level2)
    write_json(output_dir / "pairwise_target_vs_wrong.json", analysis["pairwise_target_vs_wrong"])
    write_json(output_dir / "bucket_analysis.json", analysis["bucket_analysis"])
    write_json(output_dir / "runtime.json", analysis["runtime"])
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return summary

from types import SimpleNamespace

import chess
import pytest
import torch

from src.graph.graph_builder import build_graph
from src.inference.chess_gat_inference import (
    check_user_move,
    compute_metric_parity,
    empty_error_analysis_metrics,
    error_category,
    evaluate_rows_with_logits,
    finalize_error_analysis_metrics,
    hit_label,
    is_legal_uci,
    is_mate_in_one_row,
    puzzle_row_to_graph,
    rank_bucket,
    summarize_prediction,
    target_rank_from_logits,
    update_error_analysis_metrics,
    verify_target_checkmate,
)


def _row(fen, target_move, themes="mate mateIn1 oneMove", mate_depth=1):
    return SimpleNamespace(
        FEN=fen,
        TargetMove=target_move,
        Themes=themes,
        MateDepth=mate_depth,
    )


def test_ui_graph_helper_matches_official_graph_builder_for_known_target():
    move_to_idx = {"g6g7": 0, "e2e4": 1}
    row = _row(
        "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1",
        "g6g7",
    )

    expected = build_graph(row.FEN, row.TargetMove, move_to_idx)
    actual = puzzle_row_to_graph(row, move_to_idx)

    assert torch.equal(actual.x, expected.x)
    assert torch.equal(actual.edge_index, expected.edge_index)
    assert torch.equal(actual.edge_attr, expected.edge_attr)
    assert torch.equal(actual.global_features, expected.global_features)
    assert torch.equal(actual.y, expected.y)


def test_oov_target_graph_builds_without_crashing_and_rank_is_oov():
    move_to_idx = {"e2e4": 0}
    row = _row(
        "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1",
        "g6g7",
    )

    graph = puzzle_row_to_graph(row, move_to_idx)
    rank = target_rank_from_logits(
        torch.tensor([1.0]),
        row.TargetMove,
        move_to_idx,
    )

    assert graph.y.item() == 0
    assert graph.target_move == "g6g7"
    assert graph.target_oov is True
    assert rank is None


def test_raw_top1_is_not_replaced_when_it_is_illegal():
    board = chess.Board()
    move_to_idx = {"e2e5": 0, "g1f3": 1, "e2e4": 2}
    idx_to_move = {0: "e2e5", 1: "g1f3", 2: "e2e4"}
    logits = torch.tensor([10.0, 9.0, 8.0])

    summary = summarize_prediction(
        logits=logits,
        board=board,
        target_move="g1f3",
        move_to_idx=move_to_idx,
        idx_to_move=idx_to_move,
        k=3,
    )

    assert summary["topk"][0]["move"] == "e2e5"
    assert summary["topk"][0]["legal"] is False
    assert summary["top1_label"] == "MISS"
    assert summary["top3_label"] == "HIT"
    assert summary["best_legal"]["move"] == "g1f3"
    assert summary["best_legal"]["ground_truth"] is True
    assert summary["top3_hit"] is True
    assert summary["target_rank"] == 2


@pytest.mark.parametrize(
    ("rank", "top1", "top3", "top5", "bucket"),
    [
        (1, "HIT", "HIT", "HIT", "rank_1"),
        (2, "MISS", "HIT", "HIT", "rank_2_3"),
        (4, "MISS", "MISS", "HIT", "rank_4_5"),
        (7, "MISS", "MISS", "MISS", "rank_6_10"),
        (None, "OOV", "OOV", "OOV", "TARGET_OOV"),
    ],
)
def test_hit_labels_and_rank_buckets(rank, top1, top3, top5, bucket):
    assert hit_label(rank, 1) == top1
    assert hit_label(rank, 3) == top3
    assert hit_label(rank, 5) == top5
    assert rank_bucket(rank) == bucket


@pytest.mark.parametrize(
    ("rank", "legal", "category"),
    [
        (1, True, "TOP1_CORRECT"),
        (2, True, "TOP1_WRONG_LEGAL"),
        (2, False, "TOP1_ILLEGAL"),
        (None, False, "TARGET_OOV"),
    ],
)
def test_error_category_helper(rank, legal, category):
    assert error_category(rank, legal) == category


def test_user_move_validation_distinguishes_statuses():
    board = chess.Board()

    assert check_user_move(board, "bad", "e2e4")["status"] == "INVALID_SYNTAX"
    assert check_user_move(board, "e2e5", "e2e4")["status"] == "ILLEGAL_MOVE"
    assert check_user_move(board, "g1f3", "e2e4")["status"] == "LEGAL_BUT_WRONG"
    assert check_user_move(board, "e2e4", "e2e4")["status"] == "CORRECT"


def test_mate_in_one_metadata_and_checkmate_verification():
    row = _row(
        "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1",
        "g6g7",
    )

    assert is_mate_in_one_row(row) is True
    assert verify_target_checkmate(row.FEN, row.TargetMove) is True


def test_batch_metric_helper_counts_oov_and_illegal_top1():
    rows = [
        _row(chess.STARTING_FEN, "g1f3", themes="opening", mate_depth=3),
        _row(chess.STARTING_FEN, "b1c3", themes="opening", mate_depth=3),
    ]
    move_to_idx = {"e2e5": 0, "g1f3": 1}
    idx_to_move = {0: "e2e5", 1: "g1f3"}
    logits = [
        torch.tensor([10.0, 9.0]),
        torch.tensor([10.0, 9.0]),
    ]

    metrics = evaluate_rows_with_logits(
        rows=rows,
        logits_list=logits,
        move_to_idx=move_to_idx,
        idx_to_move=idx_to_move,
        k=2,
    )

    assert metrics["total"] == 2
    assert metrics["evaluable"] == 1
    assert metrics["oov"] == 1
    assert metrics["illegal_top1"] == 2
    assert metrics["top1"] == 0
    assert metrics["top3"] == 1
    assert metrics["top5"] == 1


def test_error_analysis_aggregation_and_parity_infrastructure():
    rows = [
        _row(chess.STARTING_FEN, "g1f3", themes="mateIn1 sacrifice", mate_depth=1),
        _row(chess.STARTING_FEN, "e2e4", themes="mateIn2", mate_depth=2),
        _row(chess.STARTING_FEN, "b1c3", themes="mateIn2", mate_depth=2),
    ]
    move_to_idx = {"g1f3": 0, "e2e4": 1, "e2e5": 2}
    idx_to_move = {0: "g1f3", 1: "e2e4", 2: "e2e5"}
    logits = [
        torch.tensor([9.0, 2.0, 1.0]),
        torch.tensor([8.0, 7.0, 6.0]),
        torch.tensor([8.0, 7.0, 6.0]),
    ]

    metrics = empty_error_analysis_metrics()
    for row, row_logits in zip(rows, logits):
        summary = summarize_prediction(
            logits=row_logits,
            board=chess.Board(row.FEN),
            target_move=row.TargetMove,
            move_to_idx=move_to_idx,
            idx_to_move=idx_to_move,
            k=3,
        )
        update_error_analysis_metrics(metrics, row, summary)
    finalized = finalize_error_analysis_metrics(metrics)

    assert finalized["total"] == 3
    assert finalized["evaluable"] == 2
    assert finalized["oov"] == 1
    assert finalized["error_categories"]["TOP1_CORRECT"] == 1
    assert finalized["error_categories"]["TOP1_WRONG_LEGAL"] == 1
    assert finalized["rank_buckets"]["rank_2_3"] == 1
    assert finalized["rates"]["top1"] == pytest.approx(0.5)
    assert finalized["rates"]["top3"] == pytest.approx(1.0)
    assert finalized["rates"]["best_legal_top1"] == pytest.approx(0.5)

    parity = compute_metric_parity(
        finalized,
        reference={"examples": 2, "top1": 0.5, "top3": 1.0, "top5": 1.0},
    )
    assert parity["status"] == "PASS"


@pytest.mark.parametrize(
    ("move", "expected"),
    [
        ("e2e4", True),
        ("e2e5", False),
    ],
)
def test_legal_uci_helper(move, expected):
    assert is_legal_uci(chess.Board(), move) is expected

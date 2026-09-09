from types import SimpleNamespace

import pytest
import torch

from src.inference.no_timing_multimodel import (
    MODEL_OPTIONS,
    best_legal_summary,
    error_analysis_record,
    finalize_prediction_metrics,
    legal_candidate_summary,
)


def _row():
    return SimpleNamespace(
        PuzzleId="p1",
        FEN="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        TargetMove="e2e4",
        MateDepth=1,
        Rating=1200,
        Themes="mate mateIn1",
    )


def test_model_options_include_all_no_timing_modes():
    assert MODEL_OPTIONS == [
        "Model A Raw",
        "Model A Best-Legal",
        "Model A2 Masked",
        "Model A3 Legal Scorer",
    ]


def test_a3_legal_candidate_summary_ranks_candidate_order():
    board_row = _row()
    scores = torch.tensor([0.1, 0.8, 0.2])
    candidates = ["g1f3", "e2e4", "d2d4"]

    summary = legal_candidate_summary(
        scores=scores,
        candidate_uci=candidates,
        board=__import__("chess").Board(board_row.FEN),
        target_move=board_row.TargetMove,
        k=3,
    )

    assert summary["model_mode"] == "Model A3 Legal Scorer"
    assert summary["target_rank"] == 1
    assert summary["top1_hit"] is True
    assert summary["legal_candidate_count"] == 3
    assert summary["illegal_top1_rate"] == 0.0


def test_best_legal_summary_filters_illegal_global_logits():
    row = _row()
    bundle = SimpleNamespace(
        idx_to_move={
            0: "a1a8",
            1: "g1f3",
            2: "e2e4",
        }
    )
    logits = torch.tensor([10.0, 8.0, 7.0])

    summary = best_legal_summary(logits, bundle, row, k=2)

    assert summary["model_mode"] == "Model A Best-Legal"
    assert summary["topk"][0]["move"] == "g1f3"
    assert summary["top1_legal"] is True
    assert summary["target_rank"] == 2


def test_prediction_metrics_and_error_record_format():
    row = _row()
    prediction = {
        "model_mode": "Model A3 Legal Scorer",
        "topk": [{"move": "e2e4", "score": 1.0, "legal": True}],
        "target_rank": 1,
        "top1_hit": True,
        "top3_hit": True,
        "top5_hit": True,
        "top1_legal": True,
        "legal_candidate_count": 20,
    }

    metrics = finalize_prediction_metrics([prediction])
    record = error_analysis_record(row, prediction)

    assert metrics["N"] == 1
    assert metrics["top1"] == pytest.approx(1.0)
    assert metrics["median_legal_target_rank"] == 1
    assert record["PuzzleId"] == "p1"
    assert record["PredictedTop1"] == "e2e4"
    assert record["LegalCandidates"] == 20

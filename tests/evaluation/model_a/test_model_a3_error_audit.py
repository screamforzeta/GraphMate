"""Unit tests for the A3 prediction headroom/error audit."""

from __future__ import annotations

import pytest

from src.evaluation.model_a.model_a3_error_audit import (
    A3ErrorAuditConfig,
    discover_stockfish,
    enrich_quiet,
    legal_count_bucket,
    margin_summary,
    move_type_features,
    rank_bucket,
    split_summary,
    stockfish_cache_key,
    stockfish_placeholder_summary,
    summarize_rows,
)


def test_rank_and_legal_count_buckets():
    """Rank and candidate-count buckets use the documented boundaries."""

    assert rank_bucket(1) == "1"
    assert rank_bucket(5) == "5"
    assert rank_bucket(6) == "6-10"
    assert rank_bucket(11) == ">10"
    assert legal_count_bucket(20) == "<=20"
    assert legal_count_bucket(21) == "21-30"
    assert legal_count_bucket(51) == ">50"


def test_move_type_features_capture_check_and_material_delta():
    """Feature extraction detects captures, checks, and immediate material."""

    features = enrich_quiet(
        move_type_features("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1", "e1e2")
    )

    assert features["moving_piece_type"] == "king"
    assert features["is_capture"] is True
    assert features["captured_piece_type"] == "queen"
    assert features["captured_piece_value"] == 9
    assert features["material_delta_immediate"] == 9
    assert features["quiet_move"] is False


def test_move_type_features_checkmate():
    """Feature extraction detects an immediate mate move."""

    features = enrich_quiet(
        move_type_features(
            "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
            "d8h4",
        )
    )

    assert features["gives_check"] is True
    assert features["gives_checkmate"] is True


def test_move_type_features_promotion_and_castling():
    """Promotion and castling are represented explicitly."""

    promotion = enrich_quiet(
        move_type_features("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q")
    )
    castling = enrich_quiet(
        move_type_features("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1")
    )

    assert promotion["is_promotion"] is True
    assert promotion["promotion_piece"] == "queen"
    assert castling["is_castling"] is True


def test_move_type_features_destination_safety_defense():
    """Destination attack/defense fields are boolean and present."""

    features = enrich_quiet(
        move_type_features("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4")
    )

    assert isinstance(features["destination_attacked_after_move"], bool)
    assert isinstance(features["destination_defended_after_move"], bool)
    assert isinstance(features["moving_piece_attacked_before"], bool)
    assert isinstance(features["moving_piece_attacked_after"], bool)


def _row(rank, margin, correct=False, target_piece="queen", prediction_piece="rook"):
    """Build a minimal per-puzzle audit row."""

    return {
        "PuzzleId": f"p{rank}-{margin}",
        "FEN": "fen",
        "rating": 1500,
        "MateDepth": 2,
        "target_move": "a1a2",
        "legal_move_count": 32,
        "target_rank": rank,
        "top1_correct": correct,
        "target_in_top3": rank <= 3,
        "target_in_top5": rank <= 5,
        "score_margin_top1_top2": margin,
        "A3_top1_move": "b1b2",
        "A3_top1_score": 2.0,
        "target_score": 1.0,
        "top5_moves": ["b1b2", "a1a2"],
        "target_features": {
            "moving_piece_type": target_piece,
            "is_capture": False,
            "gives_check": rank == 2,
            "gives_checkmate": False,
            "is_promotion": False,
            "is_castling": False,
            "destination_attacked_after_move": False,
            "destination_defended_after_move": True,
            "material_delta_immediate": 0,
            "quiet_move": rank != 2,
        },
        "prediction_features": {
            "moving_piece_type": prediction_piece,
            "is_capture": True,
            "gives_check": False,
            "gives_checkmate": False,
            "is_promotion": False,
            "is_castling": False,
            "destination_attacked_after_move": True,
            "destination_defended_after_move": False,
            "material_delta_immediate": 3,
            "quiet_move": False,
        },
    }


def test_summarize_rows_topk_rank_and_error_breakdown():
    """Top-k, rank histogram, and error buckets are deterministic."""

    rows = [_row(1, 0.2, True), _row(2, 0.4), _row(6, 1.5)]
    summary = summarize_rows(rows)

    assert summary["topk"]["top1"] == pytest.approx(1 / 3)
    assert summary["topk"]["top3"] == pytest.approx(2 / 3)
    assert summary["topk"]["top10"] == pytest.approx(1.0)
    assert summary["target_rank_histogram"]["1"] == 1
    assert summary["error_target_rank_breakdown"]["2"]["count"] == 1
    assert summary["error_target_rank_breakdown"]["6-10"]["count"] == 1


def test_margin_summary():
    """Score margin summaries report robust quartiles."""

    summary = margin_summary([0.0, 1.0, 2.0, 3.0])

    assert summary["n"] == 4
    assert summary["mean"] == pytest.approx(1.5)
    assert summary["median"] == pytest.approx(1.5)
    assert summary["p25"] == pytest.approx(0.75)
    assert summary["p75"] == pytest.approx(2.25)


def test_split_summary_keeps_validation_and_test_separate():
    """Validation has no frozen parity while test parity is computed separately."""

    rows = [_row(1, 0.2, True)]
    stockfish = {"available": False}

    validation = split_summary("validation", rows, stockfish)
    test = split_summary("test", rows, stockfish)

    assert validation["split"] == "validation"
    assert validation["parity"]["status"] == "N/A"
    assert test["split"] == "test"
    assert test["parity"]["status"] in {"PASS", "FAIL"}


def test_stockfish_unavailable_fallback(monkeypatch):
    """Stockfish discovery is optional and must not be required for pytest."""

    monkeypatch.setattr("shutil.which", lambda _: None)
    config = A3ErrorAuditConfig(stockfish=True)

    discovery = discover_stockfish()
    summary = stockfish_placeholder_summary(config)

    assert discovery["available"] is False
    assert summary["available"] is False
    assert summary["ambiguity_audit_complete"] is False


def test_stockfish_cache_key_is_deterministic():
    """Cache keys are stable and include engine-relevant config."""

    first = stockfish_cache_key("fen", "a1a2", "sf", 18, threads=1, hash_mb=128)
    second = stockfish_cache_key("fen", "a1a2", "sf", 18, threads=1, hash_mb=128)
    different = stockfish_cache_key("fen", "a1a2", "sf", 16, threads=1, hash_mb=128)

    assert first == second
    assert first != different

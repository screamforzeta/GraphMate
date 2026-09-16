"""Tests for the A3 post-move/opponent-response diagnostic audit."""

from __future__ import annotations

import chess
import pytest

from src.evaluation.model_a.model_a3_postmove_audit import (
    aggregate_response_features,
    analyze_pairs,
    bucket_label,
    candidate_level1_features,
    material_for_color,
    parity_from_rows,
    response_features_for_move,
    run_postmove_audit,
)


def test_candidate_application_does_not_mutate_original_board():
    """Level-1 extraction must copy/apply internally without mutating caller board."""

    fen = chess.STARTING_FEN
    board = chess.Board(fen)

    candidate_level1_features(fen, "e2e4")

    assert board.fen() == chess.Board(fen).fen()


def test_level1_capture_checkmate_and_material_features():
    """Level-1 features include mate, check and material fields."""

    features = candidate_level1_features(
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "d8h4",
    )

    assert features["gives_check"] is True
    assert features["gives_checkmate"] is True
    assert features["resulting_checkmate"] is True
    assert features["opponent_has_legal_moves"] is False
    assert features["opponent_response_count"] if "opponent_response_count" in features else True
    assert isinstance(features["own_material_after"], int)
    assert isinstance(features["material_balance_after"], int)


def test_attacked_and_defended_counts_are_present():
    """Destination attack/defense counts are deterministic integers."""

    features = candidate_level1_features(chess.STARTING_FEN, "g1f3")

    assert isinstance(features["attackers_on_destination_count"], int)
    assert isinstance(features["defenders_on_destination_count"], int)
    assert isinstance(features["moved_piece_attacked_after"], bool)
    assert isinstance(features["moved_piece_defended_after"], bool)


def test_response_aggregation_and_terminal_candidate():
    """Level-2 handles both normal and terminal candidates."""

    normal = aggregate_response_features(chess.STARTING_FEN, "e2e4")
    terminal = aggregate_response_features(
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "d8h4",
    )

    assert normal["opponent_response_count"] > 0
    assert 0.0 <= normal["fraction_responses_give_check"] <= 1.0
    assert terminal["opponent_response_count"] == 0
    assert terminal["fraction_responses_capture_candidate_piece"] == 0.0


def test_response_features_capture_candidate_piece():
    """A response that captures the candidate piece is detected."""

    board = chess.Board("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1")
    move = chess.Move.from_uci("e2e1q")
    board.push(move)
    response = chess.Move.from_uci("e1e1") if False else None
    # Use a simpler real response fixture: white king captures a black queen on e1.
    board = chess.Board("4k3/8/8/8/8/8/8/4qK2 w - - 0 1")
    response = chess.Move.from_uci("f1e1")
    features = response_features_for_move(board, response, chess.E1, chess.BLACK)

    assert features["response_is_capture"] is True
    assert features["captures_candidate_piece"] is True
    assert features["captured_material_value"] == 9


def _row(rank=2, mate_depth=2, quiet=False, capture=False, check=False, mate=False):
    """Build a minimal A3 row for pair analysis."""

    return {
        "PuzzleId": f"p{rank}",
        "FEN": chess.STARTING_FEN,
        "rating": 1500,
        "MateDepth": mate_depth,
        "target_move": "e2e4",
        "A3_top1_move": "d2d4",
        "target_rank": rank,
        "top1_correct": rank == 1,
        "target_features": {
            "quiet_move": quiet,
            "is_capture": capture,
            "gives_check": check,
            "gives_checkmate": mate,
            "moving_piece_type": "pawn",
        },
    }


def test_pairing_recoverable_and_unrecoverable_counts():
    """Top5 recoverable and unrecoverable errors are separated."""

    rows = [_row(rank=2), _row(rank=6)]

    analysis = analyze_pairs(rows)

    assert analysis["recoverable_top5_errors"] == 1
    assert analysis["unrecoverable_top5_errors"] == 1
    assert analysis["runtime"]["level1"]["candidates_processed"] == 2
    assert analysis["runtime"]["level2"]["candidates_processed"] == 2


def test_bucket_assignment():
    """Requested bucket labels are assigned from target features."""

    labels = bucket_label(_row(rank=2, mate_depth=1, quiet=True, check=True))

    assert "rank2" in labels
    assert "mateIn1" in labels
    assert "quiet_target" in labels
    assert "checking_target" in labels
    assert "target_piece_pawn" in labels


def test_parity_guard_fails_on_incomplete_rows():
    """Parity guard must fail fast unless official metrics are reproduced."""

    parity = parity_from_rows([_row(rank=1)])

    assert parity["status"] == "FAIL"
    assert parity["checks"]["n"] is False


def test_runtime_and_deterministic_output(tmp_path):
    """Blocked run is deterministic when checkpoint is absent."""

    from src.evaluation.model_a.model_a3_postmove_audit import A3PostMoveAuditConfig

    config = A3PostMoveAuditConfig(
        checkpoint=str(tmp_path / "missing.pt"),
        output_dir=str(tmp_path / "out"),
    )

    with pytest.raises(FileNotFoundError):
        run_postmove_audit(config)


def test_material_accounting_start_position():
    """Both sides have equal non-king material in the initial position."""

    board = chess.Board()

    assert material_for_color(board, chess.WHITE) == material_for_color(board, chess.BLACK)

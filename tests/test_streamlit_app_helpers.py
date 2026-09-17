from types import SimpleNamespace

import chess
import pandas as pd
import pytest

from src.streamlit_app.graph_metadata import graph_representation_metadata
from src.streamlit_app.model_registry import MODEL_REGISTRY, model_table_rows
from src.streamlit_app.puzzle_inference import (
    invalidate_prediction_if_fen_changed,
    row_for_current_fen,
    run_reference_line_rollout,
)
from src.streamlit_app.puzzle_sequence import (
    PuzzleSession,
    filter_by_mate_depth,
    legal_move_options,
    move_history_rows,
    solution_moves_from_lichess_moves,
    solution_san_sequence,
    transformed_solver_fen,
)
from src.streamlit_app.results_loader import (
    canonical_mate_depth_rows,
    canonical_rating_rows,
    canonical_result_snapshot,
    format_metric_rows,
)


def test_model_registry_contains_frozen_family_and_a1_is_diagnostic():
    rows = model_table_rows()
    keys = [row["key"] for row in rows]

    assert keys == ["A", "A1", "A2", "A3", "A4", "B"]
    assert MODEL_REGISTRY["A1"].runnable_in_streamlit is True
    assert MODEL_REGISTRY["A3"].display_name == "A3 - Legal Move Scorer"
    assert MODEL_REGISTRY["B"].display_name == "B - Timing-Aware Legal Move Scorer"
    assert "not a separate architecture" in MODEL_REGISTRY["A1"].status
    assert MODEL_REGISTRY["B"].timing_used is True


def test_graph_representation_metadata_matches_node15_edge5_global4():
    metadata = graph_representation_metadata()

    assert metadata["nodes"] == 64
    assert metadata["node_dim"] == 15
    assert metadata["edge_dim"] == 5
    assert metadata["global_dim"] == 4
    assert "is_check" not in metadata["node_features"]
    assert metadata["edge_features"] == ["legal_move", "attack", "defend", "pin", "check_line"]


def test_mate_depth_filter_supports_mate_in_1_to_5():
    df = pd.DataFrame({"MateDepth": [1, 2, 3, 4, 5]})

    assert len(filter_by_mate_depth(df, "All")) == 5
    for depth in range(1, 6):
        filtered = filter_by_mate_depth(df, f"Mate in {depth}")
        assert filtered["MateDepth"].tolist() == [depth]


def test_lichess_setup_move_solver_fen_and_solution_sequence():
    original = chess.STARTING_FEN
    moves = "e2e4 e7e5 g1f3"

    solver_fen = transformed_solver_fen(original, moves)
    solution = solution_moves_from_lichess_moves(moves)

    board = chess.Board(original)
    board.push(chess.Move.from_uci("e2e4"))
    assert solver_fen == board.fen()
    assert solution == ["e7e5", "g1f3"]


def test_solution_san_sequence_and_puzzle_session_undo_reset():
    fen = transformed_solver_fen(chess.STARTING_FEN, "e2e4 e7e5 g1f3")
    solution = ["e7e5", "g1f3"]
    san_rows = solution_san_sequence(fen, solution)
    session = PuzzleSession.create(fen, solution)

    assert san_rows[0]["san"] == "e5"
    session.play("e7e5")
    assert session.is_exact_solution_prefix()
    assert not session.is_complete_solution()
    session.play("g1f3")
    assert session.is_complete_solution()
    session.undo()
    assert session.human_moves == ["e7e5"]
    session.reset()
    assert session.human_moves == []
    assert session.current_fen == fen


def test_legal_move_options_are_san_first():
    options = legal_move_options(chess.STARTING_FEN)

    e4 = next(option for option in options if option["uci"] == "e2e4")
    assert e4["san"] == "e4"
    assert e4["label"] == "e4"
    assert "e2e4" not in e4["label"]


def test_play_solver_move_auto_applies_reference_reply_and_history_rows():
    fen = transformed_solver_fen(chess.STARTING_FEN, "e2e4 e7e5 g1f3")
    session = PuzzleSession.create(fen, ["e7e5", "g1f3"])

    result = session.play_solver_move_with_reference_reply("e7e5")
    rows = move_history_rows(session.start_fen, session.human_moves, session.auto_reply_indices)

    assert result["status"] == "COMPLETE"
    assert result["san"] == "e5"
    assert result["auto_reply"] == "g1f3"
    assert result["auto_reply_san"] == "Nf3"
    assert session.human_moves == ["e7e5", "g1f3"]
    assert rows[0]["san"] == "e5"
    assert rows[0]["source"] == "Your move"
    assert rows[1]["san"] == "Nf3"
    assert rows[1]["source"] == "Reference reply"


def test_play_solver_move_rejects_wrong_move_without_mutation():
    fen = transformed_solver_fen(chess.STARTING_FEN, "e2e4 e7e5 g1f3")
    session = PuzzleSession.create(fen, ["e7e5", "g1f3"])

    result = session.play_solver_move_with_reference_reply("c7c5")

    assert result["status"] == "INCORRECT"
    assert session.human_moves == []


def test_current_fen_changes_after_solver_move_and_reference_reply():
    fen = transformed_solver_fen(chess.STARTING_FEN, "e2e4 e7e5 g1f3")
    session = PuzzleSession.create(fen, ["e7e5", "g1f3"])

    result = session.play_solver_move_with_reference_reply("e7e5")
    expected_board = chess.Board(fen)
    expected_board.push(chess.Move.from_uci("e7e5"))
    expected_after_user = expected_board.fen()
    expected_board.push(chess.Move.from_uci("g1f3"))

    assert result["fen_after_user_move"] == expected_after_user
    assert session.current_fen == expected_board.fen()
    assert result["fen_after_turn"] == expected_board.fen()
    assert "b8c6" in session.legal_moves()


def test_row_for_current_fen_and_prediction_invalidation():
    row = SimpleNamespace(FEN=chess.STARTING_FEN, TargetMove="e2e4", PuzzleId="x")
    board = chess.Board(chess.STARTING_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    current_row = row_for_current_fen(row, board.fen(), target_move="e7e5")
    state = {"model_prediction_fen": chess.STARTING_FEN, "model_result": {"old": True}, "a4_result": {"old": True}}

    invalidated = invalidate_prediction_if_fen_changed(state, board.fen())

    assert current_row.FEN == board.fen()
    assert current_row.TargetMove == "e7e5"
    assert invalidated is True
    assert state["model_result"] is None
    assert state["a4_result"] is None


def test_reference_line_rollout_uses_successive_fens_without_mutating_session():
    fen = transformed_solver_fen(chess.STARTING_FEN, "e2e4 e7e5 g1f3")
    session = PuzzleSession.create(fen, ["e7e5", "g1f3"])
    seen_fens = []

    def predictor(position_fen):
        seen_fens.append(position_fen)
        return "e7e5"

    rollout = run_reference_line_rollout(fen, ["e7e5", "g1f3"], predictor)

    assert rollout["solved"] is True
    assert seen_fens == [fen]
    assert session.current_fen == fen


def test_reference_line_rollout_reports_failure_ply():
    fen = transformed_solver_fen(chess.STARTING_FEN, "e2e4 e7e5 g1f3")

    rollout = run_reference_line_rollout(fen, ["e7e5", "g1f3"], lambda _fen: "c7c5")

    assert rollout["solved"] is False
    assert rollout["failure_ply"] == 1


def test_illegal_puzzle_session_move_raises():
    session = PuzzleSession.create(chess.STARTING_FEN, ["e2e4"])

    with pytest.raises(ValueError, match="Illegal move"):
        session.play("e2e5")


def test_result_snapshot_labels_terminal_and_timing_values():
    snapshot = canonical_result_snapshot()

    assert snapshot["A4_test_end_to_end_top1"] > snapshot["A3_test_top1"]
    assert snapshot["B_synthetic_top1"] < snapshot["A3_test_top1"]


def test_performance_tables_use_real_canonical_values():
    depth_rows = canonical_mate_depth_rows()
    rating_rows = canonical_rating_rows()
    formatted = format_metric_rows(depth_rows[:1])

    assert depth_rows[0]["Puzzle depth"] == "Mate in 1"
    assert depth_rows[0]["A3"] == pytest.approx(0.7955)
    assert depth_rows[0]["A4"] == pytest.approx(0.9393)
    assert rating_rows[-1]["Rating"] == "2400+"
    assert rating_rows[-1]["A4"] == pytest.approx(0.45)
    assert formatted[0]["Improvement"] == "14.38%"

from types import SimpleNamespace

import chess
import pandas as pd
import pytest

from src.streamlit_app.graph_metadata import graph_representation_metadata
from src.streamlit_app.model_registry import MODEL_REGISTRY, model_table_rows
from src.streamlit_app.puzzle_sequence import (
    PuzzleSession,
    filter_by_mate_depth,
    solution_moves_from_lichess_moves,
    solution_san_sequence,
    transformed_solver_fen,
)
from src.streamlit_app.results_loader import canonical_result_snapshot


def test_model_registry_contains_frozen_family_and_a1_is_diagnostic():
    rows = model_table_rows()
    keys = [row["key"] for row in rows]

    assert keys == ["A", "A1", "A2", "A3", "A4", "B"]
    assert MODEL_REGISTRY["A1"].runnable_in_streamlit is True
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


def test_illegal_puzzle_session_move_raises():
    session = PuzzleSession.create(chess.STARTING_FEN, ["e2e4"])

    with pytest.raises(ValueError, match="Illegal move"):
        session.play("e2e5")


def test_result_snapshot_labels_terminal_and_timing_values():
    snapshot = canonical_result_snapshot()

    assert snapshot["A4_test_end_to_end_top1"] > snapshot["A3_test_top1"]
    assert snapshot["B_synthetic_top1"] < snapshot["A3_test_top1"]

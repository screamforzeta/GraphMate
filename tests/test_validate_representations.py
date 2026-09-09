import pandas as pd
import torch
from torch_geometric.data import Data

from src.validation.representations import (
    EXPECTED_EDGE_FEATURE_DIM,
    EXPECTED_GLOBAL_SHAPE,
    EXPECTED_NODE_SHAPE,
    fen_matches,
    sample_indices,
    validate_graph,
    validate_puzzle_row,
)


def test_sample_indices_is_deterministic_and_bounded():
    first = sample_indices(
        total=10,
        sample_size=4,
        seed=42,
    )
    second = sample_indices(
        total=10,
        sample_size=4,
        seed=42,
    )

    assert first == second
    assert first == sorted(first)
    assert len(first) == 4
    assert max(first) < 10


def test_fen_matches_normalized_python_chess_output():
    fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"

    assert fen_matches(
        fen,
        fen,
    )


def test_validate_puzzle_row_accepts_correct_lichess_semantics():
    row = pd.Series(
        {
            "OriginalFEN": (
                "rnbqkbnr/pppppppp/8/8/8/8/"
                "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            ),
            "FEN": (
                "rnbqkbnr/pppppppp/8/8/4P3/8/"
                "PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ),
            "Moves": "e2e4 e7e5",
            "TargetMove": "e7e5",
        }
    )

    assert validate_puzzle_row(row) == []


def test_validate_puzzle_row_reports_target_mismatch():
    row = pd.Series(
        {
            "OriginalFEN": (
                "rnbqkbnr/pppppppp/8/8/8/8/"
                "PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            ),
            "FEN": (
                "rnbqkbnr/pppppppp/8/8/4P3/8/"
                "PPPP1PPP/RNBQKBNR b KQkq - 0 1"
            ),
            "Moves": "e2e4 e7e5",
            "TargetMove": "d7d5",
        }
    )

    assert "TargetMove mismatch" in validate_puzzle_row(row)


def test_validate_graph_accepts_expected_shapes_and_multilabel_counts():
    graph = Data(
        x=torch.zeros(EXPECTED_NODE_SHAPE, dtype=torch.float),
        edge_index=torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.long,
        ),
        edge_attr=torch.tensor(
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.float,
        ),
        y=torch.tensor(0, dtype=torch.long),
    )
    graph.global_features = torch.zeros(
        EXPECTED_GLOBAL_SHAPE,
        dtype=torch.float,
    )

    failures, multilabel = validate_graph(graph)

    assert failures == []
    assert multilabel == {
        "single": 1,
        "two": 1,
        "three_plus": 1,
    }


def test_validate_graph_rejects_duplicate_edges_and_empty_labels():
    graph = Data(
        x=torch.zeros(EXPECTED_NODE_SHAPE, dtype=torch.float),
        edge_index=torch.tensor(
            [
                [0, 0],
                [1, 1],
            ],
            dtype=torch.long,
        ),
        edge_attr=torch.zeros(
            (2, EXPECTED_EDGE_FEATURE_DIM),
            dtype=torch.float,
        ),
        y=torch.tensor(0, dtype=torch.long),
    )
    graph.global_features = torch.zeros(
        EXPECTED_GLOBAL_SHAPE,
        dtype=torch.float,
    )

    failures, _ = validate_graph(graph)

    assert "duplicate edge" in failures
    assert "edge without active label" in failures

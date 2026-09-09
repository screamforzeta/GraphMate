import pandas as pd

from src.graph.move_encoder import compute_oov_stats
from src.data.preprocess.prepare_puzzles_dataset import (
    extract_target_move,
    transform_lichess_puzzle,
)


def test_lichess_puzzle_uses_position_after_setup_move():
    row = pd.Series(
        {
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/"
            "PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "e2e4 e7e5",
        }
    )

    transformed = transform_lichess_puzzle(row)

    assert transformed["OriginalFEN"] == row["FEN"]
    assert transformed["FEN"].startswith(
        "rnbqkbnr/pppppppp/8/8/4P3/8/"
    )
    assert transformed["TargetMove"] == "e7e5"
    assert transformed["InvalidReason"] is None


def test_lichess_puzzle_rejects_illegal_target_move():
    row = pd.Series(
        {
            "FEN": "rnbqkbnr/pppppppp/8/8/8/8/"
            "PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "Moves": "e2e4 e2e5",
        }
    )

    transformed = transform_lichess_puzzle(row)

    assert transformed["InvalidReason"] == "illegal_target_move"


def test_extract_target_move_supports_castling_and_promotion_strings():
    assert extract_target_move("e2e4 e1g1") == "e1g1"
    assert extract_target_move("a2a4 e7e8q") == "e7e8q"


def test_lichess_transform_supports_en_passant_target():
    row = pd.Series(
        {
            "FEN": "8/3p4/8/4P3/8/8/8/K6k b - - 0 1",
            "Moves": "d7d5 e5d6",
        }
    )

    transformed = transform_lichess_puzzle(row)

    assert transformed["TargetMove"] == "e5d6"
    assert transformed["InvalidReason"] is None


def test_compute_oov_stats_counts_validation_targets(tmp_path):
    csv_path = tmp_path / "val.csv"
    pd.DataFrame(
        {
            "TargetMove": [
                "e2e4",
                "e7e8q",
                "e1g1",
                "e5d6",
            ]
        }
    ).to_csv(csv_path, index=False)

    stats = compute_oov_stats(csv_path, {"e2e4": 0, "e1g1": 1})

    assert stats["input_targets"] == 4
    assert stats["oov_targets"] == 2
    assert stats["oov_percentage"] == 50.0
    assert stats["examples"] == ["e5d6", "e7e8q"]

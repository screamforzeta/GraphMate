"""Dataset adapters for LLM chess benchmarks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.streamlit_app.puzzle_sequence import solution_moves_from_lichess_moves


@dataclass(frozen=True)
class BenchmarkPuzzle:
    """Canonical benchmark puzzle record."""

    puzzle_id: str
    initial_solver_fen: str
    target_move: str
    reference_moves_uci: list[str]
    mate_depth: int | None = None
    rating: int | None = None
    source: str = "lichess"


def dataset_fingerprint(path):
    """Return SHA256 fingerprint for one dataset file."""

    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lichess_test(
    path="data/final/puzzles/test.csv",
    limit=None,
    offset=0,
    move_vocab_path="resources/move_encoder/move_to_idx.json",
    shared_gnn_population=True,
):
    """Load the frozen Lichess test split without running model inference."""

    dataframe = pd.read_csv(path)
    if shared_gnn_population and Path(move_vocab_path).exists():
        move_to_idx = json.loads(Path(move_vocab_path).read_text(encoding="utf-8"))
        dataframe = dataframe[dataframe["TargetMove"].astype(str).isin(move_to_idx)]
    rows = dataframe.iloc[int(offset) :]
    if limit is not None:
        rows = rows.iloc[: int(limit)]
    puzzles = []
    for row in rows.itertuples(index=False):
        puzzles.append(
            BenchmarkPuzzle(
                puzzle_id=str(row.PuzzleId),
                initial_solver_fen=str(row.FEN),
                target_move=str(row.TargetMove),
                reference_moves_uci=solution_moves_from_lichess_moves(row.Moves),
                mate_depth=int(row.MateDepth) if hasattr(row, "MateDepth") else None,
                rating=int(row.Rating) if hasattr(row, "Rating") else None,
            )
        )
    return puzzles


def validate_heldout_classic_record(record):
    """Validate one future held-out classic puzzle schema record."""

    required = {"puzzle_id", "source", "initial_solver_fen", "reference_moves_uci", "mate_depth"}
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"Missing heldout fields: {missing}")
    if not isinstance(record["reference_moves_uci"], list) or not record["reference_moves_uci"]:
        raise ValueError("reference_moves_uci must be a non-empty list")
    return True

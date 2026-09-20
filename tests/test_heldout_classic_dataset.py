import json

import pytest

from src.data.heldout_classic import (
    HeldoutProblem,
    board_state_key,
    build_heldout_dataset,
    canonical_dataset_fingerprint,
    detect_lichess_overlap,
    select_stratified,
    validate_problem,
)


MATE1_FEN = "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"
MATE1_MOVE = "g7f8"


def problem(**overrides):
    payload = {
        "heldout_id": "h1",
        "source": "synthetic_test_source",
        "source_problem_id": "p1",
        "fen": MATE1_FEN,
        "side_to_move": "white",
        "mate_depth": 1,
        "solution_uci": [MATE1_MOVE],
        "solution_plies": 1,
        "dataset_version": "test",
    }
    payload.update(overrides)
    return HeldoutProblem(**payload)


def test_valid_complete_solution_line():
    assert validate_problem(problem()) == []


def test_invalid_fen_and_side_to_move_handling():
    assert validate_problem(problem(fen="not a fen")) == ["invalid_fen"]
    assert "side_to_move_mismatch" in validate_problem(problem(side_to_move="black"))


def test_illegal_truncated_wrong_depth_and_not_mate():
    assert "illegal_move_at_ply_1" in validate_problem(problem(solution_uci=["e2e4"]))
    assert "truncated_solution_line" in validate_problem(problem(mate_depth=2, solution_uci=[MATE1_MOVE]))
    assert "mate_depth_mismatch" in validate_problem(problem(mate_depth=2, solution_uci=[MATE1_MOVE]))
    assert "line_not_checkmate" in validate_problem(problem(solution_uci=["g7g8"]))


def test_duplicate_keys_and_lichess_overlap_detection():
    key = board_state_key(MATE1_FEN)
    lichess_keys = {
        "train": {"exact": {MATE1_FEN}, "normalized": {key}, "original_exact": set()},
        "val": {"exact": set(), "normalized": set(), "original_exact": {MATE1_FEN}},
        "test": {"exact": set(), "normalized": set(), "original_exact": set()},
    }

    reasons = detect_lichess_overlap(problem(), lichess_keys)

    assert "lichess_train_solver_fen_exact_overlap" in reasons
    assert "lichess_train_solver_fen_normalized_overlap" in reasons
    assert "lichess_val_original_fen_exact_overlap" in reasons


def test_deterministic_stratified_sampling_and_fingerprint():
    problems = [
        problem(heldout_id=f"h{i}", source_problem_id=f"p{i}", mate_depth=1 if i < 3 else 2)
        for i in range(6)
    ]

    first = select_stratified(problems, per_depth=2, seed=42)
    second = select_stratified(problems, per_depth=2, seed=42)
    fp1 = canonical_dataset_fingerprint([item.to_row() for item in first])
    fp2 = canonical_dataset_fingerprint([item.to_row() for item in second])

    assert [item.heldout_id for item in first] == [item.heldout_id for item in second]
    assert len(first) == 4
    assert fp1 == fp2


def test_build_manifest_rejects_invalid_duplicate_and_overlap(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {
            "source": "public_source",
            "source_problem_id": "ok",
            "fen": MATE1_FEN,
            "mate_depth": 1,
            "solution_uci": [MATE1_MOVE],
        },
        {
            "source": "public_source",
            "source_problem_id": "duplicate",
            "fen": MATE1_FEN,
            "mate_depth": 1,
            "solution_uci": [MATE1_MOVE],
        },
        {
            "source": "public_source",
            "source_problem_id": "bad",
            "fen": MATE1_FEN,
            "mate_depth": 1,
            "solution_uci": ["g7g8"],
        },
    ]
    with source.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    monkeypatch.setattr(
        "src.data.heldout_classic.load_lichess_overlap_keys",
        lambda: {
            "train": {"exact": set(), "normalized": set(), "original_exact": set()},
            "val": {"exact": set(), "normalized": set(), "original_exact": set()},
            "test": {"exact": set(), "normalized": set(), "original_exact": set()},
        },
    )

    manifest = build_heldout_dataset(
        source,
        data_root=tmp_path / "heldout",
        dataset_version="test_v1",
        source_name="public_source",
        source_url="https://example.test/source",
        source_license="test-license",
    )

    assert manifest["dataset_status"] == "VALIDATED"
    assert manifest["accepted_count"] == 1
    assert manifest["rejected_count"] == 2
    assert manifest["model_outputs_used_for_selection"] is False
    assert manifest["dataset_used_for_model_tuning"] is False
    assert (tmp_path / "heldout" / "final" / "heldout_classic.csv").exists()
    assert (tmp_path / "heldout" / "final" / "manifest.json").exists()


def test_lifecycle_status_not_frozen_by_builder(monkeypatch, tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "source": "public_source",
                "source_problem_id": "ok",
                "fen": MATE1_FEN,
                "mate_depth": 1,
                "solution_uci": [MATE1_MOVE],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.data.heldout_classic.load_lichess_overlap_keys",
        lambda: {
            "train": {"exact": set(), "normalized": set(), "original_exact": set()},
            "val": {"exact": set(), "normalized": set(), "original_exact": set()},
            "test": {"exact": set(), "normalized": set(), "original_exact": set()},
        },
    )

    manifest = build_heldout_dataset(source, data_root=tmp_path / "heldout")

    assert manifest["dataset_status"] == "VALIDATED"
    assert manifest["dataset_status"] != "FROZEN"

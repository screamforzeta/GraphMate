import json
import itertools
from pathlib import Path

import chess
import pytest

from src.data.heldout_sources import yacpdb_candidate_build as build
from src.data.heldout_sources.yacpdb import YacpdbCandidate


def make_candidate(depth, index):
    board = chess.Board(None)
    board.turn = chess.WHITE
    board.clear_stack()
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
    squares = [square for square in chess.SQUARES if square not in {chess.A1, chess.H8}]
    first, second = list(itertools.combinations(squares, 2))[(depth - 1) * 25 + index]
    board.set_piece_at(first, chess.Piece(chess.KNIGHT, chess.WHITE))
    board.set_piece_at(second, chess.Piece(chess.BISHOP, chess.WHITE))
    source_id = str(depth * 100000 + index)
    return {
        "candidate": YacpdbCandidate(
            heldout_id=f"heldout_{source_id}",
            source_problem_id=source_id,
            fen=board.fen(),
            mate_depth=depth,
            stipulation=f"#{depth}",
            key_move_uci="a1b1",
            source_solution_raw="1.Ka1-b1 !",
            composer=f"Composer {index % 5}",
            source_reference=f"Source {index % 7}",
            publication_date=str(1900 + index % 50),
            source_url=f"https://www.yacpdb.org/#{source_id}",
            license=None,
            raw_record={"id": source_id},
        ),
        "parser_method": "popeye_coordinate",
        "key_token": "Ka1-b1",
        "piece_count": 4,
        "is_capture_key": False,
        "is_check_key": False,
        "is_promotion_key": False,
        "is_castling_key": False,
    }


def make_pool(per_depth=25):
    return {depth: [make_candidate(depth, index) for index in range(per_depth)] for depth in range(1, 11)}


def empty_lichess_keys():
    return {"train": {"exact": set(), "normalized": set(), "original_exact": set()}}


def test_deterministic_selection_counts_and_ordering():
    pool = make_pool(25)

    first = build.select_candidates(pool, per_depth=20, seed=42)
    second = build.select_candidates(pool, per_depth=20, seed=42)

    assert {depth: len(rows) for depth, rows in first.items()} == {depth: 20 for depth in range(1, 11)}
    assert sum(len(rows) for rows in first.values()) == 200
    assert {
        depth: [row["candidate"].source_problem_id for row in rows]
        for depth, rows in first.items()
    } == {
        depth: [row["candidate"].source_problem_id for row in rows]
        for depth, rows in second.items()
    }
    assert [row["candidate"].source_problem_id for row in first[1]] == sorted(
        [row["candidate"].source_problem_id for row in first[1]],
        key=int,
    )


def test_insufficient_clean_candidates_fail_closed():
    pool = make_pool(19)

    with pytest.raises(RuntimeError, match="Insufficient clean candidates"):
        build.select_candidates(pool, per_depth=20, seed=42)


def test_availability_fingerprint_mismatch_fail_closed(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "yacpdb_availability.json").write_text(
        json.dumps({"scan_fingerprint": "wrong", "status": "COMPLETE", "completed_depths": list(range(1, 11)), "config": {}})
    )

    with pytest.raises(RuntimeError, match="fingerprint"):
        build.load_availability(tmp_path, build.EXPECTED_AVAILABILITY_FINGERPRINT)


def test_selected_audits_fail_closed_for_duplicates_and_contamination():
    config = build.CandidateBuildConfig(output_root=Path("."), per_depth=1)
    rows = build.selected_rows({depth: [make_candidate(depth, depth)] for depth in range(1, 11)}, config)
    assert build.audit_selected_rows(rows, empty_lichess_keys())["duplicate_source_ids"] == 0

    duplicate_id_rows = [dict(row) for row in rows]
    duplicate_id_rows[1]["source_problem_id"] = duplicate_id_rows[0]["source_problem_id"]
    with pytest.raises(RuntimeError, match="Duplicate source_problem_id"):
        build.audit_selected_rows(duplicate_id_rows, empty_lichess_keys())

    duplicate_fen_rows = [dict(row) for row in rows]
    duplicate_fen_rows[1]["fen"] = duplicate_fen_rows[0]["fen"]
    with pytest.raises(RuntimeError, match="Exact FEN duplicate"):
        build.audit_selected_rows(duplicate_fen_rows, empty_lichess_keys())

    contaminated = {"train": {"exact": {rows[0]["fen"]}, "normalized": set(), "original_exact": set()}}
    with pytest.raises(RuntimeError, match="Lichess contamination"):
        build.audit_selected_rows(rows, contaminated)


def test_dataset_fingerprint_is_deterministic_and_timestamp_independent():
    config = build.CandidateBuildConfig(output_root=Path("."), per_depth=1)
    rows = build.selected_rows({depth: [make_candidate(depth, depth)] for depth in range(1, 11)}, config)
    manifest_a = {"created_at": "2020", "dataset_fingerprint": build.dataset_fingerprint(rows, config)}
    manifest_b = {"created_at": "2030", "dataset_fingerprint": build.dataset_fingerprint(rows, config)}

    assert manifest_a["dataset_fingerprint"] == manifest_b["dataset_fingerprint"]


def test_full_build_writes_manifest_selected_ids_and_queue(monkeypatch, tmp_path):
    availability = {
        "scan_fingerprint": build.EXPECTED_AVAILABILITY_FINGERPRINT,
        "status": "COMPLETE",
        "completed_depths": list(range(1, 11)),
        "config": {
            "scan_version": build.SCAN_VERSION,
            "importer_version": build.YACPDB_IMPORTER_VERSION,
            "key_extractor_version": build.KEY_EXTRACTOR_VERSION,
        },
        "depths": {str(depth): {"clean_unique_eligible": 25} for depth in range(1, 11)},
    }

    monkeypatch.setattr(build, "load_availability", lambda output_root, expected: availability)
    monkeypatch.setattr(build, "reconstruct_clean_candidate_pool", lambda output_root, artifact: (make_pool(25), {}))
    monkeypatch.setattr(build, "load_lichess_overlap_keys", lambda: empty_lichess_keys())
    monkeypatch.setattr(build, "write_markdown_report", lambda manifest, rows, path: None)

    manifest = build.build_candidate_dataset(build.CandidateBuildConfig(output_root=tmp_path))
    final_dir = tmp_path / "final" / build.DATASET_VERSION

    assert manifest["lifecycle_status"] == "VALIDATED_NOT_FROZEN"
    assert manifest["frozen"] is False
    assert manifest["forced_mate_verified"] is False
    assert manifest["lifecycle_status"] != "FROZEN"
    assert manifest["total_selected_count"] == 200
    assert manifest["selected_count_per_depth"] == {str(depth): 20 for depth in range(1, 11)}
    assert (final_dir / "manifest.json").exists()
    assert (final_dir / "selected_ids.json").exists()
    assert (final_dir / "dataset.jsonl").exists()
    assert sum(1 for _ in (final_dir / "forced_mate_verification_queue.jsonl").open()) == 200

import json
from pathlib import Path

import chess
import pytest

from src.verification import popeye


MATE1_FEN = "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"


def row(heldout_id="h1", source_id="1", depth=1, fen=MATE1_FEN, key="g7f8"):
    return {
        "heldout_id": heldout_id,
        "source_problem_id": source_id,
        "mate_depth": depth,
        "fen": fen,
        "key_move_uci": key,
    }


def write_dataset(root: Path, rows: list[dict], fingerprint=popeye.EXPECTED_DATASET_FINGERPRINT):
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps({"dataset_fingerprint": fingerprint, "lifecycle_status": "VALIDATED_NOT_FROZEN"}),
        encoding="utf-8",
    )
    with (root / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    (root / "dataset.csv").write_text("heldout_id\n" + "\n".join(item["heldout_id"] for item in rows) + "\n", encoding="utf-8")
    (root / "selected_ids.json").write_text(json.dumps({"selected": [item["source_problem_id"] for item in rows]}), encoding="utf-8")


def parent_result(item, reason="TIMEOUT", fingerprint="parent-fp", dataset_fingerprint=popeye.EXPECTED_DATASET_FINGERPRINT):
    return {
        "heldout_id": item["heldout_id"],
        "source_problem_id": item["source_problem_id"],
        "mate_depth": item["mate_depth"],
        "dataset_fingerprint": dataset_fingerprint,
        "verification_config_fingerprint": fingerprint,
        "timeout_seconds": 300,
        "verification_status": "FAILED" if reason == "TIMEOUT" else "VERIFIED",
        "verification_reason": reason,
        "forced_mate_verified": reason != "TIMEOUT",
    }


def write_results(path: Path, rows: list[dict]):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def test_input_adapter_simple_position_and_depths():
    text = popeye.popeye_input_for_row(row(depth=1))
    assert "BeginProblem" in text
    assert "Option NoBoard" in text
    assert "Stipulation #1" in text
    assert "Forsyth 7k/6Q1/6K1/8/8/8/8/8" in text

    text10 = popeye.popeye_input_for_row(row(depth=10))
    assert "Stipulation #10" in text10


def test_input_adapter_all_standard_pieces_and_knight_mapping():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

    assert popeye.fen_to_popeye_forsyth(fen) == "rsbqkbsr/pppppppp/8/8/8/8/PPPPPPPP/RSBQKBSR"


def test_input_adapter_rejects_unrepresented_state_and_malformed_fen():
    with pytest.raises(ValueError, match="white to move"):
        popeye.fen_to_popeye_forsyth("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    with pytest.raises(ValueError, match="castling"):
        popeye.fen_to_popeye_forsyth("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    with pytest.raises(ValueError, match="en-passant"):
        popeye.fen_to_popeye_forsyth("7k/8/8/3pP3/8/8/8/7K w - d6 0 1")
    with pytest.raises(ValueError):
        popeye.fen_to_popeye_forsyth("not a fen")


def test_process_runner_success_timeout_nonzero_and_missing(monkeypatch):
    class Completed:
        stdout = "ok"
        stderr = "warn"
        returncode = 0

    monkeypatch.setattr(popeye.subprocess, "run", lambda *args, **kwargs: Completed())
    result = popeye.run_popeye_process("/bin/echo", "input", 1)
    assert result.stdout == "ok"
    assert result.stderr == "warn"
    assert result.return_code == 0
    assert result.stdout_type == "str"
    assert result.stderr_type == "str"

    class BytesCompleted:
        stdout = "Popeye café".encode()
        stderr = "avertissement".encode()
        returncode = 1

    monkeypatch.setattr(popeye.subprocess, "run", lambda *args, **kwargs: BytesCompleted())
    result = popeye.run_popeye_process("/bin/echo", "input", 1)
    assert result.stdout == "Popeye café"
    assert result.stderr == "avertissement"
    assert result.return_code == 1
    assert result.stdout_type == "bytes"
    assert result.stderr_type == "bytes"

    def timeout(*args, **kwargs):
        raise popeye.subprocess.TimeoutExpired("cmd", 1, output=b"partial \xff", stderr=b"late \xff")

    monkeypatch.setattr(popeye.subprocess, "run", timeout)
    result = popeye.run_popeye_process("/bin/echo", "input", 1)
    assert result.timed_out is True
    assert result.stdout == "partial �"
    assert result.stderr == "late �"
    assert result.stdout_type == "bytes"
    assert result.stderr_type == "bytes"

    def missing(*args, **kwargs):
        raise OSError("missing executable")

    monkeypatch.setattr(popeye.subprocess, "run", missing)
    result = popeye.run_popeye_process("/missing", "input", 1)
    assert "missing executable" in result.stderr


def test_popeye_banner_version_parsing_rejects_invalid_outputs():
    identity = popeye.parse_popeye_banner("Popeye Linux-7.0.0-31-generic-unknown-64Bit v4.103 (1024 MB)")
    assert identity.version == "v4.103"
    assert identity.banner == "Popeye Linux-7.0.0-31-generic-unknown-64Bit v4.103 (1024 MB)"

    with pytest.raises(RuntimeError, match="missing valid"):
        popeye.parse_popeye_banner("error opening input file: No such file or directory")
    with pytest.raises(RuntimeError, match="empty output"):
        popeye.parse_popeye_banner("")
    with pytest.raises(RuntimeError, match="missing valid"):
        popeye.parse_popeye_banner("some unrelated solver banner")


def test_output_parser_one_key_multiple_no_solution_and_mismatch():
    parsed = popeye.parse_popeye_output("1.Qg7-f8 #\nsolution finished.", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_UNIQUE_KEY_MATCH"
    assert parsed.verified_keys_uci == ["g7f8"]

    parsed = popeye.parse_popeye_output("1.Qg7-f8 #\n1.Kg6-h6 #\nsolution finished.", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY"
    assert set(parsed.verified_keys_uci) == {"g7f8", "g6h6"}

    parsed = popeye.parse_popeye_output("1.Kg6-h6 #\nsolution finished.", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_KEY_MISMATCH"

    tree_output = "1.Qg7-f8 #\n  1...Kh8-g8\n  2.Kg6-h6 #\nsolution finished."
    parsed = popeye.parse_popeye_output(tree_output, "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_UNIQUE_KEY_MATCH"
    assert parsed.verified_keys_uci == ["g7f8"]

    parsed = popeye.parse_popeye_output("No solution", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "NO_SOLUTION"

    parsed = popeye.parse_popeye_output("header only", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "UNSUPPORTED_POPEYE_OUTPUT"


def test_root_key_extraction_uses_only_popeye_root_solution_lines():
    output_145182 = """Popeye Linux-7.0.0-31-generic-unknown-64Bit v4.103 (1024 MB)
heldout_id=yacpdb_classic_v1_0011 source_problem_id=145182

1.Rg8-g7 # !
1.Rg8-h8 # !
1.Nd7-f6 # !

solution finished. Time = 0.006 s
"""
    board_145182 = chess.Board("4K1R1/3N1N1k/3R3P/8/8/8/8/8 w - - 0 1")
    keys_145182 = [
        popeye.popeye_token_to_uci(token, board_145182)
        for token in popeye.extract_root_key_tokens(output_145182)
    ]
    assert keys_145182 == ["g8g7", "g8h8", "d7f6"]

    output_67081 = """Popeye Linux-7.0.0-31-generic-unknown-64Bit v4.103 (1024 MB)
heldout_id=yacpdb_classic_v1_0122 source_problem_id=67081

1.Qe4-f4 !
  1...Kh8-g8
  2.Qf4-f7 +
  2...Kg8-h8
  3.Qf7-f8 #
1.Qb7-f7 !
  1...Kh8-g8
  2.Qf7-f8 #

solution finished. Time = 3.000 s
"""
    tokens = popeye.extract_root_key_tokens(output_67081)
    assert tokens == ["Qe4-f4", "Qb7-f7"]
    assert "Qf4-f7" not in tokens
    assert "Qf7-f8" not in tokens


def test_config_fingerprint_and_resume_identity(tmp_path):
    config = popeye.PopeyeRunConfig("abc", "/bin/py", "v4.89", "Popeye test v4.89", 10)
    same = popeye.PopeyeRunConfig("abc", "/bin/py", "v4.89", "Popeye test v4.89", 10)
    other = popeye.PopeyeRunConfig("abc", "/bin/py", "v4.89", "Popeye test v4.89", 20)
    assert popeye.config_fingerprint(config) == popeye.config_fingerprint(same)
    assert popeye.config_fingerprint(config) != popeye.config_fingerprint(other)

    path = tmp_path / "results.jsonl"
    fp = popeye.config_fingerprint(config)
    path.write_text(json.dumps({"heldout_id": "h1", "verification_config_fingerprint": fp}) + "\n")
    assert "h1" in popeye.load_existing_results(path, fp)
    with pytest.raises(RuntimeError, match="incompatible"):
        popeye.load_existing_results(path, "different")


def test_retry_selection_is_strict_and_fail_closed(tmp_path):
    rows = [row("h1", "1"), row("h2", "2"), row("h3", "3")]
    parent = tmp_path / "parent.jsonl"
    write_results(
        parent,
        [
            parent_result(rows[0], "TIMEOUT"),
            parent_result(rows[1], "VERIFIED_UNIQUE_KEY_MATCH"),
            parent_result(rows[2], "POPEYE_ERROR"),
        ],
    )
    selected, provenance = popeye.select_timeout_retry_rows(parent, rows, popeye.EXPECTED_DATASET_FINGERPRINT)
    assert [item["heldout_id"] for item in selected] == ["h1"]
    assert provenance["retry_selected_count"] == 1
    assert provenance["parent_verification_config_fingerprint"] == "parent-fp"
    assert provenance["parent_timeout_seconds"] == 300

    write_results(parent, [parent_result(rows[0], "TIMEOUT"), parent_result(rows[0], "TIMEOUT")])
    with pytest.raises(RuntimeError, match="Duplicate"):
        popeye.select_timeout_retry_rows(parent, rows, popeye.EXPECTED_DATASET_FINGERPRINT)

    write_results(parent, [dict(parent_result(rows[0], "TIMEOUT"), heldout_id="unknown")])
    with pytest.raises(RuntimeError, match="unknown"):
        popeye.select_timeout_retry_rows(parent, rows, popeye.EXPECTED_DATASET_FINGERPRINT)

    write_results(parent, [parent_result(rows[0], "TIMEOUT", dataset_fingerprint="wrong")])
    with pytest.raises(RuntimeError, match="fingerprint"):
        popeye.select_timeout_retry_rows(parent, rows, popeye.EXPECTED_DATASET_FINGERPRINT)


def test_retry_config_fingerprint_depends_on_parent_and_timeout():
    base = popeye.PopeyeRunConfig(
        "dataset",
        "/py",
        "v4.103",
        "Popeye v4.103",
        1200,
        verification_pass="timeout_retry",
        parent_verification_config_fingerprint="parent-a",
        parent_timeout_seconds=300,
        parent_results_path="parent.jsonl",
        retry_selection_reason="TIMEOUT",
    )
    different_parent = popeye.PopeyeRunConfig(
        "dataset",
        "/py",
        "v4.103",
        "Popeye v4.103",
        1200,
        verification_pass="timeout_retry",
        parent_verification_config_fingerprint="parent-b",
        parent_timeout_seconds=300,
        parent_results_path="parent.jsonl",
        retry_selection_reason="TIMEOUT",
    )
    different_timeout = popeye.PopeyeRunConfig(
        "dataset",
        "/py",
        "v4.103",
        "Popeye v4.103",
        300,
        verification_pass="timeout_retry",
        parent_verification_config_fingerprint="parent-a",
        parent_timeout_seconds=300,
        parent_results_path="parent.jsonl",
        retry_selection_reason="TIMEOUT",
    )
    assert popeye.config_fingerprint(base) != "7500621cd95b841a0a105ec37f02b821ea3ca0fd116a78f7d94c244c7d9f7bb1"
    assert popeye.config_fingerprint(base) != popeye.config_fingerprint(different_parent)
    assert popeye.config_fingerprint(base) != popeye.config_fingerprint(different_timeout)


def test_orchestration_structured_results_summary_and_byte_integrity(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    rows = [row(f"h{i}", str(i), 1) for i in range(200)]
    write_dataset(dataset_dir, rows)
    before = popeye.canonical_file_hashes(dataset_dir)

    monkeypatch.setattr(popeye, "popeye_identity", lambda executable: popeye.PopeyeIdentity("v4.test", "Popeye test v4.test"))
    monkeypatch.setattr(
        popeye,
        "run_popeye_process",
        lambda executable, input_text, timeout: popeye.ProcessResult("1.Qg7-f8 #\nsolution finished.", "", 0, 0.01),
    )
    monkeypatch.setattr(popeye, "write_markdown_report", lambda *args, **kwargs: None)

    summary = popeye.run_verification(dataset_dir, tmp_path / "verification", "/fake/popeye", 10)
    after = popeye.canonical_file_hashes(dataset_dir)

    assert before == after
    assert summary["total"] == 200
    assert summary["popeye_version"] == "v4.test"
    assert summary["popeye_banner"] == "Popeye test v4.test"
    assert summary["verified_forced_mate"] == 200
    assert summary["unique_key_match"] == 200
    assert summary["by_mate_depth"]["1"]["total"] == 200
    assert (tmp_path / "verification" / "results.jsonl").exists()
    assert (tmp_path / "verification" / "summary.json").exists()


def test_retry_run_uses_existing_result_semantics_and_keeps_hashes(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    rows = [row(f"h{i}", str(i), 1) for i in range(200)]
    write_dataset(dataset_dir, rows)
    before = popeye.canonical_file_hashes(dataset_dir)
    parent = tmp_path / "parent.jsonl"
    write_results(
        parent,
        [
            parent_result(rows[0], "TIMEOUT"),
            parent_result(rows[1], "TIMEOUT"),
            parent_result(rows[2], "VERIFIED_UNIQUE_KEY_MATCH"),
        ],
    )

    monkeypatch.setattr(popeye, "popeye_identity", lambda executable: popeye.PopeyeIdentity("v4.test", "Popeye test v4.test"))
    outputs = iter(
        [
            popeye.ProcessResult("1.Qg7-f8 #\nsolution finished.", "", 0, 0.01),
            popeye.ProcessResult("", "", None, 1200.0, timed_out=True),
        ]
    )
    monkeypatch.setattr(popeye, "run_popeye_process", lambda executable, input_text, timeout: next(outputs))
    monkeypatch.setattr(popeye, "write_markdown_report", lambda *args, **kwargs: None)

    summary = popeye.run_verification(
        dataset_dir,
        tmp_path / "retry",
        "/fake/popeye",
        1200,
        retry_timeouts_from=parent,
    )
    after = popeye.canonical_file_hashes(dataset_dir)

    assert before == after
    assert summary["verification_pass"] == "timeout_retry"
    assert summary["parent_verification_config_fingerprint"] == "parent-fp"
    assert summary["retry_selected_count"] == 2
    assert summary["total"] == 2
    assert summary["verified_forced_mate"] == 1
    assert summary["timeouts"] == 1
    assert summary["unique_key_match"] == 1
    result_rows = [json.loads(line) for line in (tmp_path / "retry" / "results.jsonl").read_text().splitlines()]
    assert result_rows[1]["verification_reason"] == "TIMEOUT"


def test_dataset_fingerprint_required(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_dataset(dataset_dir, [row()], fingerprint="wrong")

    with pytest.raises(RuntimeError, match="Dataset fingerprint mismatch"):
        popeye.verify_dataset_fingerprint(dataset_dir, popeye.EXPECTED_DATASET_FINGERPRINT)

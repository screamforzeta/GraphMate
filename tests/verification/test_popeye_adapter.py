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

    def timeout(*args, **kwargs):
        raise popeye.subprocess.TimeoutExpired("cmd", 1, output="partial", stderr="late")

    monkeypatch.setattr(popeye.subprocess, "run", timeout)
    result = popeye.run_popeye_process("/bin/echo", "input", 1)
    assert result.timed_out is True
    assert result.stdout == "partial"
    assert result.stderr == "late"

    def missing(*args, **kwargs):
        raise OSError("missing executable")

    monkeypatch.setattr(popeye.subprocess, "run", missing)
    result = popeye.run_popeye_process("/missing", "input", 1)
    assert "missing executable" in result.stderr


def test_output_parser_one_key_multiple_no_solution_and_mismatch():
    parsed = popeye.parse_popeye_output("1.Qg7-f8 #\nsolution finished.", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_UNIQUE_KEY_MATCH"
    assert parsed.verified_keys_uci == ["g7f8"]

    parsed = popeye.parse_popeye_output("1.Qg7-f8 #\n1.Kg6-h6 #\nsolution finished.", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY"
    assert set(parsed.verified_keys_uci) == {"g7f8", "g6h6"}

    parsed = popeye.parse_popeye_output("1.Kg6-h6 #\nsolution finished.", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "VERIFIED_KEY_MISMATCH"

    parsed = popeye.parse_popeye_output("No solution", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "NO_SOLUTION"

    parsed = popeye.parse_popeye_output("header only", "", MATE1_FEN, "g7f8")
    assert parsed.verification_reason == "UNSUPPORTED_POPEYE_OUTPUT"


def test_config_fingerprint_and_resume_identity(tmp_path):
    config = popeye.PopeyeRunConfig("abc", "/bin/py", "4.89", 10)
    same = popeye.PopeyeRunConfig("abc", "/bin/py", "4.89", 10)
    other = popeye.PopeyeRunConfig("abc", "/bin/py", "4.89", 20)
    assert popeye.config_fingerprint(config) == popeye.config_fingerprint(same)
    assert popeye.config_fingerprint(config) != popeye.config_fingerprint(other)

    path = tmp_path / "results.jsonl"
    fp = popeye.config_fingerprint(config)
    path.write_text(json.dumps({"heldout_id": "h1", "verification_config_fingerprint": fp}) + "\n")
    assert "h1" in popeye.load_existing_results(path, fp)
    with pytest.raises(RuntimeError, match="incompatible"):
        popeye.load_existing_results(path, "different")


def test_orchestration_structured_results_summary_and_byte_integrity(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    rows = [row(f"h{i}", str(i), 1) for i in range(200)]
    write_dataset(dataset_dir, rows)
    before = popeye.canonical_file_hashes(dataset_dir)

    monkeypatch.setattr(popeye, "popeye_version", lambda executable: "Popeye 4.test")
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
    assert summary["verified_forced_mate"] == 200
    assert summary["unique_key_match"] == 200
    assert summary["by_mate_depth"]["1"]["total"] == 200
    assert (tmp_path / "verification" / "results.jsonl").exists()
    assert (tmp_path / "verification" / "summary.json").exists()


def test_dataset_fingerprint_required(tmp_path):
    dataset_dir = tmp_path / "dataset"
    write_dataset(dataset_dir, [row()], fingerprint="wrong")

    with pytest.raises(RuntimeError, match="Dataset fingerprint mismatch"):
        popeye.verify_dataset_fingerprint(dataset_dir, popeye.EXPECTED_DATASET_FINGERPRINT)

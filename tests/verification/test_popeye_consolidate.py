import json
from pathlib import Path

import pytest

from src.verification.classic_scoring import is_accepted_classic_key
from src.verification import popeye_consolidate as consolidate


FEN = "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"


def dataset_row(idx, depth=1, key="g7f8"):
    return {
        "heldout_id": f"h{idx}",
        "source_problem_id": f"p{idx}",
        "mate_depth": depth,
        "fen": FEN,
        "key_move_uci": key,
    }


def result_for(row, reason="VERIFIED_UNIQUE_KEY_MATCH", pass_fp="p1", timeout=300, keys=None):
    if keys is None:
        keys = [row["key_move_uci"]] if reason != "TIMEOUT" else []
    return {
        "heldout_id": row["heldout_id"],
        "source_problem_id": row["source_problem_id"],
        "mate_depth": row["mate_depth"],
        "canonical_fen": row["fen"],
        "source_key_move_uci": row["key_move_uci"],
        "dataset_fingerprint": consolidate.EXPECTED_DATASET_FINGERPRINT,
        "verification_config_fingerprint": pass_fp,
        "timeout_seconds": timeout,
        "verification_status": "FAILED" if reason == "TIMEOUT" else "VERIFIED",
        "verification_reason": reason,
        "forced_mate_verified": reason != "TIMEOUT",
        "verified_keys_uci": keys,
        "source_key_in_verified_keys": row["key_move_uci"] in keys,
        "runtime_seconds": 1.0,
    }


def write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_dataset(root: Path, rows):
    root.mkdir(parents=True)
    write_json(root / "manifest.json", {"dataset_fingerprint": consolidate.EXPECTED_DATASET_FINGERPRINT})
    write_jsonl(root / "dataset.jsonl", rows)
    (root / "dataset.csv").write_text("heldout_id\n" + "\n".join(row["heldout_id"] for row in rows) + "\n", encoding="utf-8")
    write_json(root / "selected_ids.json", {"ids": [row["heldout_id"] for row in rows]})


def make_fixture(tmp_path):
    dataset = [dataset_row(i, depth=(i % 10) + 1) for i in range(200)]
    dataset[1]["key_move_uci"] = "a1a2"
    dataset[2]["key_move_uci"] = "b1b2"
    dataset_dir = tmp_path / "dataset"
    write_dataset(dataset_dir, dataset)
    p1 = [result_for(row) for row in dataset]
    p1[1] = result_for(dataset[1], "TIMEOUT")
    p1[2] = result_for(dataset[2], "TIMEOUT")
    p2 = [
        result_for(dataset[1], "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY", pass_fp="p2", timeout=1200, keys=["a1a2", "c1c2"]),
        result_for(dataset[2], "TIMEOUT", pass_fp="p2", timeout=1200, keys=[]),
    ]
    pass1_results = tmp_path / "pass1.jsonl"
    pass2_results = tmp_path / "pass2.jsonl"
    pass1_summary = tmp_path / "pass1_summary.json"
    pass2_summary = tmp_path / "pass2_summary.json"
    write_jsonl(pass1_results, p1)
    write_jsonl(pass2_results, p2)
    write_json(pass1_summary, {"verification_config_fingerprint": "p1", "timeout_seconds": 300, "total": 200})
    write_json(
        pass2_summary,
        {
            "verification_config_fingerprint": "p2",
            "parent_verification_config_fingerprint": "p1",
            "timeout_seconds": 1200,
            "retry_selected_count": 2,
            "retry_selected_heldout_ids": ["h1", "h2"],
            "total": 2,
        },
    )
    return dataset_dir, pass1_results, pass1_summary, pass2_results, pass2_summary


def test_consolidation_precedence_accepted_keys_and_hashes(tmp_path):
    fixture = make_fixture(tmp_path)
    output = tmp_path / "out"

    summary = consolidate.consolidate(*fixture, output)
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    by_id = {row["heldout_id"]: row for row in rows}

    assert by_id["h0"]["resolved_by_pass"] == "pass1_300s"
    assert by_id["h1"]["resolved_by_pass"] == "pass2_retry_1200s"
    assert by_id["h1"]["accepted_key_basis"] == "POPEYE_VERIFIED_MULTIPLE"
    assert by_id["h1"]["accepted_key_moves_uci"] == ["a1a2", "c1c2"]
    assert by_id["h2"]["resolved_by_pass"] == "unresolved_timeout"
    assert by_id["h2"]["accepted_key_moves_uci"] == ["b1b2"]
    assert by_id["h2"]["accepted_key_basis"] == "YACPDB_SOURCE_UNVERIFIED_TIMEOUT"
    assert by_id["h2"]["forced_mate_verified"] is False
    assert summary["total_problems"] == 200
    assert summary["verified_total"] == 199
    assert summary["verified_multiple_keys_including_source"] == 1
    assert summary["unresolved_timeout"] == 1
    assert summary["freeze_readiness"] == "READY_WITH_DOCUMENTED_TIMEOUTS"
    assert (output / "accepted_keys.jsonl").exists()
    assert (output / "multi_key_cases.jsonl").exists()
    assert (output / "residual_timeout_cases.jsonl").exists()


def test_pass2_cannot_replace_completed_pass1(tmp_path):
    dataset_dir, p1, p1s, p2, p2s = make_fixture(tmp_path)
    pass2_rows = [json.loads(line) for line in p2.read_text().splitlines()]
    pass2_rows.append(result_for(dataset_row(0), "VERIFIED_UNIQUE_KEY_MATCH", pass_fp="p2", timeout=1200))
    write_jsonl(p2, pass2_rows)

    with pytest.raises(RuntimeError, match="not a Pass-1 timeout"):
        consolidate.consolidate(dataset_dir, p1, p1s, p2, p2s, tmp_path / "out")


def test_parent_fingerprint_dataset_fingerprint_duplicate_and_missing_fail_closed(tmp_path):
    dataset_dir, p1, p1s, p2, p2s = make_fixture(tmp_path)
    write_json(p2s, {"verification_config_fingerprint": "p2", "parent_verification_config_fingerprint": "wrong", "retry_selected_count": 2})
    with pytest.raises(RuntimeError, match="parent fingerprint"):
        consolidate.consolidate(dataset_dir, p1, p1s, p2, p2s, tmp_path / "out1")

    dataset_dir, p1, p1s, p2, p2s = make_fixture(tmp_path / "case2")
    rows = [json.loads(line) for line in p1.read_text().splitlines()]
    rows[0]["dataset_fingerprint"] = "wrong"
    write_jsonl(p1, rows)
    with pytest.raises(RuntimeError, match="dataset fingerprint"):
        consolidate.consolidate(dataset_dir, p1, p1s, p2, p2s, tmp_path / "out2")

    dataset_dir, p1, p1s, p2, p2s = make_fixture(tmp_path / "case3")
    rows = [json.loads(line) for line in p1.read_text().splitlines()]
    rows.append(rows[0])
    write_jsonl(p1, rows)
    with pytest.raises(RuntimeError, match="Duplicate"):
        consolidate.consolidate(dataset_dir, p1, p1s, p2, p2s, tmp_path / "out3")


def test_verified_key_must_contain_source_key_and_order_is_deterministic(tmp_path):
    dataset_dir, p1, p1s, p2, p2s = make_fixture(tmp_path)
    p2_rows = [json.loads(line) for line in p2.read_text().splitlines()]
    p2_rows[0]["verified_keys_uci"] = ["c1c2", "d1d2"]
    p2_rows[0]["source_key_in_verified_keys"] = False
    write_jsonl(p2, p2_rows)
    with pytest.raises(RuntimeError, match="does not contain source key"):
        consolidate.consolidate(dataset_dir, p1, p1s, p2, p2s, tmp_path / "out")


def test_classic_scoring_helper():
    assert is_accepted_classic_key("a1a2", ["a1a2", "c1c2"])
    assert is_accepted_classic_key("c1c2", ["a1a2", "c1c2"])
    assert not is_accepted_classic_key("h1h2", ["a1a2", "c1c2"])

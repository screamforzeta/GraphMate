import json
from pathlib import Path

import pytest

from src.verification import freeze_classic_benchmark as freeze
from src.verification.popeye_consolidate import ACCEPTED_KEY_POLICY_VERSION, CONSOLIDATION_VERSION, EXPECTED_DATASET_FINGERPRINT


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def make_dataset(root: Path, mutate_depth=False):
    rows = []
    for idx in range(200):
        depth = (idx % 10) + 1
        rows.append(
            {
                "heldout_id": f"h{idx}",
                "source_problem_id": f"p{idx}",
                "mate_depth": 1 if mutate_depth else depth,
                "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
                "key_move_uci": "a1a2",
            }
        )
    write_json(root / "manifest.json", {"dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT})
    write_jsonl(root / "dataset.jsonl", rows)
    (root / "dataset.csv").write_text("heldout_id\n" + "\n".join(row["heldout_id"] for row in rows) + "\n", encoding="utf-8")
    write_json(root / "selected_ids.json", {"ids": [row["heldout_id"] for row in rows]})


def make_consolidated(root: Path, **overrides):
    summary = {
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "total_problems": 200,
        "verified_total": 193,
        "verified_unique_key": 188,
        "verified_multiple_keys_including_source": 5,
        "unresolved_timeout": 7,
        "key_mismatch": 0,
        "popeye_errors": 0,
        "unverifiable": 0,
        "accepted_key_policy_version": ACCEPTED_KEY_POLICY_VERSION,
        "consolidation_version": CONSOLIDATION_VERSION,
        "freeze_readiness": "READY_WITH_DOCUMENTED_TIMEOUTS",
        "pass1": {"verification_config_fingerprint": "p1", "timeout_seconds": 300},
        "pass2": {"verification_config_fingerprint": "p2", "timeout_seconds": 1200},
    }
    summary.update(overrides)
    write_json(root / "summary.json", summary)
    for name in ("results.jsonl", "accepted_keys.jsonl", "multi_key_cases.jsonl", "residual_timeout_cases.jsonl"):
        write_jsonl(root / name, [{"heldout_id": "h0", "accepted_key_moves_uci": ["a1a2"]}])


def test_freeze_manifest_and_verify_mode(tmp_path):
    dataset = tmp_path / "dataset"
    consolidated = tmp_path / "consolidated"
    manifest_path = dataset / "freeze_manifest.json"
    make_dataset(dataset)
    make_consolidated(consolidated)

    manifest = freeze.write_freeze_manifest(dataset, consolidated, manifest_path)
    verified = freeze.verify_freeze_manifest(dataset, consolidated, manifest_path)

    assert manifest["lifecycle"] == "FROZEN"
    assert manifest["freeze_fingerprint"] == verified["freeze_fingerprint"]
    assert manifest["verification"]["unresolved_timeout"] == 7


def test_freeze_rejects_canonical_mutation_and_distribution(tmp_path):
    dataset = tmp_path / "dataset"
    consolidated = tmp_path / "consolidated"
    make_dataset(dataset)
    make_consolidated(consolidated)
    manifest_path = dataset / "freeze_manifest.json"
    freeze.write_freeze_manifest(dataset, consolidated, manifest_path)
    (dataset / "dataset.csv").write_text("changed", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Freeze fingerprint|Canonical hash"):
        freeze.verify_freeze_manifest(dataset, consolidated, manifest_path)

    bad_dataset = tmp_path / "bad_dataset"
    make_dataset(bad_dataset, mutate_depth=True)
    with pytest.raises(RuntimeError, match="20 problems per MateDepth"):
        freeze.build_freeze_manifest(bad_dataset, consolidated)


def test_freeze_rejects_consolidated_mutations_and_bad_versions(tmp_path):
    dataset = tmp_path / "dataset"
    consolidated = tmp_path / "consolidated"
    make_dataset(dataset)
    make_consolidated(consolidated)
    manifest_path = dataset / "freeze_manifest.json"
    freeze.write_freeze_manifest(dataset, consolidated, manifest_path)
    (consolidated / "accepted_keys.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Freeze fingerprint|Consolidated artifact"):
        freeze.verify_freeze_manifest(dataset, consolidated, manifest_path)

    make_consolidated(consolidated, accepted_key_policy_version="wrong")
    with pytest.raises(RuntimeError, match="accepted-key policy"):
        freeze.build_freeze_manifest(dataset, consolidated)

    make_consolidated(consolidated, consolidation_version="wrong")
    with pytest.raises(RuntimeError, match="consolidation version"):
        freeze.build_freeze_manifest(dataset, consolidated)


def test_freeze_rejects_bad_counts_and_failure_states(tmp_path):
    dataset = tmp_path / "dataset"
    consolidated = tmp_path / "consolidated"
    make_dataset(dataset)
    make_consolidated(consolidated, total_problems=199)
    with pytest.raises(RuntimeError, match="total_problems"):
        freeze.build_freeze_manifest(dataset, consolidated)

    make_consolidated(consolidated, key_mismatch=1)
    with pytest.raises(RuntimeError, match="key mismatches"):
        freeze.build_freeze_manifest(dataset, consolidated)

    make_consolidated(consolidated, popeye_errors=1)
    with pytest.raises(RuntimeError, match="Popeye errors"):
        freeze.build_freeze_manifest(dataset, consolidated)

    make_consolidated(consolidated, unresolved_timeout=7)
    manifest = freeze.build_freeze_manifest(dataset, consolidated)
    assert manifest["verification"]["unresolved_timeout"] == 7


def test_freeze_fingerprint_deterministic_and_created_at_excluded(tmp_path):
    dataset = tmp_path / "dataset"
    consolidated = tmp_path / "consolidated"
    make_dataset(dataset)
    make_consolidated(consolidated)
    first = freeze.build_freeze_manifest(dataset, consolidated)
    second = dict(first)
    second["created_at"] = "2099-01-01T00:00:00+00:00"

    assert freeze.deterministic_hash(freeze.freeze_identity_payload(first)) == freeze.deterministic_hash(freeze.freeze_identity_payload(second))


def test_incompatible_refreeze_fails_closed(tmp_path):
    dataset = tmp_path / "dataset"
    consolidated = tmp_path / "consolidated"
    make_dataset(dataset)
    make_consolidated(consolidated)
    manifest_path = dataset / "freeze_manifest.json"
    freeze.write_freeze_manifest(dataset, consolidated, manifest_path)
    existing = json.loads(manifest_path.read_text())
    existing["freeze_fingerprint"] = "different"
    write_json(manifest_path, existing)

    with pytest.raises(RuntimeError, match="incompatible"):
        freeze.write_freeze_manifest(dataset, consolidated, manifest_path)


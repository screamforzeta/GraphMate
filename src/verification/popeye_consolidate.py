"""Consolidate two Popeye verification passes into a derived benchmark view."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.verification.classic_scoring import is_accepted_classic_key


CONSOLIDATION_VERSION = "yacpdb_popeye_consolidation_v1"
ACCEPTED_KEY_POLICY_VERSION = "accepted_classic_keys_v1"
EXPECTED_DATASET_FINGERPRINT = "bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a"
CANONICAL_FILES = ("dataset.jsonl", "dataset.csv", "selected_ids.json")


def sha256_file(path: Path) -> str:
    """Return SHA256 for a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hashes(dataset_dir: Path) -> dict[str, str]:
    """Return hashes for immutable canonical dataset files."""

    return {name: sha256_file(dataset_dir / name) for name in CANONICAL_FILES}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""

    if not path.exists():
        raise RuntimeError(f"Missing JSONL file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic JSONL rows."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""

    if not path.exists():
        raise RuntimeError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load canonical dataset rows and manifest."""

    manifest = read_json(dataset_dir / "manifest.json")
    if manifest.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
        raise RuntimeError("Canonical dataset fingerprint mismatch")
    rows = read_jsonl(dataset_dir / "dataset.jsonl")
    ids = [row["heldout_id"] for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200:
        raise RuntimeError("Canonical dataset must contain exactly 200 unique heldout IDs")
    counts = Counter(int(row["mate_depth"]) for row in rows)
    if any(counts[depth] != 20 for depth in range(1, 11)):
        raise RuntimeError("Canonical dataset must contain exactly 20 rows per MateDepth")
    return rows, manifest


def unique_by_heldout(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    """Index rows by heldout_id and fail on duplicates."""

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        heldout_id = str(row.get("heldout_id") or "")
        if not heldout_id:
            raise RuntimeError(f"{label} row missing heldout_id")
        if heldout_id in indexed:
            raise RuntimeError(f"Duplicate heldout_id in {label}: {heldout_id}")
        indexed[heldout_id] = row
    return indexed


def validate_passes(
    dataset_rows: list[dict[str, Any]],
    pass1_rows: list[dict[str, Any]],
    pass1_summary: dict[str, Any],
    pass2_rows: list[dict[str, Any]],
    pass2_summary: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Validate consolidation invariants and return indexed rows."""

    dataset = unique_by_heldout(dataset_rows, "dataset")
    pass1 = unique_by_heldout(pass1_rows, "pass1")
    pass2 = unique_by_heldout(pass2_rows, "pass2")
    if set(pass1) != set(dataset):
        raise RuntimeError("Pass 1 must contain exactly one result for every canonical problem")
    for row in list(pass1.values()) + list(pass2.values()):
        if row.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
            raise RuntimeError("Verification result dataset fingerprint mismatch")
    timeout_ids = {heldout_id for heldout_id, row in pass1.items() if row.get("verification_reason") == "TIMEOUT"}
    if not set(pass2).issubset(timeout_ids):
        raise RuntimeError("Pass 2 contains a result that was not a Pass-1 timeout")
    if set(pass2) != timeout_ids:
        raise RuntimeError("Pass 2 must contain exactly the Pass-1 timeout retry set")
    pass1_fp = str(pass1_summary.get("verification_config_fingerprint") or "")
    pass2_parent = str(pass2_summary.get("parent_verification_config_fingerprint") or "")
    if not pass1_fp or pass2_parent != pass1_fp:
        raise RuntimeError("Pass-2 parent fingerprint does not match Pass 1")
    declared = pass2_summary.get("retry_selected_heldout_ids")
    if declared is not None and set(declared) != set(pass2):
        raise RuntimeError("Pass-2 retry_selected_heldout_ids does not match Pass-2 rows")
    if pass2_summary.get("retry_selected_count") is not None and int(pass2_summary["retry_selected_count"]) != len(pass2):
        raise RuntimeError("Pass-2 retry_selected_count does not match Pass-2 rows")
    return dataset, pass1, pass2


def accepted_keys_for(row: dict[str, Any], source_key: str) -> tuple[list[str], str]:
    """Return accepted keys and basis for a final verification row."""

    reason = row.get("verification_reason")
    keys = sorted(set(str(key) for key in row.get("verified_keys_uci", []) if key))
    if reason == "VERIFIED_UNIQUE_KEY_MATCH":
        if keys != [source_key]:
            raise RuntimeError("Verified unique-key result does not match source key")
        return keys, "POPEYE_VERIFIED_UNIQUE"
    if reason == "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY":
        if source_key not in keys:
            raise RuntimeError("Verified multiple-key result does not contain source key")
        return keys, "POPEYE_VERIFIED_MULTIPLE"
    if reason == "TIMEOUT":
        return [source_key], "YACPDB_SOURCE_UNVERIFIED_TIMEOUT"
    raise RuntimeError(f"Unsupported final verification reason for accepted keys: {reason}")


def final_record(
    dataset_row: dict[str, Any],
    pass1_row: dict[str, Any],
    pass2_row: dict[str, Any] | None,
    pass1_fp: str,
    pass2_fp: str | None,
) -> dict[str, Any]:
    """Build one derived final verification row."""

    if pass1_row.get("verification_reason") == "TIMEOUT":
        final = pass2_row
        resolved_by = "pass2_retry_1200s" if final and final.get("verification_reason") != "TIMEOUT" else "unresolved_timeout"
    else:
        final = pass1_row
        resolved_by = "pass1_300s"
    if final is None:
        raise RuntimeError("Missing final verification row")
    accepted, basis = accepted_keys_for(final, dataset_row["key_move_uci"])
    final_reason = final.get("verification_reason")
    final_status = final.get("verification_status")
    forced = bool(final.get("forced_mate_verified"))
    if final_reason == "TIMEOUT":
        final_status = "VERIFICATION_INCONCLUSIVE_TIMEOUT"
        forced = False
    return {
        "heldout_id": dataset_row["heldout_id"],
        "source_problem_id": dataset_row["source_problem_id"],
        "mate_depth": dataset_row["mate_depth"],
        "canonical_fen": dataset_row["fen"],
        "source_key_move_uci": dataset_row["key_move_uci"],
        "final_verification_status": final_status,
        "final_verification_reason": final_reason,
        "forced_mate_verified": forced,
        "verified_keys_uci": final.get("verified_keys_uci", []),
        "source_key_in_verified_keys": bool(final.get("source_key_in_verified_keys")),
        "accepted_key_moves_uci": accepted,
        "accepted_key_basis": basis,
        "resolved_by_pass": resolved_by,
        "pass1_verification_config_fingerprint": pass1_fp,
        "pass2_verification_config_fingerprint": pass2_fp if pass1_row.get("verification_reason") == "TIMEOUT" else None,
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "pass1_runtime_seconds": pass1_row.get("runtime_seconds"),
        "pass2_runtime_seconds": pass2_row.get("runtime_seconds") if pass2_row else None,
    }


def summarize(rows: list[dict[str, Any]], pass1_summary: dict[str, Any], pass2_summary: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    """Return deterministic consolidated summary."""

    by_depth: dict[str, dict[str, int]] = {}
    for depth in range(1, 11):
        depth_rows = [row for row in rows if int(row["mate_depth"]) == depth]
        by_depth[str(depth)] = {
            "total": len(depth_rows),
            "verified": sum(bool(row["forced_mate_verified"]) for row in depth_rows),
            "verified_unique": sum(row["accepted_key_basis"] == "POPEYE_VERIFIED_UNIQUE" for row in depth_rows),
            "verified_multiple": sum(row["accepted_key_basis"] == "POPEYE_VERIFIED_MULTIPLE" for row in depth_rows),
            "unresolved_timeout": sum(row["accepted_key_basis"] == "YACPDB_SOURCE_UNVERIFIED_TIMEOUT" for row in depth_rows),
        }
    key_mismatch = sum(row["final_verification_reason"] == "VERIFIED_KEY_MISMATCH" for row in rows)
    popeye_errors = sum(row["final_verification_reason"] == "POPEYE_ERROR" for row in rows)
    unverifiable = sum(row["final_verification_status"] == "UNVERIFIABLE" for row in rows)
    return {
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "canonical_hashes": hashes,
        "total_problems": len(rows),
        "verified_total": sum(bool(row["forced_mate_verified"]) for row in rows),
        "verified_unique_key": sum(row["accepted_key_basis"] == "POPEYE_VERIFIED_UNIQUE" for row in rows),
        "verified_multiple_keys_including_source": sum(row["accepted_key_basis"] == "POPEYE_VERIFIED_MULTIPLE" for row in rows),
        "unresolved_timeout": sum(row["accepted_key_basis"] == "YACPDB_SOURCE_UNVERIFIED_TIMEOUT" for row in rows),
        "key_mismatch": key_mismatch,
        "popeye_errors": popeye_errors,
        "unverifiable": unverifiable,
        "by_mate_depth": by_depth,
        "pass1": {
            "timeout_seconds": pass1_summary.get("timeout_seconds"),
            "verification_config_fingerprint": pass1_summary.get("verification_config_fingerprint"),
            "counts": {key: pass1_summary.get(key) for key in ("total", "verified_forced_mate", "failed", "timeouts", "unique_key_match", "multiple_keys_including_source")},
        },
        "pass2": {
            "timeout_seconds": pass2_summary.get("timeout_seconds"),
            "verification_config_fingerprint": pass2_summary.get("verification_config_fingerprint"),
            "parent_verification_config_fingerprint": pass2_summary.get("parent_verification_config_fingerprint"),
            "counts": {key: pass2_summary.get(key) for key in ("total", "verified_forced_mate", "failed", "timeouts", "unique_key_match", "multiple_keys_including_source")},
        },
        "accepted_key_policy_version": ACCEPTED_KEY_POLICY_VERSION,
        "consolidation_version": CONSOLIDATION_VERSION,
        "canonical_dataset_mutated": False,
        "freeze_readiness": "READY_WITH_DOCUMENTED_TIMEOUTS" if key_mismatch == 0 and popeye_errors == 0 and unverifiable == 0 else "NOT_READY",
    }


def write_reports(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Write explicit multi-key and residual-timeout reports."""

    multi = [row for row in rows if row["accepted_key_basis"] == "POPEYE_VERIFIED_MULTIPLE"]
    timeouts = [row for row in rows if row["accepted_key_basis"] == "YACPDB_SOURCE_UNVERIFIED_TIMEOUT"]
    write_jsonl(output_dir / "multi_key_cases.jsonl", multi)
    write_jsonl(output_dir / "residual_timeout_cases.jsonl", timeouts)


def consolidate(
    dataset_dir: Path,
    pass1_results: Path,
    pass1_summary: Path,
    pass2_results: Path,
    pass2_summary: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the derived consolidated verification artifacts."""

    before = canonical_hashes(dataset_dir)
    dataset_rows, _manifest = load_dataset(dataset_dir)
    p1_summary = read_json(pass1_summary)
    p2_summary = read_json(pass2_summary)
    dataset, p1, p2 = validate_passes(dataset_rows, read_jsonl(pass1_results), p1_summary, read_jsonl(pass2_results), p2_summary)
    p1_fp = str(p1_summary["verification_config_fingerprint"])
    p2_fp = str(p2_summary["verification_config_fingerprint"])
    rows = [
        final_record(dataset[heldout_id], p1[heldout_id], p2.get(heldout_id), p1_fp, p2_fp)
        for heldout_id in sorted(dataset)
    ]
    after = canonical_hashes(dataset_dir)
    if before != after:
        raise RuntimeError("Canonical dataset mutated during consolidation")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "results.jsonl", rows)
    write_jsonl(output_dir / "accepted_keys.jsonl", [
        {
            "heldout_id": row["heldout_id"],
            "source_problem_id": row["source_problem_id"],
            "mate_depth": row["mate_depth"],
            "source_key_move_uci": row["source_key_move_uci"],
            "accepted_key_moves_uci": row["accepted_key_moves_uci"],
            "accepted_key_basis": row["accepted_key_basis"],
        }
        for row in rows
    ])
    write_reports(rows, output_dir)
    summary = summarize(rows, p1_summary, p2_summary, before)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/heldout_classic/final/yacpdb_classic_v1")
    parser.add_argument("--pass1-results", required=True)
    parser.add_argument("--pass1-summary", required=True)
    parser.add_argument("--pass2-results", required=True)
    parser.add_argument("--pass2-summary", required=True)
    parser.add_argument("--output-dir", default="data/heldout_classic/verification/yacpdb_classic_v1/consolidated")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    summary = consolidate(
        Path(args.dataset_dir),
        Path(args.pass1_results),
        Path(args.pass1_summary),
        Path(args.pass2_results),
        Path(args.pass2_summary),
        Path(args.output_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

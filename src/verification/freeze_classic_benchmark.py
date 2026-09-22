"""Freeze and verify the YACPDB classic held-out benchmark identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.verification.popeye_consolidate import (
    ACCEPTED_KEY_POLICY_VERSION,
    CONSOLIDATION_VERSION,
    EXPECTED_DATASET_FINGERPRINT,
    canonical_hashes,
    read_json,
    read_jsonl,
)


FREEZE_SCHEMA_VERSION = "yacpdb_classic_freeze_manifest_v1"
DATASET_NAME = "yacpdb_classic_heldout"
DATASET_VERSION = "yacpdb_classic_v1"
BENCHMARK_FILES = (
    "results.jsonl",
    "accepted_keys.jsonl",
    "multi_key_cases.jsonl",
    "residual_timeout_cases.jsonl",
    "summary.json",
)


def sha256_file(path: Path) -> str:
    """Return SHA256 for a file."""

    if not path.exists():
        raise RuntimeError(f"Missing benchmark-defining artifact: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_hash(payload: dict[str, Any]) -> str:
    """Return SHA256 over stable JSON content."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dataset_stats(dataset_dir: Path) -> dict[str, Any]:
    """Validate canonical dataset shape and return statistics."""

    rows = read_jsonl(dataset_dir / "dataset.jsonl")
    ids = [row["heldout_id"] for row in rows]
    if len(rows) != 200 or len(set(ids)) != 200:
        raise RuntimeError("Canonical dataset must contain exactly 200 unique problems")
    distribution = {str(depth): 0 for depth in range(1, 11)}
    for row in rows:
        depth = str(int(row["mate_depth"]))
        if depth not in distribution:
            raise RuntimeError("Canonical dataset contains MateDepth outside #1..#10")
        distribution[depth] += 1
    if any(count != 20 for count in distribution.values()):
        raise RuntimeError("Canonical dataset must contain exactly 20 problems per MateDepth")
    return {"total": len(rows), "mate_depth_distribution": distribution}


def validate_consolidated_summary(summary: dict[str, Any]) -> None:
    """Fail closed if consolidated verification cannot support freeze."""

    if summary.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
        raise RuntimeError("Consolidated summary dataset fingerprint mismatch")
    if summary.get("accepted_key_policy_version") != ACCEPTED_KEY_POLICY_VERSION:
        raise RuntimeError("Wrong accepted-key policy version")
    if summary.get("consolidation_version") != CONSOLIDATION_VERSION:
        raise RuntimeError("Wrong consolidation version")
    if int(summary.get("total_problems", -1)) != 200:
        raise RuntimeError("Consolidated summary total_problems must be 200")
    if int(summary.get("key_mismatch", -1)) != 0:
        raise RuntimeError("Cannot freeze with key mismatches")
    if int(summary.get("popeye_errors", -1)) != 0:
        raise RuntimeError("Cannot freeze with Popeye errors")
    if int(summary.get("unverifiable", -1)) != 0:
        raise RuntimeError("Cannot freeze with unverifiable cases")
    if summary.get("freeze_readiness") != "READY_WITH_DOCUMENTED_TIMEOUTS":
        raise RuntimeError("Consolidated summary is not freeze-ready")


def benchmark_artifact_hashes(consolidated_dir: Path) -> dict[str, str]:
    """Return hashes for all benchmark-defining consolidated artifacts."""

    return {name: sha256_file(consolidated_dir / name) for name in BENCHMARK_FILES}


def freeze_identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return timestamp-free freeze identity payload."""

    return {
        "freeze_schema_version": manifest["freeze_schema_version"],
        "dataset_name": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "lifecycle": manifest["lifecycle"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "canonical_hashes": manifest["canonical_hashes"],
        "dataset_statistics": manifest["dataset_statistics"],
        "selection": manifest["selection"],
        "verification": manifest["verification"],
        "consolidated_artifact_hashes": manifest["consolidated_artifact_hashes"],
        "post_freeze_rules": manifest["post_freeze_rules"],
    }


def build_freeze_manifest(dataset_dir: Path, consolidated_dir: Path) -> dict[str, Any]:
    """Build a freeze manifest from canonical and consolidated artifacts."""

    candidate_manifest = read_json(dataset_dir / "manifest.json")
    if candidate_manifest.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
        raise RuntimeError("Candidate manifest dataset fingerprint mismatch")
    stats = dataset_stats(dataset_dir)
    summary = read_json(consolidated_dir / "summary.json")
    validate_consolidated_summary(summary)
    canonical = canonical_hashes(dataset_dir)
    artifact_hashes = benchmark_artifact_hashes(consolidated_dir)
    manifest = {
        "freeze_schema_version": FREEZE_SCHEMA_VERSION,
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "lifecycle": "FROZEN",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": EXPECTED_DATASET_FINGERPRINT,
        "canonical_hashes": canonical,
        "dataset_statistics": {"source": "YACPDB", **stats},
        "selection": {
            "seed": 42,
            "per_depth": 20,
            "mate_depths": list(range(1, 11)),
            "selection_performed_before_heldout_model_evaluation": True,
        },
        "verification": {
            "popeye_version": "v4.103",
            "pass1_fingerprint": summary["pass1"]["verification_config_fingerprint"],
            "pass1_timeout_seconds": summary["pass1"]["timeout_seconds"],
            "pass2_fingerprint": summary["pass2"]["verification_config_fingerprint"],
            "pass2_timeout_seconds": summary["pass2"]["timeout_seconds"],
            "consolidation_version": summary["consolidation_version"],
            "accepted_key_policy_version": summary["accepted_key_policy_version"],
            "verified_total": summary["verified_total"],
            "verified_unique": summary["verified_unique_key"],
            "verified_multiple": summary["verified_multiple_keys_including_source"],
            "unresolved_timeout": summary["unresolved_timeout"],
            "key_mismatch": summary["key_mismatch"],
            "popeye_errors": summary["popeye_errors"],
        },
        "consolidated_artifact_hashes": artifact_hashes,
        "freeze_fingerprint_procedure": "SHA256 over stable JSON containing canonical hashes, dataset fingerprint/statistics, selection metadata, verification provenance/totals, consolidated artifact hashes, schema/version identifiers, and post-freeze rules. created_at is excluded.",
        "post_freeze_rules": {
            "selected_problem_ids_immutable": True,
            "canonical_fens_immutable": True,
            "mate_depths_immutable": True,
            "source_keys_immutable": True,
            "accepted_key_sets_immutable": True,
            "verification_classifications_immutable": True,
            "accepted_key_policy_version_immutable": True,
            "future_corrections_require_new_version": "yacpdb_classic_v2",
            "model_outputs_excluded_from_benchmark_identity": True,
        },
    }
    manifest["freeze_fingerprint"] = deterministic_hash(freeze_identity_payload(manifest))
    return manifest


def write_freeze_manifest(dataset_dir: Path, consolidated_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Create or verify an immutable freeze manifest."""

    manifest = build_freeze_manifest(dataset_dir, consolidated_dir)
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if existing.get("freeze_fingerprint") != manifest["freeze_fingerprint"]:
            raise RuntimeError("Existing freeze manifest is incompatible with current benchmark state")
        return existing
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return manifest


def verify_freeze_manifest(dataset_dir: Path, consolidated_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify an existing freeze manifest without modifying artifacts."""

    if not manifest_path.exists():
        raise RuntimeError(f"Missing freeze manifest: {manifest_path}")
    expected = build_freeze_manifest(dataset_dir, consolidated_dir)
    actual = read_json(manifest_path)
    if actual.get("freeze_fingerprint") != expected["freeze_fingerprint"]:
        raise RuntimeError("Freeze fingerprint mismatch")
    if actual.get("canonical_hashes") != expected["canonical_hashes"]:
        raise RuntimeError("Canonical hash mismatch")
    if actual.get("consolidated_artifact_hashes") != expected["consolidated_artifact_hashes"]:
        raise RuntimeError("Consolidated artifact hash mismatch")
    return actual


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/heldout_classic/final/yacpdb_classic_v1")
    parser.add_argument("--consolidated-dir", default="data/heldout_classic/verification/yacpdb_classic_v1/consolidated")
    parser.add_argument("--manifest-path", default="data/heldout_classic/final/yacpdb_classic_v1/freeze_manifest.json")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    consolidated_dir = Path(args.consolidated_dir)
    manifest_path = Path(args.manifest_path)
    manifest = verify_freeze_manifest(dataset_dir, consolidated_dir, manifest_path) if args.verify else write_freeze_manifest(dataset_dir, consolidated_dir, manifest_path)
    print(json.dumps({"lifecycle": manifest["lifecycle"], "freeze_fingerprint": manifest["freeze_fingerprint"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

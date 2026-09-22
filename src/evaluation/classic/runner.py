"""Run identity, persistence, and guard rails for classic evaluation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.classic.core import (
    CLASSIC_EVALUATION_PROTOCOL_VERSION,
    EXPECTED_FREEZE_FINGERPRINT,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/classic_benchmark/yacpdb_classic_v1")
ALLOWED_INVALID_INFRASTRUCTURE_REASONS = {"HTTP_404_RUNTIME_FAILURE"}


def stable_hash(payload: dict[str, Any]) -> str:
    """Return a stable SHA256 digest for a JSON payload."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_run_identity(
    model_metadata: dict[str, Any],
    *,
    freeze_fingerprint: str = EXPECTED_FREEZE_FINGERPRINT,
    protocol_version: str = CLASSIC_EVALUATION_PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Build the identity that defines one official classic run."""

    payload = {
        "benchmark_version": "yacpdb_classic_v1",
        "freeze_fingerprint": freeze_fingerprint,
        "model_id": model_metadata["model_id"],
        "model_family": model_metadata["model_family"],
        "model_version": model_metadata.get("model_version", "unknown"),
        "evaluation_protocol_version": protocol_version,
        "accepted_key_policy_version": "accepted_classic_keys_v1",
        "checkpoint_sha256": model_metadata.get("checkpoint_sha256")
        or model_metadata.get("a4_checkpoint_sha256")
        or model_metadata.get("model_digest"),
        "timing_protocol": model_metadata.get("timing_protocol"),
        "prompt_version": model_metadata.get("prompt_version"),
        "parser_version": model_metadata.get("parser_version"),
    }
    payload["run_id"] = f"{payload['model_id']}_{stable_hash(payload)[:16]}"
    return payload


def invalid_marker(path: Path) -> Path:
    """Return the sidecar invalid-run marker path for an attempt directory."""

    return Path(path) / "INVALID_INFRASTRUCTURE_RUN.json"


def read_invalid_marker(path: Path) -> dict[str, Any] | None:
    """Read and validate an invalid infrastructure marker if present."""

    marker = invalid_marker(path)
    if not marker.exists():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "INVALID_INFRASTRUCTURE_RUN":
        raise RuntimeError(f"Invalid run marker has wrong status: {marker}")
    if payload.get("reason") not in ALLOWED_INVALID_INFRASTRUCTURE_REASONS:
        raise RuntimeError(f"Invalid run marker reason is not retryable: {marker}")
    return payload


def attempt_dir_for(base_dir: Path, attempt: int) -> Path:
    """Return the directory name for one execution attempt."""

    if attempt == 1:
        return base_dir
    return base_dir.with_name(f"{base_dir.name}_attempt{attempt}")


def select_attempt_directory(base_dir: Path, resume: bool) -> tuple[Path, dict[str, Any]]:
    """Select a guarded execution-attempt directory for a scientific run ID."""

    if not base_dir.exists():
        return base_dir, {
            "execution_attempt": 1,
            "previous_attempt_status": None,
            "previous_attempt_reason": None,
            "scientific_run_id": base_dir.name,
        }

    attempt = 1
    last_dir = base_dir
    invalid_seen = False
    while attempt_dir_for(base_dir, attempt).exists():
        last_dir = attempt_dir_for(base_dir, attempt)
        completed = last_dir / "COMPLETED"
        invalid = read_invalid_marker(last_dir)
        if completed.exists() and invalid is None:
            if resume:
                return last_dir, {
                    "execution_attempt": attempt,
                    "previous_attempt_status": None,
                    "previous_attempt_reason": None,
                    "scientific_run_id": base_dir.name,
                }
            raise RuntimeError(f"Official classic run already completed: {last_dir}")
        if invalid is not None:
            invalid_seen = True
            attempt += 1
            continue
        if resume:
            return last_dir, {
                "execution_attempt": attempt,
                "previous_attempt_status": None,
                "previous_attempt_reason": None,
                "scientific_run_id": base_dir.name,
            }
        raise RuntimeError(
            "Existing official attempt is not completed and not explicitly invalid; "
            "use --resume or add an explicit invalid infrastructure marker."
        )

    if resume and invalid_seen:
        raise RuntimeError(f"Cannot resume invalid infrastructure run: {base_dir}")
    previous_invalid = read_invalid_marker(last_dir)
    return attempt_dir_for(base_dir, attempt), {
        "execution_attempt": attempt,
        "previous_attempt_status": previous_invalid.get("status") if previous_invalid else None,
        "previous_attempt_reason": previous_invalid.get("reason") if previous_invalid else None,
        "scientific_run_id": base_dir.name,
    }


def prepare_official_run_directory(
    model_metadata: dict[str, Any],
    *,
    official: bool,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    resume: bool = False,
    freeze_fingerprint: str = EXPECTED_FREEZE_FINGERPRINT,
) -> Path:
    """Create a guarded output directory for an official run."""

    if not official:
        raise RuntimeError("Classic held-out evaluation requires explicit --official.")
    identity = build_run_identity(model_metadata, freeze_fingerprint=freeze_fingerprint)
    scientific_run_id = identity["run_id"]
    base_dir = Path(output_root) / str(model_metadata["model_id"]) / scientific_run_id
    run_dir, attempt = select_attempt_directory(base_dir, resume=resume)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_identity": identity,
        "scientific_run_id": scientific_run_id,
        "execution_attempt": attempt["execution_attempt"],
        "execution_attempt_id": run_dir.name,
        "previous_attempt_status": attempt["previous_attempt_status"],
        "previous_attempt_reason": attempt["previous_attempt_reason"],
        "model_metadata": model_metadata,
        "official": True,
        "resume": bool(resume),
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing.get("run_identity") != identity:
            raise RuntimeError("Existing run config identity mismatch")
        if existing.get("scientific_run_id", identity["run_id"]) != scientific_run_id:
            raise RuntimeError("Existing scientific run ID mismatch")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def write_prediction_records(path: Path, records) -> None:
    """Persist prediction records as deterministic JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record.heldout_id in seen:
                raise RuntimeError(f"Duplicate prediction for heldout_id: {record.heldout_id}")
            seen.add(record.heldout_id)
            handle.write(json.dumps(record.to_json(), sort_keys=True, ensure_ascii=False) + "\n")


def append_prediction_record(path: Path, record) -> None:
    """Append one prediction record after checking for duplicate IDs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if path.exists():
        completed = {
            row["heldout_id"]
            for row in read_prediction_records(path)
        }
    if record.heldout_id in completed:
        raise RuntimeError(f"Duplicate prediction for heldout_id: {record.heldout_id}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_json(), sort_keys=True, ensure_ascii=False) + "\n")


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    """Read persisted prediction JSONL records."""

    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mark_completed(run_dir: Path) -> None:
    """Mark an official run complete after predictions and summaries exist."""

    Path(run_dir, "COMPLETED").write_text("completed\n", encoding="utf-8")

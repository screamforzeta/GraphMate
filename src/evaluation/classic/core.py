"""Core data contracts and scoring for the frozen classic benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.verification.classic_scoring import is_accepted_classic_key
from src.verification.freeze_classic_benchmark import verify_freeze_manifest
from src.verification.popeye_consolidate import (
    ACCEPTED_KEY_POLICY_VERSION,
    EXPECTED_DATASET_FINGERPRINT,
    read_json,
    read_jsonl,
)


CLASSIC_EVALUATION_PROTOCOL_VERSION = "classic_heldout_eval_v1"
EXPECTED_FREEZE_FINGERPRINT = "57ff2b7c725a4ef802817581fac9430f8c6416a6de54dafcb1deb7b471716b60"
DEFAULT_DATASET_DIR = Path("data/heldout_classic/final/yacpdb_classic_v1")
DEFAULT_CONSOLIDATED_DIR = Path("data/heldout_classic/verification/yacpdb_classic_v1/consolidated")
DEFAULT_FREEZE_MANIFEST = DEFAULT_DATASET_DIR / "freeze_manifest.json"
VERIFIED_BASES = {"POPEYE_VERIFIED_UNIQUE", "POPEYE_VERIFIED_MULTIPLE"}


@dataclass(frozen=True)
class ClassicSample:
    """One immutable classic benchmark position with accepted-key metadata."""

    heldout_id: str
    source_problem_id: str
    fen: str
    mate_depth: int
    source_key_move_uci: str
    accepted_key_moves_uci: tuple[str, ...]
    accepted_key_basis: str
    verification_status: str
    verification_reason: str
    forced_mate_verified: bool
    dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT

    @property
    def verification_category(self) -> str:
        """Return the reporting category used by classic stratification."""

        if self.accepted_key_basis == "POPEYE_VERIFIED_UNIQUE":
            return "verified_unique"
        if self.accepted_key_basis == "POPEYE_VERIFIED_MULTIPLE":
            return "verified_multiple"
        if self.accepted_key_basis == "YACPDB_SOURCE_UNVERIFIED_TIMEOUT":
            return "unresolved_timeout_source_only"
        return "other"


@dataclass(frozen=True)
class ClassicPredictionRecord:
    """Common persisted prediction schema shared by GNN and LLM systems."""

    benchmark_version: str
    freeze_fingerprint: str
    heldout_id: str
    source_problem_id: str
    mate_depth: int
    model_id: str
    model_family: str
    model_version: str
    predicted_move_uci: str | None
    accepted_key_moves_uci: tuple[str, ...]
    source_key_move_uci: str
    is_correct: bool
    is_legal: bool | None
    parse_success: bool | None
    verification_status: str
    accepted_key_basis: str
    runtime_seconds: float | None = None
    ranked_moves_uci: tuple[str, ...] = ()
    first_accepted_key_rank: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["accepted_key_moves_uci"] = list(self.accepted_key_moves_uci)
        payload["ranked_moves_uci"] = list(self.ranked_moves_uci)
        return payload


def verify_frozen_benchmark(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    consolidated_dir: Path = DEFAULT_CONSOLIDATED_DIR,
    manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
    expected_freeze_fingerprint: str = EXPECTED_FREEZE_FINGERPRINT,
) -> dict[str, Any]:
    """Verify frozen benchmark identity before any official evaluation."""

    manifest = verify_freeze_manifest(dataset_dir, consolidated_dir, manifest_path)
    if manifest.get("lifecycle") != "FROZEN":
        raise RuntimeError("Classic benchmark lifecycle is not FROZEN")
    if manifest.get("freeze_fingerprint") != expected_freeze_fingerprint:
        raise RuntimeError("Classic benchmark freeze fingerprint mismatch")
    if manifest.get("verification", {}).get("accepted_key_policy_version") != ACCEPTED_KEY_POLICY_VERSION:
        raise RuntimeError("Accepted-key policy mismatch")
    return manifest


def _index_by_heldout(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        heldout_id = str(row.get("heldout_id") or "")
        if not heldout_id:
            raise RuntimeError(f"{label} row missing heldout_id")
        if heldout_id in indexed:
            raise RuntimeError(f"Duplicate heldout_id in {label}: {heldout_id}")
        indexed[heldout_id] = row
    return indexed


def _validate_manifest_for_fixture(dataset_dir: Path) -> None:
    manifest = read_json(dataset_dir / "manifest.json")
    if manifest.get("dataset_fingerprint") != EXPECTED_DATASET_FINGERPRINT:
        raise RuntimeError("Classic fixture/canonical manifest fingerprint mismatch")


def load_classic_samples(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    consolidated_dir: Path = DEFAULT_CONSOLIDATED_DIR,
    manifest_path: Path = DEFAULT_FREEZE_MANIFEST,
    require_frozen: bool = True,
    expected_freeze_fingerprint: str = EXPECTED_FREEZE_FINGERPRINT,
) -> list[ClassicSample]:
    """Load canonical rows joined to consolidated accepted-key metadata."""

    if require_frozen:
        verify_frozen_benchmark(dataset_dir, consolidated_dir, manifest_path, expected_freeze_fingerprint)
    else:
        _validate_manifest_for_fixture(dataset_dir)

    canonical = _index_by_heldout(read_jsonl(dataset_dir / "dataset.jsonl"), "canonical dataset")
    accepted = _index_by_heldout(read_jsonl(consolidated_dir / "accepted_keys.jsonl"), "accepted keys")
    if set(canonical) != set(accepted):
        raise RuntimeError("Canonical and consolidated accepted-key IDs disagree")

    samples: list[ClassicSample] = []
    for heldout_id in sorted(canonical):
        source = canonical[heldout_id]
        final = accepted[heldout_id]
        checks = {
            "source_problem_id": (source.get("source_problem_id"), final.get("source_problem_id")),
            "mate_depth": (int(source.get("mate_depth")), int(final.get("mate_depth"))),
            "fen": (source.get("fen"), final.get("canonical_fen")),
            "source_key_move_uci": (source.get("key_move_uci"), final.get("source_key_move_uci")),
        }
        for field_name, (left, right) in checks.items():
            if left != right:
                raise RuntimeError(f"Canonical/consolidated mismatch for {heldout_id}: {field_name}")
        accepted_keys = tuple(str(move) for move in final.get("accepted_key_moves_uci", []) if move)
        if not accepted_keys:
            raise RuntimeError(f"Accepted-key row has no accepted moves: {heldout_id}")
        samples.append(
            ClassicSample(
                heldout_id=heldout_id,
                source_problem_id=str(source["source_problem_id"]),
                fen=str(source["fen"]),
                mate_depth=int(source["mate_depth"]),
                source_key_move_uci=str(source["key_move_uci"]),
                accepted_key_moves_uci=accepted_keys,
                accepted_key_basis=str(final["accepted_key_basis"]),
                verification_status=str(final["final_verification_status"]),
                verification_reason=str(final["final_verification_reason"]),
                forced_mate_verified=bool(final["forced_mate_verified"]),
                dataset_fingerprint=str(final.get("dataset_fingerprint", EXPECTED_DATASET_FINGERPRINT)),
            )
        )
    return samples


def score_classic_prediction(predicted_uci: str | None, accepted_key_moves_uci: tuple[str, ...] | list[str]) -> bool:
    """Score one Top-1 prediction using accepted-key set membership."""

    if predicted_uci is None:
        return False
    return is_accepted_classic_key(predicted_uci, list(accepted_key_moves_uci))


def _rank_of_first_accepted(ranked_moves: tuple[str, ...], accepted: tuple[str, ...]) -> int | None:
    accepted_set = set(accepted)
    for index, move in enumerate(ranked_moves, start=1):
        if move in accepted_set:
            return index
    return None


def build_prediction_record(
    sample: ClassicSample,
    model_metadata: dict[str, Any],
    predicted_move_uci: str | None,
    *,
    ranked_moves_uci: list[str] | tuple[str, ...] = (),
    is_legal: bool | None = None,
    parse_success: bool | None = None,
    runtime_seconds: float | None = None,
    diagnostics: dict[str, Any] | None = None,
    freeze_fingerprint: str = EXPECTED_FREEZE_FINGERPRINT,
) -> ClassicPredictionRecord:
    """Build a common prediction record without mutating benchmark data."""

    ranked = tuple(ranked_moves_uci)
    top1 = predicted_move_uci if predicted_move_uci is not None else (ranked[0] if ranked else None)
    return ClassicPredictionRecord(
        benchmark_version="yacpdb_classic_v1",
        freeze_fingerprint=freeze_fingerprint,
        heldout_id=sample.heldout_id,
        source_problem_id=sample.source_problem_id,
        mate_depth=sample.mate_depth,
        model_id=str(model_metadata["model_id"]),
        model_family=str(model_metadata["model_family"]),
        model_version=str(model_metadata.get("model_version", "unknown")),
        predicted_move_uci=top1,
        accepted_key_moves_uci=sample.accepted_key_moves_uci,
        source_key_move_uci=sample.source_key_move_uci,
        is_correct=score_classic_prediction(top1, sample.accepted_key_moves_uci),
        is_legal=is_legal,
        parse_success=parse_success,
        verification_status=sample.verification_status,
        accepted_key_basis=sample.accepted_key_basis,
        runtime_seconds=runtime_seconds,
        ranked_moves_uci=ranked,
        first_accepted_key_rank=_rank_of_first_accepted(ranked, sample.accepted_key_moves_uci),
        diagnostics=diagnostics or {},
    )


def _accuracy(records: list[ClassicPredictionRecord]) -> dict[str, Any]:
    total = len(records)
    correct = sum(record.is_correct for record in records)
    return {"total": total, "top1_correct": correct, "top1_accuracy": correct / total if total else None}


def aggregate_predictions(records: list[ClassicPredictionRecord]) -> dict[str, Any]:
    """Aggregate classic metrics by all required frozen strata."""

    all_metrics = _accuracy(records)
    verified = [record for record in records if record.accepted_key_basis in VERIFIED_BASES]
    ranked = [record for record in records if record.ranked_moves_uci]
    ranks = [record.first_accepted_key_rank for record in ranked if record.first_accepted_key_rank is not None]
    if ranked:
        all_metrics["top3_accuracy"] = sum(
            any(move in set(record.accepted_key_moves_uci) for move in record.ranked_moves_uci[:3])
            for record in ranked
        ) / len(ranked)
        all_metrics["top5_accuracy"] = sum(
            any(move in set(record.accepted_key_moves_uci) for move in record.ranked_moves_uci[:5])
            for record in ranked
        ) / len(ranked)
    if ranks:
        ordered = sorted(ranks)
        mid = len(ordered) // 2
        all_metrics["mean_accepted_key_rank"] = sum(ranks) / len(ranks)
        all_metrics["median_accepted_key_rank"] = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    return {
        "all_200": all_metrics,
        "popeye_verified_193": _accuracy(verified),
        "by_mate_depth": {
            str(depth): _accuracy([record for record in records if record.mate_depth == depth])
            for depth in range(1, 11)
        },
        "by_verification_category": {
            basis: _accuracy([record for record in records if record.accepted_key_basis == basis])
            for basis in sorted({record.accepted_key_basis for record in records})
        },
        "diagnostics": {
            "legal_false": sum(record.is_legal is False for record in records),
            "parse_failure": sum(record.parse_success is False for record in records),
            "verification_category_counts": dict(Counter(record.accepted_key_basis for record in records)),
        },
    }


def prediction_record_from_json(payload: dict[str, Any]) -> ClassicPredictionRecord:
    """Rebuild a prediction record from persisted JSON."""

    return ClassicPredictionRecord(
        benchmark_version=payload["benchmark_version"],
        freeze_fingerprint=payload["freeze_fingerprint"],
        heldout_id=payload["heldout_id"],
        source_problem_id=payload["source_problem_id"],
        mate_depth=int(payload["mate_depth"]),
        model_id=payload["model_id"],
        model_family=payload["model_family"],
        model_version=payload["model_version"],
        predicted_move_uci=payload.get("predicted_move_uci"),
        accepted_key_moves_uci=tuple(payload.get("accepted_key_moves_uci", [])),
        source_key_move_uci=payload["source_key_move_uci"],
        is_correct=bool(payload["is_correct"]),
        is_legal=payload.get("is_legal"),
        parse_success=payload.get("parse_success"),
        verification_status=payload["verification_status"],
        accepted_key_basis=payload["accepted_key_basis"],
        runtime_seconds=payload.get("runtime_seconds"),
        ranked_moves_uci=tuple(payload.get("ranked_moves_uci", [])),
        first_accepted_key_rank=payload.get("first_accepted_key_rank"),
        diagnostics=payload.get("diagnostics", {}),
    )

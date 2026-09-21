"""Build the deterministic YACPDB candidate held-out dataset.

This module consumes only the completed bounded availability scan cache. It
does not contact YACPDB, change parser semantics, run engines, or run models.
The output is a candidate dataset ready for independent forced-mate review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess

from src.data.heldout_classic import board_state_key, detect_lichess_overlap, load_lichess_overlap_keys
from src.data.heldout_sources.yacpdb import (
    KEY_EXTRACTOR_VERSION,
    YACPDB_IMPORTER_VERSION,
    candidate_to_problem,
    normalize_yacpdb_record,
)
from src.data.heldout_sources.yacpdb_availability import (
    ENDPOINT_CLASSIFICATION,
    SCAN_VERSION,
    evaluate_record,
)


DATASET_NAME = "yacpdb_heldout_classic"
DATASET_VERSION = "yacpdb_classic_v1"
LIFECYCLE_STATUS = "VALIDATED_NOT_FROZEN"
SCHEMA_VERSION = "yacpdb_candidate_schema_v1"
VALIDATOR_VERSION = "yacpdb_candidate_builder_v1"
SELECTION_SEED = 42
PER_DEPTH = 20
EXPECTED_AVAILABILITY_FINGERPRINT = "a7694af29724e10907d3b51d759d1ef4f38492e35cfb410b40150400696a2880"
DEFAULT_OUTPUT_ROOT = Path("data/heldout_classic")


@dataclass(frozen=True)
class CandidateBuildConfig:
    """Immutable configuration for candidate dataset construction."""

    output_root: Path = DEFAULT_OUTPUT_ROOT
    dataset_version: str = DATASET_VERSION
    availability_fingerprint: str = EXPECTED_AVAILABILITY_FINGERPRINT
    per_depth: int = PER_DEPTH
    seed: int = SELECTION_SEED


def semantic_fingerprint(payload: dict[str, Any]) -> str:
    """Return a SHA256 fingerprint over stable semantic JSON content."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_availability(output_root: Path, expected_fingerprint: str) -> dict[str, Any]:
    """Load and verify the authoritative availability artifact."""

    path = output_root / "processed" / "yacpdb_availability.json"
    if not path.exists():
        raise RuntimeError(f"Missing availability artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("scan_fingerprint") != expected_fingerprint:
        raise RuntimeError("Availability fingerprint mismatch")
    if payload.get("status") != "COMPLETE":
        raise RuntimeError("Availability scan is not complete")
    if sorted(payload.get("completed_depths", [])) != list(range(1, 11)):
        raise RuntimeError("Availability scan does not cover MateDepth 1..10")
    config = payload.get("config") or {}
    if config.get("scan_version") != SCAN_VERSION:
        raise RuntimeError("Incompatible availability scan version")
    if config.get("importer_version") != YACPDB_IMPORTER_VERSION:
        raise RuntimeError("Incompatible YACPDB importer version")
    if config.get("key_extractor_version") != KEY_EXTRACTOR_VERSION:
        raise RuntimeError("Incompatible YACPDB key extractor version")
    return payload


def cached_page_paths(output_root: Path, depth: int) -> list[Path]:
    """Return cached raw page paths for one MateDepth."""

    depth_dir = output_root / "raw" / "yacpdb" / "availability" / f"mate_{depth}"
    paths = sorted(path for path in depth_dir.glob("page_*.json") if not path.name.endswith(".metadata.json"))
    if not paths:
        raise RuntimeError(f"Missing cached YACPDB pages for MateDepth {depth}: {depth_dir}")
    return paths


def load_cached_records(output_root: Path, depth: int) -> list[dict[str, Any]]:
    """Load unique cached source records for one MateDepth."""

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in cached_page_paths(output_root, depth):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("success"):
            raise RuntimeError(f"Cached page is not successful: {path}")
        for record in (payload.get("result") or {}).get("entries") or []:
            source_id = str(record.get("id") or record.get("source_problem_id") or record.get("yacpdb_id") or "")
            if not source_id or source_id in seen_ids:
                continue
            seen_ids.add(source_id)
            records.append(record)
    return records


def numeric_source_id(source_problem_id: str) -> int:
    """Return numeric YACPDB ID for canonical ordering."""

    return int(str(source_problem_id))


def reconstruct_clean_candidate_pool(
    output_root: Path,
    availability: dict[str, Any],
    lichess_keys: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    """Reconstruct clean unique eligible candidates from cached scan records."""

    lichess_keys = load_lichess_overlap_keys() if lichess_keys is None else lichess_keys
    pool: dict[int, list[dict[str, Any]]] = {}
    audit = {
        "duplicate_source_ids": 0,
        "exact_fen_duplicates": 0,
        "normalized_position_duplicates": 0,
        "lichess_contamination": 0,
        "contamination_reasons": {},
    }
    global_ids: set[str] = set()
    global_exact: set[str] = set()
    global_normalized: set[str] = set()
    contamination_reasons = Counter()
    for depth in range(1, 11):
        clean: list[dict[str, Any]] = []
        for record in load_cached_records(output_root, depth):
            evaluation = evaluate_record(record, depth)
            if evaluation.status != "eligible" or evaluation.candidate is None:
                continue
            candidate, reason = normalize_yacpdb_record(record)
            if reason or candidate is None:
                continue
            problem = candidate_to_problem(candidate, DATASET_VERSION)
            overlaps = detect_lichess_overlap(problem, lichess_keys)
            if overlaps:
                audit["lichess_contamination"] += 1
                contamination_reasons.update(overlaps)
                continue
            source_id = candidate.source_problem_id
            exact = candidate.fen
            normalized = board_state_key(candidate.fen)
            if source_id in global_ids:
                audit["duplicate_source_ids"] += 1
                continue
            if exact in global_exact:
                audit["exact_fen_duplicates"] += 1
                continue
            if normalized in global_normalized:
                audit["normalized_position_duplicates"] += 1
                continue
            global_ids.add(source_id)
            global_exact.add(exact)
            global_normalized.add(normalized)
            board = chess.Board(candidate.fen)
            move = chess.Move.from_uci(candidate.key_move_uci)
            board.push(move)
            clean.append(
                {
                    "candidate": candidate,
                    "parser_method": evaluation.parser_method,
                    "key_token": evaluation.key_token,
                    "piece_count": evaluation.piece_count,
                    "is_capture_key": evaluation.is_capture_key,
                    "is_check_key": board.is_check(),
                    "is_promotion_key": evaluation.is_promotion_key,
                    "is_castling_key": evaluation.is_castling_key,
                }
            )
        expected = int(availability["depths"][str(depth)]["clean_unique_eligible"])
        if len(clean) != expected:
            raise RuntimeError(f"Reconstructed clean pool mismatch for MateDepth {depth}: {len(clean)} != {expected}")
        pool[depth] = sorted(clean, key=lambda item: numeric_source_id(item["candidate"].source_problem_id))
    audit["contamination_reasons"] = dict(contamination_reasons)
    return pool, audit


def select_candidates(pool: dict[int, list[dict[str, Any]]], per_depth: int, seed: int) -> dict[int, list[dict[str, Any]]]:
    """Select a deterministic seeded sample after canonical ID ordering."""

    rng = random.Random(seed)
    selected: dict[int, list[dict[str, Any]]] = {}
    for depth in range(1, 11):
        bucket = list(pool.get(depth, []))
        if len(bucket) < per_depth:
            raise RuntimeError(f"Insufficient clean candidates for MateDepth {depth}: {len(bucket)} < {per_depth}")
        sample = rng.sample(bucket, per_depth)
        selected[depth] = sorted(sample, key=lambda item: numeric_source_id(item["candidate"].source_problem_id))
    return selected


def row_for_candidate(item: dict[str, Any], heldout_index: int, config: CandidateBuildConfig) -> dict[str, Any]:
    """Return one canonical candidate dataset row."""

    candidate = item["candidate"]
    board = chess.Board(candidate.fen)
    source_metadata = {
        "raw_record": candidate.raw_record,
        "source_solution_preservation_policy": "Preserved locally in generated candidate artifact; redistribution license status remains unclear.",
    }
    return {
        "heldout_id": f"{config.dataset_version}_{heldout_index:04d}",
        "dataset_version": config.dataset_version,
        "source": "YACPDB",
        "source_problem_id": candidate.source_problem_id,
        "fen": candidate.fen,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "mate_depth": candidate.mate_depth,
        "stipulation": candidate.stipulation,
        "problem_type": "directmate",
        "key_move_uci": candidate.key_move_uci,
        "source_solution_raw": candidate.source_solution_raw,
        "solution_tree": None,
        "principal_line_uci": [],
        "composer": candidate.composer,
        "author": candidate.composer,
        "publication": candidate.source_reference,
        "source_reference": candidate.source_reference,
        "publication_date": candidate.publication_date,
        "source_url": candidate.source_url,
        "license": candidate.license,
        "license_status": "UNCLEAR",
        "validation_status": "KEY_VALIDATED",
        "source_claimed_mate": True,
        "key_validation_status": "KEY_VALIDATED",
        "line_validation_status": "NOT_NORMALIZED",
        "forced_mate_verified": False,
        "forced_mate_verification_status": "NOT_VERIFIED_ENGINE_NOT_USED",
        "duplicate_status": "unique",
        "contamination_status": "clean",
        "parser_method": item["parser_method"],
        "key_token": item["key_token"],
        "piece_count": item["piece_count"],
        "key_is_capture": item["is_capture_key"],
        "key_is_check": item["is_check_key"],
        "key_is_promotion": item["is_promotion_key"],
        "key_is_castling": item["is_castling_key"],
        "selection_seed": config.seed,
        "selection_strategy": "canonical_numeric_yacpdb_id_order_then_seeded_sample_per_mate_depth",
        "availability_scan_fingerprint": config.availability_fingerprint,
        "source_metadata": source_metadata,
        "warnings": ["SOURCE_CLAIMED_MATE", "NOT_VERIFIED_ENGINE_NOT_USED"],
    }


def selected_rows(selected: dict[int, list[dict[str, Any]]], config: CandidateBuildConfig) -> list[dict[str, Any]]:
    """Return selected rows sorted by MateDepth and selected source ID."""

    rows: list[dict[str, Any]] = []
    index = 1
    for depth in range(1, 11):
        for item in selected[depth]:
            rows.append(row_for_candidate(item, index, config))
            index += 1
    return rows


def dataset_fingerprint(rows: list[dict[str, Any]], config: CandidateBuildConfig) -> str:
    """Fingerprint the selected semantic dataset content."""

    stable_rows = [
        {
            "dataset_version": row["dataset_version"],
            "source_problem_id": row["source_problem_id"],
            "fen": row["fen"],
            "mate_depth": row["mate_depth"],
            "key_move_uci": row["key_move_uci"],
            "parser_method": row["parser_method"],
        }
        for row in rows
    ]
    return semantic_fingerprint(
        {
            "dataset_version": config.dataset_version,
            "selected_rows": stable_rows,
            "schema_version": SCHEMA_VERSION,
            "parser_version": YACPDB_IMPORTER_VERSION,
            "key_extractor_version": KEY_EXTRACTOR_VERSION,
            "validator_version": VALIDATOR_VERSION,
            "availability_scan_fingerprint": config.availability_fingerprint,
            "selection_seed": config.seed,
            "selection_algorithm": "canonical_numeric_yacpdb_id_order_then_seeded_sample_per_mate_depth",
        }
    )


def audit_selected_rows(rows: list[dict[str, Any]], lichess_keys: dict[str, dict[str, set[str]]] | None = None) -> dict[str, Any]:
    """Fail closed if selected rows contain duplicates or Lichess contamination."""

    lichess_keys = load_lichess_overlap_keys() if lichess_keys is None else lichess_keys
    ids = [row["source_problem_id"] for row in rows]
    fens = [row["fen"] for row in rows]
    normalized = [board_state_key(row["fen"]) for row in rows]
    contamination = Counter()
    for row in rows:
        problem = candidate_to_problem(normalize_yacpdb_record(row)[0], row["dataset_version"]) if False else None
        probe = type("Probe", (), {"fen": row["fen"]})()
        for reason in detect_lichess_overlap(probe, lichess_keys):  # type: ignore[arg-type]
            contamination[reason] += 1
    audit = {
        "duplicate_source_ids": len(ids) - len(set(ids)),
        "exact_fen_duplicates": len(fens) - len(set(fens)),
        "normalized_position_duplicates": len(normalized) - len(set(normalized)),
        "lichess_contamination": sum(contamination.values()),
        "contamination_reasons": dict(contamination),
        "all_key_validated": all(row["key_validation_status"] == "KEY_VALIDATED" for row in rows),
        "all_forced_mate_verified": all(row["forced_mate_verified"] is True for row in rows),
    }
    if audit["duplicate_source_ids"]:
        raise RuntimeError("Duplicate source_problem_id in selected dataset")
    if audit["exact_fen_duplicates"]:
        raise RuntimeError("Exact FEN duplicate in selected dataset")
    if audit["normalized_position_duplicates"]:
        raise RuntimeError("Normalized-position duplicate in selected dataset")
    if audit["lichess_contamination"]:
        raise RuntimeError("Lichess contamination in selected dataset")
    if not audit["all_key_validated"]:
        raise RuntimeError("Selected dataset contains non-key-validated rows")
    if audit["all_forced_mate_verified"]:
        raise RuntimeError("Forced-mate verification was unexpectedly marked true")
    return audit


def distribution_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute descriptive post-selection statistics without resampling."""

    pieces = [int(row["piece_count"]) for row in rows]
    composers = {row["composer"] for row in rows if row.get("composer")}
    publications = {row["publication"] for row in rows if row.get("publication")}
    years = []
    for row in rows:
        text = str(row.get("publication_date") or "")
        if text[:4].isdigit():
            years.append(int(text[:4]))
    return {
        "mate_depth_distribution": dict(Counter(str(row["mate_depth"]) for row in rows)),
        "piece_count": {
            "mean": sum(pieces) / len(pieces),
            "median": statistics.median(pieces),
            "min": min(pieces),
            "max": max(pieces),
        },
        "key_characteristics": {
            "capture": sum(bool(row["key_is_capture"]) for row in rows),
            "checking": sum(bool(row["key_is_check"]) for row in rows),
            "promotion": sum(bool(row["key_is_promotion"]) for row in rows),
            "castling": sum(bool(row["key_is_castling"]) for row in rows),
        },
        "parser_methods": dict(Counter(str(row["parser_method"]) for row in rows)),
        "provenance": {
            "distinct_composers": len(composers),
            "distinct_publications": len(publications),
            "year_min": min(years) if years else None,
            "year_max": max(years) if years else None,
            "year_populated": len(years),
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as deterministic UTF-8 JSONL."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact CSV derived from the canonical JSONL."""

    fields = [
        "heldout_id",
        "dataset_version",
        "source",
        "source_problem_id",
        "fen",
        "side_to_move",
        "mate_depth",
        "stipulation",
        "problem_type",
        "key_move_uci",
        "composer",
        "publication",
        "publication_date",
        "source_url",
        "validation_status",
        "key_validation_status",
        "line_validation_status",
        "forced_mate_verification_status",
        "duplicate_status",
        "contamination_status",
        "parser_method",
        "key_token",
        "piece_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_markdown_report(manifest: dict[str, Any], rows: list[dict[str, Any]], path: Path) -> None:
    """Write a concise human-readable candidate build report."""

    selected = manifest["selected_source_ids_by_depth"]
    lines = [
        "# YACPDB Candidate Dataset Build",
        "",
        f"Lifecycle: `{manifest['lifecycle_status']}`",
        f"Dataset version: `{manifest['dataset_version']}`",
        f"Dataset fingerprint: `{manifest['dataset_fingerprint']}`",
        "",
        "This build selects 20 clean unique eligible YACPDB candidates for each MateDepth #1 through #10 from the completed bounded availability scan. It is not frozen and has not been engine-verified.",
        "",
        "## Selection",
        "",
        f"- Availability scan fingerprint: `{manifest['availability_scan_fingerprint']}`",
        f"- Seed: `{manifest['selection_seed']}`",
        f"- Canonical ordering: `{manifest['canonical_ordering_rule']}`",
        f"- Strategy: `{manifest['deterministic_selection_algorithm']}`",
        "",
        "## Selected IDs",
        "",
    ]
    for depth in range(1, 11):
        lines.append(f"- MateIn{depth}: {', '.join(selected[str(depth)])}")
    stats = manifest["distribution"]
    lines.extend(
        [
            "",
            "## Distribution",
            "",
            f"- MateDepth: {stats['mate_depth_distribution']}",
            f"- Piece count: mean={stats['piece_count']['mean']:.2f}, median={stats['piece_count']['median']}, min={stats['piece_count']['min']}, max={stats['piece_count']['max']}",
            f"- Key characteristics: {stats['key_characteristics']}",
            f"- Parser methods: {stats['parser_methods']}",
            f"- Provenance: {stats['provenance']}",
            "",
            "## Integrity",
            "",
            f"- Duplicate audit: {manifest['duplicate_audit']}",
            f"- Lichess contamination audit: {manifest['contamination_audit']}",
            "- Forced-mate status: `NOT_VERIFIED_ENGINE_NOT_USED` for every row.",
            "- No GNN, LLM, Stockfish, Popeye, or target-dependent selection was used.",
            "",
            "## Review Index",
            "",
            "| Heldout ID | YACPDB ID | MateDepth | Composer | Source | FEN | Pieces | Key token | Key UCI | Parser | Validation | Duplicate | Contamination | Forced mate | Warnings |",
            "|---|---:|---:|---|---|---|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        warnings = ", ".join(row["warnings"])
        lines.append(
            f"| `{row['heldout_id']}` | {row['source_problem_id']} | {row['mate_depth']} | "
            f"{row.get('composer') or ''} | {row.get('publication') or ''} | `{row['fen']}` | "
            f"{row['piece_count']} | `{row.get('key_token') or ''}` | `{row['key_move_uci']}` | "
            f"{row.get('parser_method') or ''} | {row['validation_status']} | {row['duplicate_status']} | "
            f"{row['contamination_status']} | {row['forced_mate_verification_status']} | {warnings} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_candidate_dataset(config: CandidateBuildConfig) -> dict[str, Any]:
    """Build and persist the deterministic YACPDB candidate dataset."""

    availability = load_availability(config.output_root, config.availability_fingerprint)
    pool, pool_audit = reconstruct_clean_candidate_pool(config.output_root, availability)
    selected = select_candidates(pool, config.per_depth, config.seed)
    rows = selected_rows(selected, config)
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 selected rows, found {len(rows)}")
    counts = Counter(row["mate_depth"] for row in rows)
    if any(counts[depth] != config.per_depth for depth in range(1, 11)):
        raise RuntimeError("Selected MateDepth distribution is not exactly 20 per depth")
    selected_audit = audit_selected_rows(rows)
    fingerprint = dataset_fingerprint(rows, config)
    distribution = distribution_stats(rows)
    final_dir = config.output_root / "final" / config.dataset_version
    final_dir.mkdir(parents=True, exist_ok=True)
    selected_ids_by_depth = {
        str(depth): [item["candidate"].source_problem_id for item in selected[depth]]
        for depth in range(1, 11)
    }
    selected_ids_payload = {
        "dataset_version": config.dataset_version,
        "selection_seed": config.seed,
        "selection_strategy": "canonical_numeric_yacpdb_id_order_then_seeded_sample_per_mate_depth",
        "canonical_ordering_rule": "numeric YACPDB source_problem_id ascending before sampling; selected rows sorted by MateDepth then numeric source_problem_id",
        "availability_scan_fingerprint": config.availability_fingerprint,
        "selected_ids": selected_ids_by_depth,
    }
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_version": config.dataset_version,
        "lifecycle_status": LIFECYCLE_STATUS,
        "source": "YACPDB",
        "source_endpoint_classification": ENDPOINT_CLASSIFICATION,
        "availability_scan_fingerprint": config.availability_fingerprint,
        "candidate_pool_counts_per_depth": {str(depth): availability["depths"][str(depth)]["clean_unique_eligible"] for depth in range(1, 11)},
        "selected_count_per_depth": {str(depth): counts[depth] for depth in range(1, 11)},
        "total_selected_count": len(rows),
        "selection_seed": config.seed,
        "deterministic_selection_algorithm": "Sort clean candidates by numeric YACPDB source_problem_id, sample 20 per depth with Python random.Random(seed=42), then sort selected rows by depth and numeric ID.",
        "canonical_ordering_rule": "numeric YACPDB source_problem_id ascending",
        "selected_source_ids_by_depth": selected_ids_by_depth,
        "schema_version": SCHEMA_VERSION,
        "parser_version": YACPDB_IMPORTER_VERSION,
        "normalizer_version": YACPDB_IMPORTER_VERSION,
        "key_extractor_version": KEY_EXTRACTOR_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "duplicate_audit": {
            "duplicate_source_ids": selected_audit["duplicate_source_ids"],
            "exact_fen_duplicates": selected_audit["exact_fen_duplicates"],
            "normalized_position_duplicates": selected_audit["normalized_position_duplicates"],
        },
        "contamination_audit": {
            "lichess_contamination": selected_audit["lichess_contamination"],
            "contamination_reasons": selected_audit["contamination_reasons"],
        },
        "pool_audit": pool_audit,
        "provenance_license_status": "UNCLEAR",
        "source_solution_preservation_policy": "source_solution_raw is preserved in generated local candidate artifacts; redistribution status is not asserted.",
        "dataset_fingerprint": fingerprint,
        "dataset_fingerprint_procedure": "SHA256 over canonical JSON containing dataset version, selected source IDs, normalized FENs, MateDepth, key_move_uci, parser/key-extractor/validator versions, availability scan fingerprint, seed, and selection algorithm.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_outputs_used_for_selection": False,
        "GNN_inference_performed": False,
        "LLM_inference_performed": False,
        "Stockfish_used": False,
        "Popeye_used": False,
        "forced_mate_verified": False,
        "ready_for_forced_mate_verification": True,
        "frozen": False,
        "distribution": distribution,
        "canonical_dataset_path": str(final_dir / "dataset.jsonl"),
        "derived_csv_path": str(final_dir / "dataset.csv"),
        "selected_ids_path": str(final_dir / "selected_ids.json"),
        "verification_queue_path": str(final_dir / "forced_mate_verification_queue.jsonl"),
    }
    (final_dir / "selected_ids.json").write_text(json.dumps(selected_ids_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_jsonl(final_dir / "dataset.jsonl", rows)
    write_csv(final_dir / "dataset.csv", rows)
    queue_rows = [
        {
            "heldout_id": row["heldout_id"],
            "source_problem_id": row["source_problem_id"],
            "fen": row["fen"],
            "side_to_move": row["side_to_move"],
            "mate_depth": row["mate_depth"],
            "stipulation": row["stipulation"],
            "key_move_uci": row["key_move_uci"],
            "forced_mate_verification_status": "NOT_VERIFIED_ENGINE_NOT_USED",
        }
        for row in rows
    ]
    write_jsonl(final_dir / "forced_mate_verification_queue.jsonl", queue_rows)
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(manifest, rows, Path("generic_info/yacpdb_candidate_build.md"))
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--availability-fingerprint", default=EXPECTED_AVAILABILITY_FINGERPRINT)
    parser.add_argument("--per-depth", type=int, default=PER_DEPTH)
    parser.add_argument("--seed", type=int, default=SELECTION_SEED)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    manifest = build_candidate_dataset(
        CandidateBuildConfig(
            output_root=Path(args.output_root),
            dataset_version=args.dataset_version,
            availability_fingerprint=args.availability_fingerprint,
            per_depth=args.per_depth,
            seed=args.seed,
        )
    )
    print(
        json.dumps(
            {
                "dataset_version": manifest["dataset_version"],
                "lifecycle_status": manifest["lifecycle_status"],
                "total_selected_count": manifest["total_selected_count"],
                "dataset_fingerprint": manifest["dataset_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

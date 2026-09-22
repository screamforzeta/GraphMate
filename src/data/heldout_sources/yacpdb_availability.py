"""Bounded YACPDB availability scan for orthodox directmates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess

from src.data.heldout_classic import board_state_key, detect_lichess_overlap, load_lichess_overlap_keys
from src.data.heldout_sources.yacpdb import (
    KEY_EXTRACTOR_VERSION,
    YACPDB_IMPORTER_VERSION,
    YacpdbCandidate,
    candidate_to_problem,
    classify_stipulation,
    composer,
    normalize_yacpdb_record,
    publication_date,
    record_fen,
    source_reference,
    source_url_for_id,
)
from src.data.heldout_sources.yacpdb_client import DEFAULT_USER_AGENT, YacpdbClient, YacpdbResponse
from src.data.heldout_sources.yacpdb_solution import extract_structural_key


SCAN_VERSION = "yacpdb_availability_scan_v1"
QUERY_ORDER = "UNKNOWN"
ENDPOINT_CLASSIFICATION = "PUBLIC_BUT_UNDOCUMENTED"
DEFAULT_OUTPUT_ROOT = Path("data/heldout_classic")


@dataclass
class RecordEvaluation:
    """Evaluation outcome for one raw YACPDB source record."""

    source_problem_id: str
    mate_depth: int
    status: str
    reason: str | None = None
    candidate: YacpdbCandidate | None = None
    parser_method: str | None = None
    key_token: str | None = None
    piece_count: int | None = None
    is_capture_key: bool = False
    is_check_key: bool = False
    is_promotion_key: bool = False
    is_castling_key: bool = False
    warnings: list[str] = field(default_factory=list)


def scan_fingerprint(config: dict[str, Any]) -> str:
    """Return a deterministic fingerprint for scan identity."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_for_depth(depth: int) -> str:
    """Return the frozen per-depth YACPDB QL query."""

    return f'Stip("^#{depth}$")'


def response_from_cache(path: Path) -> YacpdbResponse:
    """Load a cached raw JSON response as a client response."""

    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return YacpdbResponse(
        method="GET",
        url=str(metadata.get("endpoint") or path),
        status=int(metadata.get("status") or 200),
        content_type=str(metadata.get("content_type") or "application/json"),
        body=path.read_bytes(),
    )


def cache_page_response(
    client: YacpdbClient,
    response: YacpdbResponse,
    depth_dir: Path,
    depth: int,
    page: int,
    query: str,
    overwrite: bool = False,
) -> Path:
    """Cache one raw page response and return its path."""

    stem = f"page_{page:04d}"
    raw_path, _metadata_path = client.cache_raw_response(
        response,
        depth_dir,
        stem,
        metadata={
            "mate_depth": depth,
            "safe_request_parameters": {"q": query, "p": page},
            "endpoint_classification": ENDPOINT_CLASSIFICATION,
        },
        overwrite=overwrite,
    )
    return raw_path


def load_or_fetch_page(
    client: YacpdbClient,
    depth_dir: Path,
    depth: int,
    page: int,
    query: str,
) -> tuple[YacpdbResponse, bool]:
    """Return one page response, reusing cache when available."""

    raw_path = depth_dir / f"page_{page:04d}.json"
    if raw_path.exists():
        return response_from_cache(raw_path), True
    response = client.search(query, page=page)
    cache_page_response(client, response, depth_dir, depth, page, query)
    return response, False


def evaluate_record(record: dict[str, Any], depth: int) -> RecordEvaluation:
    """Apply the frozen source/position/key filters to one record."""

    source_id = str(record.get("id") or record.get("source_problem_id") or record.get("yacpdb_id") or "")
    source_depth, source_reason = classify_stipulation(record)
    if source_reason or source_depth != depth:
        return RecordEvaluation(source_id, depth, "source_filter_rejected", source_reason or "UNSUPPORTED_STIPULATION")
    fen, _side_to_move, fen_reason = record_fen(record)
    if fen_reason or not fen:
        return RecordEvaluation(source_id, depth, "position_rejected", fen_reason or "UNSUPPORTED_ALGEBRAIC_POSITION")
    board = chess.Board(fen)
    key, key_reason = extract_structural_key(str(record.get("solution") or record.get("source_solution_raw") or ""), board)
    if key_reason or key is None:
        return RecordEvaluation(source_id, depth, "key_rejected", key_reason)
    move = chess.Move.from_uci(key.key_move_uci)
    if move not in board.legal_moves:
        return RecordEvaluation(source_id, depth, "key_illegal", "KEY_NOT_LEGAL")
    candidate, reason = normalize_yacpdb_record(record)
    if reason or candidate is None:
        return RecordEvaluation(source_id, depth, "key_illegal", reason or "KEY_NOT_LEGAL")
    board.push(move)
    return RecordEvaluation(
        source_problem_id=source_id,
        mate_depth=depth,
        status="eligible",
        candidate=candidate,
        parser_method=key.parse_method,
        key_token=key.key_token,
        piece_count=len(chess.Board(fen).piece_map()),
        is_capture_key=chess.Board(fen).is_capture(move),
        is_check_key=board.is_check(),
        is_promotion_key=move.promotion is not None,
        is_castling_key=chess.Board(fen).is_castling(move),
        warnings=["SOURCE_CLAIMED_MATE", "NOT_VERIFIED_ENGINE_NOT_USED"],
    )


def empty_depth_stats(depth: int, query: str) -> dict[str, Any]:
    """Create a per-depth stats container."""

    return {
        "mate_depth": depth,
        "query": query,
        "source_reported_query_count": 0,
        "records_inspected": 0,
        "unique_records": 0,
        "source_filter_pass": 0,
        "source_filter_rejected": 0,
        "position_normalized": 0,
        "position_rejected": 0,
        "key_extracted": 0,
        "key_rejected": 0,
        "key_legal": 0,
        "key_illegal": 0,
        "eligible": 0,
        "eligible_before_dedup": 0,
        "unique_eligible_positions": 0,
        "eligible_before_contamination": 0,
        "contaminated": 0,
        "clean_eligible": 0,
        "clean_unique_eligible": 0,
        "duplicate_yacpdb_ids": 0,
        "exact_fen_duplicates": 0,
        "normalized_position_duplicates": 0,
        "rejection_reasons": {},
        "parser_methods": {},
        "key_stats": {},
        "position_stats": {},
        "contamination": {},
        "source_ids_inspected": [],
        "representative_rejections": {},
    }


def update_funnel(stats: dict[str, Any], evaluation: RecordEvaluation) -> None:
    """Update funnel counters from one record evaluation."""

    stats["records_inspected"] += 1
    stats["unique_records"] += 1
    if evaluation.status == "source_filter_rejected":
        stats["source_filter_rejected"] += 1
    else:
        stats["source_filter_pass"] += 1
        if evaluation.status == "position_rejected":
            stats["position_rejected"] += 1
        else:
            stats["position_normalized"] += 1
            if evaluation.status == "key_rejected":
                stats["key_rejected"] += 1
            else:
                stats["key_extracted"] += 1
                if evaluation.status == "key_illegal":
                    stats["key_illegal"] += 1
                else:
                    stats["key_legal"] += 1
                    stats["eligible"] += 1
                    stats["eligible_before_dedup"] += 1
    if evaluation.reason:
        reasons = Counter(stats["rejection_reasons"])
        reasons[evaluation.reason] += 1
        stats["rejection_reasons"] = dict(reasons)
        examples = stats["representative_rejections"].setdefault(evaluation.reason, [])
        if len(examples) < 5:
            examples.append(evaluation.source_problem_id)


def summarize_depth_posthoc(
    stats: dict[str, Any],
    evaluations: list[RecordEvaluation],
    lichess_keys: dict[str, dict[str, set[str]]],
) -> list[dict[str, Any]]:
    """Compute duplicate, contamination, position, key, and review stats."""

    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    clean_normalized: set[str] = set()
    piece_counts: list[int] = []
    parser_methods = Counter()
    key_stats = Counter()
    contamination = Counter()
    review_rows: list[dict[str, Any]] = []
    for evaluation in evaluations:
        if evaluation.status != "eligible" or evaluation.candidate is None:
            continue
        candidate = evaluation.candidate
        problem = candidate_to_problem(candidate, "availability_scan")
        fen = candidate.fen
        normalized = board_state_key(fen)
        if fen in seen_exact:
            stats["exact_fen_duplicates"] += 1
        if normalized in seen_normalized:
            stats["normalized_position_duplicates"] += 1
        seen_exact.add(fen)
        seen_normalized.add(normalized)
        overlaps = detect_lichess_overlap(problem, lichess_keys)
        if overlaps:
            stats["contaminated"] += 1
            for overlap in overlaps:
                contamination[overlap] += 1
        else:
            stats["clean_eligible"] += 1
            clean_normalized.add(normalized)
            review_rows.append(
                {
                    "yacpdb_id": candidate.source_problem_id,
                    "mate_depth": candidate.mate_depth,
                    "composer": candidate.composer,
                    "source": candidate.source_reference,
                    "fen": candidate.fen,
                    "piece_count": evaluation.piece_count,
                    "key_token": evaluation.key_token,
                    "key_move_uci": candidate.key_move_uci,
                    "parser_method": evaluation.parser_method,
                    "validation_status": candidate.validation_status,
                    "line_validation_status": candidate.line_validation_status,
                    "forced_mate_verification_status": candidate.forced_mate_verification_status,
                    "duplicate_status": "unique_position" if normalized not in clean_normalized else "duplicate_position",
                    "contamination_status": "clean",
                    "warnings": evaluation.warnings,
                }
            )
        if evaluation.piece_count is not None:
            piece_counts.append(evaluation.piece_count)
        if evaluation.parser_method:
            parser_methods[evaluation.parser_method] += 1
        key_stats["capture_keys"] += int(evaluation.is_capture_key)
        key_stats["checking_keys"] += int(evaluation.is_check_key)
        key_stats["promotion_keys"] += int(evaluation.is_promotion_key)
        key_stats["castling_keys"] += int(evaluation.is_castling_key)
    stats["unique_eligible_positions"] = len(seen_normalized)
    stats["eligible_before_contamination"] = stats["eligible"]
    stats["clean_unique_eligible"] = len(clean_normalized)
    stats["parser_methods"] = dict(parser_methods)
    stats["key_stats"] = dict(key_stats)
    stats["contamination"] = dict(contamination)
    if piece_counts:
        stats["position_stats"] = {
            "piece_count_min": min(piece_counts),
            "piece_count_median": statistics.median(piece_counts),
            "piece_count_mean": sum(piece_counts) / len(piece_counts),
            "piece_count_max": max(piece_counts),
            "side_to_move": {"white": len(piece_counts), "black": 0},
        }
    return sorted(review_rows, key=lambda row: int(row["yacpdb_id"]))[:2]


def scan_depth(
    client: YacpdbClient,
    output_root: Path,
    depth: int,
    per_depth_cap: int,
    lichess_keys: dict[str, dict[str, set[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Scan one MateDepth up to the configured raw unique-record cap."""

    query = query_for_depth(depth)
    depth_dir = output_root / "raw" / "yacpdb" / "availability" / f"mate_{depth}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    stats = empty_depth_stats(depth, query)
    seen_ids: set[str] = set()
    evaluations: list[RecordEvaluation] = []
    page = 1
    completed = True
    while len(seen_ids) < per_depth_cap:
        response, _cached = load_or_fetch_page(client, depth_dir, depth, page, query)
        payload = response.json()
        if not payload.get("success"):
            completed = False
            break
        result = payload.get("result") or {}
        stats["source_reported_query_count"] = int(result.get("count") or 0)
        entries = result.get("entries") or []
        if not entries:
            break
        for record in entries:
            source_id = str(record.get("id") or "")
            if source_id in seen_ids:
                stats["duplicate_yacpdb_ids"] += 1
                continue
            if len(seen_ids) >= per_depth_cap:
                break
            seen_ids.add(source_id)
            stats["source_ids_inspected"].append(source_id)
            evaluation = evaluate_record(record, depth)
            evaluations.append(evaluation)
            update_funnel(stats, evaluation)
        if len(seen_ids) >= stats["source_reported_query_count"]:
            break
        page += 1
    review_rows = summarize_depth_posthoc(stats, evaluations, lichess_keys)
    return stats, review_rows, completed


def aggregate_rejections(depths: dict[str, dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Aggregate rejection reasons across depths for report tables."""

    reasons = sorted({reason for stats in depths.values() for reason in stats["rejection_reasons"]})
    table: dict[str, dict[str, int]] = {}
    for reason in reasons:
        row = {f"MateIn{depth}": int(depths[str(depth)]["rejection_reasons"].get(reason, 0)) for depth in range(1, 11)}
        row["Total"] = sum(row.values())
        table[reason] = row
    return table


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    """Write the human-readable availability report."""

    lines = [
        "# YACPDB Availability Scan",
        "",
        f"Status: `{payload['status']}`",
        "",
        "Availability is measured within a bounded discovery sample of up to "
        f"{payload['config']['per_depth_cap']} unique source records per MateDepth. Query order is `UNKNOWN`; this is not a random sample.",
        "",
        "## Funnel",
        "",
        "| MateDepth | Source count | Inspected | Filter pass | Position OK | Key extracted | Key legal | Unique eligible | Clean unique eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = Counter()
    for depth in range(1, 11):
        stats = payload["depths"][str(depth)]
        lines.append(
            f"| {depth} | {stats['source_reported_query_count']} | {stats['records_inspected']} | "
            f"{stats['source_filter_pass']} | {stats['position_normalized']} | {stats['key_extracted']} | "
            f"{stats['key_legal']} | {stats['unique_eligible_positions']} | {stats['clean_unique_eligible']} |"
        )
        for key in ("source_reported_query_count", "records_inspected", "source_filter_pass", "position_normalized", "key_extracted", "key_legal", "unique_eligible_positions", "clean_unique_eligible"):
            totals[key] += int(stats[key])
    lines.append(
        f"| TOTAL | {totals['source_reported_query_count']} | {totals['records_inspected']} | "
        f"{totals['source_filter_pass']} | {totals['position_normalized']} | {totals['key_extracted']} | "
        f"{totals['key_legal']} | {totals['unique_eligible_positions']} | {totals['clean_unique_eligible']} |"
    )
    lines.extend(["", "## Rejections", ""])
    header = "| Rejection reason | " + " | ".join(f"#{depth}" for depth in range(1, 11)) + " | Total |"
    sep = "|---" + "|---:" * 11 + "|"
    lines.extend([header, sep])
    for reason, row in payload["rejection_table"].items():
        values = " | ".join(str(row[f"MateIn{depth}"]) for depth in range(1, 11))
        lines.append(f"| `{reason}` | {values} | {row['Total']} |")
    lines.extend(["", "## Depth Feasibility", ""])
    for depth in range(1, 11):
        count = payload["depths"][str(depth)]["clean_unique_eligible"]
        if count >= 20:
            label = ">= 20 clean candidates"
        elif count >= 10:
            label = "10-19 clean candidates"
        elif count >= 1:
            label = "1-9 clean candidates"
        else:
            label = "0 clean candidates"
        lines.append(f"- MateIn{depth}: {count} - {label}")
    lines.extend(["", "## Manual Review Sample", ""])
    for row in payload["manual_review_sample"]:
        lines.append(f"- MateIn{row['mate_depth']} ID {row['yacpdb_id']}: `{row['key_token']}` -> `{row['key_move_uci']}`, pieces={row['piece_count']}, parser={row['parser_method']}")
    lines.extend(["", "## Integrity", "", "- No parser/normalizer semantic changes were made during the scan.", "- No final dataset was built or frozen.", "- No model, LLM, Stockfish, or Popeye inference was run."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_scan(output_root: Path, per_depth_cap: int, timeout: float) -> dict[str, Any]:
    """Run the bounded availability scan for MateDepth 1..10."""

    config = {
        "scan_version": SCAN_VERSION,
        "endpoint": "https://yacpdb.org/gateway/ql",
        "endpoint_classification": ENDPOINT_CLASSIFICATION,
        "query_order": QUERY_ORDER,
        "per_depth_cap": per_depth_cap,
        "page_size_observed": 100,
        "queries": {str(depth): query_for_depth(depth) for depth in range(1, 11)},
        "importer_version": YACPDB_IMPORTER_VERSION,
        "key_extractor_version": KEY_EXTRACTOR_VERSION,
        "server_side_filtering": "Stip only; conservative filters applied locally",
    }
    fingerprint = scan_fingerprint(config)
    processed_dir = output_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    state_path = processed_dir / "yacpdb_availability_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("scan_fingerprint") != fingerprint:
            raise RuntimeError("Incompatible YACPDB availability resume state")
    state_path.write_text(json.dumps({"scan_fingerprint": fingerprint, "config": config}, indent=2, sort_keys=True), encoding="utf-8")

    client = YacpdbClient(timeout=timeout, user_agent=DEFAULT_USER_AGENT, retries=2, sleep_seconds=1.0)
    lichess_keys = load_lichess_overlap_keys()
    depths: dict[str, dict[str, Any]] = {}
    manual_review: list[dict[str, Any]] = []
    completed_depths: list[int] = []
    incomplete_depths: list[int] = []
    for depth in range(1, 11):
        stats, review_rows, completed = scan_depth(client, output_root, depth, per_depth_cap, lichess_keys)
        depths[str(depth)] = stats
        manual_review.extend(review_rows)
        (completed_depths if completed else incomplete_depths).append(depth)
    status = "COMPLETE" if not incomplete_depths else "PARTIAL"
    payload = {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scan_fingerprint": fingerprint,
        "config": config,
        "completed_depths": completed_depths,
        "incomplete_depths": incomplete_depths,
        "depths": depths,
        "rejection_table": aggregate_rejections(depths),
        "manual_review_sample": manual_review,
    }
    availability_path = processed_dir / "yacpdb_availability.json"
    availability_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(payload, Path("generic_info/classic_benchmark/yacpdb/yacpdb_availability.md"))
    return payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--per-depth-cap", type=int, default=200)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for the bounded availability scan."""

    args = parse_args()
    payload = run_scan(Path(args.output_root), args.per_depth_cap, args.timeout)
    print(json.dumps({"status": payload["status"], "scan_fingerprint": payload["scan_fingerprint"]}, indent=2, sort_keys=True))
    return 0 if payload["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())

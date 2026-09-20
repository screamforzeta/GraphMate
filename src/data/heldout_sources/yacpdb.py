"""Conservative YACPDB importer for external held-out directmates.

YACPDB records are chess compositions. This module accepts only orthodox
directmates `#1` through `#10` with an unambiguous legal key move. It does not
retrieve live website data, run engines, or parse full solution trees in v1.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import pandas as pd

from src.data.heldout_classic import (
    FINAL_COLUMNS,
    HeldoutProblem,
    board_state_key,
    canonical_dataset_fingerprint,
    dataset_fingerprint,
    detect_lichess_overlap,
    load_lichess_overlap_keys,
    read_source_records,
    select_stratified,
    stable_heldout_id,
)
from src.data.heldout_sources.yacpdb_position import normalize_algebraic_position
from src.data.heldout_sources.yacpdb_solution import extract_structural_key, resolve_key_token, structurally_identify_key


YACPDB_IMPORTER_VERSION = "yacpdb_importer_v2"
KEY_EXTRACTOR_VERSION = "yacpdb_structural_key_extractor_v2"
DIRECTMATE_RE = re.compile(r"^#(?P<depth>10|[1-9])$")
MOVE_NUMBER_RE = re.compile(r"^\d+\.(?:\.\.)?")
POPEYE_MOVE_RE = re.compile(
    r"^(?P<piece>[KQRBSN])?(?P<src>[a-h][1-8])(?P<sep>[-x*])(?P<dst>[a-h][1-8])(?:=(?P<promo>[QRBSN]))?$"
)
PROMOTION_MAP = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "S": chess.KNIGHT, "N": chess.KNIGHT}
PIECE_MAP = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "S": chess.KNIGHT, "N": chess.KNIGHT}
UNSUPPORTED_STIP_PREFIXES = ("h#", "s#", "hs#", "ser-", "serh", "ser-s", "=")
UNSUPPORTED_CONDITION_TOKENS = {
    "circe",
    "anticirce",
    "andernach",
    "isardam",
    "madrasi",
    "grasshopper",
    "nightrider",
    "imitator",
    "take&make",
    "kobul",
    "patrol",
}


@dataclass(frozen=True)
class YacpdbCandidate:
    """Normalized YACPDB candidate before final sampling."""

    heldout_id: str
    source_problem_id: str
    fen: str
    mate_depth: int
    stipulation: str
    key_move_uci: str
    source_solution_raw: str
    composer: str | None
    source_reference: str | None
    publication_date: str | None
    source_url: str | None
    license: str | None
    raw_record: dict[str, Any]
    validation_status: str = "KEY_VALIDATED"
    key_validation_status: str = "KEY_VALIDATED"
    line_validation_status: str = "NOT_NORMALIZED"
    forced_mate_verification_status: str = "NOT_VERIFIED_ENGINE_NOT_USED"


def source_url_for_id(problem_id: str) -> str:
    """Return a stable YACPDB reference URL."""

    return f"https://www.yacpdb.org/# {problem_id}".replace("# ", "#")


def classify_stipulation(record: dict[str, Any]) -> tuple[int | None, str | None]:
    """Accept only orthodox directmate #1..#10 stipulations."""

    stipulation = str(record.get("stipulation") or record.get("stip") or "").strip()
    lowered = stipulation.lower()
    if not stipulation:
        return None, "MISSING_STIPULATION"
    if lowered.startswith(UNSUPPORTED_STIP_PREFIXES):
        return None, "UNSUPPORTED_NON_DIRECTMATE"
    match = DIRECTMATE_RE.fullmatch(stipulation)
    if not match:
        return None, "UNSUPPORTED_STIPULATION"
    conditions_text = " ".join(str(record.get(key) or "") for key in ("conditions", "condition", "fairy", "keywords")).lower()
    if any(token in conditions_text for token in UNSUPPORTED_CONDITION_TOKENS):
        return None, "UNSUPPORTED_FAIRY_CONDITION"
    if record.get("twins") or record.get("twin"):
        return None, "UNSUPPORTED_TWIN"
    return int(match.group("depth")), None


def strip_comments(text: str) -> str:
    """Remove Popeye brace comments while preserving surrounding tokens."""

    return re.sub(r"\{[^}]*\}", " ", text)


def first_solution_token(solution: str) -> tuple[str | None, str | None]:
    """Return the first actual solution token or a deterministic failure."""

    text = strip_comments(solution).strip()
    if not text:
        return None, "NO_SOLUTION"
    tokens = [token.strip() for token in text.replace("\n", " ").split() if token.strip()]
    if not tokens:
        return None, "NO_SOLUTION"
    index = 0
    if MOVE_NUMBER_RE.fullmatch(tokens[index]):
        index += 1
    elif re.match(r"^\d+\.", tokens[index]):
        tokens[index] = re.sub(r"^\d+\.(?:\.\.)?", "", tokens[index], count=1)
    if index >= len(tokens):
        return None, "NO_KEY_TOKEN"
    token = tokens[index].strip()
    if token.lower().startswith(("try", "set", "threat")):
        return None, "UNSUPPORTED_SOLUTION_SYNTAX"
    if "?" in token:
        return None, "TRY_OR_REFUTATION_TOKEN"
    return token.rstrip("!+#"), None


def parse_key_token(token: str, board: chess.Board) -> tuple[str | None, str | None]:
    """Convert one YACPDB/Popeye key token to UCI if unambiguous and legal."""

    uci, reason, _method = resolve_key_token(token, board)
    return uci, reason


def extract_key_uci(record: dict[str, Any], fen: str) -> tuple[str | None, str | None]:
    """Extract and validate the solution key from a YACPDB record."""

    raw_solution = str(record.get("solution") or record.get("source_solution_raw") or "").strip()
    try:
        board = chess.Board(fen)
    except Exception:
        return None, "INVALID_FEN"
    key, failure = extract_structural_key(raw_solution, board)
    if failure or key is None:
        return None, failure
    return key.key_move_uci, None


def record_fen(record: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    """Return FEN, side-to-move text, and deterministic failure if unsupported."""

    fen = str(record.get("fen") or record.get("FEN") or "").strip()
    if fen:
        try:
            board = chess.Board(fen)
        except Exception:
            return None, None, "INVALID_FEN"
        return fen, "white" if board.turn == chess.WHITE else "black", None
    normalized, reason = normalize_algebraic_position(record)
    if reason or normalized is None:
        return None, None, reason
    return normalized.fen, normalized.side_to_move, None


def source_reference(record: dict[str, Any]) -> str | None:
    """Return a compact source reference from flat or YACPDB nested metadata."""

    source = record.get("source") or record.get("publication")
    if isinstance(source, dict):
        parts = [str(source.get("name") or "").strip()]
        for key in ("issue", "problemid"):
            if source.get(key) not in (None, ""):
                parts.append(str(source[key]).strip())
        return " / ".join(part for part in parts if part)
    return source


def publication_date(record: dict[str, Any]) -> str | None:
    """Return publication date/year from flat or nested YACPDB source metadata."""

    if record.get("date") or record.get("year") or record.get("publication_date"):
        return record.get("date") or record.get("year") or record.get("publication_date")
    source = record.get("source")
    if isinstance(source, dict):
        date = source.get("date")
        if isinstance(date, dict):
            return "-".join(str(date[key]) for key in ("year", "month", "day") if key in date)
        if date:
            return str(date)
    return None


def composer(record: dict[str, Any]) -> str | None:
    """Return composer metadata from flat or YACPDB author fields."""

    if record.get("composer") or record.get("author"):
        return record.get("composer") or record.get("author")
    authors = record.get("authors")
    if isinstance(authors, list):
        return "; ".join(str(author) for author in authors)
    return None


def normalize_yacpdb_record(record: dict[str, Any]) -> tuple[YacpdbCandidate | None, str | None]:
    """Convert one raw YACPDB-like record into a key-validated candidate."""

    depth, reason = classify_stipulation(record)
    if reason:
        return None, reason
    fen, side_to_move, fen_error = record_fen(record)
    if fen_error or not fen:
        return None, fen_error or "MISSING_FEN"
    key_uci, key_error = extract_key_uci(record, fen)
    if key_error:
        return None, key_error
    problem_id = str(record.get("source_problem_id") or record.get("id") or record.get("yacpdb_id") or "").strip()
    if not problem_id:
        return None, "MISSING_YACPDB_ID"
    return (
        YacpdbCandidate(
            heldout_id=stable_heldout_id("YACPDB", problem_id),
            source_problem_id=problem_id,
            fen=fen,
            mate_depth=int(depth),
            stipulation=f"#{depth}",
            key_move_uci=str(key_uci),
            source_solution_raw=str(record.get("solution") or record.get("source_solution_raw") or ""),
            composer=composer(record),
            source_reference=source_reference(record),
            publication_date=publication_date(record),
            source_url=record.get("source_url") or source_url_for_id(problem_id),
            license=record.get("license") or record.get("provenance_license"),
            raw_record=record,
        ),
        None,
    )


def candidate_to_problem(candidate: YacpdbCandidate, dataset_version: str) -> HeldoutProblem:
    """Convert a key-validated YACPDB candidate into the generic held-out schema."""

    return HeldoutProblem(
        heldout_id=candidate.heldout_id,
        source="YACPDB",
        source_problem_id=candidate.source_problem_id,
        fen=candidate.fen,
        side_to_move="white" if chess.Board(candidate.fen).turn == chess.WHITE else "black",
        mate_depth=candidate.mate_depth,
        stipulation=candidate.stipulation,
        problem_type="directmate",
        key_move_uci=candidate.key_move_uci,
        solution_uci=[],
        solution_plies=0,
        source_solution_raw=candidate.source_solution_raw,
        solution_tree=None,
        principal_line_uci=None,
        source_url=candidate.source_url,
        source_reference=candidate.source_reference,
        author=candidate.composer,
        publication=candidate.source_reference,
        publication_date=candidate.publication_date,
        license=candidate.license,
        validation_status="KEY_VALIDATED",
        dataset_version=dataset_version,
        key_validation_status="KEY_VALIDATED",
        line_validation_status="NOT_NORMALIZED",
        forced_mate_verification_status="NOT_VERIFIED_ENGINE_NOT_USED",
        source_metadata=candidate.raw_record,
    )


def scan_availability(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Scan YACPDB availability by MateDepth with conservative filters."""

    by_depth = {str(depth): Counter() for depth in range(1, 11)}
    rejection_reasons = Counter()
    candidates: list[YacpdbCandidate] = []
    for record in records:
        raw_stip = str(record.get("stipulation") or record.get("stip") or "")
        match = DIRECTMATE_RE.fullmatch(raw_stip.strip())
        depth_key = match.group("depth") if match else "unknown"
        if depth_key in by_depth:
            by_depth[depth_key]["discovered"] += 1
        candidate, reason = normalize_yacpdb_record(record)
        if reason:
            rejection_reasons[reason] += 1
            if depth_key in by_depth:
                by_depth[depth_key][reason] += 1
            continue
        candidates.append(candidate)
        key = str(candidate.mate_depth)
        by_depth[key]["orthodox_filter_pass"] += 1
        by_depth[key]["position_valid"] += 1
        by_depth[key]["key_extractable"] += 1
        by_depth[key]["key_legal"] += 1
        by_depth[key]["eligible"] += 1
    return {
        "importer_version": YACPDB_IMPORTER_VERSION,
        "key_extractor_version": KEY_EXTRACTOR_VERSION,
        "availability_by_depth": {f"MateIn{depth}": dict(by_depth[str(depth)]) for depth in range(1, 11)},
        "rejection_reasons": dict(rejection_reasons),
        "eligible_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def write_availability(records: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    """Write availability JSON and return the payload."""

    output_root.mkdir(parents=True, exist_ok=True)
    payload = scan_availability(records)
    path = output_root / "processed" / "yacpdb_availability.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def build_yacpdb_dataset(
    records: list[dict[str, Any]],
    output_root: Path,
    dataset_version: str,
    per_depth: int | None,
    seed: int,
) -> dict[str, Any]:
    """Build a VALIDATED_NOT_FROZEN YACPDB candidate dataset from local records."""

    availability = scan_availability(records)
    candidates = [
        YacpdbCandidate(**candidate)
        for candidate in availability["candidates"]
    ]
    seen_ids: set[str] = set()
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    lichess_keys = load_lichess_overlap_keys()
    accepted: list[HeldoutProblem] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        reasons = []
        if candidate.source_problem_id in seen_ids:
            reasons.append("duplicate_yacpdb_id")
        if candidate.fen in seen_exact:
            reasons.append("duplicate_exact_fen")
        key = board_state_key(candidate.fen)
        if key in seen_normalized:
            reasons.append("duplicate_normalized_position")
        reasons.extend(detect_lichess_overlap(candidate_to_problem(candidate, dataset_version), lichess_keys))
        if reasons:
            rejected.append({"source_problem_id": candidate.source_problem_id, "reasons": reasons})
            continue
        seen_ids.add(candidate.source_problem_id)
        seen_exact.add(candidate.fen)
        seen_normalized.add(key)
        accepted.append(candidate_to_problem(candidate, dataset_version))
    selected = select_stratified(accepted, per_depth, seed)
    final_dir = output_root / "final"
    processed_dir = output_root / "processed"
    final_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    rows = [problem.to_row() for problem in selected]
    pd.DataFrame(rows, columns=FINAL_COLUMNS).to_csv(final_dir / "heldout_classic.csv", index=False)
    (processed_dir / "yacpdb_rejected_records.json").write_text(json.dumps(rejected, indent=2, sort_keys=True), encoding="utf-8")
    manual_review = manual_review_sample(selected)
    (processed_dir / "yacpdb_manual_review.json").write_text(json.dumps(manual_review, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "dataset_name": "heldout_classic_mate_in_n",
        "dataset_version": dataset_version,
        "lifecycle_status": "VALIDATED",
        "source": "YACPDB",
        "yacpdb_retrieval_method": "local exported/cache records; no live API assumed",
        "source_query_filter": "orthodox directmates #1..#10 only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_candidate_count": len(records),
        "availability": {key: value for key, value in availability.items() if key != "candidates"},
        "selected_count": len(selected),
        "selected_ids": [problem.source_problem_id for problem in selected],
        "final_mate_depth_distribution": dict(Counter(str(problem.mate_depth) for problem in selected)),
        "duplicate_and_overlap_rejections": rejected,
        "schema_version": "heldout_classic_v1",
        "parser_version": YACPDB_IMPORTER_VERSION,
        "key_extractor_version": KEY_EXTRACTOR_VERSION,
        "validator_version": "heldout_classic_validator_v1",
        "seed": seed,
        "dataset_fingerprint": canonical_dataset_fingerprint(rows),
        "license_status": "UNCLEAR",
        "model_outputs_used_for_selection": False,
        "GNN_inference_performed": False,
        "LLM_inference_performed": False,
        "forced_mate_verified": False,
    }
    (final_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def manual_review_sample(problems: list[HeldoutProblem], per_depth: int = 3) -> list[dict[str, Any]]:
    """Return deterministic manual-review examples per MateDepth."""

    sample = []
    by_depth: dict[int, list[HeldoutProblem]] = defaultdict(list)
    for problem in problems:
        by_depth[problem.mate_depth].append(problem)
    for depth in sorted(by_depth):
        for problem in sorted(by_depth[depth], key=lambda item: item.heldout_id)[:per_depth]:
            sample.append(
                {
                    "source_problem_id": problem.source_problem_id,
                    "composer": problem.author,
                    "source_reference": problem.source_reference,
                    "stipulation": problem.stipulation,
                    "fen": problem.fen,
                    "board": str(chess.Board(problem.fen)),
                    "key_move_uci": problem.key_move_uci,
                    "source_solution_raw": problem.source_solution_raw,
                    "key_validation_status": problem.key_validation_status,
                    "line_validation_status": problem.line_validation_status,
                    "forced_mate_verification_status": problem.forced_mate_verification_status,
                }
            )
    return sample


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read local YACPDB export/cache records."""

    return read_source_records(path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["discover", "availability", "build"], required=True)
    parser.add_argument("--source", required=True, help="Local YACPDB export/cache JSON/JSONL/CSV.")
    parser.add_argument("--output-root", default="data/heldout_classic")
    parser.add_argument("--dataset-version", default="yacpdb_v1")
    parser.add_argument("--per-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    records = read_records(Path(args.source))
    output_root = Path(args.output_root)
    if args.mode == "discover":
        raw_dir = output_root / "raw" / "yacpdb"
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "mode": "discover",
            "source": str(args.source),
            "record_count": len(records),
            "retrieval_method": "local export/cache import; no live YACPDB API assumed",
            "next_step": "run --mode availability",
        }
        (raw_dir / "discovery_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.mode == "availability":
        payload = write_availability(records, output_root)
        print(json.dumps({key: value for key, value in payload.items() if key != "candidates"}, indent=2, sort_keys=True))
        return 0
    manifest = build_yacpdb_dataset(records, output_root, args.dataset_version, args.per_depth, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

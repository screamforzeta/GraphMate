"""Build and validate an external held-out classic Mate-in-N dataset.

The dataset is independent from Lichess train/validation/test data. This module
only imports source-provided problems, validates chess semantics, detects
duplicates/contamination, and writes canonical artifacts. It never runs models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import pandas as pd


DATASET_NAME = "heldout_classic_mate_in_n"
SCHEMA_VERSION = "heldout_classic_v1"
VALIDATOR_VERSION = "heldout_classic_validator_v1"
DEFAULT_DATA_ROOT = Path("data/heldout_classic")
LICHESS_SPLITS = {
    "train": Path("data/final/puzzles/train.csv"),
    "val": Path("data/final/puzzles/val.csv"),
    "test": Path("data/final/puzzles/test.csv"),
}
FINAL_COLUMNS = [
    "heldout_id",
    "source",
    "source_problem_id",
    "fen",
    "side_to_move",
    "mate_depth",
    "stipulation",
    "problem_type",
    "key_move_uci",
    "solution_uci",
    "solution_plies",
    "source_solution_raw",
    "solution_tree",
    "principal_line_uci",
    "source_url",
    "source_reference",
    "author",
    "publication",
    "publication_date",
    "license",
    "validation_status",
    "dataset_version",
    "key_validation_status",
    "line_validation_status",
    "forced_mate_verification_status",
    "source_metadata_json",
]


@dataclass(frozen=True)
class HeldoutProblem:
    """Canonical external held-out problem."""

    heldout_id: str
    source: str
    source_problem_id: str
    fen: str
    side_to_move: str
    mate_depth: int
    solution_uci: list[str]
    solution_plies: int
    stipulation: str | None = None
    problem_type: str | None = None
    key_move_uci: str | None = None
    source_solution_raw: str | None = None
    solution_tree: dict[str, Any] | None = None
    principal_line_uci: list[str] | None = None
    source_url: str | None = None
    source_reference: str | None = None
    author: str | None = None
    publication: str | None = None
    publication_date: str | None = None
    license: str | None = None
    validation_status: str = "LINE_VALIDATED"
    dataset_version: str = "v0"
    key_validation_status: str = "NOT_APPLICABLE"
    line_validation_status: str = "LINE_VALIDATED"
    forced_mate_verification_status: str = "NOT_VERIFIED_ENGINE_NOT_USED"
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Return a deterministic CSV row."""

        payload = asdict(self)
        payload["solution_uci"] = json.dumps(self.solution_uci, separators=(",", ":"))
        payload["solution_tree"] = json.dumps(self.solution_tree, sort_keys=True, separators=(",", ":")) if self.solution_tree is not None else ""
        payload["principal_line_uci"] = json.dumps(self.principal_line_uci, separators=(",", ":")) if self.principal_line_uci is not None else ""
        payload["source_metadata_json"] = json.dumps(self.source_metadata, sort_keys=True, separators=(",", ":"))
        payload.pop("source_metadata")
        return payload


def board_state_key(fen: str) -> str:
    """Return a normalized board-state key preserving legal state fields.

    The key includes board placement, side to move, castling rights, and
    en-passant square. It excludes halfmove/fullmove counters.
    """

    board = chess.Board(fen)
    return " ".join(board.fen().split()[:4])


def dataset_fingerprint(path: Path) -> str:
    """Return SHA256 for one file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_dataset_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Return deterministic fingerprint for canonical final rows."""

    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_solution_moves(raw: Any) -> list[str]:
    """Parse a source solution field into UCI move strings."""

    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    text = "" if raw is None else str(raw).strip()
    if not text:
        return []
    try:
        decoded = json.loads(text)
        if isinstance(decoded, list):
            return [str(item).strip() for item in decoded if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [token.strip() for token in text.replace(",", " ").split() if token.strip()]


def side_to_move_from_fen(fen: str) -> str:
    """Return `white` or `black` from a FEN."""

    return "white" if chess.Board(fen).turn == chess.WHITE else "black"


def stable_heldout_id(source: str, source_problem_id: str) -> str:
    """Return a stable held-out ID from source metadata."""

    digest = hashlib.sha256(f"{source}:{source_problem_id}".encode("utf-8")).hexdigest()[:12]
    return f"heldout_{digest}"


def normalize_source_record(record: dict[str, Any], dataset_version: str) -> tuple[HeldoutProblem | None, str | None]:
    """Normalize one raw source record into the canonical schema."""

    source = str(record.get("source") or record.get("source_name") or "").strip()
    source_problem_id = str(record.get("source_problem_id") or record.get("problem_id") or record.get("id") or "").strip()
    fen = str(record.get("fen") or record.get("FEN") or "").strip()
    mate_depth_raw = record.get("mate_depth") or record.get("MateDepth") or record.get("mate_in")
    solution = parse_solution_moves(record.get("solution_uci") or record.get("solution") or record.get("moves"))
    if not source:
        return None, "missing_source"
    if not source_problem_id:
        return None, "missing_source_problem_id"
    if not fen:
        return None, "missing_fen"
    if mate_depth_raw is None or str(mate_depth_raw).strip() == "":
        return None, "missing_mate_depth"
    try:
        mate_depth = int(mate_depth_raw)
    except (TypeError, ValueError):
        return None, "invalid_mate_depth"
    if mate_depth < 1:
        return None, "invalid_mate_depth"
    problem = HeldoutProblem(
        heldout_id=str(record.get("heldout_id") or stable_heldout_id(source, source_problem_id)),
        source=source,
        source_problem_id=source_problem_id,
        fen=fen,
        side_to_move=str(record.get("side_to_move") or "").strip() or side_to_move_from_fen(fen),
        mate_depth=mate_depth,
        solution_uci=solution,
        solution_plies=len(solution),
        source_url=record.get("source_url"),
        source_reference=record.get("source_reference") or record.get("publication"),
        author=record.get("author") or record.get("composer"),
        publication=record.get("publication") or record.get("source_publication"),
        publication_date=record.get("publication_date") or record.get("year"),
        license=record.get("license") or record.get("provenance_license"),
        dataset_version=dataset_version,
        source_metadata={key: value for key, value in record.items() if key not in FINAL_COLUMNS},
    )
    return problem, None


def validate_problem(problem: HeldoutProblem) -> list[str]:
    """Validate one canonical problem's chess semantics."""

    failures: list[str] = []
    try:
        board = chess.Board(problem.fen)
    except Exception:
        return ["invalid_fen"]
    if problem.side_to_move not in {"white", "black"}:
        failures.append("invalid_side_to_move")
    elif problem.side_to_move != side_to_move_from_fen(problem.fen):
        failures.append("side_to_move_mismatch")
    if not problem.solution_uci:
        failures.append("missing_solution")
        return failures
    solver_color = board.turn
    solver_moves = 0
    for index, move_uci in enumerate(problem.solution_uci):
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            failures.append(f"invalid_uci_at_ply_{index + 1}")
            return failures
        if move not in board.legal_moves:
            failures.append(f"illegal_move_at_ply_{index + 1}")
            return failures
        moving_color = board.turn
        board.push(move)
        if moving_color == solver_color:
            solver_moves += 1
    if not board.is_checkmate():
        failures.append("line_not_checkmate")
    if solver_moves != problem.mate_depth:
        failures.append("mate_depth_mismatch")
    if len(problem.solution_uci) < 2 * problem.mate_depth - 1:
        failures.append("truncated_solution_line")
    final_mover_color = not board.turn
    if final_mover_color != solver_color:
        failures.append("final_mating_move_not_by_solver")
    return failures


def read_source_records(path: Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, or CSV source records."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("problems", [])
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported source format: {path.suffix}")


def load_lichess_overlap_keys(split_paths: dict[str, Path] = LICHESS_SPLITS) -> dict[str, dict[str, set[str]]]:
    """Load exact and normalized FEN keys for existing Lichess splits."""

    result: dict[str, dict[str, set[str]]] = {}
    for split, path in split_paths.items():
        exact: set[str] = set()
        normalized: set[str] = set()
        original_exact: set[str] = set()
        if path.exists():
            df = pd.read_csv(path)
            if "FEN" in df.columns:
                exact.update(str(value) for value in df["FEN"].dropna())
                normalized.update(board_state_key(str(value)) for value in df["FEN"].dropna())
            if "OriginalFEN" in df.columns:
                original_exact.update(str(value) for value in df["OriginalFEN"].dropna())
        result[split] = {
            "exact": exact,
            "normalized": normalized,
            "original_exact": original_exact,
        }
    return result


def detect_lichess_overlap(problem: HeldoutProblem, lichess_keys: dict[str, dict[str, set[str]]]) -> list[str]:
    """Return deterministic Lichess overlap reasons for one held-out problem."""

    reasons = []
    key = board_state_key(problem.fen)
    for split, keys in lichess_keys.items():
        if problem.fen in keys["exact"]:
            reasons.append(f"lichess_{split}_solver_fen_exact_overlap")
        if key in keys["normalized"]:
            reasons.append(f"lichess_{split}_solver_fen_normalized_overlap")
        if problem.fen in keys["original_exact"]:
            reasons.append(f"lichess_{split}_original_fen_exact_overlap")
    return reasons


def select_stratified(problems: list[HeldoutProblem], per_depth: int | None, seed: int) -> list[HeldoutProblem]:
    """Deterministically sample up to `per_depth` problems per MateDepth."""

    if per_depth is None:
        return sorted(problems, key=lambda item: item.heldout_id)
    rng = random.Random(seed)
    selected = []
    by_depth: dict[int, list[HeldoutProblem]] = defaultdict(list)
    for problem in problems:
        by_depth[problem.mate_depth].append(problem)
    for depth in sorted(by_depth):
        bucket = sorted(by_depth[depth], key=lambda item: item.heldout_id)
        if len(bucket) > per_depth:
            bucket = sorted(rng.sample(bucket, per_depth), key=lambda item: item.heldout_id)
        selected.extend(bucket)
    return sorted(selected, key=lambda item: (item.mate_depth, item.heldout_id))


def build_heldout_dataset(
    source_path: Path,
    data_root: Path = DEFAULT_DATA_ROOT,
    dataset_version: str = "v1",
    source_name: str | None = None,
    source_url: str | None = None,
    source_license: str | None = None,
    per_depth: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Import, validate, decontaminate, and write held-out dataset artifacts."""

    raw_records = read_source_records(source_path)
    final_dir = data_root / "final"
    processed_dir = data_root / "processed"
    final_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    lichess_keys = load_lichess_overlap_keys()
    accepted_candidates: list[HeldoutProblem] = []
    rejected: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    for index, raw in enumerate(raw_records):
        enriched = dict(raw)
        if source_name and not enriched.get("source"):
            enriched["source"] = source_name
        if source_url and not enriched.get("source_url"):
            enriched["source_url"] = source_url
        if source_license and not enriched.get("license"):
            enriched["license"] = source_license
        problem, error = normalize_source_record(enriched, dataset_version)
        if error:
            rejected.append({"row_index": index, "reason": error, "raw": raw})
            continue
        failures = validate_problem(problem)
        if problem.fen in seen_exact:
            failures.append("duplicate_exact_fen")
        normalized_key = board_state_key(problem.fen)
        if normalized_key in seen_normalized:
            failures.append("duplicate_normalized_position")
        failures.extend(detect_lichess_overlap(problem, lichess_keys))
        if failures:
            rejected.append({"heldout_id": problem.heldout_id, "reason": ";".join(sorted(set(failures))), "raw": raw})
            continue
        seen_exact.add(problem.fen)
        seen_normalized.add(normalized_key)
        accepted_candidates.append(problem)
    selected = select_stratified(accepted_candidates, per_depth, seed)
    rows = [problem.to_row() for problem in selected]
    final_csv = final_dir / "heldout_classic.csv"
    pd.DataFrame(rows, columns=FINAL_COLUMNS).to_csv(final_csv, index=False)
    rejected_path = processed_dir / "rejected_records.jsonl"
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fingerprint = canonical_dataset_fingerprint(rows)
    rejection_counts = Counter(row["reason"] for row in rejected)
    depth_distribution = Counter(str(problem.mate_depth) for problem in selected)
    manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_version": dataset_version,
        "dataset_status": "VALIDATED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(source_path),
            "name": source_name,
            "url": source_url,
            "license": source_license,
            "fingerprint": dataset_fingerprint(source_path),
        },
        "schema_version": SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "selection_procedure": "validate_all_then_optional_stratified_sample_by_mate_depth",
        "random_seed": seed,
        "per_depth": per_depth,
        "raw_count": len(raw_records),
        "accepted_count": len(selected),
        "candidate_valid_count": len(accepted_candidates),
        "rejected_count": len(rejected),
        "rejection_reasons": dict(rejection_counts),
        "duplicate_policy": "exact FEN and normalized board-state key (first four FEN fields)",
        "lichess_overlap_policy": "compare held-out FEN against solver-facing FEN exact/normalized and OriginalFEN exact for train/val/test",
        "mate_depth_distribution": dict(sorted(depth_distribution.items(), key=lambda item: int(item[0]))),
        "dataset_fingerprint": fingerprint,
        "final_csv": str(final_csv),
        "rejected_records": str(rejected_path),
        "model_outputs_used_for_selection": False,
        "dataset_used_for_model_tuning": False,
        "forced_mate_verified": False,
        "forced_mate_note": "Legal principal line ending in checkmate is validated; forced mate against all defenses is not engine-verified.",
    }
    manifest_path = final_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="External public source file (.json/.jsonl/.csv).")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset-version", default="v1")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--source-license", default=None)
    parser.add_argument("--per-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    manifest = build_heldout_dataset(
        source_path=Path(args.source),
        data_root=Path(args.data_root),
        dataset_version=args.dataset_version,
        source_name=args.source_name,
        source_url=args.source_url,
        source_license=args.source_license,
        per_depth=args.per_depth,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

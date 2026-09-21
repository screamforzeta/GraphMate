"""Popeye adapter for independent YACPDB forced-mate verification.

The adapter reads the fixed candidate dataset, converts canonical FEN positions
to documented Popeye input, runs an external Popeye executable when available,
and writes separate verification artifacts. It never mutates the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess


ADAPTER_VERSION = "popeye_adapter_v1"
OUTPUT_PARSER_VERSION = "popeye_output_parser_v1"
EXPECTED_DATASET_FINGERPRINT = "bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a"
DEFAULT_DATASET_DIR = Path("data/heldout_classic/final/yacpdb_classic_v1")
DEFAULT_VERIFICATION_ROOT = Path("data/heldout_classic/verification/yacpdb_classic_v1/popeye")
POPEYE_EXECUTABLE_CANDIDATES = ("popeye", "py", "py.exe")
STANDARD_PIECES = set("PNBRQKpnbrqk")
STATUS_VERIFIED = "VERIFIED"
STATUS_FAILED = "FAILED"
STATUS_UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class PopeyeRunConfig:
    """Configuration identity for one Popeye verification run."""

    dataset_fingerprint: str
    executable_path: str
    popeye_version: str
    popeye_banner: str | None
    timeout_seconds: float
    adapter_version: str = ADAPTER_VERSION
    output_parser_version: str = OUTPUT_PARSER_VERSION
    invocation_options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessResult:
    """Raw process execution result."""

    stdout: str
    stderr: str
    return_code: int | None
    runtime_seconds: float
    timed_out: bool = False


@dataclass(frozen=True)
class ParsedPopeyeOutput:
    """Parsed verification and key information from Popeye output."""

    verification_status: str
    verification_reason: str
    forced_mate_verified: bool
    verified_keys_uci: list[str]
    output_supported: bool = True


@dataclass(frozen=True)
class PopeyeIdentity:
    """Stable Popeye executable identity extracted from a real banner."""

    version: str
    banner: str


def sha256_file(path: Path) -> str:
    """Return SHA256 hash for a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_fingerprint(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA256 over stable JSON content."""

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def locate_popeye(explicit_path: str | None = None) -> str | None:
    """Return a Popeye executable path if one is available."""

    if explicit_path:
        candidate = Path(explicit_path)
        return str(candidate) if candidate.exists() else None
    for name in POPEYE_EXECUTABLE_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


POPEYE_BANNER_RE = re.compile(r"^(?P<banner>Popeye\b.*?\b(?P<version>v\d+(?:\.\d+)+)\b.*)$", re.MULTILINE)


def parse_popeye_banner(text: str) -> PopeyeIdentity:
    """Extract Popeye version identity from a real stdout/stderr banner."""

    if not text or not text.strip():
        raise RuntimeError("Popeye banner detection failed: empty output")
    match = POPEYE_BANNER_RE.search(text)
    if not match:
        raise RuntimeError("Popeye banner detection failed: missing valid Popeye banner")
    return PopeyeIdentity(version=match.group("version"), banner=match.group("banner").strip())


def popeye_identity_from_output(stdout: str, stderr: str = "") -> PopeyeIdentity:
    """Return stable Popeye identity from process output."""

    return parse_popeye_banner("\n".join(part for part in (stdout, stderr) if part))


def popeye_identity(executable_path: str, timeout_seconds: float = 10.0) -> PopeyeIdentity:
    """Identify Popeye by running a supported stdin problem and parsing banner."""

    probe = "\n".join(
        [
            "BeginProblem",
            "Option NoBoard",
            "Stipulation #1",
            "Forsyth 7k/6Q1/6K1/8/8/8/8/8",
            "EndProblem",
            "",
        ]
    )
    result = run_popeye_process(executable_path, probe, timeout_seconds)
    if result.timed_out:
        raise RuntimeError("Popeye banner detection failed: timeout")
    if result.return_code not in (0, None):
        raise RuntimeError(f"Popeye banner detection failed: return code {result.return_code}")
    return popeye_identity_from_output(result.stdout, result.stderr)


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Load the candidate dataset manifest."""

    path = dataset_dir / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset_fingerprint(dataset_dir: Path, expected_fingerprint: str) -> dict[str, Any]:
    """Load manifest and fail closed if the semantic fingerprint differs."""

    manifest = load_manifest(dataset_dir)
    actual = manifest.get("dataset_fingerprint")
    if actual != expected_fingerprint:
        raise RuntimeError(f"Dataset fingerprint mismatch: {actual} != {expected_fingerprint}")
    return manifest


def load_dataset_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load canonical candidate dataset rows."""

    path = dataset_dir / "dataset.jsonl"
    if not path.exists():
        raise RuntimeError(f"Missing dataset: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_file_hashes(dataset_dir: Path) -> dict[str, str]:
    """Hash canonical dataset files that must remain byte-identical."""

    names = ("dataset.jsonl", "dataset.csv", "selected_ids.json")
    return {name: sha256_file(dataset_dir / name) for name in names}


def assert_canonical_hashes_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    """Fail closed if canonical dataset bytes changed."""

    if before != after:
        raise RuntimeError("Canonical dataset files were mutated during verification")


def fen_to_popeye_forsyth(fen: str) -> str:
    """Convert canonical FEN to Popeye's orthodox Forsyth board field."""

    board = chess.Board(fen)
    if board.turn != chess.WHITE:
        raise ValueError("UNVERIFIABLE_INPUT_SEMANTICS: Popeye directmate adapter expects white to move")
    if board.castling_rights:
        raise ValueError("UNVERIFIABLE_INPUT_SEMANTICS: castling rights are not represented by this adapter")
    if board.ep_square is not None:
        raise ValueError("UNVERIFIABLE_INPUT_SEMANTICS: en-passant state is not represented by this adapter")
    placement = board.board_fen()
    if any(char.isalpha() and char not in STANDARD_PIECES for char in placement):
        raise ValueError("UNVERIFIABLE_INPUT_SEMANTICS: non-orthodox piece in FEN")
    return placement.replace("n", "s").replace("N", "S")


def popeye_input_for_row(row: dict[str, Any]) -> str:
    """Return documented Popeye input for one orthodox directmate row."""

    forsyth = fen_to_popeye_forsyth(row["fen"])
    depth = int(row["mate_depth"])
    if depth < 1:
        raise ValueError("UNVERIFIABLE_INPUT_SEMANTICS: mate depth must be positive")
    return "\n".join(
        [
            "BeginProblem",
            f'Remark heldout_id={row["heldout_id"]} source_problem_id={row["source_problem_id"]}',
            "Option NoBoard",
            f"Stipulation #{depth}",
            f"Forsyth {forsyth}",
            "EndProblem",
            "",
        ]
    )


MOVE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[KQRBSN])?[a-h][1-8][-x*][a-h][1-8](?:=[QRBSN])?[+#]?(?![A-Za-z0-9])"
)


def popeye_token_to_uci(token: str, board: chess.Board) -> str | None:
    """Convert one Popeye move token to UCI if legal and unambiguous."""

    cleaned = token.strip().rstrip("+#")
    match = re.fullmatch(r"(?P<piece>[KQRBSN])?(?P<src>[a-h][1-8])[-x*](?P<dst>[a-h][1-8])(?:=(?P<promo>[QRBSN]))?", cleaned)
    if not match:
        return None
    promo_map = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "S": chess.KNIGHT, "N": chess.KNIGHT}
    move = chess.Move.from_uci(match.group("src") + match.group("dst") + (chess.piece_symbol(promo_map[match.group("promo")]) if match.group("promo") else ""))
    if move in board.legal_moves:
        return move.uci()
    return None


def parse_popeye_output(stdout: str, stderr: str, fen: str, source_key_uci: str) -> ParsedPopeyeOutput:
    """Parse Popeye output conservatively and extract solution keys."""

    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    if "no solution" in lowered or "no solutions" in lowered or "0 solutions" in lowered:
        return ParsedPopeyeOutput(STATUS_FAILED, "NO_SOLUTION", False, [])
    if "error" in lowered and "solution" not in lowered:
        return ParsedPopeyeOutput(STATUS_FAILED, "POPEYE_ERROR", False, [])
    if "solution finished" not in lowered and "solution" not in lowered:
        return ParsedPopeyeOutput(STATUS_UNVERIFIABLE, "UNSUPPORTED_POPEYE_OUTPUT", False, [], output_supported=False)
    board = chess.Board(fen)
    keys: list[str] = []
    for token in MOVE_TOKEN_RE.findall(text):
        uci = popeye_token_to_uci(token, board)
        if uci and uci not in keys:
            keys.append(uci)
    if not keys:
        return ParsedPopeyeOutput(STATUS_UNVERIFIABLE, "OUTPUT_PARSE_ERROR", False, [])
    if len(keys) == 1 and keys[0] == source_key_uci:
        return ParsedPopeyeOutput(STATUS_VERIFIED, "VERIFIED_UNIQUE_KEY_MATCH", True, keys)
    if source_key_uci in keys:
        return ParsedPopeyeOutput(STATUS_VERIFIED, "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY", True, keys)
    return ParsedPopeyeOutput(STATUS_VERIFIED, "VERIFIED_KEY_MISMATCH", True, keys)


def run_popeye_process(executable_path: str, input_text: str, timeout_seconds: float) -> ProcessResult:
    """Run Popeye once and capture process output."""

    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [executable_path],
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
        return ProcessResult(completed.stdout, completed.stderr, completed.returncode, time.perf_counter() - start)
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(exc.stdout or "", exc.stderr or "", None, time.perf_counter() - start, timed_out=True)
    except OSError as exc:
        return ProcessResult("", str(exc), None, time.perf_counter() - start)


def config_fingerprint(config: PopeyeRunConfig) -> str:
    """Return deterministic verification configuration fingerprint."""

    return semantic_fingerprint(asdict(config))


def result_from_process(
    row: dict[str, Any],
    config: PopeyeRunConfig,
    config_fp: str,
    process_result: ProcessResult,
    input_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    """Convert process and parser outputs into one structured result row."""

    if process_result.timed_out:
        status = STATUS_FAILED
        reason = "TIMEOUT"
        parsed = ParsedPopeyeOutput(status, reason, False, [])
    elif process_result.return_code not in (0, None):
        status = STATUS_FAILED
        reason = "POPEYE_ERROR"
        parsed = ParsedPopeyeOutput(status, reason, False, [])
    else:
        parsed = parse_popeye_output(process_result.stdout, process_result.stderr, row["fen"], row["key_move_uci"])
    return {
        "heldout_id": row["heldout_id"],
        "source_problem_id": row["source_problem_id"],
        "mate_depth": row["mate_depth"],
        "canonical_fen": row["fen"],
        "source_key_move_uci": row["key_move_uci"],
        "popeye_version": config.popeye_version,
        "popeye_banner": config.popeye_banner,
        "timeout_seconds": config.timeout_seconds,
        "runtime_seconds": process_result.runtime_seconds,
        "return_code": process_result.return_code,
        "verification_status": parsed.verification_status,
        "verification_reason": parsed.verification_reason,
        "forced_mate_verified": parsed.forced_mate_verified,
        "verified_keys_uci": parsed.verified_keys_uci,
        "verified_key_count": len(parsed.verified_keys_uci),
        "source_key_in_verified_keys": row["key_move_uci"] in parsed.verified_keys_uci,
        "input_artifact": str(input_path),
        "stdout_artifact": str(stdout_path),
        "stderr_artifact": str(stderr_path),
        "dataset_fingerprint": config.dataset_fingerprint,
        "verification_config_fingerprint": config_fp,
    }


def summarize_results(results: list[dict[str, Any]], config: PopeyeRunConfig, config_fp: str) -> dict[str, Any]:
    """Aggregate Popeye verification results."""

    runtimes = [float(row["runtime_seconds"]) for row in results]
    by_depth: dict[str, dict[str, Any]] = {}
    for depth in range(1, 11):
        depth_rows = [row for row in results if int(row["mate_depth"]) == depth]
        by_depth[str(depth)] = {
            "total": len(depth_rows),
            "verified_forced_mate": sum(bool(row["forced_mate_verified"]) for row in depth_rows),
            "failed": sum(row["verification_status"] == STATUS_FAILED for row in depth_rows),
            "unverifiable": sum(row["verification_status"] == STATUS_UNVERIFIABLE for row in depth_rows),
            "timeouts": sum(row["verification_reason"] == "TIMEOUT" for row in depth_rows),
            "popeye_errors": sum(row["verification_reason"] == "POPEYE_ERROR" for row in depth_rows),
            "unique_key_match": sum(row["verification_reason"] == "VERIFIED_UNIQUE_KEY_MATCH" for row in depth_rows),
            "multiple_keys_including_source": sum(row["verification_reason"] == "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY" for row in depth_rows),
            "key_mismatch": sum(row["verification_reason"] == "VERIFIED_KEY_MISMATCH" for row in depth_rows),
        }
    return {
        "dataset_fingerprint": config.dataset_fingerprint,
        "verification_config_fingerprint": config_fp,
        "popeye_version": config.popeye_version,
        "popeye_banner": config.popeye_banner,
        "timeout_seconds": config.timeout_seconds,
        "total": len(results),
        "verified_forced_mate": sum(bool(row["forced_mate_verified"]) for row in results),
        "failed": sum(row["verification_status"] == STATUS_FAILED for row in results),
        "unverifiable": sum(row["verification_status"] == STATUS_UNVERIFIABLE for row in results),
        "timeouts": sum(row["verification_reason"] == "TIMEOUT" for row in results),
        "popeye_errors": sum(row["verification_reason"] == "POPEYE_ERROR" for row in results),
        "unique_key_match": sum(row["verification_reason"] == "VERIFIED_UNIQUE_KEY_MATCH" for row in results),
        "multiple_keys_including_source": sum(row["verification_reason"] == "VERIFIED_MULTIPLE_KEYS_INCLUDES_SOURCE_KEY" for row in results),
        "key_mismatch": sum(row["verification_reason"] == "VERIFIED_KEY_MISMATCH" for row in results),
        "by_mate_depth": by_depth,
        "runtime": {
            "total": sum(runtimes) if runtimes else 0.0,
            "mean": statistics.mean(runtimes) if runtimes else 0.0,
            "median": statistics.median(runtimes) if runtimes else 0.0,
            "max": max(runtimes) if runtimes else 0.0,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_markdown_report(
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    path: Path,
    executable_path: str | None,
    install_note: str,
    canonical_mutated: bool,
) -> None:
    """Write human-readable Popeye verification report."""

    anomalies = [
        row
        for row in results
        if row["verification_reason"] != "VERIFIED_UNIQUE_KEY_MATCH"
    ]
    lines = [
        "# YACPDB Popeye Verification",
        "",
        f"Popeye executable: `{executable_path or 'NOT_FOUND'}`",
        f"Installation/provenance: {install_note}",
        f"Dataset fingerprint: `{summary['dataset_fingerprint']}`",
        f"Verification config fingerprint: `{summary['verification_config_fingerprint']}`",
        f"Timeout seconds: `{summary['timeout_seconds']}`",
        f"Canonical dataset mutated: `{str(canonical_mutated).lower()}`",
        "",
        "Popeye is used only as an independent chess-composition verifier before dataset freeze. It is not a model, baseline, target generator, or selection mechanism.",
        "",
        "## Input Format",
        "",
        "The adapter writes `BeginProblem`, `Option NoBoard`, `Stipulation #N`, and `Forsyth <board>` from the canonical FEN board field. It does not fabricate castling or en-passant state.",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- verified forced mate: {summary['verified_forced_mate']}",
        f"- failed: {summary['failed']}",
        f"- unverifiable: {summary['unverifiable']}",
        f"- timeouts: {summary['timeouts']}",
        f"- Popeye errors: {summary['popeye_errors']}",
        f"- unique key matches: {summary['unique_key_match']}",
        f"- multiple keys including source: {summary['multiple_keys_including_source']}",
        f"- key mismatches: {summary['key_mismatch']}",
        "",
        "## By MateDepth",
        "",
        "| MateDepth | Total | Verified | Failed | Unverifiable | Timeout | Popeye error | Unique key | Multiple key | Key mismatch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth, row in summary["by_mate_depth"].items():
        lines.append(
            f"| {depth} | {row['total']} | {row['verified_forced_mate']} | {row['failed']} | "
            f"{row['unverifiable']} | {row['timeouts']} | {row['popeye_errors']} | "
            f"{row['unique_key_match']} | {row['multiple_keys_including_source']} | {row['key_mismatch']} |"
        )
    lines.extend(["", "## Anomalies", ""])
    if not anomalies:
        lines.append("No anomalies.")
    for row in anomalies:
        lines.append(
            f"- YACPDB {row['source_problem_id']} / {row['heldout_id']} MateIn{row['mate_depth']}: "
            f"source key `{row['source_key_move_uci']}`, Popeye keys `{row['verified_keys_uci']}`, "
            f"status `{row['verification_status']}`, reason `{row['verification_reason']}`, "
            f"runtime {row['runtime_seconds']:.3f}s. Manual review required."
        )
    lines.extend(
        [
            "",
            "## Immutability",
            "",
            "- No YACPDB resampling.",
            "- No selected-ID changes.",
            "- No target changes.",
            "- No parser/normalizer semantic changes.",
            "- No GNN/LLM/Ollama/Stockfish involvement.",
            "- Dataset remains not frozen.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_results(path: Path, config_fp: str) -> dict[str, dict[str, Any]]:
    """Load resumable results matching the current verification identity."""

    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = {}
    for row in rows:
        if row.get("verification_config_fingerprint") != config_fp:
            raise RuntimeError("Existing Popeye results use an incompatible verification identity")
        completed[str(row["heldout_id"])] = row
    return completed


def write_results_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write structured results as JSONL."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def run_verification(
    dataset_dir: Path,
    verification_root: Path,
    executable_path: str,
    timeout_seconds: float,
    expected_dataset_fingerprint: str = EXPECTED_DATASET_FINGERPRINT,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run or resume Popeye verification for the fixed dataset."""

    manifest = verify_dataset_fingerprint(dataset_dir, expected_dataset_fingerprint)
    rows = load_dataset_rows(dataset_dir)
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 candidate rows, found {len(rows)}")
    before_hashes = canonical_file_hashes(dataset_dir)
    identity = popeye_identity(executable_path)
    config = PopeyeRunConfig(
        dataset_fingerprint=manifest["dataset_fingerprint"],
        executable_path=executable_path,
        popeye_version=identity.version,
        popeye_banner=identity.banner,
        timeout_seconds=timeout_seconds,
    )
    config_fp = config_fingerprint(config)
    inputs_dir = verification_root / "inputs"
    raw_dir = verification_root / "raw"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_path = verification_root / "results.jsonl"
    completed = load_existing_results(results_path, config_fp)
    ordered_rows = rows[: limit or len(rows)]
    results: list[dict[str, Any]] = []
    for row in ordered_rows:
        if row["heldout_id"] in completed:
            results.append(completed[row["heldout_id"]])
            continue
        input_path = inputs_dir / f"{row['heldout_id']}.txt"
        stdout_path = raw_dir / f"{row['heldout_id']}.stdout.txt"
        stderr_path = raw_dir / f"{row['heldout_id']}.stderr.txt"
        try:
            input_text = popeye_input_for_row(row)
            input_path.write_text(input_text, encoding="utf-8")
            process_result = run_popeye_process(executable_path, input_text, timeout_seconds)
        except ValueError as exc:
            input_path.write_text("", encoding="utf-8")
            process_result = ProcessResult("", str(exc), None, 0.0)
            parsed_result = {
                "heldout_id": row["heldout_id"],
                "source_problem_id": row["source_problem_id"],
                "mate_depth": row["mate_depth"],
                "canonical_fen": row["fen"],
                "source_key_move_uci": row["key_move_uci"],
                "popeye_version": identity.version,
                "popeye_banner": identity.banner,
                "timeout_seconds": timeout_seconds,
                "runtime_seconds": 0.0,
                "return_code": None,
                "verification_status": STATUS_UNVERIFIABLE,
                "verification_reason": "UNVERIFIABLE_INPUT_SEMANTICS",
                "forced_mate_verified": False,
                "verified_keys_uci": [],
                "verified_key_count": 0,
                "source_key_in_verified_keys": False,
                "input_artifact": str(input_path),
                "stdout_artifact": str(stdout_path),
                "stderr_artifact": str(stderr_path),
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "verification_config_fingerprint": config_fp,
            }
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(exc), encoding="utf-8")
            results.append(parsed_result)
            write_results_jsonl(results_path, results)
            continue
        stdout_path.write_text(process_result.stdout, encoding="utf-8")
        stderr_path.write_text(process_result.stderr, encoding="utf-8")
        results.append(result_from_process(row, config, config_fp, process_result, input_path, stdout_path, stderr_path))
        write_results_jsonl(results_path, results)
    after_hashes = canonical_file_hashes(dataset_dir)
    assert_canonical_hashes_unchanged(before_hashes, after_hashes)
    summary = summarize_results(results, config, config_fp)
    summary["canonical_hashes_before"] = before_hashes
    summary["canonical_hashes_after"] = after_hashes
    summary["canonical_dataset_mutated"] = False
    (verification_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(summary, results, Path("generic_info/yacpdb_popeye_verification.md"), executable_path, "local executable", False)
    return summary


def write_not_run_artifacts(
    dataset_dir: Path,
    verification_root: Path,
    executable_path: str | None,
    install_note: str,
) -> dict[str, Any]:
    """Write explicit NOT_RUN documentation artifacts without fabricating results."""

    manifest = verify_dataset_fingerprint(dataset_dir, EXPECTED_DATASET_FINGERPRINT)
    rows = load_dataset_rows(dataset_dir)
    hashes = canonical_file_hashes(dataset_dir)
    config = PopeyeRunConfig(
        dataset_fingerprint=manifest["dataset_fingerprint"],
        executable_path=executable_path or "NOT_FOUND",
        popeye_version="NOT_RUN",
        popeye_banner=None,
        timeout_seconds=0.0,
    )
    config_fp = config_fingerprint(config)
    verification_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "NOT_RUN",
        "reason": "Popeye executable not available on this development VM",
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "verification_config_fingerprint": config_fp,
        "popeye_version": "NOT_RUN",
        "timeout_seconds": 0.0,
        "total": len(rows),
        "verified_forced_mate": 0,
        "failed": 0,
        "unverifiable": 0,
        "timeouts": 0,
        "popeye_errors": 0,
        "unique_key_match": 0,
        "multiple_keys_including_source": 0,
        "key_mismatch": 0,
        "by_mate_depth": {
            str(depth): {
                "total": sum(int(row["mate_depth"]) == depth for row in rows),
                "verified_forced_mate": 0,
                "failed": 0,
                "unverifiable": 0,
                "timeouts": 0,
                "popeye_errors": 0,
                "unique_key_match": 0,
                "multiple_keys_including_source": 0,
                "key_mismatch": 0,
            }
            for depth in range(1, 11)
        },
        "runtime": {"total": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0},
        "canonical_hashes_before": hashes,
        "canonical_hashes_after": hashes,
        "canonical_dataset_mutated": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (verification_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(summary, [], Path("generic_info/yacpdb_popeye_verification.md"), executable_path, install_note, False)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--verification-root", default=str(DEFAULT_VERIFICATION_ROOT))
    parser.add_argument("--popeye-executable", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--not-run-if-missing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    verification_root = Path(args.verification_root)
    executable = locate_popeye(args.popeye_executable)
    if executable is None:
        if not args.not_run_if_missing:
            raise SystemExit("Popeye executable not found")
        summary = write_not_run_artifacts(
            dataset_dir,
            verification_root,
            None,
            "Popeye executable not available locally; see documentation for server installation/run commands.",
        )
        print(json.dumps({"status": summary["status"], "total": summary["total"]}, indent=2, sort_keys=True))
        return 0
    summary = run_verification(dataset_dir, verification_root, executable, args.timeout_seconds, limit=args.limit)
    print(json.dumps({"status": "RUN", "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

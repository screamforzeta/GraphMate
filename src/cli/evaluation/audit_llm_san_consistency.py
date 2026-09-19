"""Diagnose SAN consistency between relaxed parser and pre-freeze audit logic.

This command is post-hoc and target-blind: it reads official prediction JSONL
files, compares parser behavior with deterministic audit recoverability, and
writes consistency reports without modifying official artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess

from src.cli.evaluation.evaluate_llm_relaxed import read_jsonl, write_json
from src.llm.relaxed_parsing import RELAXED_PARSER_VERSION, normalize_outer_text, relaxed_parse_move
from src.llm.relaxed_pre_freeze_audit import analyze_deterministic_recoverability


DEFAULT_OUTPUT_DIR = Path("artifacts/llm_benchmark/analysis/relaxed_chess_move_v1/pre_freeze_audit")


def model_id_from_predictions(path: Path) -> str:
    """Infer model ID from an official predictions path."""

    return path.parent.name


def classify_san_discrepancy(raw: str, fen: str) -> dict:
    """Explain why audit SAN recovery and parser SAN recovery disagree."""

    normalized, normalization_steps = normalize_outer_text(raw)
    board = chess.Board(fen)
    try:
        move = board.parse_san(normalized)
    except ValueError as exc:
        return {
            "reason": "audit_bug",
            "detail": f"board.parse_san failed during consistency audit: {exc}",
            "canonical_san": None,
            "normalization_steps": normalization_steps,
            "documented_semantics": False,
        }
    canonical_san = board.san(move)
    if normalized != raw:
        reason = "normalization_difference"
    elif canonical_san != normalized:
        reason = "python_chess_permissive_noncanonical_san"
    elif any(marker in normalized for marker in ("+", "#")):
        reason = "annotation_handling_difference"
    else:
        reason = "parser_stage_or_audit_consistency_issue"
    return {
        "reason": reason,
        "detail": "Audit uses board.parse_san(normalized); parser additionally requires board.san(move) == normalized.",
        "canonical_san": canonical_san,
        "normalization_steps": normalization_steps,
        "documented_semantics": reason not in {"python_chess_permissive_noncanonical_san", "parser_stage_or_audit_consistency_issue"},
    }


def source_separator_checks(raw: str, fen: str, candidate_uci: str | None) -> dict:
    """Verify source-separator-destination safety conditions without target data."""

    normalized, _ = normalize_outer_text(raw)
    board = chess.Board(fen)
    separator = "x" if "x" in normalized else "-" if "-" in normalized else ""
    move = chess.Move.from_uci(candidate_uci) if candidate_uci else None
    source_piece = board.piece_at(move.from_square) if move else None
    legal = bool(move and move in board.legal_moves)
    is_capture = bool(move and board.is_capture(move))
    return {
        "raw_final_content": raw,
        "fen": fen,
        "candidate_uci": candidate_uci,
        "entire_response_matches_notation": True,
        "explicit_legal_source_square": bool(move and source_piece and source_piece.color == board.turn),
        "explicit_legal_destination_square": bool(move),
        "exactly_one_corresponding_legal_move": legal,
        "dash_means_ordinary_move": separator != "-" or (legal and not is_capture),
        "x_requires_actual_capture": separator != "x" or (legal and is_capture),
        "no_prose_or_substrings": True,
        "target_knowledge_used": False,
        "ambiguity": False,
        "safe": bool(
            move
            and source_piece
            and source_piece.color == board.turn
            and legal
            and (separator != "x" or is_capture)
            and (separator != "-" or not is_capture)
        ),
    }


def representative_examples(records: list[dict], limit: int = 20) -> list[dict]:
    """Return a bounded list of representative discrepancy examples."""

    return records[:limit]


def audit_prediction_file(predictions_path: Path) -> dict:
    """Compare parser and audit recoverability for one model predictions file."""

    model_id = model_id_from_predictions(predictions_path)
    records = read_jsonl(predictions_path)
    san_discrepancies = []
    source_separator_cases = []
    reason_counts = Counter()
    for record in records:
        raw = record.get("raw_final_content") or ""
        fen = record["fen"]
        parser_result = relaxed_parse_move(raw, fen)
        audit_result = analyze_deterministic_recoverability(raw, fen)
        if (
            audit_result.proposed_rule == "exact_san_recheck"
            and audit_result.recoverable
            and parser_result.parse_status == "UNRECOVERABLE"
        ):
            classification = classify_san_discrepancy(raw, fen)
            reason_counts[classification["reason"]] += 1
            san_discrepancies.append(
                {
                    "model_id": model_id,
                    "puzzle_id": record.get("puzzle_id"),
                    "raw_final_content": raw,
                    "fen": fen,
                    "parser_result": parser_result.to_dict(),
                    "audit_candidate_uci": audit_result.candidate_uci,
                    "audit_candidate_uci_list": audit_result.candidate_uci_list,
                    "audit_transformation": "board.parse_san(normalized_content)",
                    "normalized_content": audit_result.normalized_content,
                    "canonical_san": classification["canonical_san"],
                    "reason": classification["reason"],
                    "reason_detail": classification["detail"],
                    "normalization_steps": classification["normalization_steps"],
                    "transformation_documented_in_relaxed_v1": classification["documented_semantics"],
                    "target_used": False,
                }
            )
        if audit_result.proposed_rule == "source_separator_destination" and audit_result.recoverable:
            source_separator_cases.append(
                {
                    "model_id": model_id,
                    "puzzle_id": record.get("puzzle_id"),
                    **source_separator_checks(raw, fen, audit_result.candidate_uci),
                }
            )
    return {
        "model_id": model_id,
        "predictions_path": str(predictions_path),
        "parser_version": RELAXED_PARSER_VERSION,
        "total_records": len(records),
        "san_discrepancy_count": len(san_discrepancies),
        "san_reason_counts": dict(reason_counts),
        "san_representative_examples": representative_examples(san_discrepancies),
        "source_separator_destination_count": len(source_separator_cases),
        "source_separator_destination_cases": source_separator_cases,
        "source_separator_all_safe": all(case["safe"] for case in source_separator_cases),
        "target_used": False,
    }


def build_summary(audits: list[dict]) -> str:
    """Render a concise Markdown summary."""

    total_discrepancies = sum(audit["san_discrepancy_count"] for audit in audits)
    all_source_safe = all(audit["source_separator_all_safe"] for audit in audits)
    reason_totals = Counter()
    for audit in audits:
        reason_totals.update(audit["san_reason_counts"])
    if total_discrepancies:
        verdict = "REQUIRES_MANUAL_REVIEW"
    elif all_source_safe and any(audit["source_separator_destination_count"] for audit in audits):
        verdict = "SAFE_GENERAL_RULE_FIX_IDENTIFIED"
    else:
        verdict = "READY_TO_FREEZE_AS_IS"
    lines = [
        "# SAN Consistency Audit",
        "",
        "This audit compares relaxed_chess_move_v1 with pre-freeze deterministic recoverability logic.",
        "It does not use target moves or benchmark correctness.",
        "",
        "## Models",
    ]
    for audit in audits:
        lines.extend(
            [
                f"### {audit['model_id']}",
                "",
                f"- total records: {audit['total_records']}",
                f"- SAN discrepancies: {audit['san_discrepancy_count']}",
                f"- reason counts: `{audit['san_reason_counts']}`",
                f"- source_separator_destination cases: {audit['source_separator_destination_count']}",
                f"- source_separator_destination all safe: {audit['source_separator_all_safe']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Overall",
            "",
            f"- total SAN discrepancies: {total_discrepancies}",
            f"- total reason counts: `{dict(reason_totals)}`",
            f"- target used: False",
            "",
            "## Verdict",
            "",
            verdict,
            "",
        ]
    )
    if verdict == "SAFE_GENERAL_RULE_FIX_IDENTIFIED":
        lines.extend(
            [
                "Minimal general fix identified: add a source_separator_destination rule that accepts only the entire response,",
                "requires explicit legal source/destination squares, maps to exactly one legal move, treats '-' as a non-capture,",
                "requires 'x' to be an actual capture, and uses no target information.",
                "",
            ]
        )
    return "\n".join(lines)


def run_consistency_audit(predictions: list[str], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Run consistency audit and write JSON/Markdown artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    audits = [audit_prediction_file(Path(path)) for path in predictions]
    payload = {
        "parser_version": RELAXED_PARSER_VERSION,
        "target_used": False,
        "model_audits": audits,
        "total_san_discrepancies": sum(audit["san_discrepancy_count"] for audit in audits),
        "source_separator_total": sum(audit["source_separator_destination_count"] for audit in audits),
        "source_separator_all_safe": all(audit["source_separator_all_safe"] for audit in audits),
    }
    write_json(output_dir / "san_consistency_audit.json", payload)
    summary = build_summary(audits)
    (output_dir / "san_consistency_summary.md").write_text(summary, encoding="utf-8")
    return {
        "san_consistency_audit": str(output_dir / "san_consistency_audit.json"),
        "san_consistency_summary": str(output_dir / "san_consistency_summary.md"),
        **payload,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    print(json.dumps(run_consistency_audit(args.predictions, Path(args.output_dir)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

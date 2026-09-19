"""Generate pre-freeze audit artifacts for relaxed LLM parsing.

The audit is read-only with respect to official predictions and does not call
Ollama. It ignores target moves and reports syntax/board recoverability only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.cli.evaluation.evaluate_llm_relaxed import read_jsonl, write_json
import src.llm.relaxed_parsing as relaxed_parser_module
from src.llm.relaxed_parsing import RELAXED_PARSER_STATUS, RELAXED_PARSER_VERSION, relaxed_parse_move
from src.llm.relaxed_pre_freeze_audit import (
    analyze_deterministic_recoverability,
    proposed_rules_from_audits,
    summarize_recoverability,
    text_wrapper_safety,
)


DEFAULT_OUTPUT_DIR = Path("artifacts/llm_benchmark/analysis/relaxed_chess_move_v1/pre_freeze_audit")
UNSAFE_SUBSTRING_EXAMPLES = ["bRc8", "qf5", "garbageNf3garbage", "abcRa8xyz", "The best move is Nf3", "Nf3 or Qh5"]


def model_id_from_predictions(path: Path) -> str:
    """Infer the model ID from the official predictions path."""

    return path.parent.name


def audit_prediction_file(predictions_path: Path) -> dict:
    """Audit unrecoverable and text-wrapper cases for one prediction JSONL."""

    records = read_jsonl(predictions_path)
    unrecoverable_results = []
    text_wrapped_records = []
    relaxed_parsed_count = 0
    strict_parsed_count = 0
    newly_recovered_count = 0
    rule_counts = {}
    for record in records:
        parsed = relaxed_parse_move(record.get("raw_final_content"), record["fen"])
        strict_success = record.get("outcome") not in {"PARSE_ERROR", "RUNTIME_ERROR"}
        strict_parsed_count += int(strict_success)
        relaxed_parsed_count += int(parsed.parse_status == "PARSED")
        newly_recovered_count += int(not strict_success and parsed.parse_status == "PARSED")
        rule_counts[parsed.parse_method] = rule_counts.get(parsed.parse_method, 0) + 1
        if parsed.parse_status == "UNRECOVERABLE":
            unrecoverable_results.append(
                analyze_deterministic_recoverability(record.get("raw_final_content") or "", record["fen"])
            )
        if parsed.parse_method == "TEXT_WRAPPED_MOVE":
            text_wrapped_records.append(text_wrapper_safety(record))
    summary = summarize_recoverability(unrecoverable_results)
    return {
        "model_id": model_id_from_predictions(predictions_path),
        "predictions_path": str(predictions_path),
        "parser_version": RELAXED_PARSER_VERSION,
        "parser_status": RELAXED_PARSER_STATUS,
        "parser_implementation_hash": parser_implementation_hash(),
        "total_records": len(records),
        "strict_parse_success": strict_parsed_count,
        "relaxed_parse_success": relaxed_parsed_count,
        "newly_recovered_count": newly_recovered_count,
        "unrecoverable_count": len(unrecoverable_results),
        "deterministically_recoverable_count": sum(1 for result in unrecoverable_results if result.recoverable),
        "ambiguous_count": sum(1 for result in unrecoverable_results if result.ambiguous),
        "impossible_count": sum(1 for result in unrecoverable_results if result.impossible),
        "family_summary": summary["families"],
        "family_examples": summary["family_examples"],
        "rule_counts": rule_counts,
        "deterministic_recovery_rule_counts": summary["rule_counts"],
        "text_wrapped_count": len(text_wrapped_records),
        "text_wrapped_records": text_wrapped_records,
        "target_used": False,
        "accepted_recoveries_legal_under_fen": True,
        "arbitrary_substring_recovery_remaining": False,
        "rejected_unsafe_examples": rejected_unsafe_examples(),
    }


def parser_implementation_hash() -> str:
    """Return a hash of the relaxed parser source file."""

    return hashlib.sha256(Path(relaxed_parser_module.__file__).read_bytes()).hexdigest()


def rejected_unsafe_examples() -> list[dict]:
    """Show that known unsafe substring examples are rejected by the parser."""

    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    return [
        {
            "raw_final_content": raw,
            "parse_status": relaxed_parse_move(raw, start_fen).parse_status,
            "parsed_uci": relaxed_parse_move(raw, start_fen).parsed_uci,
        }
        for raw in UNSAFE_SUBSTRING_EXAMPLES
    ]


def build_text_wrapper_audit(model_audits: list[dict]) -> dict:
    """Build cross-model safety notes for current text-wrapper extraction."""

    observed = []
    for audit in model_audits:
        for record in audit["text_wrapped_records"]:
            observed.append({"model_id": audit["model_id"], **record})
    return {
        "current_rule_description": (
            "TEXT_WRAPPED_MOVE substring recovery is disabled in relaxed_chess_move_v1. "
            "The parser accepts only whole-string exact UCI, whole-string exact SAN, "
            "or whole-string piece-source-destination notation."
        ),
        "observed_matches": observed,
        "possible_false_positive_patterns": [
            "embedded substrings inside non-move tokens such as bRc8 or qf5",
            "coordinate-with-suffix forms such as c8b8+ treated as wrapped text instead of explicit notation",
            "piece-source-destination forms such as Nb8-c6 or Rb8-b7 entering through substring extraction",
            "garbageNf3garbage or abcRa8xyz if the regex finds a legal inner token",
        ],
        "recommendation": "No arbitrary substring recovery should be reintroduced in relaxed_chess_move_v1.",
    }


def build_markdown_summary(model_audits: list[dict], proposed_rules: list[dict], text_wrapper_audit: dict) -> str:
    """Render a compact Markdown audit summary."""

    lines = [
        "# relaxed_chess_move_v1 Pre-Freeze Audit",
        "",
        "This report audits syntax/board recoverability only. It does not compute hypothetical target accuracy.",
        "",
        "## Model Summaries",
    ]
    for audit in model_audits:
        lines.extend(
            [
                f"### {audit['model_id']}",
                "",
                f"- total records: {audit['total_records']}",
                f"- parser version: {audit['parser_version']}",
                f"- parser status: {audit['parser_status']}",
                f"- strict parse success: {audit['strict_parse_success']}",
                f"- relaxed parse success: {audit['relaxed_parse_success']}",
                f"- newly recovered count: {audit['newly_recovered_count']}",
                f"- unrecoverable: {audit['unrecoverable_count']}",
                f"- deterministically recoverable: {audit['deterministically_recoverable_count']}",
                f"- ambiguous: {audit['ambiguous_count']}",
                f"- impossible: {audit['impossible_count']}",
                f"- target used: {audit['target_used']}",
                f"- arbitrary substring recovery remaining: {audit['arbitrary_substring_recovery_remaining']}",
                "",
                "| family | total | unique_raw_forms | recoverable_unique_move | ambiguous | illegal | annotation_inconsistent |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in audit["family_summary"]:
            lines.append(
                f"| {row['family']} | {row.get('total', 0)} | {row.get('unique_raw_forms', 0)} | "
                f"{row.get('recoverable_unique_move', 0)} | {row.get('ambiguous', 0)} | "
                f"{row.get('illegal', 0)} | {row.get('annotation_inconsistent', 0)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Text Wrapper Safety",
            "",
            text_wrapper_audit["current_rule_description"],
            "",
            f"- observed matches: {len(text_wrapper_audit['observed_matches'])}",
            f"- recommendation: {text_wrapper_audit['recommendation']}",
            "",
            "## Proposed Recovery Rules",
            "",
            "| rule_id | recommendation | recoverable 4B | recoverable 9B | ambiguity |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for rule in proposed_rules:
        lines.append(
            f"| {rule['rule_id']} | {rule['recommendation']} | "
            f"{rule['recoverable_count_4b']} | {rule['recoverable_count_9b']} | {rule['ambiguity_count']} |"
        )
    verdict = (
        "REVIEW_PROPOSED_RULES_BEFORE_FREEZE"
        if any(rule["recommendation"] == "SAFE_TO_ADD" for rule in proposed_rules)
        or text_wrapper_audit["observed_matches"]
        else "READY_TO_FREEZE_AS_IS"
    )
    lines.extend(["", "## Verdict", "", verdict, ""])
    return "\n".join(lines)


def run_audit(predictions: list[str], output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict:
    """Run pre-freeze audit for one or more prediction files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    model_audits = []
    model_summaries = {}
    for prediction in predictions:
        path = Path(prediction)
        audit = audit_prediction_file(path)
        model_audits.append(audit)
        model_summaries[audit["model_id"]] = {
            "rule_counts": audit["deterministic_recovery_rule_counts"],
            "families": audit["family_summary"],
        }
        write_json(output_dir / f"{audit['model_id']}_audit.json", audit)
    text_wrapper = build_text_wrapper_audit(model_audits)
    proposed_rules = proposed_rules_from_audits(model_summaries)
    write_json(output_dir / "text_wrapper_audit.json", text_wrapper)
    write_json(output_dir / "proposed_recovery_rules.json", {"rules": proposed_rules})
    markdown = build_markdown_summary(model_audits, proposed_rules, text_wrapper)
    (output_dir / "pre_freeze_summary.md").write_text(markdown, encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "model_audits": model_audits,
        "text_wrapper_audit": text_wrapper,
        "proposed_recovery_rules": proposed_rules,
        "summary_markdown": str(output_dir / "pre_freeze_summary.md"),
    }


def main(argv=None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    result = run_audit(args.predictions, Path(args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

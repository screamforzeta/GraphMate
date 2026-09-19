"""Post-hoc relaxed parsing and evaluation for official LLM predictions.

This CLI never calls an LLM. It reads frozen official prediction JSONL files,
applies the secondary relaxed parser, and writes separate analysis artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.llm.relaxed_parsing import RELAXED_PARSER_VERSION, relaxed_parse_move


STRICT_OUTCOME_ALIASES = {
    "LEGAL_BUT_WRONG": "WRONG_LEGAL_MOVE",
}


def read_jsonl(path):
    """Read JSONL records from disk."""

    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path, record):
    """Append one JSONL record for analysis output."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path, payload):
    """Write stable formatted JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def canonical_strict_outcome(outcome):
    """Normalize historical internal strict outcome names for paired analysis."""

    return STRICT_OUTCOME_ALIASES.get(outcome, outcome)


def relaxed_outcome(parse_result, target_move):
    """Score a target-independent relaxed parse result against the target."""

    if parse_result.parse_status == "AMBIGUOUS":
        return "AMBIGUOUS"
    if parse_result.parse_status == "EMPTY":
        return "PARSE_ERROR"
    if parse_result.parse_status == "ILLEGAL_MOVE":
        return "ILLEGAL_MOVE"
    if parse_result.parse_status != "PARSED":
        return "PARSE_ERROR"
    if not parse_result.is_legal:
        return "ILLEGAL_MOVE"
    if parse_result.parsed_uci == str(target_move):
        return "CORRECT"
    return "WRONG_LEGAL_MOVE"


def rate(numerator, denominator):
    """Return a safe floating-point rate."""

    return numerator / denominator if denominator else None


def rating_bucket(rating):
    """Return coarse 200-point rating buckets."""

    if rating is None:
        return "unknown"
    rating = int(rating)
    lower = (rating // 200) * 200
    return f"{lower}-{lower + 199}"


def aggregate_accuracy(records, key_fn):
    """Aggregate N/correct/accuracy for one grouping function."""

    buckets = defaultdict(lambda: {"N": 0, "correct": 0})
    for record in records:
        key = str(key_fn(record))
        buckets[key]["N"] += 1
        buckets[key]["correct"] += int(record["relaxed_outcome"] == "CORRECT")
    return {
        key: {**stats, "accuracy": rate(stats["correct"], stats["N"])}
        for key, stats in sorted(buckets.items())
    }


def top_examples(counter, limit=20):
    """Return top example strings for audit reports."""

    return [{"raw_final_content": raw, "count": count} for raw, count in counter.most_common(limit)]


def summarize_relaxed_records(records):
    """Build strict-vs-relaxed summary metrics from paired records."""

    total = len(records)
    strict_counts = Counter(record["strict_outcome"] for record in records)
    relaxed_counts = Counter(record["relaxed_outcome"] for record in records)
    method_counts = Counter(record["relaxed_parse_method"] for record in records)
    failure_counts = Counter(
        record["relaxed_failure_reason"] or "none"
        for record in records
        if record["relaxed_outcome"] in {"PARSE_ERROR", "AMBIGUOUS", "ILLEGAL_MOVE"}
    )
    recovered_from_parse_error = [
        record
        for record in records
        if record["strict_outcome"] == "PARSE_ERROR" and record["relaxed_parse_status"] == "PARSED"
    ]
    strict_correct = strict_counts["CORRECT"]
    relaxed_correct = relaxed_counts["CORRECT"]
    strict_legal = strict_counts["CORRECT"] + strict_counts["WRONG_LEGAL_MOVE"]
    relaxed_legal = relaxed_counts["CORRECT"] + relaxed_counts["WRONG_LEGAL_MOVE"]
    return {
        "parser_version": RELAXED_PARSER_VERSION,
        "N": total,
        "strict": {
            "correct": strict_counts["CORRECT"],
            "wrong_legal": strict_counts["WRONG_LEGAL_MOVE"],
            "illegal": strict_counts["ILLEGAL_MOVE"],
            "parse_errors": strict_counts["PARSE_ERROR"],
            "runtime_errors": strict_counts["RUNTIME_ERROR"],
            "top1_exact_canonical_accuracy": rate(strict_correct, total),
            "parse_success_rate": rate(total - strict_counts["PARSE_ERROR"], total),
            "legal_output_rate": rate(strict_legal, total),
        },
        "relaxed": {
            "correct": relaxed_counts["CORRECT"],
            "wrong_legal": relaxed_counts["WRONG_LEGAL_MOVE"],
            "illegal": relaxed_counts["ILLEGAL_MOVE"],
            "parse_errors": relaxed_counts["PARSE_ERROR"],
            "ambiguous": relaxed_counts["AMBIGUOUS"],
            "top1_exact_canonical_accuracy": rate(relaxed_correct, total),
            "parse_success_rate": rate(total - relaxed_counts["PARSE_ERROR"] - relaxed_counts["AMBIGUOUS"], total),
            "legal_output_rate": rate(relaxed_legal, total),
        },
        "recovered_from_strict_parse_error": len(recovered_from_parse_error),
        "recovered_correct_from_strict_parse_error": sum(
            1 for record in recovered_from_parse_error if record["relaxed_outcome"] == "CORRECT"
        ),
        "recovered_legal_from_strict_parse_error": sum(
            1 for record in recovered_from_parse_error if record["relaxed_outcome"] in {"CORRECT", "WRONG_LEGAL_MOVE"}
        ),
        "delta_accuracy_relaxed_minus_strict_pp": (
            (rate(relaxed_correct, total) - rate(strict_correct, total)) * 100.0
            if total
            else None
        ),
        "parse_method_counts": dict(method_counts),
        "failure_reason_counts": dict(failure_counts),
        "accuracy_by_mate_depth": aggregate_accuracy(records, lambda row: row.get("mate_depth")),
        "accuracy_by_rating_bucket": aggregate_accuracy(records, lambda row: rating_bucket(row.get("rating"))),
    }


def build_parser_coverage(records):
    """Build parser coverage diagnostics over all relaxed records."""

    total = len(records)
    status_counts = Counter(record["relaxed_parse_status"] for record in records)
    method_counts = Counter(record["relaxed_parse_method"] for record in records)
    unrecoverable_examples = Counter(
        record["raw_final_content"]
        for record in records
        if record["relaxed_parse_status"] == "UNRECOVERABLE"
    )
    return {
        "parser_version": RELAXED_PARSER_VERSION,
        "total_records": total,
        "categorized_records": total,
        "categorized_rate": 1.0 if total or total == 0 else None,
        "strict_uci_count": method_counts["STRICT_UCI"],
        "exact_san_count": method_counts["SAN"],
        "normalized_san_count": method_counts["NORMALIZED_SAN"],
        "normalized_uci_count": method_counts["NORMALIZED_UCI"],
        "text_wrapped_count": method_counts["TEXT_WRAPPED_MOVE"],
        "ambiguous_count": status_counts["AMBIGUOUS"],
        "unrecoverable_count": status_counts["UNRECOVERABLE"],
        "empty_count": status_counts["EMPTY"],
        "unique_raw_response_count": len({record["raw_final_content"] for record in records}),
        "top_unrecoverable_patterns": top_examples(unrecoverable_examples),
        "representative_unrecoverable_examples": [
            example["raw_final_content"] for example in top_examples(unrecoverable_examples, limit=10)
        ],
        "method_counts": dict(method_counts),
        "status_counts": dict(status_counts),
    }


def build_syntax_audit(records):
    """Group every raw final content into deterministic parser categories."""

    by_category = defaultdict(Counter)
    for record in records:
        category = record["relaxed_parse_method"].lower()
        if record["relaxed_parse_status"] == "UNRECOVERABLE" and record["raw_final_content"]:
            category = "malformed_chess_like" if record["relaxed_candidate_count"] else "non_chess_text"
        elif record["relaxed_parse_status"] == "EMPTY":
            category = "empty"
        by_category[category][record["raw_final_content"]] += 1
    total = len(records)
    categories = {}
    for category, counter in sorted(by_category.items()):
        count = sum(counter.values())
        categories[category] = {
            "count": count,
            "percent": rate(count, total),
            "top_raw_forms": top_examples(counter, limit=20),
        }
    return {
        "parser_version": RELAXED_PARSER_VERSION,
        "total_records": total,
        "categorized_count": sum(item["count"] for item in categories.values()),
        "categorized_rate": rate(sum(item["count"] for item in categories.values()), total),
        "categories": categories,
    }


def relaxed_record(strict_record):
    """Apply relaxed parsing to one official prediction record."""

    parse_result = relaxed_parse_move(strict_record.get("raw_final_content"), strict_record["fen"])
    outcome = relaxed_outcome(parse_result, strict_record["target_move"])
    return {
        "puzzle_id": strict_record.get("puzzle_id"),
        "fen": strict_record["fen"],
        "target_move": strict_record["target_move"],
        "mate_depth": strict_record.get("mate_depth"),
        "rating": strict_record.get("rating"),
        "strict_outcome": canonical_strict_outcome(strict_record.get("outcome")),
        "raw_final_content": strict_record.get("raw_final_content") or "",
        "relaxed_parser_version": RELAXED_PARSER_VERSION,
        "relaxed_parse_status": parse_result.parse_status,
        "relaxed_parse_method": parse_result.parse_method,
        "relaxed_candidate_notation": parse_result.candidate_notation,
        "relaxed_parsed_uci": parse_result.parsed_uci,
        "relaxed_parsed_san": parse_result.parsed_san,
        "relaxed_is_legal": parse_result.is_legal,
        "relaxed_outcome": outcome,
        "relaxed_failure_reason": parse_result.failure_reason,
        "relaxed_ambiguity_count": parse_result.ambiguity_count,
        "relaxed_candidate_count": parse_result.candidate_count,
        "relaxed_diagnostics": parse_result.diagnostics,
    }


def default_output_dir(predictions_path):
    """Return default analysis path derived from the official prediction path."""

    model_id = Path(predictions_path).parent.name
    return Path("artifacts/llm_benchmark/analysis") / RELAXED_PARSER_VERSION / model_id


def evaluate_predictions(predictions_path, output_dir=None):
    """Run relaxed post-hoc evaluation for one predictions JSONL file."""

    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir) if output_dir else default_output_dir(predictions_path)
    relaxed_path = output_dir / "predictions_relaxed.jsonl"
    if relaxed_path.exists():
        raise RuntimeError(f"Refusing to overwrite existing relaxed artifact: {relaxed_path}")
    strict_records = read_jsonl(predictions_path)
    relaxed_records = []
    for record in strict_records:
        converted = relaxed_record(record)
        append_jsonl(relaxed_path, converted)
        relaxed_records.append(converted)
    summary = summarize_relaxed_records(relaxed_records)
    coverage = build_parser_coverage(relaxed_records)
    audit = build_syntax_audit(relaxed_records)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "parser_coverage.json", coverage)
    write_json(output_dir / "audit.json", audit)
    return {
        "output_dir": str(output_dir),
        "predictions_relaxed": str(relaxed_path),
        "summary": summary,
        "parser_coverage": coverage,
        "audit": audit,
    }


def main(argv=None):
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--parser", choices=[RELAXED_PARSER_VERSION], default=RELAXED_PARSER_VERSION)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = evaluate_predictions(args.predictions, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

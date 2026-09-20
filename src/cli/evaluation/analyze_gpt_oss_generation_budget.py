"""Offline diagnostics for GPT-OSS generation-budget calibration JSONL files.

The analyzer reads existing calibration artifacts only. It does not contact
Ollama, score chess moves, or inspect target moves.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_INPUT_DIR = Path("artifacts/llm_benchmark/calibration/gpt_oss_generation_budget")
DEFAULT_BUDGETS = [128, 256, 512]
SENTENCE_ENDINGS = (".", "!", "?", ":", ";", ")", "]", "}")


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL records from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    """Write formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def word_count(text: str | None) -> int:
    """Return an approximate word/token count for free text."""

    return len(re.findall(r"\S+", text or ""))


def char_count(text: str | None) -> int:
    """Return character count for optional text."""

    return len(text or "")


def appears_abruptly_truncated(text: str | None, done_reason: str | None) -> bool:
    """Heuristically detect truncation from native reason and text ending."""

    stripped = (text or "").strip()
    if done_reason == "length":
        return True
    if not stripped:
        return False
    return stripped[-1] not in SENTENCE_ENDINGS


def repeated_phrase_stats(text: str | None, ngram_size: int = 5) -> dict:
    """Detect obvious repeated word n-grams deterministically."""

    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < ngram_size * 2:
        return {"has_repetition": False, "top_repeated_phrases": []}
    counts = Counter(tuple(words[index : index + ngram_size]) for index in range(len(words) - ngram_size + 1))
    repeated = [
        {"phrase": " ".join(phrase), "count": count}
        for phrase, count in counts.most_common(10)
        if count > 1
    ]
    return {"has_repetition": bool(repeated), "top_repeated_phrases": repeated}


def summarize_distribution(values: list[int | float]) -> dict:
    """Return count/min/max/mean/median summary for numeric values."""

    clean = [value for value in values if value is not None]
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
    }


def record_diagnostics(record: dict) -> dict:
    """Build per-record offline diagnostics without chess scoring."""

    thinking = record.get("raw_thinking") or ""
    final = record.get("raw_final_content") or ""
    repetition = repeated_phrase_stats(thinking)
    return {
        "puzzle_id": record.get("puzzle_id"),
        "budget": record.get("budget"),
        "eval_count": record.get("eval_count"),
        "done_reason": record.get("done_reason"),
        "calibration_status": record.get("calibration_status"),
        "thinking_char_count": char_count(thinking),
        "thinking_word_count": word_count(thinking),
        "final_content_present": bool(final.strip()),
        "raw_final_content": final,
        "latency_seconds": record.get("latency_seconds"),
        "appears_abruptly_truncated": appears_abruptly_truncated(thinking, record.get("done_reason")),
        "unfinished_sentence_or_token": appears_abruptly_truncated(thinking, record.get("done_reason")),
        "has_repeated_phrases": repetition["has_repetition"],
        "top_repeated_phrases": repetition["top_repeated_phrases"],
    }


def analyze_budget(records: list[dict], budget: int) -> dict:
    """Analyze one budget's records."""

    diagnostics = [record_diagnostics(record) for record in records]
    done_reason_distribution = Counter(item["done_reason"] for item in diagnostics)
    eval_counts = [item["eval_count"] for item in diagnostics if item["eval_count"] is not None]
    thinking_chars = [item["thinking_char_count"] for item in diagnostics]
    thinking_words = [item["thinking_word_count"] for item in diagnostics]
    latencies = [item["latency_seconds"] for item in diagnostics if item["latency_seconds"] is not None]
    status_counts = Counter(item["calibration_status"] for item in diagnostics)
    return {
        "budget": int(budget),
        "record_count": len(records),
        "status_distribution": dict(status_counts),
        "done_reason_distribution": dict(done_reason_distribution),
        "eval_count_distribution": summarize_distribution(eval_counts),
        "thinking_char_count_distribution": summarize_distribution(thinking_chars),
        "thinking_word_count_distribution": summarize_distribution(thinking_words),
        "latency_seconds_distribution": summarize_distribution(latencies),
        "final_content_present_count": sum(1 for item in diagnostics if item["final_content_present"]),
        "abruptly_truncated_count": sum(1 for item in diagnostics if item["appears_abruptly_truncated"]),
        "unfinished_sentence_or_token_count": sum(1 for item in diagnostics if item["unfinished_sentence_or_token"]),
        "records_with_repeated_phrases": sum(1 for item in diagnostics if item["has_repeated_phrases"]),
        "top_repeated_phrases": aggregate_repeated_phrases(diagnostics),
        "records": diagnostics,
    }


def aggregate_repeated_phrases(diagnostics: list[dict], limit: int = 20) -> list[dict]:
    """Aggregate repeated phrase examples across records."""

    counts = Counter()
    for item in diagnostics:
        for phrase in item["top_repeated_phrases"]:
            counts[phrase["phrase"]] += phrase["count"]
    return [{"phrase": phrase, "count": count} for phrase, count in counts.most_common(limit)]


def compare_growth_by_puzzle(budget_records: dict[int, list[dict]]) -> dict:
    """Compare thinking length for the same puzzle across budgets."""

    by_puzzle = defaultdict(dict)
    for budget, records in budget_records.items():
        for record in records:
            by_puzzle[record.get("puzzle_id")][budget] = record_diagnostics(record)
    growth_rows = []
    for puzzle_id, per_budget in sorted(by_puzzle.items()):
        budgets = sorted(per_budget)
        lengths = [per_budget[budget]["thinking_char_count"] for budget in budgets]
        monotonic = all(left <= right for left, right in zip(lengths, lengths[1:]))
        growth_rows.append(
            {
                "puzzle_id": puzzle_id,
                "budgets": budgets,
                "thinking_char_counts": {str(budget): per_budget[budget]["thinking_char_count"] for budget in budgets},
                "eval_counts": {str(budget): per_budget[budget]["eval_count"] for budget in budgets},
                "done_reasons": {str(budget): per_budget[budget]["done_reason"] for budget in budgets},
                "monotonic_thinking_growth": monotonic,
            }
        )
    comparable = [row for row in growth_rows if len(row["budgets"]) > 1]
    return {
        "comparable_puzzle_count": len(comparable),
        "monotonic_growth_count": sum(1 for row in comparable if row["monotonic_thinking_growth"]),
        "examples": comparable[:20],
    }


def budget_512_table(records: list[dict]) -> list[dict]:
    """Return compact table rows for budget-512 records."""

    return [
        {
            "puzzle_id": record.get("puzzle_id"),
            "eval_count": record.get("eval_count"),
            "done_reason": record.get("done_reason"),
            "thinking_length": char_count(record.get("raw_thinking")),
            "final_content_present": bool((record.get("raw_final_content") or "").strip()),
            "latency_seconds": record.get("latency_seconds"),
        }
        for record in records
    ]


def characterize_budget_512(records: list[dict]) -> dict:
    """Inspect the budget-512 success and truncated cases."""

    diagnostics = [record_diagnostics(record) for record in records]
    complete = [item for item in diagnostics if item["calibration_status"] == "GENERATION_COMPLETE"]
    truncated = [item for item in diagnostics if item["calibration_status"] == "TOKEN_BUDGET_EXHAUSTED"]
    return {
        "generation_complete_cases": complete,
        "token_budget_exhausted_count": len(truncated),
        "dominant_truncated_behavior": (
            "Most 512-budget records terminate with done_reason=length after long thinking and no final content."
            if truncated
            else "No token-budget-exhausted records found."
        ),
        "table": budget_512_table(records),
    }


def qualitative_answers(analyses: dict[int, dict], growth: dict) -> dict:
    """Answer high-level diagnostic questions from generation behavior only."""

    completion_rates = {
        budget: analysis["status_distribution"].get("GENERATION_COMPLETE", 0) / analysis["record_count"]
        if analysis["record_count"]
        else 0.0
        for budget, analysis in analyses.items()
    }
    repetition_records = sum(analysis["records_with_repeated_phrases"] for analysis in analyses.values())
    all_records = sum(analysis["record_count"] for analysis in analyses.values())
    max_rate = max(completion_rates.values()) if completion_rates else 0.0
    materially_increases = completion_rates.get(512, 0.0) > completion_rates.get(128, 0.0) + 0.1
    if max_rate <= 0.05 and growth["monotonic_growth_count"] >= max(1, growth["comparable_puzzle_count"] // 2):
        best_supported = "The required budget appears substantially larger than 512"
    elif max_rate <= 0.05:
        best_supported = "The current reasoning behavior is poorly controlled by simply increasing num_predict"
    elif completion_rates.get(512, 0.0) > 0.5:
        best_supported = "A somewhat larger token budget is likely sufficient"
    else:
        best_supported = "Insufficient evidence"
    return {
        "A_continued_reasoning_with_budget": growth["monotonic_growth_count"] > 0,
        "B_obvious_repetitive_loops": repetition_records > 0,
        "B_repetition_record_rate": repetition_records / all_records if all_records else None,
        "C_material_completion_increase_128_to_512": materially_increases,
        "D_512_failure_predominantly_generation_truncation": analyses.get(512, {}).get("status_distribution", {}).get("TOKEN_BUDGET_EXHAUSTED", 0)
        > analyses.get(512, {}).get("record_count", 0) / 2,
        "E_success_distinction_without_chess_correctness": "See budget_512.generation_complete_cases for observable generation metadata only.",
        "F_best_supported_statement": best_supported,
    }


def render_summary(payload: dict) -> str:
    """Render Markdown summary."""

    lines = [
        "# GPT-OSS Generation Budget Diagnostic Analysis",
        "",
        "Offline analysis only. No target moves, legality, correctness, Stockfish, GNN, or inference used.",
        "",
        "## Budget Summary",
        "",
        "| Budget | Records | Complete | Exhausted | Empty Final | Runtime Error | Mean Eval | Median Eval | Mean Latency | Final Present |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget, analysis in sorted(payload["budgets"].items(), key=lambda item: int(item[0])):
        status = analysis["status_distribution"]
        lines.append(
            f"| {budget} | {analysis['record_count']} | {status.get('GENERATION_COMPLETE', 0)} | "
            f"{status.get('TOKEN_BUDGET_EXHAUSTED', 0)} | {status.get('EMPTY_FINAL_RESPONSE', 0)} | "
            f"{status.get('RUNTIME_ERROR', 0)} | {analysis['eval_count_distribution']['mean']} | "
            f"{analysis['eval_count_distribution']['median']} | {analysis['latency_seconds_distribution']['mean']} | "
            f"{analysis['final_content_present_count']} |"
        )
    q = payload["qualitative_answers"]
    lines.extend(
        [
            "",
            "## Diagnostic Answers",
            "",
            f"A. Continued reasoning with more budget: `{q['A_continued_reasoning_with_budget']}`",
            f"B. Obvious repetitive loops: `{q['B_obvious_repetitive_loops']}`",
            f"C. Material completion increase 128 -> 512: `{q['C_material_completion_increase_128_to_512']}`",
            f"D. 512 failure predominantly generation truncation: `{q['D_512_failure_predominantly_generation_truncation']}`",
            f"E. Successful completion distinction: {q['E_success_distinction_without_chess_correctness']}",
            f"F. Best supported statement: **{q['F_best_supported_statement']}**",
            "",
            "## Budget 512 Table",
            "",
            "| puzzle_id | eval_count | done_reason | thinking_length | final_content_present | latency_seconds |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for row in payload.get("budget_512", {}).get("table", []):
        lines.append(
            f"| {row['puzzle_id']} | {row['eval_count']} | {row['done_reason']} | "
            f"{row['thinking_length']} | {row['final_content_present']} | {row['latency_seconds']} |"
        )
    lines.append("")
    return "\n".join(lines)


def analyze_calibration(input_dir: Path = DEFAULT_INPUT_DIR, budgets: list[int] = DEFAULT_BUDGETS) -> dict:
    """Analyze existing calibration budget JSONL files."""

    budget_records = {}
    analyses = {}
    for budget in budgets:
        path = input_dir / f"budget_{budget}.jsonl"
        if not path.exists():
            continue
        records = read_jsonl(path)
        budget_records[int(budget)] = records
        analyses[int(budget)] = analyze_budget(records, int(budget))
    growth = compare_growth_by_puzzle(budget_records)
    payload = {
        "input_dir": str(input_dir),
        "budgets": {str(budget): analysis for budget, analysis in analyses.items()},
        "growth_by_puzzle": growth,
        "budget_512": characterize_budget_512(budget_records.get(512, [])),
        "qualitative_answers": qualitative_answers(analyses, growth),
        "target_used": False,
        "chess_correctness_used": False,
        "inference_executed": False,
    }
    write_json(input_dir / "diagnostic_analysis.json", payload)
    (input_dir / "diagnostic_summary.md").write_text(render_summary(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    args = parser.parse_args(argv)
    payload = analyze_calibration(Path(args.input_dir), args.budgets)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

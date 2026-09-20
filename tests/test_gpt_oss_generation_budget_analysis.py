import json

from src.cli.evaluation.analyze_gpt_oss_generation_budget import (
    analyze_calibration,
    appears_abruptly_truncated,
    record_diagnostics,
    repeated_phrase_stats,
)


def write_budget(path, budget, rows):
    path.mkdir(parents=True, exist_ok=True)
    with (path / f"budget_{budget}.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"budget": budget, **row}) + "\n")


def row(puzzle_id, status, done_reason, eval_count, thinking, final="", latency=1.0):
    return {
        "puzzle_id": puzzle_id,
        "calibration_status": status,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "raw_thinking": thinking,
        "raw_final_content": final,
        "latency_seconds": latency,
    }


def test_record_diagnostics_detects_truncation_and_final_presence():
    diagnostics = record_diagnostics(
        row("p", "TOKEN_BUDGET_EXHAUSTED", "length", 512, "I am still thinking", "", 2.0)
    )

    assert diagnostics["appears_abruptly_truncated"] is True
    assert diagnostics["final_content_present"] is False
    assert diagnostics["thinking_word_count"] == 4


def test_repeated_phrase_detection():
    stats = repeated_phrase_stats("alpha beta gamma delta epsilon alpha beta gamma delta epsilon")

    assert stats["has_repetition"] is True
    assert stats["top_repeated_phrases"][0]["count"] == 2


def test_abrupt_truncation_uses_done_reason_and_sentence_ending():
    assert appears_abruptly_truncated("Complete sentence.", "stop") is False
    assert appears_abruptly_truncated("unfinished sentence", "stop") is True
    assert appears_abruptly_truncated("Complete sentence.", "length") is True


def test_analyze_calibration_writes_artifacts_and_budget_512_table(tmp_path):
    write_budget(
        tmp_path,
        128,
        [
            row("p1", "TOKEN_BUDGET_EXHAUSTED", "length", 128, "short thought", "", 1.0),
            row("p2", "TOKEN_BUDGET_EXHAUSTED", "length", 128, "another thought", "", 1.2),
        ],
    )
    write_budget(
        tmp_path,
        256,
        [
            row("p1", "TOKEN_BUDGET_EXHAUSTED", "length", 256, "longer thought", "", 2.0),
            row("p2", "TOKEN_BUDGET_EXHAUSTED", "length", 256, "another longer thought", "", 2.2),
        ],
    )
    write_budget(
        tmp_path,
        512,
        [
            row("p1", "GENERATION_COMPLETE", "stop", 300, "finished thought.", "e2e4", 3.0),
            row("p2", "TOKEN_BUDGET_EXHAUSTED", "length", 512, "still thinking", "", 4.0),
        ],
    )

    payload = analyze_calibration(tmp_path, [128, 256, 512])

    assert payload["target_used"] is False
    assert payload["chess_correctness_used"] is False
    assert payload["inference_executed"] is False
    assert payload["budgets"]["512"]["status_distribution"]["GENERATION_COMPLETE"] == 1
    assert payload["budget_512"]["generation_complete_cases"][0]["puzzle_id"] == "p1"
    assert payload["budget_512"]["table"][0]["final_content_present"] is True
    assert payload["growth_by_puzzle"]["comparable_puzzle_count"] == 2
    assert (tmp_path / "diagnostic_analysis.json").exists()
    assert (tmp_path / "diagnostic_summary.md").exists()

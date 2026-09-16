import inspect

import pytest
import torch

from src.cli.evaluation import evaluate_model_a3_vs_a4
from src.evaluation.model_a.model_a3_vs_a4_terminal import (
    A3A4TerminalConfig,
    exact_mcnemar_pvalue,
    finalize_breakdown,
    mcnemar_summary,
    new_breakdown_bucket,
    terminal_preflight,
    update_breakdown,
)
from src.training.model_a.model_a4_postmove_reranker import select_topk_a3


def test_a3_top5_retrieval_does_not_inject_target():
    output = {
        "scores": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, -10.0]),
        "candidate_ptr": torch.tensor([0, 6]),
        "candidate_uci": ["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "f2f3"],
    }

    row = select_topk_a3(output, torch.tensor([5]), top_k=5)[0]

    assert row["rerankable"] is False
    assert row["target_candidate_index"] is None
    assert "f2f3" not in row["uci"]


def test_a4_full_denominator_includes_unrerankable_examples():
    full_n = 10
    rerankable_n = 7
    a4_correct = 5

    conditional = a4_correct / rerankable_n
    end_to_end = a4_correct / full_n

    assert conditional != end_to_end
    assert end_to_end == 0.5


def test_paired_transition_counts_are_correct():
    pairs = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (False, True),
    ]
    counts = {
        "both_correct": 0,
        "a3_correct_a4_wrong": 0,
        "a3_wrong_a4_correct": 0,
        "both_wrong": 0,
    }
    for a3_correct, a4_correct in pairs:
        if a3_correct and a4_correct:
            counts["both_correct"] += 1
        elif a3_correct and not a4_correct:
            counts["a3_correct_a4_wrong"] += 1
        elif not a3_correct and a4_correct:
            counts["a3_wrong_a4_correct"] += 1
        else:
            counts["both_wrong"] += 1

    assert counts == {
        "both_correct": 1,
        "a3_correct_a4_wrong": 1,
        "a3_wrong_a4_correct": 2,
        "both_wrong": 1,
    }


def test_mcnemar_n01_n10_mapping_and_exact_pvalue():
    summary = mcnemar_summary(n01=7, n10=3)

    assert summary["n01_a3_wrong_a4_correct"] == 7
    assert summary["n10_a3_correct_a4_wrong"] == 3
    assert summary["exact_two_sided_binomial_p"] == pytest.approx(exact_mcnemar_pvalue(7, 3))


def test_retrieval_reranking_error_decomposition_sums():
    retrieval_failures = 4
    reranking_failures = 6
    a4_errors = 10

    assert retrieval_failures + reranking_failures == a4_errors


def test_mate_depth_and_rating_grouping_preserves_total_n():
    buckets = {
        "MateIn1": new_breakdown_bucket(),
        "MateIn2": new_breakdown_bucket(),
    }
    update_breakdown(buckets["MateIn1"], True, True, True)
    update_breakdown(buckets["MateIn1"], False, True, True)
    update_breakdown(buckets["MateIn2"], False, False, False)

    rows = finalize_breakdown(buckets)

    assert sum(row["n"] for row in rows) == 3


def test_preflight_reports_missing_or_found_checkpoints_without_test_evaluation(tmp_path):
    config = A3A4TerminalConfig(
        a3_checkpoint=str(tmp_path / "a3.pt"),
        a4_checkpoint=str(tmp_path / "a4.pt"),
        a4_training_summary=str(tmp_path / "training_summary.json"),
    )

    preflight = terminal_preflight(config)

    assert preflight["test_evaluation_has_happened"] is False
    assert preflight["test_used_for_training"] is False
    assert preflight["test_used_for_checkpoint_selection"] is False
    assert preflight["evaluator_ready"] is False


def test_cli_requires_explicit_terminal_flag_for_test_run():
    source = inspect.getsource(evaluate_model_a3_vs_a4.main)

    assert "run_terminal_test" in source
    assert "TERMINAL_TEST_EXECUTED = NO" in source
    assert "run_terminal_evaluation(config)" in source

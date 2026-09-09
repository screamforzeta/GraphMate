import json

import pytest
import torch

from src.evaluation.model_a.model_a_vs_a2_vs_a3 import (
    MODEL_A3_REFERENCE,
    SHARED_TEST_N,
    a3_parity_diagnostics,
    a3_topk_and_ranks,
    compute_a3_deltas_pp,
    finalize_bucket,
    new_metric_bucket,
    parity_status_a3,
    render_report,
    update_bucket,
    write_outputs,
)
from src.training.model_a.legal_mask import build_legal_mask_from_indices


def test_shared_test_n_reference_is_frozen():
    assert SHARED_TEST_N == 8610
    assert MODEL_A3_REFERENCE["n"] == 8610


def test_a3_candidate_topk_and_legal_rank_are_native():
    scores = torch.tensor([0.1, 0.9, 0.2, 0.7, 0.6])
    ptr = torch.tensor([0, 3, 5])
    targets = torch.tensor([2, 0])

    metrics = a3_topk_and_ranks(scores, ptr, targets)

    assert metrics["correct"][1] == 1
    assert metrics["correct"][3] == 2
    assert metrics["correct"][5] == 2
    assert metrics["ranks"] == [2, 1]
    assert metrics["illegal_top1"] == 0


def test_a3_topk_does_not_move_scores_to_cpu_before_ranking(monkeypatch):
    scores = torch.tensor([0.1, 0.9, 0.2])
    ptr = torch.tensor([0, 3])
    targets = torch.tensor([2])

    def fail_cpu(self):
        raise AssertionError("A3 ranking must stay on the current tensor device.")

    monkeypatch.setattr(torch.Tensor, "cpu", fail_cpu)

    metrics = a3_topk_and_ranks(scores, ptr, targets)

    assert metrics["ranks"] == [2]


def test_a3_parity_passes_for_reference_metrics():
    actual = dict(MODEL_A3_REFERENCE)

    parity = parity_status_a3(actual)

    assert parity["status"] == "PASS"
    assert all(
        item["result"] == "PASS"
        for item in parity["diagnostics"]
    )


def test_a3_loss_amp_sized_diff_passes():
    actual = dict(MODEL_A3_REFERENCE)
    actual["a3_loss"] += 1.940300793368266e-7

    parity = parity_status_a3(actual)

    assert parity["status"] == "PASS"
    loss_row = [
        item
        for item in parity["diagnostics"]
        if item["metric"] == "a3_loss"
    ][0]
    assert loss_row["tolerance"] == pytest.approx(1e-6)
    assert loss_row["result"] == "PASS"


def test_a3_loss_diff_above_metric_tolerance_fails():
    actual = dict(MODEL_A3_REFERENCE)
    actual["a3_loss"] += 1.1e-6

    parity = parity_status_a3(actual)

    assert parity["status"] == "FAIL"
    failed = [
        item
        for item in parity["diagnostics"]
        if item["result"] == "FAIL"
    ]
    assert [item["metric"] for item in failed] == ["a3_loss"]


def test_a3_top1_small_diff_still_fails():
    actual = dict(MODEL_A3_REFERENCE)
    actual["a3_top1"] += 1e-8

    parity = parity_status_a3(actual)

    assert parity["status"] == "FAIL"
    failed = [
        item
        for item in parity["diagnostics"]
        if item["result"] == "FAIL"
    ]
    assert [item["metric"] for item in failed] == ["a3_top1"]


def test_a3_n_mismatch_fails():
    actual = dict(MODEL_A3_REFERENCE)
    actual["n"] += 1

    parity = parity_status_a3(actual)

    assert parity["status"] == "FAIL"
    failed = [
        item
        for item in parity["diagnostics"]
        if item["result"] == "FAIL"
    ]
    assert [item["metric"] for item in failed] == ["n"]


def test_a3_parity_diagnostics_report_each_official_field():
    diagnostics = a3_parity_diagnostics(dict(MODEL_A3_REFERENCE))

    assert [item["metric"] for item in diagnostics] == list(MODEL_A3_REFERENCE)
    assert all(item["abs_diff"] == 0 for item in diagnostics)
    assert all(item["result"] == "PASS" for item in diagnostics)


def test_a3_parity_fails_on_metric_drift():
    actual = dict(MODEL_A3_REFERENCE)
    actual["a3_top1"] += 0.001

    parity = parity_status_a3(actual)

    assert parity["status"] == "FAIL"
    failed = [
        item
        for item in parity["diagnostics"]
        if item["result"] == "FAIL"
    ]
    assert [item["metric"] for item in failed] == ["a3_top1"]


def test_a3_parity_aggregate_boolean_fails_on_missing_metric():
    actual = dict(MODEL_A3_REFERENCE)
    actual.pop("a3_loss")

    parity = parity_status_a3(actual)

    assert parity["status"] == "FAIL"
    failed = [
        item
        for item in parity["diagnostics"]
        if item["result"] == "FAIL"
    ]
    assert failed[0]["metric"] == "a3_loss"
    assert failed[0]["actual"] is None


def test_update_bucket_accumulates_all_four_systems():
    bucket = new_metric_bucket()
    targets = torch.tensor([2])
    legal_mask = build_legal_mask_from_indices([[2, 3, 4]], num_classes=5)
    logits_a = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
    logits_a2 = torch.tensor([[1.0, 0.0, 5.0, 4.0, 3.0]])
    scores_a3 = torch.tensor([0.2, 0.9, 0.1])
    candidate_ptr = torch.tensor([0, 3])
    target_indices_a3 = torch.tensor([1])
    loss_a3 = torch.tensor(0.5)

    update_bucket(
        bucket,
        targets,
        legal_mask,
        logits_a,
        logits_a2,
        scores_a3,
        candidate_ptr,
        target_indices_a3,
        loss_a3,
    )
    result = finalize_bucket(bucket)

    assert result["raw_top1"] == pytest.approx(0.0)
    assert result["best_legal_top1"] == pytest.approx(1.0)
    assert result["a2_masked_top1"] == pytest.approx(1.0)
    assert result["a3_top1"] == pytest.approx(1.0)
    assert result["a3_loss"] == pytest.approx(0.5)
    assert result["a3_illegal_top1_rate"] == pytest.approx(0.0)


def test_delta_calculation_reports_a3_against_all_baselines():
    metrics = {
        "raw_top1": 0.40,
        "raw_top3": 0.50,
        "raw_top5": 0.60,
        "best_legal_top1": 0.50,
        "best_legal_top3": 0.70,
        "best_legal_top5": 0.80,
        "a2_masked_top1": 0.55,
        "a2_masked_top3": 0.75,
        "a2_masked_top5": 0.85,
        "a3_top1": 0.65,
        "a3_top3": 0.86,
        "a3_top5": 0.91,
    }

    deltas = compute_a3_deltas_pp(metrics)

    assert deltas["a3_top1_minus_a_raw_top1"] == pytest.approx(25.0)
    assert deltas["a3_top1_minus_a_best_legal_top1"] == pytest.approx(15.0)
    assert deltas["a3_top1_minus_a2_masked_top1"] == pytest.approx(10.0)


def test_report_generation_contains_required_sections(tmp_path):
    metrics = {
        "n": 8610,
        "raw_top1": 0.4,
        "raw_top3": 0.5,
        "raw_top5": 0.6,
        "best_legal_top1": 0.5,
        "best_legal_top3": 0.7,
        "best_legal_top5": 0.8,
        "a2_masked_top1": 0.55,
        "a2_masked_top3": 0.75,
        "a2_masked_top5": 0.85,
        "a3_top1": 0.65,
        "a3_top3": 0.86,
        "a3_top5": 0.91,
        "a3_loss": 1.2,
        "model_a_mean_legal_rank": 4.0,
        "model_a_median_legal_rank": 2.0,
        "model_a2_mean_legal_rank": 3.0,
        "model_a2_median_legal_rank": 2.0,
        "model_a3_mean_legal_rank": 2.0,
        "model_a3_median_legal_rank": 1.0,
        "raw_illegal_top1_rate": 0.2,
        "a2_masked_illegal_top1_rate": 0.0,
        "a3_illegal_top1_rate": 0.0,
    }
    summary = {
        "shared_test_n": 8610,
        "expected_shared_test_n": 8610,
        "historical_oov": 10,
        "model_a_checkpoint": "a/best.pt",
        "model_a2_checkpoint": "a2/best.pt",
        "model_a3_checkpoint": "a3/best.pt",
        "model_a_parity": {"status": "PASS"},
        "model_a_best_legal_parity": {"status": "PASS"},
        "model_a2_parity": {"status": "PASS"},
        "model_a3_parity": {"status": "PASS"},
        "global": metrics,
        "deltas_pp": compute_a3_deltas_pp(metrics),
        "mate_depth": {"mateIn1": metrics, "mateIn2": metrics},
        "rating": {"<1200": metrics, "1200-1599": metrics},
    }

    report = render_report(summary)
    paths = write_outputs(summary, tmp_path)

    assert "# Model A vs A2 vs A3 Evaluation" in report
    assert "Candidate CE / NLL loss" in report
    assert "## A3 Parity Diagnostics" in report
    assert "## MateIn1 Focus" in report
    assert "## Rating Buckets" in report
    assert json.loads((tmp_path / "summary.json").read_text())["shared_test_n"] == 8610
    assert paths["report_md"].endswith("report.md")

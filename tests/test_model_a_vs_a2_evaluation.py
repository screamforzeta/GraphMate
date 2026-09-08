import json

import pytest
import torch

from src.evaluation.model_a_vs_a2 import (
    compute_deltas_pp,
    finalize_bucket,
    masked_topk_metrics,
    new_metric_bucket,
    parity_status,
    render_report,
    target_legal_ranks,
    update_bucket,
    validate_checkpoint,
    write_outputs,
)
from src.training.legal_mask import build_legal_mask_from_indices


def test_best_legal_topk_promotes_legal_target_after_illegal_raw_leaders():
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
    targets = torch.tensor([2])
    legal_mask = build_legal_mask_from_indices([[2, 3, 4]], num_classes=5)

    metrics, masked_logits = masked_topk_metrics(logits, targets, legal_mask)
    ranks = target_legal_ranks(logits, targets, legal_mask)

    assert logits.argmax(dim=1).item() == 0
    assert masked_logits.argmax(dim=1).item() == 2
    assert metrics[1] == pytest.approx(1.0)
    assert metrics[3] == pytest.approx(1.0)
    assert ranks == [1]


def test_illegal_classes_are_excluded_independently_per_sample():
    logits = torch.tensor(
        [
            [10.0, 9.0, 8.0],
            [10.0, 9.0, 8.0],
        ]
    )
    targets = torch.tensor([1, 0])
    legal_mask = build_legal_mask_from_indices(
        [
            [1, 2],
            [0, 2],
        ],
        num_classes=3,
    )

    metrics, masked_logits = masked_topk_metrics(logits, targets, legal_mask)

    assert masked_logits.argmax(dim=1).tolist() == [1, 0]
    assert metrics[1] == pytest.approx(1.0)


def test_update_bucket_compares_a_raw_best_legal_and_a2_masked():
    bucket = new_metric_bucket()
    targets = torch.tensor([2])
    legal_mask = build_legal_mask_from_indices([[2, 3, 4]], num_classes=5)
    logits_a = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0]])
    logits_a2 = torch.tensor([[1.0, 0.0, 5.0, 4.0, 3.0]])

    update_bucket(bucket, targets, legal_mask, logits_a, logits_a2)
    result = finalize_bucket(bucket)

    assert result["raw_top1"] == pytest.approx(0.0)
    assert result["raw_top3"] == pytest.approx(1.0)
    assert result["best_legal_top1"] == pytest.approx(1.0)
    assert result["a2_masked_top1"] == pytest.approx(1.0)
    assert result["raw_illegal_top1_rate"] == pytest.approx(1.0)
    assert result["a2_masked_illegal_top1_rate"] == pytest.approx(0.0)


def test_parity_status_uses_exact_count_based_accuracy():
    actual = {"n": 2, "raw_top1": 0.5, "raw_top3": 1.0}
    reference = {"n": 2, "raw_top1": 0.5, "raw_top3": 1.0}

    assert parity_status(actual, reference, "model")["status"] == "PASS"


def test_delta_and_report_serialization(tmp_path):
    metrics = {
        "raw_top1": 0.4,
        "raw_top3": 0.5,
        "raw_top5": 0.6,
        "best_legal_top1": 0.5,
        "best_legal_top3": 0.7,
        "best_legal_top5": 0.8,
        "a2_masked_top1": 0.55,
        "a2_masked_top3": 0.75,
        "a2_masked_top5": 0.85,
        "model_a_median_legal_rank": 2,
        "model_a2_median_legal_rank": 1,
    }
    summary = {
        "model_a_checkpoint": "a/best.pt",
        "model_a2_checkpoint": "a2/best.pt",
        "test_graphs": 2,
        "historical_oov": 0,
        "model_a_parity": {"status": "PASS"},
        "model_a2_parity": {"status": "PASS"},
        "global": metrics,
        "deltas_pp": compute_deltas_pp(metrics),
        "mate_depth": {"mateIn1": metrics},
        "rating": {"<1200": metrics},
    }

    paths = write_outputs(summary, tmp_path)
    report = render_report(summary)

    assert "Model A vs A2 Legal Evaluation" in report
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "report.md").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["test_graphs"] == 2
    assert paths["report_md"].endswith("report.md")


def test_validate_checkpoint_requires_real_file(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"checkpoint")

    assert validate_checkpoint(path, "Model").exists()
    with pytest.raises(FileNotFoundError, match="Model A"):
        validate_checkpoint(tmp_path / "missing.pt", "Model A")

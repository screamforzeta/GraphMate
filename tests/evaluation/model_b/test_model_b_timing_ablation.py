"""Regression tests for Model B timing ablation reporting."""

from __future__ import annotations

import pytest

from src.evaluation.model_b.model_b_timing_ablation import (
    MODEL_B_REFERENCE,
    build_shared_comparison_frame,
    canonical_a3_metrics,
    model_b_parity_status,
)


def _b_metrics():
    """Return official B metrics with the required source field."""

    return {"source": "B_SYNTHETIC_TIMING", **MODEL_B_REFERENCE}


def test_shared_frame_uses_distinct_metric_sources():
    """A3 and B columns must not silently share the same metrics dict."""

    frame = build_shared_comparison_frame(canonical_a3_metrics(), _b_metrics())

    top1 = next(row for row in frame if row["metric"] == "Top1")
    assert top1["a3_source"] == "OFFICIAL_A3_REFERENCE"
    assert top1["b_source"] == "B_SYNTHETIC_TIMING"
    assert top1["a3"] != top1["b"]


def test_shared_frame_rejects_b_metrics_reused_as_a3():
    """Prevent the previous reporting bug at the provenance level."""

    reused_b_as_a3 = _b_metrics()

    with pytest.raises(ValueError, match="OFFICIAL_A3_REFERENCE"):
        build_shared_comparison_frame(reused_b_as_a3, _b_metrics())


def test_shared_frame_computes_accuracy_deltas_from_inputs():
    """Delta percentage points are computed from metric values, not literals."""

    frame = build_shared_comparison_frame(canonical_a3_metrics(), _b_metrics())
    top1 = next(row for row in frame if row["metric"] == "Top1")

    expected = (
        MODEL_B_REFERENCE["top1"] - canonical_a3_metrics()["top1"]
    ) * 100
    assert top1["delta_b_minus_a3_pp"] == pytest.approx(expected)


def test_model_b_reference_values_produce_parity_pass():
    """Official terminal Model B metrics pass the parity checker."""

    parity = model_b_parity_status(MODEL_B_REFERENCE)

    assert parity["status"] == "PASS"
    assert all(item["result"] == "PASS" for item in parity["diagnostics"])


def test_model_b_real_metric_change_produces_parity_fail():
    """A genuine Top1 change must fail terminal parity."""

    changed = dict(MODEL_B_REFERENCE)
    changed["top1"] += 1e-8

    parity = model_b_parity_status(changed)

    assert parity["status"] == "FAIL"
    top1 = next(item for item in parity["diagnostics"] if item["metric"] == "top1")
    assert top1["result"] == "FAIL"

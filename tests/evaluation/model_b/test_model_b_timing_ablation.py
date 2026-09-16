"""Regression tests for Model B timing ablation reporting."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.model_b.model_b_timing_ablation import (
    MODEL_B_REFERENCE,
    _graph_metadata,
    _merge_per_puzzle,
    _transition_counts,
    _validate_puzzle_alignment,
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


def _row():
    """Return CSV-like puzzle metadata."""

    return SimpleNamespace(PuzzleId="puzzle-1", Rating=1500, MateDepth=2)


def _base_graph():
    """Return a minimal metadata graph shared by A3 and B tests."""

    return Data(fen="8/8/8/8/8/8/8/K6k w - - 0 1", target_move="a1a2")


def test_graph_metadata_allows_a3_graph_without_timing_fields():
    """A3 no-timing graphs must not need timing attributes."""

    metadata = _graph_metadata(_base_graph(), _row())

    assert metadata["PuzzleId"] == "puzzle-1"
    assert metadata["previous_move_time_seconds"] is None
    assert metadata["original_move_time_seconds"] is None
    assert metadata["previous_move_time"] is None
    assert metadata["original_move_time"] is None
    assert metadata["time_is_synthetic"] is None


def test_graph_metadata_preserves_model_b_timing_fields():
    """Timing-aware B rows keep raw and normalized timing values."""

    graph = _base_graph()
    graph.previous_move_time_seconds = torch.tensor([11.5])
    graph.original_move_time_seconds = torch.tensor([17.25])
    graph.previous_move_time = torch.tensor([-0.2])
    graph.original_move_time = torch.tensor([0.4])
    graph.time_is_synthetic = torch.tensor([1.0])

    metadata = _graph_metadata(graph, _row())

    assert metadata["previous_move_time_seconds"] == pytest.approx(11.5)
    assert metadata["original_move_time_seconds"] == pytest.approx(17.25)
    assert metadata["previous_move_time"] == pytest.approx(-0.2)
    assert metadata["original_move_time"] == pytest.approx(0.4)
    assert metadata["time_is_synthetic"] == pytest.approx(1.0)


def test_paired_merge_uses_b_synthetic_as_timing_source():
    """Final paired rows must get timing metadata from B synthetic, not A3."""

    common = {
        "PuzzleId": "puzzle-1",
        "FEN": "fen",
        "target_uci": "a1a2",
        "rating": 1500,
        "MateDepth": 2,
        "number_legal_candidates": 3,
    }
    a3 = [
        {
            **common,
            "previous_move_time_seconds": None,
            "original_move_time_seconds": None,
            "previous_move_time": None,
            "original_move_time": None,
            "time_is_synthetic": None,
            "A3_target_rank": 2,
            "A3_top1_correct": False,
            "A3_prediction": "a1b1",
        }
    ]
    synthetic = [
        {
            **common,
            "previous_move_time_seconds": 9.0,
            "original_move_time_seconds": 12.0,
            "previous_move_time": -0.1,
            "original_move_time": 0.2,
            "time_is_synthetic": 1.0,
            "B_SYNTHETIC_TIMING_target_rank": 1,
            "B_SYNTHETIC_TIMING_top1_correct": True,
            "B_SYNTHETIC_TIMING_prediction": "a1a2",
        }
    ]
    neutral = [
        {
            **common,
            "previous_move_time_seconds": 9.0,
            "original_move_time_seconds": 12.0,
            "previous_move_time": 0.0,
            "original_move_time": 0.0,
            "time_is_synthetic": 1.0,
            "B_NEUTRAL_TIMING_target_rank": 3,
            "B_NEUTRAL_TIMING_top1_correct": False,
            "B_NEUTRAL_TIMING_prediction": "a1b1",
        }
    ]

    merged = _merge_per_puzzle(a3, synthetic, neutral)
    transitions = _transition_counts(merged, "A3", "B_SYNTHETIC_TIMING")

    assert len(merged) == 1
    assert merged[0]["previous_move_time_seconds"] == 9.0
    assert merged[0]["original_move_time_seconds"] == 12.0
    assert merged[0]["previous_move_time"] == -0.1
    assert merged[0]["original_move_time"] == 0.2
    assert merged[0]["time_is_synthetic"] == 1.0
    assert transitions["left_wrong_right_correct"] == 1


def test_validate_puzzle_alignment_detects_order_without_requiring_it(monkeypatch):
    """Pairing may be by PuzzleId when order differs but sets are equal."""

    monkeypatch.setattr(
        "src.evaluation.model_b.model_b_timing_ablation.SHARED_TEST_N", 2
    )
    a3 = [{"PuzzleId": "a"}, {"PuzzleId": "b"}]
    synthetic = [{"PuzzleId": "b"}, {"PuzzleId": "a"}]

    alignment = _validate_puzzle_alignment(a3, synthetic)

    assert alignment == {"same_order": False, "same_set": True, "n": 2}

"""Regression tests for Model B timing ablation reporting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from src.evaluation.model_b.model_b_timing_ablation import (
    MODEL_B_REFERENCE,
    _graph_metadata,
    _merge_per_puzzle,
    _timing_distribution,
    _transition_counts,
    _validate_puzzle_alignment,
    build_shared_comparison_frame,
    canonical_a3_metrics,
    extract_real_move_times,
    model_b_parity_status,
    update_timing_distribution_only,
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


def test_extract_real_move_times_flattens_sequences_and_filters(tmp_path):
    """MoveTimes extraction keeps positive finite seconds and records discards."""

    path = tmp_path / "games.csv"
    path.write_text(
        "MoveTimes\n"
        "\"[1.0, 2.5, None, 0.0, -1.0]\"\n"
        "\"[3, 1e309]\"\n"
        "\"bad\"\n",
        encoding="utf-8",
    )

    audit = extract_real_move_times({"fixture": path})

    assert audit["status"] == "COMPLETE"
    assert audit["games_scanned"] == 3
    assert audit["games_with_timing"] == 2
    assert audit["raw_timing_values"] == 7
    assert audit["valid_timing_values"] == 3
    assert audit["seconds"] == [1.0, 2.5, 3.0]
    assert audit["discard_reasons"]["missing"] == 1
    assert audit["discard_reasons"]["non_finite"] == 1
    assert audit["discard_reasons"]["non_positive"] == 2
    assert audit["discard_reasons"]["parse_errors"] == 1


def test_extract_real_move_times_missing_source_is_incomplete(tmp_path):
    """Missing games source returns explicit INCOMPLETE status."""

    audit = extract_real_move_times({"missing": tmp_path / "missing.csv"})

    assert audit["status"] == "INCOMPLETE"
    assert audit["valid_timing_values"] == 0
    assert audit["warnings"]


def test_extract_real_move_times_present_source_has_positive_n(tmp_path):
    """A valid source reports non-zero real timing count."""

    path = tmp_path / "games.csv"
    path.write_text('MoveTimes\n"[4.0, 5.0]"\n', encoding="utf-8")

    audit = extract_real_move_times({"fixture": path})

    assert audit["status"] == "COMPLETE"
    assert audit["valid_timing_values"] == 2


def test_timing_distribution_statistics_are_deterministic(monkeypatch):
    """Known real/synthetic values produce deterministic descriptive stats."""

    monkeypatch.setattr(
        "src.evaluation.model_b.model_b_timing_ablation.extract_real_move_times",
        lambda: {
            "status": "COMPLETE",
            "source": "fixture",
            "unit": "seconds",
            "paths_searched": {},
            "paths_used": ["fixture"],
            "games_scanned": 1,
            "games_with_timing": 1,
            "raw_timing_values": 3,
            "valid_timing_values": 3,
            "discarded_values": 0,
            "discard_reasons": {
                "missing": 0,
                "non_numeric": 0,
                "non_finite": 0,
                "non_positive": 0,
                "parse_errors": 0,
            },
            "warnings": [],
            "seconds": [1.0, 2.0, 4.0],
        },
    )
    rows = [
        {"original_move_time_seconds": 2.0, "rating": 1000},
        {"original_move_time_seconds": 4.0, "rating": 1200},
    ]

    distribution = _timing_distribution(rows)

    assert distribution["status"] == "COMPLETE"
    assert distribution["real_seconds"]["n"] == 3
    assert distribution["real_seconds"]["mean"] == pytest.approx(7 / 3)
    assert distribution["synthetic_original_seconds"]["median"] == pytest.approx(3.0)
    assert distribution["ratios"]["synthetic_mean_over_real_mean"] == pytest.approx(9 / 7)


def test_timing_distribution_only_preserves_model_metrics(tmp_path, monkeypatch):
    """Distribution-only mode updates timing stats without touching model metrics."""

    output_dir = tmp_path / "ablation"
    output_dir.mkdir()
    summary = {
        "official_model_performance": {"A3": {"top1": 0.1}},
        "post_hoc_diagnostic_ablations": {"B_NEUTRAL_TIMING": {"top1": 0.2}},
        "transitions": {"x": 1},
        "mate_depth": {"mateIn1": {}},
        "rating": {"<1200": {}},
        "mcnemar": {"x": {}},
        "rank_delta": {"x": {}},
        "timing_distribution_vs_real_games": {"status": "OLD"},
    }
    paired_rows = [{"original_move_time_seconds": 2.0}]
    (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output_dir / "paired_test_rows.json").write_text(
        json.dumps(paired_rows), encoding="utf-8"
    )
    monkeypatch.setattr(
        "src.evaluation.model_b.model_b_timing_ablation.extract_real_move_times",
        lambda: {
            "status": "COMPLETE",
            "source": "fixture",
            "unit": "seconds",
            "paths_searched": {},
            "paths_used": ["fixture"],
            "games_scanned": 1,
            "games_with_timing": 1,
            "raw_timing_values": 1,
            "valid_timing_values": 1,
            "discarded_values": 0,
            "discard_reasons": {
                "missing": 0,
                "non_numeric": 0,
                "non_finite": 0,
                "non_positive": 0,
                "parse_errors": 0,
            },
            "warnings": [],
            "seconds": [1.0],
        },
    )

    outputs = update_timing_distribution_only(output_dir)
    updated = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert outputs["timing_distribution_status"] == "COMPLETE"
    assert updated["official_model_performance"] == summary["official_model_performance"]
    assert updated["post_hoc_diagnostic_ablations"] == summary["post_hoc_diagnostic_ablations"]
    assert updated["timing_distribution_vs_real_games"]["real_seconds"]["n"] == 1

"""Load frozen result artifacts without running evaluations."""

from __future__ import annotations

from pathlib import Path
import json


ARTIFACT_PATHS = {
    "a4_terminal": Path("artifacts/model_a4_terminal_evaluation/summary.json"),
    "model_b_timing": Path("artifacts/model_b_timing_ablation/summary.json"),
    "a3_comparison": Path("artifacts/model_a_vs_a2_vs_a3_evaluation/summary.json"),
}


def load_json_if_exists(path):
    """Load a JSON artifact if present, otherwise return None."""

    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_results():
    """Return all frozen result artifacts that are present locally."""

    return {
        name: load_json_if_exists(path)
        for name, path in ARTIFACT_PATHS.items()
    }


def canonical_result_snapshot():
    """Return immutable headline metrics documented from frozen artifacts."""

    return {
        "A3_test_top1": 0.675609756097561,
        "A3_test_top5": 0.9185830429732869,
        "A4_test_end_to_end_top1": 0.854123112659698,
        "A4_test_delta_pp": 17.851335656213696,
        "B_synthetic_top1": 0.6598141695702672,
        "B_minus_A3_pp": -1.57955805299938,
    }


def canonical_mate_depth_rows():
    """Return documented A3/A4 MateDepth frozen metrics."""

    rows = [
        ("Mate in 1", None, 0.7955, 0.9393),
        ("Mate in 2", None, 0.6978, 0.8819),
        ("Mate in 3", None, 0.6673, 0.8459),
        ("Mate in 4", None, 0.5873, 0.7839),
        ("Mate in 5", None, 0.5291, 0.7427),
    ]
    return [
        {
            "Puzzle depth": depth,
            "N": count,
            "A3": a3,
            "A4": a4,
            "Improvement": a4 - a3,
        }
        for depth, count, a3, a4 in rows
    ]


def canonical_rating_rows():
    """Return documented A3/A4 rating-band frozen metrics."""

    rows = [
        ("<1200", None, 0.8206, 0.9554),
        ("1200-1599", None, 0.6585, 0.8690),
        ("1600-1999", None, 0.5134, 0.7583),
        ("2000-2399", None, 0.4316, 0.6042),
        ("2400+", None, 0.2667, 0.4500),
    ]
    return [
        {
            "Rating": rating,
            "N": count,
            "A3": a3,
            "A4": a4,
            "Improvement": a4 - a3,
        }
        for rating, count, a3, a4 in rows
    ]


def format_metric_rows(rows):
    """Format metric rows as percentages for Streamlit display."""

    formatted = []
    for row in rows:
        item = row.copy()
        for key in ("A3", "A4", "Improvement"):
            if item.get(key) is not None:
                item[key] = f"{item[key] * 100:.2f}%"
        formatted.append(item)
    return formatted

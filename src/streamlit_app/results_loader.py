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


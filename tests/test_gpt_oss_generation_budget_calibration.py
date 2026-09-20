from types import SimpleNamespace

import pytest

from src.cli.evaluation import calibrate_gpt_oss_generation_budget as calibration


def fake_puzzles(n=80):
    return [
        SimpleNamespace(
            puzzle_id=f"p{i}",
            initial_solver_fen="8/8/8/8/8/8/8/K6k w - - 0 1",
            target_move="a1a2",
        )
        for i in range(n)
    ]


def test_deterministic_subset_and_same_subset_for_every_budget(monkeypatch, tmp_path):
    dataset = tmp_path / "test.csv"
    dataset.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(calibration, "load_lichess_test", lambda path: fake_puzzles())

    first = calibration.load_or_create_subset(dataset, tmp_path / "out", 50, 42)
    second = calibration.load_or_create_subset(dataset, tmp_path / "out", 50, 42)

    assert first == second
    assert len(first["selected"]) == 50
    assert len({row["puzzle_id"] for row in first["selected"]}) == 50


@pytest.mark.parametrize(
    ("response", "status"),
    [
        ({"error": "boom", "final_content": "e2e4", "metadata": {"done_reason": "stop"}}, "RUNTIME_ERROR"),
        ({"error": None, "final_content": "e2e4", "metadata": {"done_reason": "length"}}, "TOKEN_BUDGET_EXHAUSTED"),
        ({"error": None, "final_content": "", "metadata": {"done_reason": "stop"}}, "EMPTY_FINAL_RESPONSE"),
        ({"error": None, "final_content": "not uci but done", "metadata": {"done_reason": "stop"}}, "GENERATION_COMPLETE"),
    ],
)
def test_completion_status_ignores_chess_correctness_legality_and_target(response, status):
    assert calibration.classify_generation(response) == status


def test_budget_threshold_47_fails_48_passes():
    fail_records = [{"calibration_status": "GENERATION_COMPLETE"} for _ in range(47)]
    fail_records += [{"calibration_status": "EMPTY_FINAL_RESPONSE"} for _ in range(3)]
    pass_records = [{"calibration_status": "GENERATION_COMPLETE"} for _ in range(48)]
    pass_records += [{"calibration_status": "EMPTY_FINAL_RESPONSE"} for _ in range(2)]

    assert calibration.budget_passes(calibration.summarize_budget(fail_records, 128), 0.95) is False
    assert calibration.budget_passes(calibration.summarize_budget(pass_records, 128), 0.95) is True


def test_run_calibration_selects_first_passing_budget_and_skips_larger(monkeypatch, tmp_path):
    dataset = tmp_path / "test.csv"
    dataset.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(calibration, "load_lichess_test", lambda path: fake_puzzles(60))
    calls = []

    class FakeClient:
        def __init__(self, endpoint, timeout=120):
            pass

        def generate(self, model, prompt, system, options=None, think=None):
            calls.append(options["num_predict"])
            budget = options["num_predict"]
            index_for_budget = calls.count(budget)
            complete = budget == 256 and index_for_budget <= 48
            return {
                "final_content": "any text" if complete else "",
                "thinking": "thinking",
                "metadata": {"done_reason": "stop" if complete else "length", "eval_count": budget},
                "latency_seconds": 0.01,
                "error": None,
            }

    monkeypatch.setattr(calibration, "OllamaClient", FakeClient)

    summary = calibration.run_calibration(
        endpoint="http://localhost:11436",
        model_id="gpt_oss_20b",
        dataset_path=dataset,
        output_dir=tmp_path / "calibration",
        sample_size=50,
        seed=42,
        budgets=[128, 256, 512],
        completion_threshold=0.95,
    )

    assert summary["selected_budget"] == 256
    assert calls.count(128) == 50
    assert calls.count(256) == 50
    assert calls.count(512) == 0
    assert summary["selection_used_chess_correctness"] is False
    assert summary["target_used"] is False


def test_resume_identity_mismatch_fails_closed(monkeypatch, tmp_path):
    dataset = tmp_path / "test.csv"
    dataset.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(calibration, "load_lichess_test", lambda path: fake_puzzles(60))
    subset = calibration.load_or_create_subset(dataset, tmp_path / "out", 50, 42)
    identity = calibration.calibration_identity("gpt_oss_20b", 128, subset)
    path = tmp_path / "out" / "budget_128.jsonl"
    calibration.append_jsonl(path, {"identity": identity | {"budget": 999}, "puzzle_id": "p0"})

    with pytest.raises(RuntimeError, match="identity mismatch"):
        calibration.load_budget_records(path, identity)


def test_official_artifact_paths_cannot_be_used():
    with pytest.raises(RuntimeError, match="official"):
        calibration.assert_calibration_output_dir(
            calibration.Path("artifacts/llm_benchmark/official/gpt_oss_generation_budget")
        )

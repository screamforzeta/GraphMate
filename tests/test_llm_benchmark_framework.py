from pathlib import Path

import pytest

from src.llm.datasets import BenchmarkPuzzle, validate_heldout_classic_record
from src.cli.evaluation.benchmark_llm_chess import (
    build_preflight,
    run_runtime_calibration,
    selected_model_ids,
    validate_official_run_preconditions,
)
from src.llm.model_registry import (
    LLM_BENCHMARK_MODELS,
    discover_registry,
    generation_config_for_model,
    official_model_ready,
    verify_runtime_registry,
)
from src.llm.ollama_client import OllamaClient, extract_final_content_and_thinking, resolve_endpoint
from src.llm.parsing import parse_uci_response
from src.llm.prompting import build_prompt
from src.llm.resume import append_jsonl, assert_resume_config_matches, load_completed_keys
from src.llm.scoring import evaluate_reference_line, score_next_move, summarize_next_move_records


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PROMOTION_FEN = "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"


def test_discover_registry_uses_actual_ollama_tags():
    registry = discover_registry(
        [
            {
                "name": "qwen3.5:4b",
                "digest": LLM_BENCHMARK_MODELS["qwen_3_5_4b"]["digest"],
                "size": 3389983735,
            },
            {
                "name": "qwen3.5:9b",
                "digest": LLM_BENCHMARK_MODELS["qwen_3_5_9b"]["digest"],
                "size": 6594474711,
            },
            {
                "name": "gpt-oss:20b",
                "digest": LLM_BENCHMARK_MODELS["gpt_oss_20b"]["digest"],
                "size": 13793441244,
            },
        ]
    )

    assert registry["qwen_3_5_4b"]["ollama_model"] == "qwen3.5:4b"
    assert registry["qwen_3_5_4b"]["digest_match"] is True
    assert registry["qwen_3_5_9b"]["ollama_model"] == "qwen3.5:9b"
    assert registry["gpt_oss_20b"]["ollama_model"] == "gpt-oss:20b"


def test_registry_digest_mismatch_is_detected():
    verification = verify_runtime_registry([{"name": "qwen3.5:4b", "digest": "wrong"}])

    assert verification["qwen_3_5_4b"]["found"] is True
    assert verification["qwen_3_5_4b"]["digest_match"] is False
    assert verification["qwen_3_5_9b"]["found"] is False


def test_endpoint_resolution_cli_env_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://localhost:11436")

    assert resolve_endpoint("http://example.test:1") == "http://example.test:1"
    assert resolve_endpoint(None) == "http://localhost:11436"


def test_thinking_is_separated_from_final_content_and_not_parsed():
    final, thinking = extract_final_content_and_thinking(
        {"message": {"thinking": "I should play e2e4", "content": ""}}
    )
    parsed = parse_uci_response(final, START_FEN)

    assert thinking == "I should play e2e4"
    assert final == ""
    assert parsed.status == "PARSE_ERROR"


def test_generate_style_response_final_content_is_used():
    final, thinking = extract_final_content_and_thinking({"response": "e2e4"})

    assert final == "e2e4"
    assert thinking is None


def test_thinking_false_is_serialized_into_ollama_request(monkeypatch):
    captured = {}

    def fake_request(self, path, payload=None):
        captured.update(payload)
        return {"response": "e2e4"}

    monkeypatch.setattr(OllamaClient, "_json_request", fake_request)
    response = OllamaClient("http://localhost:11436").generate(
        "qwen3.5:4b",
        prompt="FEN:\n...\n\nReturn your move in UCI notation only.",
        system="system",
        thinking_enabled=False,
    )

    assert captured["think"] is False
    assert response["request_payload"]["think"] is False
    assert response["thinking_requested"] is False


def test_model_specific_thinking_configuration():
    qwen = generation_config_for_model("qwen_3_5_4b")
    gpt_oss = generation_config_for_model("gpt_oss_20b")

    assert qwen["think"] is False
    assert qwen["options"]["num_predict"] == 16
    assert qwen["status"] == "RUNTIME_VALIDATED"
    assert gpt_oss["think"] == "low"
    assert gpt_oss["think"] is not False
    assert gpt_oss["options"]["num_predict"] == 64
    assert gpt_oss["thinking_mode"] == "low"
    assert gpt_oss["status"] == "RUNTIME_VALIDATED"


def test_effective_endpoint_metadata_uses_ollama_url(monkeypatch):
    class Args:
        endpoint = "http://localhost:11434"
        ollama_url = "http://localhost:11436"
        timeout = 1
        dataset = "lichess-test"
        thinking_enabled = False

    monkeypatch.setattr(OllamaClient, "version", lambda self: {"version": "0.32.5"})
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: {"models": []})

    preflight = build_preflight(Args())

    assert preflight["endpoint"] == "http://localhost:11436"
    assert preflight["ollama_url"] == "http://localhost:11436"
    assert preflight["runtime_ready"]["qwen_3_5_4b"] is False
    assert preflight["runtime_ready"]["qwen_3_5_9b"] is False
    assert preflight["runtime_ready"]["gpt_oss_20b"] is False
    assert preflight["ready_for_official_benchmark"] is False


def test_official_readiness_true_when_all_tags_and_digests_match():
    models = [
        {
            "name": model["ollama_model"],
            "digest": model["digest"],
            "size": model["size"],
        }
        for model in LLM_BENCHMARK_MODELS.values()
    ]
    verification = verify_runtime_registry(models)

    assert all(official_model_ready(model_id, verification) for model_id in LLM_BENCHMARK_MODELS)


def test_preflight_ready_for_official_when_all_checks_pass(monkeypatch):
    class Args:
        endpoint = "http://localhost:11434"
        ollama_url = "http://localhost:11436"
        timeout = 1
        dataset = "lichess-test"
        thinking_enabled = False

    models = [
        {
            "name": model["ollama_model"],
            "digest": model["digest"],
            "size": model["size"],
        }
        for model in LLM_BENCHMARK_MODELS.values()
    ]
    monkeypatch.setattr(OllamaClient, "version", lambda self: {"version": "0.32.5"})
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: {"models": models})

    preflight = build_preflight(Args())

    assert preflight["runtime_ready"] == {
        "qwen_3_5_4b": True,
        "qwen_3_5_9b": True,
        "gpt_oss_20b": True,
    }
    assert preflight["ready_for_official_benchmark"] is True


def test_official_guard_refuses_digest_mismatch():
    preflight = {
        "ollama_reachable": True,
        "parser_tests_pass": True,
        "dataset_fingerprint": "abc",
        "registry_verification": {
            "qwen_3_5_4b": {
                "found": True,
                "digest_match": False,
            }
        },
    }

    with pytest.raises(RuntimeError, match="digest mismatch"):
        validate_official_run_preconditions(["qwen_3_5_4b"], preflight)


def test_runtime_smoke_model_selection_is_sequential():
    args = type("Args", (), {"all_models": True, "model": "qwen_3_5_4b"})()

    assert selected_model_ids(args) == ["qwen_3_5_4b", "qwen_3_5_9b", "gpt_oss_20b"]


def test_prompt_has_no_target_or_metadata_leakage():
    prompt = build_prompt(START_FEN)
    payload = f"{prompt['system']}\n{prompt['user']}"

    assert START_FEN in payload
    assert "TargetMove" not in payload
    assert "MateDepth" not in payload
    assert "Rating" not in payload
    assert "Themes" not in payload
    assert "legal moves" not in payload.lower()


@pytest.mark.parametrize(
    ("raw", "accepted"),
    [
        ("e2e4", True),
        (" e2e4 ", True),
        ("e7e8q", True),
        ("The best move is e2e4.", False),
        ("e2-e4", False),
        ("Nf3", False),
        ("e9e4", False),
        ("0000", False),
        ("", False),
        ("e2e4 e7e5", False),
        ("```e2e4```", False),
    ],
)
def test_strict_parser_policy(raw, accepted):
    fen = PROMOTION_FEN if raw.strip() == "e7e8q" else START_FEN
    result = parse_uci_response(raw, fen)

    assert result.parse_success is accepted


def test_legality_and_next_move_scoring():
    correct = score_next_move("e2e4", START_FEN, "e2e4")
    wrong = score_next_move("d2d4", START_FEN, "e2e4")
    illegal = score_next_move("e2e5", START_FEN, "e2e4")

    assert correct["outcome"] == "CORRECT"
    assert wrong["outcome"] == "LEGAL_BUT_WRONG"
    assert illegal["outcome"] == "ILLEGAL_MOVE"


def test_reference_line_stops_on_wrong_or_illegal_and_completes_mate_in_one():
    puzzle = BenchmarkPuzzle("smoke", START_FEN, "e2e4", ["e2e4", "e7e5"], mate_depth=1)

    solved = evaluate_reference_line(puzzle, lambda _fen: "e2e4")
    wrong = evaluate_reference_line(puzzle, lambda _fen: "d2d4")
    illegal = evaluate_reference_line(puzzle, lambda _fen: "e2e5")

    assert solved["solved"] is True
    assert solved["solver_moves_required"] == 1
    assert wrong["solved"] is False
    assert wrong["failure_step"] == 1
    assert illegal["steps"][0]["outcome"] == "ILLEGAL_MOVE"


def test_reference_line_mate_depth_semantics_and_no_chat_history_needed():
    puzzle = BenchmarkPuzzle(
        "mate2_like",
        START_FEN,
        "e2e4",
        ["e2e4", "e7e5", "g1f3"],
        mate_depth=2,
    )
    seen_fens = []

    def predictor(fen):
        seen_fens.append(fen)
        return "e2e4" if len(seen_fens) == 1 else "g1f3"

    result = evaluate_reference_line(puzzle, predictor)

    assert result["solved"] is True
    assert result["solver_moves_required"] == 2
    assert len(seen_fens) == 2
    assert seen_fens[0] != seen_fens[1]


def test_summary_counts_and_resume_guards(tmp_path):
    records = [
        score_next_move("e2e4", START_FEN, "e2e4"),
        score_next_move("d2d4", START_FEN, "e2e4"),
        score_next_move("e2e5", START_FEN, "e2e4"),
    ]
    summary = summarize_next_move_records(records)

    assert summary["N"] == 3
    assert summary["correct"] == 1
    assert summary["illegal_move_count"] == 1

    config = {
        "run_id": "r1",
        "model_id": "m",
        "ollama_model": "tag",
        "model_digest": "digest",
        "dataset": "lichess-test",
        "protocol": "next-move",
        "dataset_fingerprint": "abc",
        "prompt_version": "llm_chess_uci_v1",
        "prompt_hash": "prompt",
        "parser_version": "parser",
        "temperature": 0,
        "think": False,
        "thinking_mode": "disabled",
        "num_predict": 16,
        "generation_options": {"temperature": 0},
        "thinking_enabled": False,
        "model_generation_config": {"think": False, "num_predict": 16},
    }
    assert_resume_config_matches(config, dict(config))
    changed = dict(config)
    changed["prompt_hash"] = "other"
    with pytest.raises(ValueError):
        assert_resume_config_matches(config, changed)
    changed_thinking = dict(config)
    changed_thinking["thinking_enabled"] = True
    with pytest.raises(ValueError):
        assert_resume_config_matches(config, changed_thinking)
    changed_model_config = dict(config)
    changed_model_config["model_generation_config"] = {"think": "low", "num_predict": 64}
    with pytest.raises(ValueError):
        assert_resume_config_matches(config, changed_model_config)

    path = tmp_path / "raw_predictions.jsonl"
    append_jsonl(path, {"run_id": "r1", "model_id": "m", "puzzle_id": "p", "protocol": "next-move"})
    assert ("r1", "m", "p", "next-move") in load_completed_keys(path)


def test_heldout_classic_schema():
    assert validate_heldout_classic_record(
        {
            "puzzle_id": "classic-1",
            "source": "fixture",
            "initial_solver_fen": START_FEN,
            "reference_moves_uci": ["e2e4"],
            "mate_depth": 1,
        }
    )
    with pytest.raises(ValueError):
        validate_heldout_classic_record({"puzzle_id": "bad"})


def test_gpt_oss_calibration_stops_on_parseable_content_not_correctness(monkeypatch):
    calls = []

    class Args:
        model = "gpt_oss_20b"
        endpoint = "http://localhost:11434"
        ollama_url = "http://localhost:11436"
        timeout = 1

    def fake_generate(self, model, prompt, system, options=None, think=None):
        calls.append(options["num_predict"])
        if options["num_predict"] < 64:
            return {
                "final_content": "",
                "thinking": "still thinking e2e4",
                "thinking_requested": think,
                "thinking_returned": True,
                "metadata": {"eval_count": options["num_predict"], "done_reason": "length"},
                "latency_seconds": 0.1,
            }
        return {
            "final_content": "d2d4",
            "thinking": "done",
            "thinking_requested": think,
            "thinking_returned": True,
            "metadata": {"eval_count": options["num_predict"], "done_reason": "stop"},
            "latency_seconds": 0.1,
        }

    monkeypatch.setattr(OllamaClient, "generate", fake_generate)
    registry = {"gpt_oss_20b": LLM_BENCHMARK_MODELS["gpt_oss_20b"]}

    result = run_runtime_calibration(Args(), registry)

    assert calls == [32, 64]
    assert result["selected_budget"] == 64
    assert result["trials"][-1]["parse_status"] == "LEGAL_BUT_WRONG"

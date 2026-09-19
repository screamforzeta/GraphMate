"""CLI for local Ollama LLM chess benchmark preflight and smoke tests."""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from src.llm.datasets import dataset_fingerprint, load_lichess_test
from src.llm.model_registry import (
    LLM_BENCHMARK_MODELS,
    discover_registry,
    generation_config_for_model,
    verify_runtime_registry,
)
from src.llm.ollama_client import (
    DEFAULT_GENERATION_OPTIONS,
    DEFAULT_OLLAMA_ENDPOINT,
    DEFAULT_THINKING_ENABLED,
    GENERATION_CONFIG_STATUS,
    OllamaClient,
    resolve_endpoint,
)
from src.llm.parsing import PARSER_VERSION
from src.llm.prompting import PROMPT_TEMPLATE_VERSION, build_prompt, prompt_hash
from src.llm.scoring import evaluate_reference_line, score_next_move


GPT_OSS_CALIBRATION_BUDGETS = [32, 64, 128, 256]


def parser_tests_pass():
    """Run minimal parser self-checks used by preflight."""

    from src.llm.parsing import parse_uci_response

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    cases = {
        "e2e4": True,
        " e2e4 ": True,
        "The best move is e2e4.": False,
        "e2-e4": False,
        "Nf3": False,
        "": False,
        "e2e4 e7e5": False,
    }
    return all(parse_uci_response(raw, fen).parse_success is expected for raw, expected in cases.items())


def build_preflight(args):
    """Collect non-invasive benchmark readiness metadata."""

    endpoint = resolve_endpoint(args.ollama_url or args.endpoint)
    client = OllamaClient(endpoint, timeout=args.timeout)
    try:
        version = client.version()
        tags = client.list_models()
        reachable = True
        error = None
    except Exception as exc:
        version = None
        tags = {"models": []}
        reachable = False
        error = str(exc)
    registry = discover_registry(tags.get("models", []))
    verification = verify_runtime_registry(tags.get("models", []))
    dataset_path = Path("data/final/puzzles/test.csv")
    dataset_n = len(load_lichess_test(dataset_path)) if dataset_path.exists() else None
    fingerprint = dataset_fingerprint(dataset_path) if dataset_path.exists() else None
    return {
        "status": "LLM_CHESS_BENCHMARK_PREFLIGHT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "ollama_reachable": reachable,
        "ollama_error": error,
        "ollama_version": version,
        "endpoint": endpoint,
        "ollama_url": endpoint,
        "discovered_registry": registry,
        "frozen_registry": LLM_BENCHMARK_MODELS,
        "registry_verification": verification,
        "expected_models_found": {
            key: key in registry
            for key in LLM_BENCHMARK_MODELS
        },
        "expected_model_digest_match": {
            key: verification[key]["digest_match"]
            for key in LLM_BENCHMARK_MODELS
        },
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": prompt_hash(),
        "parser_version": PARSER_VERSION,
        "generation_options": DEFAULT_GENERATION_OPTIONS,
        "model_generation_configs": {
            model_id: model["generation_config"]
            for model_id, model in LLM_BENCHMARK_MODELS.items()
        },
        "thinking_enabled": args.thinking_enabled,
        "generation_config_status": GENERATION_CONFIG_STATUS,
        "dataset": args.dataset,
        "dataset_n": dataset_n,
        "dataset_fingerprint": fingerprint,
        "target_leakage": False,
        "mate_depth_given_to_model": False,
        "rating_given_to_model": False,
        "themes_given_to_model": False,
        "legal_moves_given_to_model": False,
        "parser_tests_pass": parser_tests_pass(),
        "official_benchmark_executed": False,
        "ready_for_smoke_test": reachable and parser_tests_pass(),
        "ready_for_official_benchmark": (
            reachable
            and parser_tests_pass()
            and all(value["found"] and value["digest_match"] for value in verification.values())
            and all(
                model["generation_config"]["status"] == "RUNTIME_VALIDATED"
                for model in LLM_BENCHMARK_MODELS.values()
            )
        ),
        "runtime_ready": {
            model_id: model["generation_config"]["status"] == "RUNTIME_VALIDATED"
            for model_id, model in LLM_BENCHMARK_MODELS.items()
        },
    }


def smoke_fixture():
    """Return one non-official smoke puzzle."""

    from src.llm.datasets import BenchmarkPuzzle

    return BenchmarkPuzzle(
        puzzle_id="NON_OFFICIAL_SMOKE_STARTPOS",
        initial_solver_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        target_move="e2e4",
        reference_moves_uci=["e2e4", "e7e5"],
        mate_depth=1,
        source="synthetic_smoke",
    )


def selected_model_ids(args):
    """Return benchmark model IDs selected for runtime smoke."""

    return list(LLM_BENCHMARK_MODELS) if args.all_models else [args.model]


def run_runtime_smoke(args, registry):
    """Run one non-official next-move request per selected discovered model."""

    endpoint = resolve_endpoint(args.ollama_url or args.endpoint)
    client = OllamaClient(endpoint, timeout=args.timeout)
    selected = selected_model_ids(args)
    puzzle = smoke_fixture()
    outputs = []
    prompt = build_prompt(puzzle.initial_solver_fen)
    for model_id in selected:
        if model_id not in registry:
            outputs.append({"model_id": model_id, "error": "model_not_discovered"})
            continue
        gen_config = generation_config_for_model(model_id)
        if gen_config["options"].get("num_predict") is None:
            outputs.append({"model_id": model_id, "error": "num_predict_pending_calibration"})
            continue
        cold = client.generate(
            registry[model_id]["ollama_model"],
            prompt=prompt["user"],
            system=prompt["system"],
            options=gen_config["options"],
            think=gen_config["think"],
        )
        warm = client.generate(
            registry[model_id]["ollama_model"],
            prompt=prompt["user"],
            system=prompt["system"],
            options=gen_config["options"],
            think=gen_config["think"],
        )
        scored = score_next_move(warm.get("final_content"), puzzle.initial_solver_fen, puzzle.target_move)
        metadata = warm.get("metadata", {})
        outputs.append(
            {
                "label": "NON_OFFICIAL_SMOKE_TEST",
                "model_id": model_id,
                "ollama_model": registry[model_id]["ollama_model"],
                "digest": registry[model_id]["digest"],
                "raw_final_content": warm.get("final_content"),
                "raw_thinking": warm.get("thinking"),
                "thinking_requested": warm.get("thinking_requested"),
                "thinking_mode": gen_config["thinking_mode"],
                "thinking_returned": warm.get("thinking_returned"),
                "generation_config": gen_config,
                "score": scored,
                "cold_latency_seconds": cold.get("latency_seconds"),
                "warm_latency_seconds": warm.get("latency_seconds"),
                "load_duration": metadata.get("load_duration"),
                "prompt_eval_count": metadata.get("prompt_eval_count"),
                "prompt_eval_duration": metadata.get("prompt_eval_duration"),
                "eval_count": metadata.get("eval_count"),
                "eval_duration": metadata.get("eval_duration"),
                "done_reason": metadata.get("done_reason"),
                "num_predict": gen_config["options"].get("num_predict"),
                "parse_status": scored["outcome"],
                "legal_status": scored["legal"],
                "num_predict_16_compatible": bool(warm.get("final_content")),
                "error": warm.get("error") or cold.get("error"),
            }
        )
    return outputs


def run_reference_line_smoke(args, registry):
    """Run a tiny non-official reference-line smoke for protocol mechanics."""

    if not registry:
        return {"label": "NON_OFFICIAL_REFERENCE_LINE_SMOKE", "error": "no_models_available"}
    model_id = selected_model_ids(args)[0]
    if model_id not in registry:
        return {"label": "NON_OFFICIAL_REFERENCE_LINE_SMOKE", "error": "model_not_discovered"}
    endpoint = resolve_endpoint(args.ollama_url or args.endpoint)
    client = OllamaClient(endpoint, timeout=args.timeout)
    puzzle = smoke_fixture()

    def predict(fen):
        prompt = build_prompt(fen)
        response = client.generate(
            registry[model_id]["ollama_model"],
            prompt["user"],
            prompt["system"],
            options=generation_config_for_model(model_id)["options"],
            think=generation_config_for_model(model_id)["think"],
        )
        return response.get("final_content")

    return {
        "label": "NON_OFFICIAL_REFERENCE_LINE_SMOKE",
        "model_id": model_id,
        "result": evaluate_reference_line(puzzle, predict),
    }


def run_runtime_calibration(args, registry):
    """Run non-official GPT-OSS num_predict calibration on smoke fixtures."""

    model_id = args.model
    if model_id != "gpt_oss_20b":
        return {"label": "NON_OFFICIAL_RUNTIME_CALIBRATION", "error": "calibration_only_for_gpt_oss_20b"}
    if model_id not in registry:
        return {"label": "NON_OFFICIAL_RUNTIME_CALIBRATION", "error": "model_not_discovered"}
    endpoint = resolve_endpoint(args.ollama_url or args.endpoint)
    client = OllamaClient(endpoint, timeout=args.timeout)
    puzzle = smoke_fixture()
    prompt = build_prompt(puzzle.initial_solver_fen)
    trials = []
    selected_budget = None
    for budget in GPT_OSS_CALIBRATION_BUDGETS:
        gen_config = generation_config_for_model(model_id, num_predict_override=budget)
        response = client.generate(
            registry[model_id]["ollama_model"],
            prompt["user"],
            prompt["system"],
            options=gen_config["options"],
            think=gen_config["think"],
        )
        scored = score_next_move(response.get("final_content"), puzzle.initial_solver_fen, puzzle.target_move)
        trial = {
            "label": "NON_OFFICIAL_RUNTIME_CALIBRATION",
            "model_id": model_id,
            "num_predict": budget,
            "thinking_requested": response.get("thinking_requested"),
            "thinking_returned": response.get("thinking_returned"),
            "raw_thinking": response.get("thinking"),
            "raw_final_content": response.get("final_content"),
            "eval_count": response.get("metadata", {}).get("eval_count"),
            "done_reason": response.get("metadata", {}).get("done_reason"),
            "parse_status": scored["outcome"],
            "legal_status": scored["legal"],
            "wall_latency": response.get("latency_seconds"),
            "calibration_success": scored["parse_success"],
        }
        trials.append(trial)
        if scored["parse_success"]:
            selected_budget = budget
            break
    return {
        "label": "NON_OFFICIAL_RUNTIME_CALIBRATION",
        "model_id": model_id,
        "criterion": "first syntactically parseable final UCI response; chess correctness ignored",
        "candidate_budgets": GPT_OSS_CALIBRATION_BUDGETS,
        "selected_budget": selected_budget,
        "trials": trials,
    }


def main(argv=None):
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen_3_5_4b")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--dataset", choices=["lichess-test", "heldout-classics"], default="lichess-test")
    parser.add_argument("--protocol", choices=["next-move", "reference-line"], default="next-move")
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/llm_benchmark/smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runtime-smoke", action="store_true")
    parser.add_argument("--runtime-calibrate", action="store_true")
    parser.add_argument("--thinking-enabled", dest="thinking_enabled", action="store_true")
    parser.add_argument("--thinking-disabled", dest="thinking_enabled", action="store_false")
    parser.set_defaults(thinking_enabled=DEFAULT_THINKING_ENABLED)
    parser.add_argument("--run-benchmark", action="store_true")
    args = parser.parse_args(argv)

    preflight = build_preflight(args)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if not args.run_benchmark:
        if args.runtime_calibrate and preflight["ollama_reachable"]:
            calibration = run_runtime_calibration(args, preflight["discovered_registry"])
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "non_official_runtime_calibration.json").write_text(
                json.dumps(calibration, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps({"NON_OFFICIAL_RUNTIME_CALIBRATION": calibration}, indent=2, sort_keys=True))
        elif (args.smoke or args.runtime_smoke) and preflight["ollama_reachable"]:
            outputs = run_runtime_smoke(args, preflight["discovered_registry"])
            reference_line = run_reference_line_smoke(args, preflight["discovered_registry"])
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "non_official_smoke.json").write_text(
                json.dumps({"single_move": outputs, "reference_line": reference_line}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps({"NON_OFFICIAL_SMOKE_TEST": outputs, "REFERENCE_LINE": reference_line}, indent=2, sort_keys=True))
        return 0
    raise SystemExit("Official benchmark execution is intentionally disabled in this implementation pass.")


if __name__ == "__main__":
    raise SystemExit(main())

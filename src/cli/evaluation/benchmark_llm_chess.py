"""CLI for local Ollama LLM chess benchmark preflight and smoke tests."""

from __future__ import annotations

import argparse
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

from src.llm.datasets import dataset_fingerprint, load_lichess_test
from src.llm.model_registry import EXPECTED_MODELS, discover_registry
from src.llm.ollama_client import DEFAULT_GENERATION_OPTIONS, DEFAULT_OLLAMA_ENDPOINT, OllamaClient
from src.llm.parsing import PARSER_VERSION
from src.llm.prompting import PROMPT_TEMPLATE_VERSION, build_prompt, prompt_hash
from src.llm.scoring import evaluate_reference_line, score_next_move


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

    client = OllamaClient(args.endpoint, timeout=args.timeout)
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
        "endpoint": args.endpoint,
        "discovered_registry": registry,
        "expected_models_found": {
            key: key in registry
            for key in EXPECTED_MODELS
        },
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": prompt_hash(),
        "parser_version": PARSER_VERSION,
        "generation_options": DEFAULT_GENERATION_OPTIONS,
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
        "ready_for_official_benchmark": reachable and parser_tests_pass() and all(key in registry for key in EXPECTED_MODELS),
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


def run_smoke(args, registry):
    """Run one non-official next-move request per selected discovered model."""

    client = OllamaClient(args.endpoint, timeout=args.timeout)
    selected = list(registry) if args.all_models else [args.model]
    puzzle = smoke_fixture()
    outputs = []
    prompt = build_prompt(puzzle.initial_solver_fen)
    for model_id in selected:
        if model_id not in registry:
            outputs.append({"model_id": model_id, "error": "model_not_discovered"})
            continue
        response = client.generate(
            registry[model_id]["ollama_model"],
            prompt=prompt["user"],
            system=prompt["system"],
        )
        scored = score_next_move(response.get("response"), puzzle.initial_solver_fen, puzzle.target_move)
        outputs.append(
            {
                "label": "NON_OFFICIAL_SMOKE_TEST",
                "model_id": model_id,
                "ollama_model": registry[model_id]["ollama_model"],
                "raw_response": response.get("response"),
                "score": scored,
                "latency_seconds": response.get("latency_seconds"),
                "error": response.get("error"),
            }
        )
    return outputs


def main(argv=None):
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen_3_5_4b")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--dataset", choices=["lichess-test", "heldout-classics"], default="lichess-test")
    parser.add_argument("--protocol", choices=["next-move", "reference-line"], default="next-move")
    parser.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/llm_benchmark/smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    args = parser.parse_args(argv)

    preflight = build_preflight(args)
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if not args.run_benchmark:
        if args.smoke and preflight["ollama_reachable"]:
            outputs = run_smoke(args, preflight["discovered_registry"])
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "non_official_smoke.json").write_text(
                json.dumps(outputs, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps({"NON_OFFICIAL_SMOKE_TEST": outputs}, indent=2, sort_keys=True))
        return 0
    raise SystemExit("Official benchmark execution is intentionally disabled in this implementation pass.")


if __name__ == "__main__":
    raise SystemExit(main())

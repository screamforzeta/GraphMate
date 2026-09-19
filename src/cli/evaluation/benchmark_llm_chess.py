"""CLI for local Ollama LLM chess benchmark preflight and smoke tests."""

from __future__ import annotations

import argparse
import json
import socket
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from src.llm.datasets import dataset_fingerprint, load_lichess_test
from src.llm.model_registry import (
    LLM_BENCHMARK_MODELS,
    discover_registry,
    generation_config_for_model,
    official_model_ready,
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
from src.llm.resume import append_jsonl, assert_resume_config_matches, load_completed_keys
from src.llm.scoring import evaluate_reference_line, score_next_move


GPT_OSS_CALIBRATION_BUDGETS = [32, 64, 128, 256]
DEFAULT_SMOKE_OUTPUT_DIR = "artifacts/llm_benchmark/smoke"
DEFAULT_OFFICIAL_OUTPUT_ROOT = "artifacts/llm_benchmark/official"
OFFICIAL_DATASET_PATHS = {
    "lichess-test": Path("data/final/puzzles/test.csv"),
}


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
            and all(official_model_ready(model_id, verification) for model_id in LLM_BENCHMARK_MODELS)
        ),
        "runtime_ready": {
            model_id: official_model_ready(model_id, verification)
            for model_id in LLM_BENCHMARK_MODELS
        },
    }


def validate_official_run_preconditions(model_ids, preflight):
    """Fail closed when official benchmark prerequisites are not satisfied."""

    if not preflight.get("ollama_reachable"):
        raise RuntimeError("Official benchmark refused: Ollama endpoint is not reachable.")
    if not preflight.get("parser_tests_pass"):
        raise RuntimeError("Official benchmark refused: parser self-tests failed.")
    for model_id in model_ids:
        verification = preflight["registry_verification"].get(model_id, {})
        if not verification.get("found"):
            raise RuntimeError(f"Official benchmark refused: model missing: {model_id}.")
        if not verification.get("digest_match"):
            raise RuntimeError(f"Official benchmark refused: digest mismatch for {model_id}.")
        if not official_model_ready(model_id, preflight["registry_verification"]):
            raise RuntimeError(f"Official benchmark refused: frozen generation config invalid for {model_id}.")
    if not preflight.get("dataset_fingerprint"):
        raise RuntimeError("Official benchmark refused: dataset fingerprint missing.")
    return True


def official_output_dir(args, model_id):
    """Return the deterministic output directory for one official run."""

    root = (
        Path(DEFAULT_OFFICIAL_OUTPUT_ROOT)
        if args.output_dir == DEFAULT_SMOKE_OUTPUT_DIR
        else Path(args.output_dir)
    )
    return root / args.protocol.replace("-", "_") / model_id


def official_run_identity(args, preflight, model_id, registry_entry):
    """Build the frozen identity used for manifest and resume checks."""

    model_config = generation_config_for_model(model_id)
    return {
        "run_id": (
            f"OFFICIAL_{args.protocol}_{args.dataset}_{model_id}_"
            f"{preflight['dataset_fingerprint'][:12]}"
        ),
        "run_label": "OFFICIAL",
        "model_id": model_id,
        "ollama_model": registry_entry["ollama_model"],
        "model_digest": registry_entry["digest"],
        "dataset": args.dataset,
        "protocol": args.protocol,
        "dataset_fingerprint": preflight["dataset_fingerprint"],
        "dataset_n": preflight["dataset_n"],
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": prompt_hash(),
        "parser_version": PARSER_VERSION,
        "temperature": model_config["options"]["temperature"],
        "think": model_config["think"],
        "thinking_mode": model_config["thinking_mode"],
        "num_predict": model_config["options"]["num_predict"],
        "generation_options": model_config["options"],
        "thinking_enabled": args.thinking_enabled,
        "model_generation_config": model_config,
    }


def load_official_records(path):
    """Load existing official JSONL records and reject mixed artifacts."""

    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_label") != "OFFICIAL":
                raise ValueError("Official benchmark refused: JSONL contains non-official records.")
            records.append(record)
    return records


def write_json(path, payload):
    """Write one stable JSON artifact."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def prepare_official_artifacts(args, preflight, model_id, registry_entry):
    """Create or validate manifest, JSONL, and summary paths for a run."""

    output_dir = official_output_dir(args, model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "output_dir": output_dir,
        "manifest": output_dir / "manifest.json",
        "predictions": output_dir / "predictions.jsonl",
        "summary": output_dir / "summary.json",
    }
    identity = official_run_identity(args, preflight, model_id, registry_entry)
    if paths["manifest"].exists():
        existing_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        assert_resume_config_matches(existing_manifest["identity"], identity)
    elif paths["predictions"].exists() and not args.resume:
        raise RuntimeError("Official benchmark refused: predictions exist without --resume.")
    else:
        manifest = {
            "run_label": "OFFICIAL",
            "status": "STARTED",
            "start_timestamp": datetime.now(timezone.utc).isoformat(),
            "end_timestamp": None,
            "hostname": socket.gethostname(),
            "ollama_endpoint": preflight["endpoint"],
            "ollama_version": preflight["ollama_version"],
            "identity": identity,
            "paths": {
                "predictions": str(paths["predictions"]),
                "summary": str(paths["summary"]),
            },
        }
        write_json(paths["manifest"], manifest)
    if paths["predictions"].exists() and not args.resume:
        raise RuntimeError("Official benchmark refused: predictions exist without --resume.")
    load_official_records(paths["predictions"])
    return identity, paths


def official_outcome(scored, runtime_error=None):
    """Map internal scorer status to the frozen official taxonomy."""

    if runtime_error:
        return "RUNTIME_ERROR"
    if scored["outcome"] == "LEGAL_BUT_WRONG":
        return "WRONG_LEGAL_MOVE"
    return scored["outcome"]


def rating_bucket(rating):
    """Return a coarse rating bucket for summary aggregation."""

    if rating is None:
        return "unknown"
    rating = int(rating)
    lower = (rating // 200) * 200
    return f"{lower}-{lower + 199}"


def summarize_official_records(records, model_id):
    """Aggregate official next-move records without fabricating Top-K metrics."""

    total = len(records)
    counts = {
        "CORRECT": 0,
        "WRONG_LEGAL_MOVE": 0,
        "ILLEGAL_MOVE": 0,
        "PARSE_ERROR": 0,
        "RUNTIME_ERROR": 0,
    }
    latencies = []
    by_mate_depth = {}
    by_rating_bucket = {}
    for record in records:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
        if record.get("latency_seconds") is not None:
            latencies.append(float(record["latency_seconds"]))
        mate_key = str(record.get("mate_depth"))
        rating_key = rating_bucket(record.get("rating"))
        for bucket, key in ((by_mate_depth, mate_key), (by_rating_bucket, rating_key)):
            bucket.setdefault(key, {"N": 0, "correct": 0})
            bucket[key]["N"] += 1
            bucket[key]["correct"] += int(bool(record.get("correct")))
    for bucket in (by_mate_depth, by_rating_bucket):
        for stats in bucket.values():
            stats["accuracy"] = stats["correct"] / stats["N"] if stats["N"] else None
    return {
        "run_label": "OFFICIAL",
        "model_id": model_id,
        "N": total,
        "correct": counts["CORRECT"],
        "wrong_legal": counts["WRONG_LEGAL_MOVE"],
        "illegal": counts["ILLEGAL_MOVE"],
        "parse_errors": counts["PARSE_ERROR"],
        "runtime_errors": counts["RUNTIME_ERROR"],
        "top1_exact_canonical_accuracy": counts["CORRECT"] / total if total else None,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "accuracy_by_mate_depth": by_mate_depth,
        "accuracy_by_rating_bucket": by_rating_bucket,
        "outcome_counts": counts,
    }


def progress_line(processed, total, started, records):
    """Format a compact terminal progress line for long official runs."""

    elapsed = time.perf_counter() - started
    latencies = [row["latency_seconds"] for row in records if row.get("latency_seconds") is not None]
    mean_latency = statistics.fmean(latencies) if latencies else 0.0
    remaining = max(total - processed, 0)
    eta = remaining * mean_latency if mean_latency else None
    counts = {}
    for row in records:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    percentage = (processed / total * 100.0) if total else 100.0
    eta_text = f"{eta:.1f}s" if eta is not None else "unknown"
    return (
        f"processed={processed}/{total} ({percentage:.1f}%) "
        f"elapsed={elapsed:.1f}s mean_latency={mean_latency:.3f}s "
        f"eta={eta_text} outcomes={counts}"
    )


def run_official_next_move_benchmark(args, preflight):
    """Execute the frozen official next-move benchmark with incremental JSONL."""

    if args.protocol != "next-move":
        raise RuntimeError("Official benchmark currently supports only protocol=next-move.")
    model_ids = selected_model_ids(args)
    validate_official_run_preconditions(model_ids, preflight)
    dataset_path = OFFICIAL_DATASET_PATHS.get(args.dataset)
    if dataset_path is None:
        raise RuntimeError(f"Official benchmark refused: unsupported dataset {args.dataset}.")

    endpoint = resolve_endpoint(args.ollama_url or args.endpoint)
    client = OllamaClient(endpoint, timeout=args.timeout)
    results = []
    for model_id in model_ids:
        registry_entry = preflight["discovered_registry"][model_id]
        identity, paths = prepare_official_artifacts(args, preflight, model_id, registry_entry)
        puzzles = load_lichess_test(dataset_path, limit=args.limit, offset=args.offset)
        completed = load_completed_keys(paths["predictions"])
        records = load_official_records(paths["predictions"])
        generation_config = generation_config_for_model(model_id)
        total = len(puzzles)
        started = time.perf_counter()
        print(
            f"OFFICIAL next-move {model_id}: starting/resuming {len(records)}/{total}",
            flush=True,
        )
        for index, puzzle in enumerate(puzzles, 1):
            key = (identity["run_id"], model_id, puzzle.puzzle_id, args.protocol)
            if key in completed:
                continue
            prompt = build_prompt(puzzle.initial_solver_fen)
            runtime_error = None
            try:
                response = client.generate(
                    registry_entry["ollama_model"],
                    prompt=prompt["user"],
                    system=prompt["system"],
                    options=generation_config["options"],
                    think=generation_config["think"],
                )
                runtime_error = response.get("error")
            except Exception as exc:
                response = {
                    "final_content": "",
                    "thinking": None,
                    "thinking_requested": generation_config["think"],
                    "thinking_returned": False,
                    "metadata": {},
                    "request_payload": None,
                    "latency_seconds": None,
                    "error": str(exc),
                }
                runtime_error = str(exc)
            scored = score_next_move(
                response.get("final_content"),
                puzzle.initial_solver_fen,
                puzzle.target_move,
            )
            outcome = official_outcome(scored, runtime_error=runtime_error)
            record = {
                "run_label": "OFFICIAL",
                "run_id": identity["run_id"],
                "model_id": model_id,
                "ollama_model": registry_entry["ollama_model"],
                "model_digest": registry_entry["digest"],
                "dataset": args.dataset,
                "protocol": args.protocol,
                "dataset_fingerprint": preflight["dataset_fingerprint"],
                "puzzle_index": args.offset + index - 1,
                "puzzle_id": puzzle.puzzle_id,
                "fen": puzzle.initial_solver_fen,
                "target_move": puzzle.target_move,
                "mate_depth": puzzle.mate_depth,
                "rating": puzzle.rating,
                "raw_final_content": response.get("final_content"),
                "raw_thinking": response.get("thinking"),
                "parsed_uci": scored["parsed_uci"],
                "parsed_san": scored["parsed_san"],
                "parse_success": False if runtime_error else scored["parse_success"],
                "legal": False if runtime_error else scored["legal"],
                "correct": False if runtime_error else scored["correct"],
                "outcome": outcome,
                "latency_seconds": response.get("latency_seconds"),
                "runtime_error": runtime_error,
                "generation_metadata": response.get("metadata", {}),
                "generation_config": generation_config,
                "request_payload": response.get("request_payload"),
                "prompt_version": PROMPT_TEMPLATE_VERSION,
                "prompt_hash": prompt_hash(),
                "parser_version": PARSER_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl(paths["predictions"], record)
            records.append(record)
            completed.add(key)
            if len(records) == total or len(records) % 50 == 0:
                print(progress_line(len(records), total, started, records), flush=True)
        summary = summarize_official_records(records, model_id)
        write_json(paths["summary"], summary)
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        manifest["status"] = "COMPLETED" if summary["N"] == total else "PARTIAL"
        manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        manifest["summary"] = summary
        write_json(paths["manifest"], manifest)
        print(json.dumps({"OFFICIAL_SUMMARY": summary}, indent=2, sort_keys=True), flush=True)
        results.append({"model_id": model_id, "output_dir": str(paths["output_dir"]), "summary": summary})
    return results


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
    parser.add_argument("--output-dir", default=DEFAULT_SMOKE_OUTPUT_DIR)
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
    result = run_official_next_move_benchmark(args, preflight)
    print(json.dumps({"OFFICIAL_BENCHMARK": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

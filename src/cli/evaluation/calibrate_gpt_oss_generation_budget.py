"""Calibrate GPT-OSS generation budget without using chess correctness.

This CLI runs a small fixed subset through GPT-OSS with candidate
`num_predict` budgets. It selects the minimum budget that lets the model emit a
non-empty final answer without exhausting the token budget.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

from src.llm.datasets import dataset_fingerprint, load_lichess_test
from src.llm.model_registry import LLM_BENCHMARK_MODELS, generation_config_for_model
from src.llm.ollama_client import OllamaClient, resolve_endpoint
from src.llm.prompting import PROMPT_TEMPLATE_VERSION, build_prompt, prompt_hash
from src.llm.resume import append_jsonl


DEFAULT_OUTPUT_DIR = Path("artifacts/llm_benchmark/calibration/gpt_oss_generation_budget")
DEFAULT_DATASET_PATH = Path("data/final/puzzles/test.csv")
DEFAULT_MODEL_ID = "gpt_oss_20b"
DEFAULT_BUDGETS = [128, 256, 512]
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.95
BASELINE_KNOWN_FAILURE = {
    "budget": 64,
    "status": "known_official_run_failure",
    "reason": "Official GPT-OSS run produced empty final content with done_reason=length for all records.",
}


def assert_calibration_output_dir(path: Path) -> None:
    """Refuse output paths that could mix with official benchmark artifacts."""

    normalized = Path(path)
    if "official" in normalized.parts:
        raise RuntimeError("Calibration output must not be inside artifacts/llm_benchmark/official.")


def load_or_create_subset(dataset_path: Path, output_dir: Path, sample_size: int, seed: int) -> dict:
    """Load an existing calibration subset or create a deterministic one."""

    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "calibration_subset.json"
    fingerprint = dataset_fingerprint(dataset_path)
    if subset_path.exists():
        subset = json.loads(subset_path.read_text(encoding="utf-8"))
        expected = {
            "dataset": "lichess-test",
            "dataset_fingerprint": fingerprint,
            "seed": seed,
            "N": sample_size,
        }
        actual = {key: subset.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError("Calibration subset identity mismatch; refusing unsafe resume.")
        return subset
    puzzles = load_lichess_test(dataset_path)
    if sample_size > len(puzzles):
        raise ValueError("sample_size exceeds dataset size")
    indices = sorted(random.Random(seed).sample(range(len(puzzles)), sample_size))
    selected = []
    for index in indices:
        puzzle = puzzles[index]
        selected.append(
            {
                "row_index": index,
                "puzzle_id": puzzle.puzzle_id,
                "fen": puzzle.initial_solver_fen,
            }
        )
    subset = {
        "dataset": "lichess-test",
        "dataset_fingerprint": fingerprint,
        "seed": seed,
        "N": sample_size,
        "selection_method": "random.Random(seed).sample(range(N), sample_size), sorted ascending",
        "selected": selected,
    }
    subset_path.write_text(json.dumps(subset, indent=2, sort_keys=True), encoding="utf-8")
    return subset


def calibration_identity(model_id: str, budget: int, subset: dict) -> dict:
    """Return fields that must match for safe resume of one budget."""

    model = LLM_BENCHMARK_MODELS[model_id]
    gen_config = generation_config_for_model(model_id, num_predict_override=budget)
    return {
        "run_type": "GPT_OSS_GENERATION_BUDGET_CALIBRATION",
        "model_id": model_id,
        "ollama_model": model["ollama_model"],
        "model_digest": model["digest"],
        "dataset": subset["dataset"],
        "dataset_fingerprint": subset["dataset_fingerprint"],
        "subset_ids": [row["puzzle_id"] for row in subset["selected"]],
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": prompt_hash(),
        "think": gen_config["think"],
        "thinking_mode": gen_config["thinking_mode"],
        "temperature": gen_config["options"]["temperature"],
        "budget": int(budget),
    }


def load_budget_records(path: Path, identity: dict) -> list[dict]:
    """Load existing budget JSONL records after validating identity."""

    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("identity") != identity:
                raise RuntimeError("Calibration JSONL identity mismatch; refusing unsafe resume.")
            records.append(row)
    return records


def classify_generation(response: dict) -> str:
    """Classify calibration completion without using chess properties."""

    if response.get("error"):
        return "RUNTIME_ERROR"
    metadata = response.get("metadata") or {}
    if metadata.get("done_reason") == "length":
        return "TOKEN_BUDGET_EXHAUSTED"
    final = (response.get("final_content") or "").strip()
    if not final:
        return "EMPTY_FINAL_RESPONSE"
    return "GENERATION_COMPLETE"


def summarize_budget(records: list[dict], budget: int) -> dict:
    """Summarize one budget's calibration records."""

    counts = {
        "GENERATION_COMPLETE": 0,
        "TOKEN_BUDGET_EXHAUSTED": 0,
        "EMPTY_FINAL_RESPONSE": 0,
        "RUNTIME_ERROR": 0,
    }
    eval_counts = []
    latencies = []
    done_reasons = {}
    for record in records:
        status = record["calibration_status"]
        counts[status] = counts.get(status, 0) + 1
        if record.get("eval_count") is not None:
            eval_counts.append(record["eval_count"])
        if record.get("latency_seconds") is not None:
            latencies.append(record["latency_seconds"])
        reason = record.get("done_reason")
        done_reasons[reason] = done_reasons.get(reason, 0) + 1
    total = len(records)
    return {
        "requested_budget": int(budget),
        "N": total,
        **counts,
        "completion_rate": counts["GENERATION_COMPLETE"] / total if total else None,
        "mean_eval_count": statistics.fmean(eval_counts) if eval_counts else None,
        "median_eval_count": statistics.median(eval_counts) if eval_counts else None,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "done_reason_distribution": done_reasons,
    }


def budget_passes(summary: dict, threshold: float) -> bool:
    """Return whether a budget satisfies the fixed completion threshold."""

    required = math.ceil(float(threshold) * summary["N"])
    return summary["GENERATION_COMPLETE"] >= required


def run_budget(client: OllamaClient, model_id: str, budget: int, subset: dict, output_dir: Path) -> tuple[list[dict], dict]:
    """Run or resume one candidate budget over the fixed subset."""

    identity = calibration_identity(model_id, budget, subset)
    path = output_dir / f"budget_{budget}.jsonl"
    existing = load_budget_records(path, identity)
    completed = {record["puzzle_id"] for record in existing}
    model = LLM_BENCHMARK_MODELS[model_id]
    gen_config = generation_config_for_model(model_id, num_predict_override=budget)
    records = list(existing)
    for row in subset["selected"]:
        if row["puzzle_id"] in completed:
            continue
        prompt = build_prompt(row["fen"])
        response = client.generate(
            model["ollama_model"],
            prompt=prompt["user"],
            system=prompt["system"],
            options=gen_config["options"],
            think=gen_config["think"],
        )
        metadata = response.get("metadata") or {}
        record = {
            "identity": identity,
            "puzzle_id": row["puzzle_id"],
            "row_index": row["row_index"],
            "fen": row["fen"],
            "budget": int(budget),
            "raw_thinking": response.get("thinking"),
            "raw_final_content": response.get("final_content"),
            "done_reason": metadata.get("done_reason"),
            "eval_count": metadata.get("eval_count"),
            "latency_seconds": response.get("latency_seconds"),
            "calibration_status": classify_generation(response),
            "runtime_error": response.get("error"),
            "metadata": metadata,
        }
        append_jsonl(path, record)
        records.append(record)
    return records, summarize_budget(records, budget)


def write_manifest(output_dir: Path, model_id: str, subset: dict, budgets: list[int], threshold: float, endpoint: str) -> None:
    """Persist calibration manifest metadata."""

    model = LLM_BENCHMARK_MODELS[model_id]
    gen_config = generation_config_for_model(model_id)
    manifest = {
        "run_type": "GPT_OSS_GENERATION_BUDGET_CALIBRATION",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "ollama_model": model["ollama_model"],
        "model_digest": model["digest"],
        "dataset": subset["dataset"],
        "dataset_fingerprint": subset["dataset_fingerprint"],
        "N": subset["N"],
        "budgets": budgets,
        "baseline_known_failure": BASELINE_KNOWN_FAILURE,
        "selection_threshold": threshold,
        "selection_rule": "minimum tested budget with >=48/50 complete when N=50",
        "selection_used_chess_correctness": False,
        "target_used": False,
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "prompt_hash": prompt_hash(),
        "thinking_mode": gen_config["thinking_mode"],
        "think": gen_config["think"],
        "temperature": gen_config["options"]["temperature"],
        "ollama_endpoint": endpoint,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def run_calibration(
    *,
    endpoint: str,
    model_id: str,
    dataset_path: Path,
    output_dir: Path,
    sample_size: int,
    seed: int,
    budgets: list[int],
    completion_threshold: float,
    timeout: int = 120,
) -> dict:
    """Run GPT-OSS generation-budget calibration."""

    assert_calibration_output_dir(output_dir)
    if model_id != DEFAULT_MODEL_ID:
        raise ValueError("This calibration is scoped to gpt_oss_20b only.")
    subset = load_or_create_subset(dataset_path, output_dir, sample_size, seed)
    write_manifest(output_dir, model_id, subset, budgets, completion_threshold, endpoint)
    client = OllamaClient(endpoint, timeout=timeout)
    budget_summaries = []
    selected_budget = None
    for budget in budgets:
        records, summary = run_budget(client, model_id, int(budget), subset, output_dir)
        budget_summaries.append(summary)
        if len(records) != sample_size:
            raise RuntimeError(f"Budget {budget} incomplete after run/resume.")
        if budget_passes(summary, completion_threshold):
            selected_budget = int(budget)
            break
    summary = {
        "run_type": "GPT_OSS_GENERATION_BUDGET_CALIBRATION",
        "baseline_known_failure": BASELINE_KNOWN_FAILURE,
        "selected_budget": selected_budget if selected_budget is not None else "NO_BUDGET_PASSES",
        "selection_threshold": completion_threshold,
        "selection_rule": "minimum tested budget with >=48/50 complete when N=50",
        "selection_used_chess_correctness": False,
        "target_used": False,
        "prompt_version": PROMPT_TEMPLATE_VERSION,
        "model_digest": LLM_BENCHMARK_MODELS[model_id]["digest"],
        "dataset_fingerprint": subset["dataset_fingerprint"],
        "thinking_mode": generation_config_for_model(model_id)["thinking_mode"],
        "temperature": generation_config_for_model(model_id)["options"]["temperature"],
        "budget_summaries": budget_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--dataset", choices=["lichess-test"], default="lichess-test")
    parser.add_argument("--model", choices=[DEFAULT_MODEL_ID], default=DEFAULT_MODEL_ID)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--completion-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)
    endpoint = resolve_endpoint(args.ollama_url)
    result = run_calibration(
        endpoint=endpoint,
        model_id=args.model,
        dataset_path=DEFAULT_DATASET_PATH,
        output_dir=Path(args.output_dir),
        sample_size=args.sample_size,
        seed=args.seed,
        budgets=args.budgets,
        completion_threshold=args.completion_threshold,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

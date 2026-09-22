"""Official evaluator for the frozen YACPDB classic held-out benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.classic.adapters import (
    A3ClassicAdapter,
    A4ClassicAdapter,
    ModelBClassicAdapter,
    MockClassicAdapter,
    QwenClassicAdapter,
)
from src.evaluation.classic.core import (
    DEFAULT_CONSOLIDATED_DIR,
    DEFAULT_DATASET_DIR,
    DEFAULT_FREEZE_MANIFEST,
    EXPECTED_FREEZE_FINGERPRINT,
    ClassicSample,
    aggregate_predictions,
    build_prediction_record,
    load_classic_samples,
    prediction_record_from_json,
    verify_frozen_benchmark,
)
from src.evaluation.classic.runner import (
    DEFAULT_OUTPUT_ROOT,
    append_prediction_record,
    build_run_identity,
    mark_completed,
    prepare_official_run_directory,
    read_prediction_records,
)


MODEL_ALIASES = {
    "MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING": "a3",
    "a3": "a3",
    "MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING": "a4",
    "MODEL_A4_POSTMOVE_RERANKER": "a4",
    "a4": "a4",
    "MODEL_B_TIMING_LEGAL_SCORER": "model_b",
    "MODEL_B_TIMING_LEGAL_MOVE_SCORER": "model_b",
    "model_b": "model_b",
    "qwen3.5:4b": "qwen3.5:4b",
    "qwen3.5:9b": "qwen3.5:9b",
}


def git_provenance() -> dict[str, object]:
    """Return best-effort git provenance without failing evaluation."""

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True)
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        commit = None
        dirty = None
    return {"git_commit": commit, "git_dirty": dirty}


def sha256_file(path: Path) -> str:
    """Return SHA256 for a local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_hashes(args, adapter) -> None:
    """Validate optional expected checkpoint hashes before prediction."""

    if isinstance(adapter, A3ClassicAdapter) and args.expected_a3_sha256:
        actual = sha256_file(Path(args.a3_checkpoint))
        if actual != args.expected_a3_sha256:
            raise RuntimeError("A3 checkpoint SHA256 mismatch")
    if isinstance(adapter, A4ClassicAdapter):
        if args.expected_a3_sha256:
            actual = sha256_file(Path(args.a3_checkpoint))
            if actual != args.expected_a3_sha256:
                raise RuntimeError("A4 retrieval A3 checkpoint SHA256 mismatch")
        if args.expected_a4_sha256:
            actual = sha256_file(Path(args.a4_checkpoint))
            if actual != args.expected_a4_sha256:
                raise RuntimeError("A4 checkpoint SHA256 mismatch")


def fixture_samples() -> list[ClassicSample]:
    """Return synthetic non-YACPDB samples for smoke testing only."""

    return [
        ClassicSample(
            heldout_id="fixture_classic_smoke_001",
            source_problem_id="fixture",
            fen="k7/8/8/8/8/8/4K3/8 w - - 0 1",
            mate_depth=1,
            source_key_move_uci="e2e3",
            accepted_key_moves_uci=("e2e3",),
            accepted_key_basis="POPEYE_VERIFIED_UNIQUE",
            verification_status="FIXTURE",
            verification_reason="FIXTURE",
            forced_mate_verified=True,
        )
    ]


def build_adapter(args):
    """Instantiate the requested model adapter."""

    model_key = MODEL_ALIASES.get(args.model)
    if model_key is None:
        raise RuntimeError(f"Unsupported classic model: {args.model}")
    if model_key == "a3":
        return A3ClassicAdapter(args.a3_checkpoint, device=args.device, amp=args.amp)
    if model_key == "a4":
        return A4ClassicAdapter(args.a3_checkpoint, args.a4_checkpoint, top_k=5, device=args.device, amp=args.amp)
    if model_key == "model_b":
        return ModelBClassicAdapter()
    if model_key in {"qwen3.5:4b", "qwen3.5:9b"}:
        return QwenClassicAdapter(model_key, endpoint=args.ollama_url, timeout=args.timeout)
    raise RuntimeError(f"Unsupported classic model: {args.model}")


def validate_official_preconditions(args, adapter) -> dict:
    """Fail before prediction when official preconditions are not met."""

    if isinstance(adapter, MockClassicAdapter):
        raise RuntimeError("Official evaluator cannot use fixture/mock adapters.")
    manifest = verify_frozen_benchmark(
        Path(args.dataset_dir),
        Path(args.consolidated_dir),
        Path(args.freeze_manifest),
        EXPECTED_FREEZE_FINGERPRINT,
    )
    samples = load_classic_samples(
        Path(args.dataset_dir),
        Path(args.consolidated_dir),
        Path(args.freeze_manifest),
        require_frozen=True,
    )
    if len(samples) != 200:
        raise RuntimeError("Official classic benchmark must contain exactly 200 samples.")
    for depth in range(1, 11):
        if sum(sample.mate_depth == depth for sample in samples) != 20:
            raise RuntimeError("Official classic benchmark must contain exactly 20 samples per MateDepth.")
    adapter.prepare()
    validate_checkpoint_hashes(args, adapter)
    return {"manifest": manifest, "samples": samples}


def write_json(path: Path, payload) -> None:
    """Write stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_summaries(run_dir: Path, records) -> dict:
    """Regenerate summaries only from persisted prediction records."""

    summary = aggregate_predictions(records)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "by_mate_depth.json", summary["by_mate_depth"])
    write_json(run_dir / "by_verification_status.json", summary["by_verification_category"])
    return summary


def evaluate_samples(args, adapter, samples, run_dir: Path, freeze_fingerprint: str) -> dict:
    """Evaluate samples with resume support and persisted records."""

    predictions_path = run_dir / "predictions.jsonl"
    persisted = read_prediction_records(predictions_path)
    completed = {row["heldout_id"] for row in persisted}
    for sample in samples:
        if sample.heldout_id.startswith("yacpdb_classic_v1_") and args.smoke_test:
            raise RuntimeError("Smoke/fixture mode cannot use frozen YACPDB heldout IDs.")
        if sample.heldout_id in completed:
            continue
        output = adapter.predict(sample)
        diagnostics = dict(output.get("diagnostics", {}))
        if output.get("predicted_move_uci") is not None and output.get("predicted_move_uci") in set(sample.accepted_key_moves_uci):
            diagnostics["outcome_category"] = "correct"
        elif output.get("parse_success") is False:
            diagnostics.setdefault("outcome_category", "parse_failure")
        elif output.get("is_legal") is False:
            diagnostics.setdefault("outcome_category", "illegal")
        elif output.get("predicted_move_uci") is not None:
            diagnostics.setdefault("outcome_category", "wrong_legal")
        else:
            diagnostics.setdefault("outcome_category", "runtime_failure")
        record = build_prediction_record(
            sample,
            adapter.metadata(),
            output.get("predicted_move_uci"),
            ranked_moves_uci=output.get("ranked_moves_uci", ()),
            is_legal=output.get("is_legal"),
            parse_success=output.get("parse_success"),
            runtime_seconds=output.get("runtime_seconds"),
            diagnostics=diagnostics,
            freeze_fingerprint=freeze_fingerprint,
        )
        append_prediction_record(predictions_path, record)
        completed.add(sample.heldout_id)
    records = [prediction_record_from_json(row) for row in read_prediction_records(predictions_path)]
    return write_summaries(run_dir, records)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", action="store_true", help="Evaluate the frozen 200-position benchmark.")
    parser.add_argument("--smoke-test", action="store_true", help="Run a safe synthetic non-YACPDB fixture smoke test.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--consolidated-dir", default=str(DEFAULT_CONSOLIDATED_DIR))
    parser.add_argument("--freeze-manifest", default=str(DEFAULT_FREEZE_MANIFEST))
    parser.add_argument("--a3-checkpoint", default="artifacts/model_a3_legal_move_scorer_no_timing/best.pt")
    parser.add_argument("--a4-checkpoint", default="artifacts/model_a4_postmove_gnn_reranker/best.pt")
    parser.add_argument("--expected-a3-sha256", default=None)
    parser.add_argument("--expected-a4-sha256", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    if args.official == args.smoke_test:
        raise RuntimeError("Choose exactly one of --official or --smoke-test.")
    adapter = build_adapter(args)
    if isinstance(adapter, ModelBClassicAdapter):
        adapter.prepare()

    if args.official:
        state = validate_official_preconditions(args, adapter)
        samples = state["samples"]
        freeze_fingerprint = state["manifest"]["freeze_fingerprint"]
        run_dir = prepare_official_run_directory(
            adapter.metadata(),
            official=True,
            output_root=Path(args.output_root),
            resume=args.resume,
            freeze_fingerprint=freeze_fingerprint,
        )
    else:
        samples = fixture_samples()
        if any(sample.heldout_id.startswith("yacpdb_classic_v1_") for sample in samples):
            raise RuntimeError("Fixture smoke test may not use frozen YACPDB IDs.")
        adapter.prepare()
        freeze_fingerprint = "SMOKE_TEST_NO_FROZEN_BENCHMARK"
        identity = build_run_identity(adapter.metadata(), freeze_fingerprint=freeze_fingerprint)
        run_dir = Path(args.output_root) / "smoke" / adapter.model_id / identity["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            run_dir / "config.json",
            {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "official": False,
                "smoke_test": True,
                "device": args.device,
                "run_identity": identity,
                "model_metadata": adapter.metadata(),
                "git": git_provenance(),
            },
        )

    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.setdefault("device", args.device)
        config.setdefault("git", git_provenance())
        config.setdefault("model_b_protocol_status", "not_applicable_timing_unavailable")
        write_json(config_path, config)
    summary = evaluate_samples(args, adapter, samples, run_dir, freeze_fingerprint)
    if args.official:
        mark_completed(run_dir)
    print(json.dumps({"run_dir": str(run_dir), "summary": summary["all_200"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

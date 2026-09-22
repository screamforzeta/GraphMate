"""Explicit artifact discovery for final experiment analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import DiscoveredArtifacts, SourceRecord


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _classic_record(root: Path, run_dir: Path, validity: str = "VALID") -> SourceRecord:
    config = read_json(run_dir / "config.json")
    ident = config.get("run_identity", {})
    meta = config.get("model_metadata", {})
    git = config.get("git", {})
    return SourceRecord(
        model_id=ident.get("model_id", meta.get("model_id", run_dir.parent.name)),
        benchmark="YACPDB classic",
        protocol=ident.get("evaluation_protocol_version", "classic_heldout_eval_v1"),
        scientific_run_id=config.get("scientific_run_id", ident.get("run_id")),
        execution_attempt=config.get("execution_attempt"),
        run_dir=str(run_dir),
        source_files=[str(run_dir / name) for name in ("config.json", "summary.json", "by_mate_depth.json", "by_verification_status.json", "predictions.jsonl")],
        checkpoint_path=meta.get("checkpoint_path"),
        checkpoint_sha256=ident.get("checkpoint_sha256", meta.get("checkpoint_sha256")),
        benchmark_version=ident.get("benchmark_version"),
        freeze_fingerprint=ident.get("freeze_fingerprint"),
        prompt_version=ident.get("prompt_version", meta.get("prompt_version")),
        parser_version=ident.get("parser_version", meta.get("parser_version")),
        git_commit=git.get("git_commit"),
        git_dirty=git.get("git_dirty"),
        official=bool(config.get("official")),
        completion_state="COMPLETED" if (run_dir / "summary.json").exists() else "UNKNOWN",
        validity_state=validity,
    )


def discover(root: Path = Path(".")) -> DiscoveredArtifacts:
    paths = {
        "model_a_vs_a2_vs_a3": root / "artifacts/model_a_vs_a2_vs_a3_evaluation/summary.json",
        "a3_final": root / "artifacts/model_a3_legal_move_scorer_no_timing/final_report.json",
        "b_final": root / "artifacts/model_b_timing_legal_move_scorer/final_report.json",
        "a4_terminal": root / "artifacts/model_a4_terminal_evaluation/summary.json",
        "timing_ablation": root / "artifacts/model_b_timing_ablation/summary.json",
        "timing_rows": root / "artifacts/model_b_timing_ablation/paired_test_rows.json",
        "llm_qwen4_summary": root / "artifacts/llm_benchmark/official/next_move/qwen_3_5_4b/summary.json",
        "llm_qwen4_manifest": root / "artifacts/llm_benchmark/official/next_move/qwen_3_5_4b/manifest.json",
        "llm_qwen9_summary": root / "artifacts/llm_benchmark/official/next_move/qwen_3_5_9b/summary.json",
        "llm_qwen9_manifest": root / "artifacts/llm_benchmark/official/next_move/qwen_3_5_9b/manifest.json",
        "classic_freeze": root / "data/heldout_classic/final/yacpdb_classic_v1/freeze_manifest.json",
        "classic_verification": root / "data/heldout_classic/verification/yacpdb_classic_v1/consolidated/summary.json",
        "classic_a3": root / "artifacts/classic_benchmark/yacpdb_classic_v1/MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING/MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING_2376b10105c6a472",
        "classic_a4": root / "artifacts/classic_benchmark/yacpdb_classic_v1/MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING/MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING_85290f244ee43a70",
        "classic_qwen4_invalid": root / "artifacts/classic_benchmark/yacpdb_classic_v1/qwen3.5:4b/qwen3.5:4b_8bf8da8d39c7e331",
        "classic_qwen4": root / "artifacts/classic_benchmark/yacpdb_classic_v1/qwen3.5:4b/qwen3.5:4b_8bf8da8d39c7e331_attempt2",
        "classic_qwen9": root / "artifacts/classic_benchmark/yacpdb_classic_v1/qwen3.5:9b/qwen3.5:9b_8016dde83ff02824",
        "classic_smoke": root / "artifacts/classic_benchmark/yacpdb_classic_v1/smoke",
        "llm_smoke": root / "artifacts/llm_benchmark/smoke",
        "gpt_oss": root / "artifacts/llm_benchmark/official/next_move/gpt_oss_20b",
    }
    missing = [str(path) for key, path in paths.items() if key not in {"classic_smoke", "llm_smoke", "gpt_oss"} and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required official artifact(s): " + ", ".join(missing))

    included = [
        SourceRecord("MODEL_A_NO_TIMING_FROZEN_BASELINE", "Lichess", "model_a_vs_a2_vs_a3_eval", str(paths["model_a_vs_a2_vs_a3"].parent), [str(paths["model_a_vs_a2_vs_a3"])], True, "COMPLETED", "VALID", checkpoint_path="artifacts/convergence_training/chess_gat_no_timing/best.pt"),
        SourceRecord("MODEL_A_BEST_LEGAL", "Lichess", "inference_only_best_legal_filter", str(paths["model_a_vs_a2_vs_a3"].parent), [str(paths["model_a_vs_a2_vs_a3"])], True, "COMPLETED", "VALID", checkpoint_path="artifacts/convergence_training/chess_gat_no_timing/best.pt", notes="Inference-only mode, not a separate trained architecture."),
        SourceRecord("MODEL_A2_LEGAL_MASK_NO_TIMING", "Lichess", "model_a_vs_a2_vs_a3_eval", str(paths["model_a_vs_a2_vs_a3"].parent), [str(paths["model_a_vs_a2_vs_a3"])], True, "COMPLETED", "VALID", checkpoint_path="artifacts/model_a2_legal_mask_no_timing/best.pt"),
        SourceRecord("MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING", "Lichess", "terminal_test", str(paths["a3_final"].parent), [str(paths["a3_final"]), str(paths["a4_terminal"])], True, "COMPLETED", "VALID", checkpoint_path="artifacts/model_a3_legal_move_scorer_no_timing/best.pt", checkpoint_sha256="4efec653a451da7585f3663847c8dc5caaa8ebffadea677617e2f496e4253b80"),
        SourceRecord("MODEL_B_TIMING_LEGAL_MOVE_SCORER", "Lichess", "terminal_test_with_synthetic_timing", str(paths["b_final"].parent), [str(paths["b_final"]), str(paths["timing_ablation"]), str(paths["timing_rows"])], True, "COMPLETED", "VALID", checkpoint_path="artifacts/model_b_timing_legal_move_scorer/best.pt"),
        SourceRecord("MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING", "Lichess", "terminal_postmove_rerank", str(paths["a4_terminal"].parent), [str(paths["a4_terminal"])], True, "COMPLETED", "VALID", checkpoint_path="artifacts/model_a4_postmove_gnn_reranker/best.pt", checkpoint_sha256="0cb73acf70487efa5c93f715a5a60c0aa09d64792d894301efaaaf47b2801e99"),
    ]
    for summary_key, manifest_key, model_id in (
        ("llm_qwen4_summary", "llm_qwen4_manifest", "qwen3.5:4b"),
        ("llm_qwen9_summary", "llm_qwen9_manifest", "qwen3.5:9b"),
    ):
        manifest = read_json(paths[manifest_key])
        ident = manifest["identity"]
        included.append(SourceRecord(
            model_id=model_id,
            benchmark="Lichess",
            protocol=ident.get("protocol", "next-move"),
            scientific_run_id=ident.get("run_id"),
            run_dir=str(paths[summary_key].parent),
            source_files=[str(paths[summary_key]), str(paths[manifest_key]), str(paths[summary_key].parent / "predictions.jsonl")],
            benchmark_version=ident.get("dataset"),
            prompt_version=ident.get("prompt_version"),
            parser_version=ident.get("parser_version"),
            checkpoint_sha256=ident.get("model_digest"),
            official=manifest.get("run_label") == "OFFICIAL",
            completion_state=manifest.get("status", "UNKNOWN"),
            validity_state="VALID",
        ))
    included.extend(_classic_record(root, paths[key]) for key in ("classic_a3", "classic_a4", "classic_qwen4", "classic_qwen9"))

    excluded = []
    invalid = paths["classic_qwen4_invalid"]
    if invalid.exists():
        excluded.append(SourceRecord(
            model_id="qwen3.5:4b",
            benchmark="YACPDB classic",
            protocol="classic_heldout_eval_v1",
            scientific_run_id="qwen3.5:4b_8bf8da8d39c7e331",
            execution_attempt=1,
            run_dir=str(invalid),
            source_files=[str(invalid / "INVALID_INFRASTRUCTURE_RUN.json")],
            official=True,
            completion_state="FAILED",
            validity_state="INVALID_INFRASTRUCTURE_RUN",
            notes="HTTP_404_RUNTIME_FAILURE; excluded from accuracy.",
        ))
    for key, reason in (("classic_smoke", "SMOKE_TEST"), ("llm_smoke", "SMOKE_TEST"), ("gpt_oss", "PROTOCOL_FAILURE_PRIMARY_COMPARISON_EXCLUSION")):
        if paths[key].exists():
            excluded.append(SourceRecord(
                model_id=key,
                benchmark="mixed",
                protocol="diagnostic",
                run_dir=str(paths[key]),
                source_files=[str(paths[key])],
                official=False,
                completion_state="DIAGNOSTIC",
                validity_state=reason,
            ))
    return DiscoveredArtifacts(root=root, included=included, excluded=excluded, paths=paths)


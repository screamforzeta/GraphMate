"""Post-hoc evaluator for Model A, Model A2, and Model A3.

Purpose:
    Compare frozen no-timing models on the same shared PyG test split without
    training, checkpoint selection, or dataset changes.
Input:
    Official best checkpoints for Model A, A2, A3, data/pyg/test, final test
    CSV, and the train-only move vocabulary.
Output:
    artifacts/model_a_vs_a2_vs_a3_evaluation/summary.json and report.md.
Run:
    Imported by src.cli.evaluation.evaluate_model_a_vs_a2_vs_a3 and unit tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import statistics

import torch
from torch_geometric.data import Batch

from src.evaluation.model_a.model_a_vs_a2 import (
    MODEL_A_CHECKPOINT,
    MODEL_A_REFERENCE,
    MODEL_A2_CHECKPOINT,
    MODEL_A2_REFERENCE,
    _mate_label,
    _rating_bucket,
    legal_rank_bucket,
    load_eval_model,
    masked_topk_metrics,
    parity_status,
    target_legal_ranks,
    validate_checkpoint,
)
from src.graph.pyg_dataset import load_move_encoder, load_pyg_dataset
from src.models.model_a.chess_legal_scorer import ChessGATLegalMoveScorer
from src.training.model_a.legal_mask import build_legal_mask_for_batch
from src.training.common.metrics import compute_topk_accuracies
from src.training.model_a.model_a3_legal_scorer import (
    build_candidate_batch,
    grouped_cross_entropy,
)


MODEL_A3_CHECKPOINT = Path("artifacts/model_a3_legal_move_scorer_no_timing/best.pt")
OUTPUT_DIR = Path("artifacts/model_a_vs_a2_vs_a3_evaluation")
SHARED_TEST_N = 8610

MODEL_A_BEST_LEGAL_REFERENCE = {
    "n": SHARED_TEST_N,
    "best_legal_top1": 0.49814169570959405,
    "best_legal_top3": 0.704994192743689,
    "best_legal_top5": 0.7955865272107717,
}
MODEL_A3_REFERENCE = {
    "n": SHARED_TEST_N,
    "a3_loss": 1.0832485489175157,
    "a3_top1": 0.675609756097561,
    "a3_top3": 0.8565621370499419,
    "a3_top5": 0.9185830429732869,
    "a3_mean_legal_rank": 2.224274099883856,
    "a3_median_legal_rank": 1.0,
    "a3_illegal_top1_rate": 0.0,
}
A3_PARITY_TOLERANCES = {
    "n": 0,
    "a3_loss": 1e-6,
    "a3_top1": 1e-12,
    "a3_top3": 1e-12,
    "a3_top5": 1e-12,
    "a3_mean_legal_rank": 1e-12,
    "a3_median_legal_rank": 1e-12,
    "a3_illegal_top1_rate": 1e-12,
}


@dataclass
class ComparisonA3Config:
    """Runtime config for post-hoc A/A2/A3 evaluation."""

    batch_size: int = 128
    device: str = "cpu"
    non_blocking: bool = False
    amp: bool = False
    output_dir: str = str(OUTPUT_DIR)


def load_eval_model_a3(checkpoint_path, device):
    """Load the official A3 legal-candidate scorer for eval-only use."""

    model = ChessGATLegalMoveScorer(dropout=0.30).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def new_metric_bucket():
    """Create aggregation counters for global or subgroup metrics."""

    return {
        "n": 0,
        "raw_top1": 0.0,
        "raw_top3": 0.0,
        "raw_top5": 0.0,
        "best_legal_top1": 0.0,
        "best_legal_top3": 0.0,
        "best_legal_top5": 0.0,
        "a2_masked_top1": 0.0,
        "a2_masked_top3": 0.0,
        "a2_masked_top5": 0.0,
        "a3_top1": 0.0,
        "a3_top3": 0.0,
        "a3_top5": 0.0,
        "a3_loss_sum": 0.0,
        "raw_illegal_top1": 0,
        "a2_masked_illegal_top1": 0,
        "a3_illegal_top1": 0,
        "legal_ranks_model_a": [],
        "legal_ranks_model_a2": [],
        "legal_ranks_model_a3": [],
        "rank_buckets_model_a": defaultdict(int),
        "rank_buckets_model_a2": defaultdict(int),
        "rank_buckets_model_a3": defaultdict(int),
    }


def a3_topk_and_ranks(scores, candidate_ptr, target_indices, top_k=(1, 3, 5)):
    """Compute native A3 candidate Top-k counts and legal target ranks.

    Parameters:
        scores: Flat candidate scores.
        candidate_ptr: Prefix offsets by graph.
        target_indices: Local target candidate indices.
        top_k: Top-k values.
    Returns:
        Dict with correct counts, one-based ranks, and illegal_top1 count.
    Side effects:
        None.
    """

    result = {
        "correct": {k: 0 for k in top_k},
        "ranks": [],
        "illegal_top1": 0,
    }
    for graph_index in range(target_indices.numel()):
        start = int(candidate_ptr[graph_index].item())
        end = int(candidate_ptr[graph_index + 1].item())
        target = int(target_indices[graph_index].item())
        group_scores = scores[start:end]
        # Keep the same device-side ordering used by the A3 terminal evaluator.
        # Moving scores to CPU before argsort can change tie/near-tie ordering.
        ordered = torch.argsort(group_scores, descending=True)
        rank = int((ordered == target).nonzero(as_tuple=False).item()) + 1
        result["ranks"].append(rank)
        for k in top_k:
            effective_k = min(k, group_scores.numel())
            if target in ordered[:effective_k].tolist():
                result["correct"][k] += 1
    return result


def update_bucket(bucket, targets, legal_mask, logits_a, logits_a2, scores_a3, candidate_ptr, target_indices_a3, loss_a3):
    """Accumulate A raw, A best-legal, A2 masked, and A3 metrics."""

    n = int(targets.numel())
    raw = compute_topk_accuracies(logits_a, targets, (1, 3, 5))
    best_legal, masked_a = masked_topk_metrics(logits_a, targets, legal_mask)
    a2_masked, masked_a2 = masked_topk_metrics(logits_a2, targets, legal_mask)
    a3 = a3_topk_and_ranks(scores_a3, candidate_ptr, target_indices_a3)

    bucket["n"] += n
    bucket["a3_loss_sum"] += float(loss_a3.item()) * n
    for k in (1, 3, 5):
        bucket[f"raw_top{k}"] += raw[k] * n
        bucket[f"best_legal_top{k}"] += best_legal[k] * n
        bucket[f"a2_masked_top{k}"] += a2_masked[k] * n
        bucket[f"a3_top{k}"] += a3["correct"][k]

    rows = torch.arange(n, device=targets.device)
    raw_top1 = logits_a.argmax(dim=1)
    masked_a2_top1 = masked_a2.argmax(dim=1)
    bucket["raw_illegal_top1"] += int((~legal_mask[rows, raw_top1]).sum().item())
    bucket["a2_masked_illegal_top1"] += int((~legal_mask[rows, masked_a2_top1]).sum().item())
    bucket["a3_illegal_top1"] += a3["illegal_top1"]

    ranks_a = target_legal_ranks(logits_a, targets, legal_mask)
    ranks_a2 = target_legal_ranks(logits_a2, targets, legal_mask)
    bucket["legal_ranks_model_a"].extend(ranks_a)
    bucket["legal_ranks_model_a2"].extend(ranks_a2)
    bucket["legal_ranks_model_a3"].extend(a3["ranks"])
    for rank in ranks_a:
        bucket["rank_buckets_model_a"][legal_rank_bucket(rank)] += 1
    for rank in ranks_a2:
        bucket["rank_buckets_model_a2"][legal_rank_bucket(rank)] += 1
    for rank in a3["ranks"]:
        bucket["rank_buckets_model_a3"][legal_rank_bucket(rank)] += 1


def finalize_bucket(bucket):
    """Convert counters into rates and rank summaries."""

    n = bucket["n"]
    if n == 0:
        return {"n": 0}
    result = {"n": n}
    for key, value in bucket.items():
        if key.startswith(("raw_top", "best_legal_top", "a2_masked_top", "a3_top")):
            result[key] = value / n
    result["a3_loss"] = bucket["a3_loss_sum"] / n
    result["raw_illegal_top1_rate"] = bucket["raw_illegal_top1"] / n
    result["a2_masked_illegal_top1_rate"] = bucket["a2_masked_illegal_top1"] / n
    result["a3_illegal_top1_rate"] = bucket["a3_illegal_top1"] / n
    for label in ("model_a", "model_a2", "model_a3"):
        ranks = bucket[f"legal_ranks_{label}"]
        result[f"{label}_mean_legal_rank"] = statistics.mean(ranks) if ranks else None
        result[f"{label}_median_legal_rank"] = statistics.median(ranks) if ranks else None
        result[f"{label}_rank_buckets"] = dict(bucket[f"rank_buckets_{label}"])
    return result


def a3_parity_diagnostics(actual, reference=MODEL_A3_REFERENCE, tolerances=A3_PARITY_TOLERANCES):
    """Return per-field A3 parity diagnostics.

    Parameters:
        actual: Dict with A3 metrics produced by the evaluator.
        reference: Frozen canonical A3 metric values.
        tolerances: Per-metric maximum absolute differences.
    Returns:
        List of per-field diagnostic dictionaries.
    Side effects:
        None.
    """

    diagnostics = []
    for key, expected in reference.items():
        actual_value = actual.get(key)
        tolerance = tolerances.get(key, 1e-12)
        if actual_value is None:
            diff = None
            passed = False
        elif key == "n":
            diff = abs(int(actual_value) - int(expected))
            passed = diff <= tolerance
        else:
            diff = abs(float(actual_value) - float(expected))
            passed = diff <= tolerance
        diagnostics.append(
            {
                "metric": key,
                "expected": expected,
                "actual": actual_value,
                "abs_diff": diff,
                "tolerance": tolerance,
                "result": "PASS" if passed else "FAIL",
            }
        )
    return diagnostics


def parity_status_a3(actual):
    """Compare A3 native metrics to frozen shared-test references."""

    diagnostics = a3_parity_diagnostics(actual)
    passed = all(item["result"] == "PASS" for item in diagnostics)
    deltas = {
        item["metric"]: item["abs_diff"]
        for item in diagnostics
        if item["metric"] != "n"
    }
    return {
        "model": "MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING",
        "status": "PASS" if passed else "FAIL",
        "actual": {key: actual.get(key) for key in MODEL_A3_REFERENCE},
        "reference": dict(MODEL_A3_REFERENCE),
        "deltas": deltas,
        "diagnostics": diagnostics,
    }


def compute_a3_deltas_pp(metrics):
    """Compute A3 percentage-point deltas against all baseline systems."""

    return {
        "a3_top1_minus_a_raw_top1": (metrics["a3_top1"] - metrics["raw_top1"]) * 100,
        "a3_top3_minus_a_raw_top3": (metrics["a3_top3"] - metrics["raw_top3"]) * 100,
        "a3_top5_minus_a_raw_top5": (metrics["a3_top5"] - metrics["raw_top5"]) * 100,
        "a3_top1_minus_a_best_legal_top1": (metrics["a3_top1"] - metrics["best_legal_top1"]) * 100,
        "a3_top3_minus_a_best_legal_top3": (metrics["a3_top3"] - metrics["best_legal_top3"]) * 100,
        "a3_top5_minus_a_best_legal_top5": (metrics["a3_top5"] - metrics["best_legal_top5"]) * 100,
        "a3_top1_minus_a2_masked_top1": (metrics["a3_top1"] - metrics["a2_masked_top1"]) * 100,
        "a3_top3_minus_a2_masked_top3": (metrics["a3_top3"] - metrics["a2_masked_top3"]) * 100,
        "a3_top5_minus_a2_masked_top5": (metrics["a3_top5"] - metrics["a2_masked_top5"]) * 100,
    }


def _slice_a3_output(output, graph_index):
    """Return one graph's candidate scores and local ptr."""

    start = int(output["candidate_ptr"][graph_index].item())
    end = int(output["candidate_ptr"][graph_index + 1].item())
    return output["scores"][start:end], torch.tensor([0, end - start], device=output["scores"].device)


def run_comparison(dataframe, config):
    """Run one-pass post-hoc comparison on the official shared PyG test split."""

    device = torch.device(config.device)
    move_to_idx = load_move_encoder()
    num_classes = len(move_to_idx)
    model_a_path = validate_checkpoint(MODEL_A_CHECKPOINT, "Model A")
    model_a2_path = validate_checkpoint(MODEL_A2_CHECKPOINT, "Model A2")
    model_a3_path = validate_checkpoint(MODEL_A3_CHECKPOINT, "Model A3")
    model_a, checkpoint_a = load_eval_model(model_a_path, num_classes, device)
    model_a2, checkpoint_a2 = load_eval_model(model_a2_path, num_classes, device)
    model_a3, checkpoint_a3 = load_eval_model_a3(model_a3_path, device)
    dataset = load_pyg_dataset(split="test")
    rows = list(dataframe.itertuples(index=False))

    global_bucket = new_metric_bucket()
    mate_buckets = defaultdict(new_metric_bucket)
    rating_buckets = defaultdict(new_metric_bucket)
    use_amp = bool(config.amp and device.type == "cuda")

    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            graphs = [
                dataset[index]
                for index in range(start, min(start + config.batch_size, len(dataset)))
            ]
            batch_rows = [rows[int(graph.source_row_index.item())] for graph in graphs]
            batch = Batch.from_data_list(graphs).to(
                device,
                non_blocking=bool(config.non_blocking and device.type == "cuda"),
            )
            legal_mask = build_legal_mask_for_batch(batch, move_to_idx, num_classes)
            candidate = build_candidate_batch(batch)
            target_indices_a3 = candidate["target_indices"].to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits_a = model_a(batch)
                logits_a2 = model_a2(batch)
                output_a3 = model_a3(batch, candidate["candidate_moves"])
                loss_a3 = grouped_cross_entropy(
                    output_a3["scores"],
                    output_a3["candidate_ptr"],
                    target_indices_a3,
                )
            update_bucket(
                global_bucket,
                batch.y,
                legal_mask,
                logits_a,
                logits_a2,
                output_a3["scores"],
                output_a3["candidate_ptr"],
                target_indices_a3,
                loss_a3,
            )

            for local_index, row in enumerate(batch_rows):
                target = batch.y[local_index:local_index + 1]
                mask = legal_mask[local_index:local_index + 1]
                a_logits = logits_a[local_index:local_index + 1]
                a2_logits = logits_a2[local_index:local_index + 1]
                a3_scores, a3_ptr = _slice_a3_output(output_a3, local_index)
                a3_target = target_indices_a3[local_index:local_index + 1]
                a3_loss = grouped_cross_entropy(a3_scores, a3_ptr, a3_target)
                update_bucket(
                    mate_buckets[_mate_label(row)],
                    target,
                    mask,
                    a_logits,
                    a2_logits,
                    a3_scores,
                    a3_ptr,
                    a3_target,
                    a3_loss,
                )
                update_bucket(
                    rating_buckets[_rating_bucket(row)],
                    target,
                    mask,
                    a_logits,
                    a2_logits,
                    a3_scores,
                    a3_ptr,
                    a3_target,
                    a3_loss,
                )

    global_metrics = finalize_bucket(global_bucket)
    summary = {
        "status": "DIAGNOSTIC_POST_HOC_EVALUATION",
        "shared_test_n": len(dataset),
        "expected_shared_test_n": SHARED_TEST_N,
        "historical_oov": int(len(dataframe) - len(dataset)),
        "model_a_checkpoint": str(model_a_path),
        "model_a2_checkpoint": str(model_a2_path),
        "model_a3_checkpoint": str(model_a3_path),
        "model_a_checkpoint_epoch": checkpoint_a.get("epoch") if isinstance(checkpoint_a, dict) else None,
        "model_a2_checkpoint_epoch": checkpoint_a2.get("epoch") if isinstance(checkpoint_a2, dict) else None,
        "model_a3_checkpoint_epoch": checkpoint_a3.get("epoch") if isinstance(checkpoint_a3, dict) else None,
        "global": global_metrics,
        "model_a_parity": parity_status(
            global_metrics,
            MODEL_A_REFERENCE,
            "MODEL_A_NO_TIMING_FROZEN_BASELINE",
        ),
        "model_a_best_legal_parity": parity_status(
            global_metrics,
            MODEL_A_BEST_LEGAL_REFERENCE,
            "MODEL_A_BEST_LEGAL",
        ),
        "model_a2_parity": parity_status(
            {
                "n": global_metrics["n"],
                "masked_top1": global_metrics["a2_masked_top1"],
                "masked_top3": global_metrics["a2_masked_top3"],
                "masked_top5": global_metrics["a2_masked_top5"],
                "masked_illegal_top1_rate": global_metrics["a2_masked_illegal_top1_rate"],
            },
            MODEL_A2_REFERENCE,
            "MODEL_A2_LEGAL_MASK_NO_TIMING",
        ),
        "model_a3_parity": parity_status_a3(
            {
                "n": global_metrics["n"],
                "a3_loss": global_metrics["a3_loss"],
                "a3_top1": global_metrics["a3_top1"],
                "a3_top3": global_metrics["a3_top3"],
                "a3_top5": global_metrics["a3_top5"],
                "a3_mean_legal_rank": global_metrics["model_a3_mean_legal_rank"],
                "a3_median_legal_rank": global_metrics["model_a3_median_legal_rank"],
                "a3_illegal_top1_rate": global_metrics["a3_illegal_top1_rate"],
            }
        ),
        "mate_depth": {
            key: finalize_bucket(value)
            for key, value in sorted(mate_buckets.items())
        },
        "rating": {
            key: finalize_bucket(value)
            for key, value in sorted(rating_buckets.items())
        },
        "deltas_pp": compute_a3_deltas_pp(global_metrics),
        "post_hoc_guardrail": "No training, no checkpoint selection, no hyperparameter tuning.",
    }
    if summary["model_a3_parity"]["status"] == "FAIL":
        summary["interpretation_blocked"] = "A3 parity failed; do not interpret subgroup deltas."
    return summary


def write_outputs(summary, output_dir=OUTPUT_DIR):
    """Write summary.json and report.md for the A/A2/A3 comparison."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    return {"summary_json": str(summary_path), "report_md": str(report_path)}


def _metric_row(label, metrics, a_key, best_key, a2_key, a3_key, precision=6):
    """Render one Markdown metric row."""

    return (
        f"| {label} | {metrics.get(a_key):.{precision}f} | "
        f"{metrics.get(best_key):.{precision}f} | "
        f"{metrics.get(a2_key):.{precision}f} | "
        f"{metrics.get(a3_key):.{precision}f} |"
    )


def render_report(summary):
    """Render the Markdown A/A2/A3 comparison report."""

    metrics = summary["global"]
    mate_in_one = summary["mate_depth"].get("mateIn1", {})
    lines = [
        "# Model A vs A2 vs A3 Evaluation",
        "",
        "## Evaluation Integrity",
        "",
        f"- Shared test N: `{summary['shared_test_n']}`",
        f"- Expected shared test N: `{summary['expected_shared_test_n']}`",
        f"- Historical OOV outside shared PyG set: `{summary['historical_oov']}`",
        f"- Model A checkpoint: `{summary['model_a_checkpoint']}`",
        f"- Model A2 checkpoint: `{summary['model_a2_checkpoint']}`",
        f"- Model A3 checkpoint: `{summary['model_a3_checkpoint']}`",
        "- Scope: test-only post-hoc analysis; no training; no model selection.",
        "",
        "## Parity",
        "",
        f"- MODEL_A_PARITY: `{summary['model_a_parity']['status']}`",
        f"- MODEL_A_BEST_LEGAL_PARITY: `{summary['model_a_best_legal_parity']['status']}`",
        f"- MODEL_A2_PARITY: `{summary['model_a2_parity']['status']}`",
        f"- MODEL_A3_PARITY: `{summary['model_a3_parity']['status']}`",
        "",
        "## Global Comparison",
        "",
        "| Metric | A Raw | A Best-Legal | A2 Masked | A3 |",
        "|---|---:|---:|---:|---:|",
        f"| Candidate CE / NLL loss | N/A | N/A | N/A | {metrics['a3_loss']:.12f} |",
        _metric_row("Top1", metrics, "raw_top1", "best_legal_top1", "a2_masked_top1", "a3_top1"),
        _metric_row("Top3", metrics, "raw_top3", "best_legal_top3", "a2_masked_top3", "a3_top3"),
        _metric_row("Top5", metrics, "raw_top5", "best_legal_top5", "a2_masked_top5", "a3_top5"),
        f"| Mean legal rank | {metrics['model_a_mean_legal_rank']:.6f} | {metrics['model_a_mean_legal_rank']:.6f} | {metrics['model_a2_mean_legal_rank']:.6f} | {metrics['model_a3_mean_legal_rank']:.6f} |",
        f"| Median legal rank | {metrics['model_a_median_legal_rank']:.6f} | {metrics['model_a_median_legal_rank']:.6f} | {metrics['model_a2_median_legal_rank']:.6f} | {metrics['model_a3_median_legal_rank']:.6f} |",
        f"| Illegal Top1 rate | {metrics['raw_illegal_top1_rate']:.6f} | 0.000000 | {metrics['a2_masked_illegal_top1_rate']:.6f} | {metrics['a3_illegal_top1_rate']:.6f} |",
        "",
        "## A3 Parity Diagnostics",
        "",
        "| Metric | Expected | Actual | Abs Diff | Tolerance | Result |",
        "|---|---:|---:|---:|---:|---|",
        *[
            (
                f"| {item['metric']} | {item['expected']} | {item['actual']} | "
                f"{item['abs_diff']} | {item['tolerance']} | {item['result']} |"
            )
            for item in summary["model_a3_parity"].get("diagnostics", [])
        ],
        "",
        "Ranking metrics require near-exact numerical parity. Candidate CE / "
        "NLL loss uses abs tolerance `1e-6` only for CUDA/AMP reproducibility; "
        "this tolerance does not affect TopK, rank, or model selection.",
        "",
        "## Global Deltas",
        "",
        json.dumps(summary["deltas_pp"], indent=2),
        "",
        "## MateDepth",
        "",
        json.dumps(summary["mate_depth"], indent=2),
        "",
        "## MateIn1 Focus",
        "",
        json.dumps(
            {
                "metrics": mate_in_one,
                "a3_top1_delta_vs_a_best_legal_pp": (
                    mate_in_one.get("a3_top1", 0.0)
                    - mate_in_one.get("best_legal_top1", 0.0)
                )
                * 100,
                "a3_top1_delta_vs_a2_pp": (
                    mate_in_one.get("a3_top1", 0.0)
                    - mate_in_one.get("a2_masked_top1", 0.0)
                )
                * 100,
            },
            indent=2,
        ),
        "",
        "## Rating Buckets",
        "",
        json.dumps(summary["rating"], indent=2),
        "",
        "## Rank Analysis",
        "",
        f"- Model A mean/median legal rank: `{metrics['model_a_mean_legal_rank']}` / `{metrics['model_a_median_legal_rank']}`",
        f"- Model A2 mean/median legal rank: `{metrics['model_a2_mean_legal_rank']}` / `{metrics['model_a2_median_legal_rank']}`",
        f"- Model A3 mean/median legal rank: `{metrics['model_a3_mean_legal_rank']}` / `{metrics['model_a3_median_legal_rank']}`",
        "",
        "## Interpretation Guardrails",
        "",
        "This evaluator does not train models, does not modify checkpoints, and does not use test, MateDepth, or rating buckets for model selection. Subgroup deltas are descriptive only and should not be read as causal claims.",
        "",
    ]
    return "\n".join(lines)

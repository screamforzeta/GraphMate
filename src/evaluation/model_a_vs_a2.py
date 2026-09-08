"""Post-hoc evaluator for Model A raw/best-legal vs Model A2 masked.

Purpose:
    Compare the frozen no-timing baseline against the legal-masked A2 ablation
    without training, changing checkpoints, or changing graph representation.
Input:
    Official Model A and A2 best checkpoints, test PyG split, final test CSV,
    and train-derived move vocabulary.
Output:
    Diagnostic summary/report artifacts under artifacts/model_a_vs_a2_evaluation.
Run:
    Imported by src.evaluate_model_a_vs_a2 and unit tests.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import statistics

import torch
from torch_geometric.data import Batch

from src.graph.pyg_dataset import load_move_encoder, load_pyg_dataset
from src.training.chess_gat_trainer import build_model
from src.training.legal_mask import apply_legal_mask, build_legal_mask_for_batch
from src.training.metrics import compute_topk_accuracies


MODEL_A_CHECKPOINT = Path("artifacts/convergence_training/chess_gat_no_timing/best.pt")
MODEL_A2_CHECKPOINT = Path("artifacts/model_a2_legal_mask_no_timing/best.pt")
OUTPUT_DIR = Path("artifacts/model_a_vs_a2_evaluation")

MODEL_A_REFERENCE = {
    "n": 8610,
    "raw_top1": 0.3961672474213732,
    "raw_top3": 0.5580720094022851,
    "raw_top5": 0.6275261325010993,
}
MODEL_A2_REFERENCE = {
    "n": 8610,
    "masked_top1": 0.4925667828106852,
    "masked_top3": 0.7149825784802852,
    "masked_top5": 0.8117305459461146,
    "masked_illegal_top1_rate": 0.0,
}


@dataclass
class ComparisonConfig:
    """Runtime config for post-hoc A-vs-A2 evaluation."""

    batch_size: int = 128
    device: str = "cpu"
    non_blocking: bool = False
    amp: bool = False
    output_dir: str = str(OUTPUT_DIR)


def validate_checkpoint(path, role):
    """Require an official checkpoint path for a comparison role."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{role} checkpoint not found: {path}")
    return path


def load_eval_model(checkpoint_path, num_classes, device):
    """Load a ChessGATNoTiming checkpoint for eval-only use."""

    model = build_model(num_classes, dropout=0.30).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def masked_topk_metrics(logits, targets, legal_mask, top_k=(1, 3, 5)):
    """Compute Top-K after removing illegal classes from the ranking."""

    masked_logits = apply_legal_mask(logits, legal_mask)
    return compute_topk_accuracies(masked_logits, targets, top_k), masked_logits


def target_legal_ranks(logits, targets, legal_mask):
    """Return one-based target ranks after legal filtering for each row."""

    ranks = []
    masked_logits = apply_legal_mask(logits, legal_mask)
    ordered = torch.argsort(masked_logits.detach().cpu(), dim=1, descending=True)
    targets_cpu = targets.detach().cpu()
    for row, target in enumerate(targets_cpu.tolist()):
        match = (ordered[row] == int(target)).nonzero(as_tuple=False)
        ranks.append(int(match[0].item()) + 1)
    return ranks


def legal_rank_bucket(rank):
    """Map legal-filtered target rank to a compact bucket."""

    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 5:
        return "rank_4_5"
    if rank <= 10:
        return "rank_6_10"
    return "rank_gt_10"


def new_metric_bucket():
    """Create aggregation counters for one global/subgroup metric bucket."""

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
        "raw_illegal_top1": 0,
        "a2_masked_illegal_top1": 0,
        "legal_ranks_model_a": [],
        "legal_ranks_model_a2": [],
        "rank_buckets_model_a": defaultdict(int),
        "rank_buckets_model_a2": defaultdict(int),
    }


def update_bucket(bucket, targets, legal_mask, logits_a, logits_a2):
    """Accumulate comparison metrics for one batch or subgroup slice."""

    n = int(targets.numel())
    raw = compute_topk_accuracies(logits_a, targets, (1, 3, 5))
    best_legal, masked_a = masked_topk_metrics(logits_a, targets, legal_mask)
    a2_masked, masked_a2 = masked_topk_metrics(logits_a2, targets, legal_mask)
    bucket["n"] += n
    for k in (1, 3, 5):
        bucket[f"raw_top{k}"] += raw[k] * n
        bucket[f"best_legal_top{k}"] += best_legal[k] * n
        bucket[f"a2_masked_top{k}"] += a2_masked[k] * n
    rows = torch.arange(n, device=targets.device)
    raw_top1 = logits_a.argmax(dim=1)
    masked_a2_top1 = masked_a2.argmax(dim=1)
    bucket["raw_illegal_top1"] += int((~legal_mask[rows, raw_top1]).sum().item())
    bucket["a2_masked_illegal_top1"] += int((~legal_mask[rows, masked_a2_top1]).sum().item())
    ranks_a = target_legal_ranks(logits_a, targets, legal_mask)
    ranks_a2 = target_legal_ranks(logits_a2, targets, legal_mask)
    bucket["legal_ranks_model_a"].extend(ranks_a)
    bucket["legal_ranks_model_a2"].extend(ranks_a2)
    for rank in ranks_a:
        bucket["rank_buckets_model_a"][legal_rank_bucket(rank)] += 1
    for rank in ranks_a2:
        bucket["rank_buckets_model_a2"][legal_rank_bucket(rank)] += 1


def finalize_bucket(bucket):
    """Convert accumulated counts to rates and rank summaries."""

    n = bucket["n"]
    if n == 0:
        return {"n": 0}
    result = {"n": n}
    for key, value in bucket.items():
        if key.startswith(("raw_top", "best_legal_top", "a2_masked_top")):
            result[key] = value / n
    result["raw_illegal_top1_rate"] = bucket["raw_illegal_top1"] / n
    result["a2_masked_illegal_top1_rate"] = bucket["a2_masked_illegal_top1"] / n
    for label in ("model_a", "model_a2"):
        ranks = bucket[f"legal_ranks_{label}"]
        result[f"{label}_mean_legal_rank"] = statistics.mean(ranks) if ranks else None
        result[f"{label}_median_legal_rank"] = statistics.median(ranks) if ranks else None
        result[f"{label}_rank_buckets"] = dict(bucket[f"rank_buckets_{label}"])
    return result


def parity_status(actual, reference, prefix):
    """Compare actual count-based metrics to a frozen reference."""

    keys = [key for key in reference if key != "n"]
    deltas = {key: actual.get(key) - reference[key] for key in keys}
    passed = actual.get("n") == reference["n"] and all(
        abs(delta) < 1e-12 for delta in deltas.values()
    )
    return {
        "model": prefix,
        "status": "PASS" if passed else "FAIL",
        "actual": {key: actual.get(key) for key in reference},
        "reference": dict(reference),
        "deltas": deltas,
    }


def _mate_label(row):
    """Return mateInN label from CSV row metadata."""

    for theme in str(getattr(row, "Themes", "")).split():
        if theme.startswith("mateIn"):
            return theme
    return f"mateIn{int(getattr(row, 'MateDepth'))}"


def _rating_bucket(row):
    """Return fixed rating bucket label."""

    rating = int(getattr(row, "Rating"))
    if rating < 1200:
        return "<1200"
    if rating < 1600:
        return "1200-1599"
    if rating < 2000:
        return "1600-1999"
    if rating < 2400:
        return "2000-2399"
    return "2400+"


def run_comparison(dataframe, config):
    """Run one-pass post-hoc comparison on the official test PyG split."""

    device = torch.device(config.device)
    move_to_idx = load_move_encoder()
    num_classes = len(move_to_idx)
    model_a_path = validate_checkpoint(MODEL_A_CHECKPOINT, "Model A")
    model_a2_path = validate_checkpoint(MODEL_A2_CHECKPOINT, "Model A2")
    model_a, checkpoint_a = load_eval_model(model_a_path, num_classes, device)
    model_a2, checkpoint_a2 = load_eval_model(model_a2_path, num_classes, device)
    dataset = load_pyg_dataset(split="test")
    rows = list(dataframe.itertuples(index=False))

    global_bucket = new_metric_bucket()
    mate_buckets = defaultdict(new_metric_bucket)
    rating_buckets = defaultdict(new_metric_bucket)
    use_amp = bool(config.amp and device.type == "cuda")

    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            graphs = [dataset[index] for index in range(start, min(start + config.batch_size, len(dataset)))]
            batch_rows = [rows[int(graph.source_row_index.item())] for graph in graphs]
            batch = Batch.from_data_list(graphs).to(
                device,
                non_blocking=bool(config.non_blocking and device.type == "cuda"),
            )
            legal_mask = build_legal_mask_for_batch(batch, move_to_idx, num_classes)
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits_a = model_a(batch)
                logits_a2 = model_a2(batch)
            update_bucket(global_bucket, batch.y, legal_mask, logits_a, logits_a2)

            for local_index, row in enumerate(batch_rows):
                target = batch.y[local_index:local_index + 1]
                mask = legal_mask[local_index:local_index + 1]
                a_logits = logits_a[local_index:local_index + 1]
                a2_logits = logits_a2[local_index:local_index + 1]
                update_bucket(mate_buckets[_mate_label(row)], target, mask, a_logits, a2_logits)
                update_bucket(rating_buckets[_rating_bucket(row)], target, mask, a_logits, a2_logits)

    global_metrics = finalize_bucket(global_bucket)
    summary = {
        "status": "DIAGNOSTIC_POST_HOC_EVALUATION",
        "model_a_checkpoint": str(model_a_path),
        "model_a2_checkpoint": str(model_a2_path),
        "model_a_checkpoint_epoch": checkpoint_a.get("epoch") if isinstance(checkpoint_a, dict) else None,
        "model_a2_checkpoint_epoch": checkpoint_a2.get("epoch") if isinstance(checkpoint_a2, dict) else None,
        "test_graphs": len(dataset),
        "historical_oov": int(len(dataframe) - len(dataset)),
        "global": global_metrics,
        "model_a_parity": parity_status(
            global_metrics,
            MODEL_A_REFERENCE,
            "MODEL_A_NO_TIMING_FROZEN_BASELINE",
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
        "mate_depth": {
            key: finalize_bucket(value)
            for key, value in sorted(mate_buckets.items())
        },
        "rating": {
            key: finalize_bucket(value)
            for key, value in sorted(rating_buckets.items())
        },
        "deltas_pp": compute_deltas_pp(global_metrics),
    }
    return summary


def compute_deltas_pp(metrics):
    """Compute primary percentage-point deltas."""

    return {
        "a_best_legal_top1_minus_a_raw_top1": (metrics["best_legal_top1"] - metrics["raw_top1"]) * 100,
        "a_best_legal_top3_minus_a_raw_top3": (metrics["best_legal_top3"] - metrics["raw_top3"]) * 100,
        "a_best_legal_top5_minus_a_raw_top5": (metrics["best_legal_top5"] - metrics["raw_top5"]) * 100,
        "a2_masked_top1_minus_a_best_legal_top1": (metrics["a2_masked_top1"] - metrics["best_legal_top1"]) * 100,
        "a2_masked_top3_minus_a_best_legal_top3": (metrics["a2_masked_top3"] - metrics["best_legal_top3"]) * 100,
        "a2_masked_top5_minus_a_best_legal_top5": (metrics["a2_masked_top5"] - metrics["best_legal_top5"]) * 100,
    }


def write_outputs(summary, output_dir=OUTPUT_DIR):
    """Write summary.json and report.md for the comparison."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    return {"summary_json": str(summary_path), "report_md": str(report_path)}


def render_report(summary):
    """Render the Markdown comparison report."""

    global_metrics = summary["global"]
    mate_in_one = summary["mate_depth"].get("mateIn1", {})
    lines = [
        "# Model A vs A2 Legal Evaluation",
        "",
        "## Evaluation Integrity",
        "",
        f"- Model A checkpoint: `{summary['model_a_checkpoint']}`",
        f"- Model A2 checkpoint: `{summary['model_a2_checkpoint']}`",
        "- Test split: `data/pyg/test`",
        f"- N: `{summary['test_graphs']}`",
        f"- OOV historical: `{summary['historical_oov']}`",
        "- Status: `DIAGNOSTIC_POST_HOC_EVALUATION`",
        "",
        "## Metric Parity",
        "",
        f"- Model A parity: `{summary['model_a_parity']['status']}`",
        f"- Model A2 parity: `{summary['model_a2_parity']['status']}`",
        "",
        "## Global Comparison",
        "",
        "| Metric | Model A Raw | Model A Best-Legal | Model A2 Masked |",
        "|---|---:|---:|---:|",
        f"| Top1 | {global_metrics['raw_top1']:.6f} | {global_metrics['best_legal_top1']:.6f} | {global_metrics['a2_masked_top1']:.6f} |",
        f"| Top3 | {global_metrics['raw_top3']:.6f} | {global_metrics['best_legal_top3']:.6f} | {global_metrics['a2_masked_top3']:.6f} |",
        f"| Top5 | {global_metrics['raw_top5']:.6f} | {global_metrics['best_legal_top5']:.6f} | {global_metrics['a2_masked_top5']:.6f} |",
        "",
        "## Global Delta",
        "",
        json.dumps(summary["deltas_pp"], indent=2),
        "",
        "## Mate-in-1 Comparison",
        "",
        json.dumps(mate_in_one, indent=2),
        "",
        "## Mate Depth Comparison",
        "",
        json.dumps(summary["mate_depth"], indent=2),
        "",
        "## Optional Target Rank Analysis",
        "",
        f"- Model A median legal rank: `{global_metrics['model_a_median_legal_rank']}`",
        f"- Model A2 median legal rank: `{global_metrics['model_a2_median_legal_rank']}`",
        "",
        "## Optional Rating Analysis",
        "",
        json.dumps(summary["rating"], indent=2),
        "",
        "## Interpretation Guardrails",
        "",
        "Model A and A2 remain frozen. No training is executed by this evaluator. "
        "This post-hoc test-set analysis must not be used to choose new hyperparameters.",
        "",
    ]
    return "\n".join(lines)

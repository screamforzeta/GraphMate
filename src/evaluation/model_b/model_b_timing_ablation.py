"""Post-hoc evaluation for Model B timing ablations.

Purpose:
    Compare the frozen A3 no-timing baseline with the official timing-aware
    Model B checkpoint, then run timing-neutral diagnostic ablations on the
    same Model B weights.
Input:
    A3 checkpoint, Model B checkpoint, data/pyg and data/pyg_puzzles_timing
    test shards, final puzzle CSV rows, and the train move vocabulary.
Output:
    JSON/Markdown reports under artifacts/model_b_timing_ablation/.
Run:
    python3 -m src.cli.evaluation.evaluate_model_b_timing_ablation
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import statistics

import pandas as pd
import torch
from torch_geometric.data import Batch

from src.evaluation.model_a.model_a_vs_a2 import _mate_label
from src.evaluation.model_a.model_a_vs_a2 import _rating_bucket
from src.evaluation.model_a.model_a_vs_a2_vs_a3 import (
    MODEL_A3_CHECKPOINT,
    MODEL_A3_REFERENCE,
    SHARED_TEST_N,
    load_eval_model_a3,
)
from src.graph.pyg_dataset import ShardedPyGDataset
from src.models.model_b.chess_timing_legal_scorer import ChessGATTimingLegalMoveScorer
from src.training.model_a.model_a3_legal_scorer import (
    build_candidate_batch,
    grouped_cross_entropy,
)


MODEL_B_CHECKPOINT = Path("artifacts/model_b_timing_legal_move_scorer/best.pt")
OUTPUT_DIR = Path("artifacts/model_b_timing_ablation")
TIMING_DATASET_ROOT = Path("data/pyg_puzzles_timing")
NO_TIMING_DATASET_ROOT = Path("data/pyg")
TEST_CSV = Path("data/final/puzzles/test.csv")
GAMES_CLEAN_CSV = Path("data/processed/games/games_clean.csv")
GAMES_FINAL_CSVS = {
    "train": Path("data/final/games/games_train.csv"),
    "val": Path("data/final/games/games_val.csv"),
    "test": Path("data/final/games/games_test.csv"),
}
GAMES_FALLBACK_CSVS = {
    "processed_clean": GAMES_CLEAN_CSV,
    "processed_metadata": Path("data/processed/games/games_metadata.csv"),
}

MODEL_B_REFERENCE = {
    "n": SHARED_TEST_N,
    "loss": 1.1394022388048426,
    "top1": 0.6598141695702672,
    "top3": 0.8484320557491289,
    "top5": 0.913704994192799,
    "mean_legal_target_rank": 2.302903600464576,
    "median_legal_target_rank": 1.0,
    "illegal_top1_rate": 0.0,
    "mean_legal_candidates": 34.1541231126597,
}
B_PARITY_TOLERANCES = {
    "n": 0,
    "loss": 1e-6,
    "top1": 1e-12,
    "top3": 1e-12,
    "top5": 1e-12,
    "mean_legal_target_rank": 1e-12,
    "median_legal_target_rank": 1e-12,
    "illegal_top1_rate": 1e-12,
    "mean_legal_candidates": 1e-12,
}
RATING_BUCKET_ORDER = ["<1200", "1200-1599", "1600-1999", "2000-2399", "2400+"]


@dataclass
class ModelBTimingAblationConfig:
    """Runtime options for read-only Model B timing ablation."""

    batch_size: int = 128
    device: str = "cpu"
    non_blocking: bool = False
    amp: bool = False
    timing_dataset_root: str = str(TIMING_DATASET_ROOT)
    no_timing_dataset_root: str = str(NO_TIMING_DATASET_ROOT)
    model_b_checkpoint: str = str(MODEL_B_CHECKPOINT)
    model_a3_checkpoint: str = str(MODEL_A3_CHECKPOINT)
    output_dir: str = str(OUTPUT_DIR)
    include_zero_timing_and_flag: bool = True
    max_error_examples_per_category: int = 8


def load_eval_model_b(checkpoint_path, device):
    """Load the official Model B checkpoint for eval-only use."""

    model = ChessGATTimingLegalMoveScorer(dropout=0.30).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def canonical_a3_metrics():
    """Return official A3 aggregate metrics in the Model B report schema."""

    return {
        "source": "OFFICIAL_A3_REFERENCE",
        "n": MODEL_A3_REFERENCE["n"],
        "loss": MODEL_A3_REFERENCE["a3_loss"],
        "top1": MODEL_A3_REFERENCE["a3_top1"],
        "top3": MODEL_A3_REFERENCE["a3_top3"],
        "top5": MODEL_A3_REFERENCE["a3_top5"],
        "mean_legal_target_rank": MODEL_A3_REFERENCE["a3_mean_legal_rank"],
        "median_legal_target_rank": MODEL_A3_REFERENCE["a3_median_legal_rank"],
        "illegal_top1_rate": MODEL_A3_REFERENCE["a3_illegal_top1_rate"],
    }


def _empty_totals():
    """Create counters for candidate-scorer metrics and paired rows."""

    return {
        "loss_sum": 0.0,
        "n": 0,
        "top1_correct": 0,
        "top3_correct": 0,
        "top5_correct": 0,
        "rank_sum": 0.0,
        "ranks": [],
        "candidate_counts": [],
    }


def _median(values):
    """Return median as float for a non-empty sequence."""

    return float(statistics.median(values)) if values else None


def _rank_predictions(scores, candidate_ptr, target_indices, candidate_uci):
    """Return per-graph top prediction, rank, and correctness flags."""

    rows = []
    for graph_index in range(target_indices.numel()):
        start = int(candidate_ptr[graph_index].item())
        end = int(candidate_ptr[graph_index + 1].item())
        target = int(target_indices[graph_index].item())
        group_scores = scores[start:end]
        ordered = torch.argsort(group_scores, descending=True)
        rank = int((ordered == target).nonzero(as_tuple=False).item()) + 1
        top_local = int(ordered[0].item())
        count = end - start
        rows.append(
            {
                "target_rank": rank,
                "top1_correct": rank == 1,
                "top3_correct": rank <= min(3, count),
                "top5_correct": rank <= min(5, count),
                "prediction_uci": candidate_uci[start + top_local],
                "candidate_count": count,
            }
        )
    return rows


def _update_totals(totals, loss, ranked):
    """Accumulate aggregate metrics from one batch."""

    batch_size = len(ranked)
    totals["loss_sum"] += float(loss.item()) * batch_size
    totals["n"] += batch_size
    for row in ranked:
        rank = int(row["target_rank"])
        totals["rank_sum"] += rank
        totals["ranks"].append(rank)
        totals["candidate_counts"].append(int(row["candidate_count"]))
        totals["top1_correct"] += int(row["top1_correct"])
        totals["top3_correct"] += int(row["top3_correct"])
        totals["top5_correct"] += int(row["top5_correct"])


def _finalize_totals(totals, source):
    """Convert counters into report metrics."""

    n = totals["n"]
    if n <= 0:
        raise ValueError("Cannot finalize zero examples.")
    candidate_counts = totals["candidate_counts"]
    return {
        "source": source,
        "n": n,
        "loss": totals["loss_sum"] / n,
        "top1": totals["top1_correct"] / n,
        "top3": totals["top3_correct"] / n,
        "top5": totals["top5_correct"] / n,
        "mean_legal_target_rank": totals["rank_sum"] / n,
        "median_legal_target_rank": _median(totals["ranks"]),
        "illegal_top1_rate": 0.0,
        "mean_legal_candidates": sum(candidate_counts) / len(candidate_counts),
        "min_legal_candidates": min(candidate_counts),
        "max_legal_candidates": max(candidate_counts),
    }


def _set_timing_mode(batch, mode):
    """Apply post-hoc timing values to a batch without changing candidates."""

    if mode == "B_SYNTHETIC_TIMING":
        return
    batch_size = int(batch.global_features.shape[0])
    device = batch.global_features.device
    zero = torch.zeros((batch_size, 1), dtype=torch.float, device=device)
    batch.previous_move_time = zero
    batch.original_move_time = zero.clone()
    if mode == "B_NEUTRAL_TIMING":
        batch.time_is_synthetic = torch.ones((batch_size, 1), dtype=torch.float, device=device)
    elif mode == "B_ZERO_TIMING_AND_FLAG":
        batch.time_is_synthetic = torch.zeros((batch_size, 1), dtype=torch.float, device=device)
    else:
        raise ValueError(f"Unknown timing mode: {mode}")


def _as_scalar(value):
    """Return a JSON-safe scalar from tensor/list metadata."""

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().view(-1)[0].item()
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _as_scalar(value[0])
        return [_as_scalar(item) for item in value]
    return value


def _optional_float_attr(graph, attribute):
    """Return an optional graph attribute as float, preserving missing as None."""

    value = getattr(graph, attribute, None)
    if value is None:
        return None
    return float(_as_scalar(value))


def _graph_metadata(graph, row):
    """Build stable per-puzzle metadata from graph attributes and CSV row.

    A3 graphs are intentionally no-timing graphs, so timing-only metadata is
    optional and stays None until B synthetic rows provide it during pairing.
    """

    return {
        "PuzzleId": str(row.PuzzleId),
        "FEN": str(getattr(graph, "fen")),
        "target_uci": str(getattr(graph, "target_move")),
        "rating": int(row.Rating),
        "MateDepth": int(row.MateDepth),
        "previous_move_time_seconds": _optional_float_attr(
            graph, "previous_move_time_seconds"
        ),
        "original_move_time_seconds": _optional_float_attr(
            graph, "original_move_time_seconds"
        ),
        "previous_move_time": _optional_float_attr(graph, "previous_move_time"),
        "original_move_time": _optional_float_attr(graph, "original_move_time"),
        "time_is_synthetic": _optional_float_attr(graph, "time_is_synthetic"),
    }


def evaluate_model_on_dataset(model, dataset, dataframe, config, mode, device):
    """Evaluate A3 or B on one dataset and return metrics plus paired rows."""

    rows = list(dataframe.itertuples(index=False))
    totals = _empty_totals()
    per_puzzle = []
    use_amp = bool(config.amp and device.type == "cuda")
    is_model_b = isinstance(model, ChessGATTimingLegalMoveScorer)

    with torch.no_grad():
        for start in range(0, len(dataset), config.batch_size):
            indices = range(start, min(start + config.batch_size, len(dataset)))
            graphs = [dataset[index] for index in indices]
            batch_rows = [rows[int(graph.source_row_index.item())] for graph in graphs]
            batch = Batch.from_data_list(graphs).to(
                device,
                non_blocking=bool(config.non_blocking and device.type == "cuda"),
            )
            if is_model_b:
                _set_timing_mode(batch, mode)
            candidate = build_candidate_batch(batch)
            target_indices = candidate["target_indices"].to(device)
            with torch.amp.autocast(device.type, enabled=use_amp):
                output = model(batch, candidate["candidate_moves"])
                loss = grouped_cross_entropy(
                    output["scores"],
                    output["candidate_ptr"],
                    target_indices,
                )
            ranked = _rank_predictions(
                output["scores"],
                output["candidate_ptr"],
                target_indices,
                output["candidate_uci"],
            )
            _update_totals(totals, loss, ranked)
            for graph, row, ranked_row in zip(graphs, batch_rows, ranked):
                metadata = _graph_metadata(graph, row)
                per_puzzle.append(
                    {
                        **metadata,
                        "number_legal_candidates": int(ranked_row["candidate_count"]),
                        f"{mode}_target_rank": int(ranked_row["target_rank"]),
                        f"{mode}_top1_correct": bool(ranked_row["top1_correct"]),
                        f"{mode}_prediction": str(ranked_row["prediction_uci"]),
                    }
                )

    return _finalize_totals(totals, mode), per_puzzle


def _merge_per_puzzle(a3_rows, synthetic_rows, neutral_rows, zero_rows=None):
    """Merge model-specific per-puzzle rows by PuzzleId and validate alignment."""

    merged = {}
    timing_keys = {
        "previous_move_time_seconds",
        "original_move_time_seconds",
        "previous_move_time",
        "original_move_time",
        "time_is_synthetic",
    }
    for source_name, source_rows in (
        ("A3", a3_rows),
        ("B_SYNTHETIC_TIMING", synthetic_rows),
        ("B_NEUTRAL_TIMING", neutral_rows),
        ("B_ZERO_TIMING_AND_FLAG", zero_rows or []),
    ):
        for row in source_rows:
            puzzle_id = row["PuzzleId"]
            if puzzle_id not in merged:
                merged[puzzle_id] = {
                    key: value
                    for key, value in row.items()
                    if not key.endswith(("_target_rank", "_top1_correct", "_prediction"))
                }
            else:
                for key in ("FEN", "target_uci", "rating", "MateDepth"):
                    if merged[puzzle_id][key] != row[key]:
                        raise ValueError(f"Puzzle alignment mismatch for {puzzle_id}: {key}")
            if source_name == "B_SYNTHETIC_TIMING":
                for key in timing_keys:
                    merged[puzzle_id][key] = row.get(key)
            for key, value in row.items():
                if key.endswith(("_target_rank", "_top1_correct", "_prediction")):
                    merged[puzzle_id][key] = value
    return list(merged.values())


def _validate_puzzle_alignment(a3_rows, synthetic_rows):
    """Fail fast unless A3 and B synthetic rows refer to the same PuzzleIds."""

    a3_ids = [row["PuzzleId"] for row in a3_rows]
    synthetic_ids = [row["PuzzleId"] for row in synthetic_rows]
    if len(a3_ids) != SHARED_TEST_N or len(synthetic_ids) != SHARED_TEST_N:
        raise ValueError(
            f"Shared-test N mismatch: A3={len(a3_ids)}, "
            f"B_SYNTHETIC_TIMING={len(synthetic_ids)}, expected={SHARED_TEST_N}"
        )
    if set(a3_ids) != set(synthetic_ids):
        missing_from_b = sorted(set(a3_ids) - set(synthetic_ids))[:10]
        missing_from_a3 = sorted(set(synthetic_ids) - set(a3_ids))[:10]
        raise ValueError(
            "A3/B PuzzleId set mismatch. "
            f"missing_from_b={missing_from_b}, missing_from_a3={missing_from_a3}"
        )
    return {
        "same_order": a3_ids == synthetic_ids,
        "same_set": True,
        "n": len(a3_ids),
    }


def _transition_counts(rows, left_prefix, right_prefix):
    """Return paired Top1 transition counts with left/right conventions."""

    counts = {
        "left_correct_right_correct": 0,
        "left_correct_right_wrong": 0,
        "left_wrong_right_correct": 0,
        "left_wrong_right_wrong": 0,
    }
    for row in rows:
        left = bool(row[f"{left_prefix}_top1_correct"])
        right = bool(row[f"{right_prefix}_top1_correct"])
        if left and right:
            counts["left_correct_right_correct"] += 1
        elif left and not right:
            counts["left_correct_right_wrong"] += 1
        elif not left and right:
            counts["left_wrong_right_correct"] += 1
        else:
            counts["left_wrong_right_wrong"] += 1
    counts["right_recovers_from_left"] = counts["left_wrong_right_correct"]
    counts["left_lost_by_right"] = counts["left_correct_right_wrong"]
    counts["net_right_minus_left"] = (
        counts["left_wrong_right_correct"] - counts["left_correct_right_wrong"]
    )
    return counts


def _rank_delta(rows, left_prefix, right_prefix):
    """Summarize paired legal-rank deltas: right rank minus left rank."""

    deltas = [
        int(row[f"{right_prefix}_target_rank"]) - int(row[f"{left_prefix}_target_rank"])
        for row in rows
    ]
    n = len(deltas)
    return {
        "n": n,
        "improved_fraction": sum(delta < 0 for delta in deltas) / n,
        "unchanged_fraction": sum(delta == 0 for delta in deltas) / n,
        "worsened_fraction": sum(delta > 0 for delta in deltas) / n,
        "mean_delta": sum(deltas) / n,
        "median_delta": statistics.median(deltas),
    }


def _mcnemar(rows, left_prefix, right_prefix):
    """Compute exact two-sided McNemar test from paired Top1 outcomes."""

    left_wrong_right_correct = 0
    left_correct_right_wrong = 0
    for row in rows:
        left = bool(row[f"{left_prefix}_top1_correct"])
        right = bool(row[f"{right_prefix}_top1_correct"])
        if not left and right:
            left_wrong_right_correct += 1
        elif left and not right:
            left_correct_right_wrong += 1
    n = left_wrong_right_correct + left_correct_right_wrong
    if n == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        smaller = min(left_wrong_right_correct, left_correct_right_wrong)
        cdf = sum(math.comb(n, k) for k in range(smaller + 1)) * (0.5 ** n)
        p_value = min(1.0, 2.0 * cdf)
        statistic = (
            (abs(left_wrong_right_correct - left_correct_right_wrong) - 1) ** 2 / n
        )
    return {
        "convention": "n01=left wrong/right correct; n10=left correct/right wrong",
        "n01": left_wrong_right_correct,
        "n10": left_correct_right_wrong,
        "statistic_chi_square_cc": statistic,
        "p_value_exact_two_sided": p_value,
    }


def _bucket_metrics(rows, prefix):
    """Compute candidate metrics for a list of per-puzzle rows."""

    if not rows:
        return {"n": 0}
    n = len(rows)
    ranks = [int(row[f"{prefix}_target_rank"]) for row in rows]
    return {
        "n": n,
        "top1": sum(bool(row[f"{prefix}_top1_correct"]) for row in rows) / n,
        "top3": sum(rank <= 3 for rank in ranks) / n,
        "top5": sum(rank <= 5 for rank in ranks) / n,
        "mean_legal_target_rank": sum(ranks) / n,
        "median_legal_target_rank": statistics.median(ranks),
        "mean_legal_candidates": sum(int(row["number_legal_candidates"]) for row in rows) / n,
    }


def _bucket_breakdown(rows, key_function):
    """Return A3/B synthetic/B neutral metrics by subgroup."""

    groups = defaultdict(list)
    for row in rows:
        groups[key_function(row)].append(row)
    result = {}
    for key, group_rows in sorted(groups.items()):
        a3 = _bucket_metrics(group_rows, "A3")
        synthetic = _bucket_metrics(group_rows, "B_SYNTHETIC_TIMING")
        neutral = _bucket_metrics(group_rows, "B_NEUTRAL_TIMING")
        result[str(key)] = {
            "A3": a3,
            "B_SYNTHETIC_TIMING": synthetic,
            "B_NEUTRAL_TIMING": neutral,
            "delta_top1_b_synthetic_minus_a3_pp": (
                synthetic["top1"] - a3["top1"]
            ) * 100,
            "delta_top1_b_neutral_minus_a3_pp": (
                neutral["top1"] - a3["top1"]
            ) * 100,
            "delta_top1_b_synthetic_minus_neutral_pp": (
                synthetic["top1"] - neutral["top1"]
            ) * 100,
        }
    return result


def _timing_quantile_breakdown(rows, timing_key, num_buckets=5):
    """Bucket rows by timing quantiles and summarize performance."""

    ordered = sorted(rows, key=lambda row: float(row[timing_key]))
    n = len(ordered)
    result = {}
    for bucket in range(num_buckets):
        start = bucket * n // num_buckets
        end = (bucket + 1) * n // num_buckets
        bucket_rows = ordered[start:end]
        label = f"Q{bucket + 1}"
        a3 = _bucket_metrics(bucket_rows, "A3")
        synthetic = _bucket_metrics(bucket_rows, "B_SYNTHETIC_TIMING")
        neutral = _bucket_metrics(bucket_rows, "B_NEUTRAL_TIMING")
        result[label] = {
            "n": len(bucket_rows),
            "min_timing": min(float(row[timing_key]) for row in bucket_rows),
            "max_timing": max(float(row[timing_key]) for row in bucket_rows),
            "mean_timing": statistics.mean(float(row[timing_key]) for row in bucket_rows),
            "mean_rating": statistics.mean(float(row["rating"]) for row in bucket_rows),
            "A3_top1": a3["top1"],
            "B_SYNTHETIC_top1": synthetic["top1"],
            "B_NEUTRAL_top1": neutral["top1"],
            "delta_b_synthetic_minus_a3_pp": (synthetic["top1"] - a3["top1"]) * 100,
            "delta_b_synthetic_minus_neutral_pp": (synthetic["top1"] - neutral["top1"]) * 100,
        }
    return result


def _pearson(xs, ys):
    """Compute Pearson correlation without external dependencies."""

    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return None
    return numerator / (x_den * y_den)


def _rankdata(values):
    """Return average ranks for Spearman correlation."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def _spearman(xs, ys):
    """Compute Spearman correlation with average ranks."""

    return _pearson(_rankdata(xs), _rankdata(ys))


def _describe(values):
    """Return descriptive statistics for numeric values."""

    if not values:
        return {"n": 0}
    ordered = sorted(float(value) for value in values)
    mean = statistics.mean(ordered)
    std = statistics.pstdev(ordered)
    if std > 0:
        skewness = statistics.mean(((value - mean) / std) ** 3 for value in ordered)
    else:
        skewness = 0.0
    return {
        "n": len(ordered),
        "mean": mean,
        "median": statistics.median(ordered),
        "std": std,
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "p50": _quantile(ordered, 0.50),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "p99": _quantile(ordered, 0.99),
        "max": max(ordered),
        "skewness": skewness,
    }


def _quantile(ordered_values, q):
    """Return linear-interpolated quantile from a sorted sequence."""

    if not ordered_values:
        return None
    position = (len(ordered_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered_values[lower]
    weight = position - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def _parse_move_times_sequence(value):
    """Parse one serialized MoveTimes list and classify every item.

    MoveTimes are persisted as Python list strings such as
    "[0.0, 1.0, None]"; JSON parsing cannot read Python None, so the
    evaluator uses ast.literal_eval and never modifies the source CSV.
    """

    stats = {
        "raw_values": 0,
        "valid_values": [],
        "discarded_missing": 0,
        "discarded_non_numeric": 0,
        "discarded_non_finite": 0,
        "discarded_non_positive": 0,
        "parse_errors": 0,
    }
    if pd.isna(value) or str(value).strip() in ("", "None", "nan"):
        stats["discarded_missing"] += 1
        return stats
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        stats["parse_errors"] += 1
        return stats
    if not isinstance(parsed, (list, tuple)):
        stats["parse_errors"] += 1
        return stats
    for item in parsed:
        stats["raw_values"] += 1
        if item is None:
            stats["discarded_missing"] += 1
            continue
        try:
            seconds = float(item)
        except (TypeError, ValueError):
            stats["discarded_non_numeric"] += 1
            continue
        if not math.isfinite(seconds):
            stats["discarded_non_finite"] += 1
            continue
        if seconds <= 0:
            stats["discarded_non_positive"] += 1
            continue
        stats["valid_values"].append(seconds)
    return stats


def extract_real_move_times(source_paths=None):
    """Extract valid real move times from persisted games CSV files.

    Parameters:
        source_paths: Optional mapping label -> CSV path. Defaults to the
            prepared final games splits.
    Returns:
        Dict with source status, counters, discard reasons, and seconds list.
    Side effects:
        Reads CSV files only.
    """

    source_paths = source_paths or GAMES_FINAL_CSVS
    result = {
        "status": "INCOMPLETE",
        "source": "final_games_splits",
        "unit": "seconds",
        "paths_searched": {key: str(path) for key, path in source_paths.items()},
        "paths_used": [],
        "games_scanned": 0,
        "games_with_timing": 0,
        "raw_timing_values": 0,
        "valid_timing_values": 0,
        "discarded_values": 0,
        "discard_reasons": {
            "missing": 0,
            "non_numeric": 0,
            "non_finite": 0,
            "non_positive": 0,
            "parse_errors": 0,
        },
        "warnings": [],
        "seconds": [],
    }
    existing = {key: Path(path) for key, path in source_paths.items() if Path(path).exists()}
    if not existing:
        result["warnings"].append("No games CSV source found.")
        return result

    for label, path in existing.items():
        try:
            dataframe = pd.read_csv(path, usecols=["MoveTimes"])
        except ValueError:
            result["warnings"].append(f"Missing MoveTimes column in {path}.")
            continue
        result["paths_used"].append(str(path))
        result["games_scanned"] += len(dataframe)
        for raw in dataframe["MoveTimes"]:
            parsed = _parse_move_times_sequence(raw)
            result["raw_timing_values"] += parsed["raw_values"]
            valid_values = parsed["valid_values"]
            if valid_values:
                result["games_with_timing"] += 1
                result["seconds"].extend(valid_values)
            result["discard_reasons"]["missing"] += parsed["discarded_missing"]
            result["discard_reasons"]["non_numeric"] += parsed["discarded_non_numeric"]
            result["discard_reasons"]["non_finite"] += parsed["discarded_non_finite"]
            result["discard_reasons"]["non_positive"] += parsed["discarded_non_positive"]
            result["discard_reasons"]["parse_errors"] += parsed["parse_errors"]

    result["valid_timing_values"] = len(result["seconds"])
    result["discarded_values"] = sum(result["discard_reasons"].values())
    if result["valid_timing_values"] > 0:
        result["status"] = "COMPLETE"
    else:
        result["warnings"].append("Games sources were found but no valid MoveTimes were extracted.")
    return result


def _timing_distribution(rows):
    """Compare synthetic test timing against locally available real game timing."""

    synthetic = [float(row["original_move_time_seconds"]) for row in rows]
    real_audit = extract_real_move_times()
    real = real_audit["seconds"]
    seconds_comparison = {
        "real": _describe(real),
        "synthetic": _describe(synthetic),
    }
    log_comparison = {
        "real": _describe([math.log1p(value) for value in real]),
        "synthetic": _describe([math.log1p(value) for value in synthetic]),
    }
    return {
        "status": real_audit["status"],
        "source": {
            key: value
            for key, value in real_audit.items()
            if key != "seconds"
        },
        "real_seconds": seconds_comparison["real"],
        "synthetic_original_seconds": seconds_comparison["synthetic"],
        "real_log1p": log_comparison["real"],
        "synthetic_original_log1p": log_comparison["synthetic"],
        "seconds_table": {
            "real": seconds_comparison["real"],
            "synthetic": seconds_comparison["synthetic"],
        },
        "log1p_table": {
            "real": log_comparison["real"],
            "synthetic": log_comparison["synthetic"],
        },
        "ratios": _timing_distribution_ratios(seconds_comparison["real"], seconds_comparison["synthetic"]),
        "ks": _ks_diagnostics(real, synthetic),
        "limitation": (
            "Real timings come from sampled Lichess games, while synthetic timings "
            "come from puzzle test graphs. The populations are not paired; this is "
            "a distributional plausibility comparison, not ground-truth timing accuracy."
        ),
    }


def _timing_distribution_ratios(real_stats, synthetic_stats):
    """Return simple descriptive synthetic/real ratios where defined."""

    ratios = {}
    for key in ("mean", "median", "p95"):
        real_value = real_stats.get(key)
        synthetic_value = synthetic_stats.get(key)
        ratios[f"synthetic_{key}_over_real_{key}"] = (
            synthetic_value / real_value
            if real_value not in (None, 0) and synthetic_value is not None
            else None
        )
    return ratios


def _ks_diagnostics(real, synthetic):
    """Return optional KS diagnostics when scipy is installed."""

    try:
        from scipy import stats
    except ImportError:
        return {
            "available": False,
            "reason": "scipy is not installed; dependency not added for this diagnostic.",
        }
    if not real or not synthetic:
        return {"available": False, "reason": "real or synthetic sample is empty."}
    seconds = stats.ks_2samp(real, synthetic)
    log_values = stats.ks_2samp(
        [math.log1p(value) for value in real],
        [math.log1p(value) for value in synthetic],
    )
    return {
        "available": True,
        "seconds": {
            "statistic": float(seconds.statistic),
            "p_value": float(seconds.pvalue),
        },
        "log1p": {
            "statistic": float(log_values.statistic),
            "p_value": float(log_values.pvalue),
        },
        "interpretation_guardrail": (
            "KS is descriptive only. Large samples and non-equivalent populations "
            "make the p-value unsuitable as a validity verdict."
        ),
    }


def _rating_shortcut(rows):
    """Quantify rating dependence of synthetic timing."""

    timings = [float(row["original_move_time_seconds"]) for row in rows]
    ratings = [float(row["rating"]) for row in rows]
    pearson = _pearson(timings, ratings)
    spearman = _spearman(timings, ratings)
    return {
        "POTENTIAL_SHORTCUT": "YES",
        "pearson_original_time_rating": pearson,
        "spearman_original_time_rating": spearman,
        "r2_linear_rating_from_timing": pearson * pearson if pearson is not None else None,
    }


def _error_examples(rows, max_per_category):
    """Collect representative paired error transitions."""

    categories = {
        "A3_wrong_to_B_synthetic_correct": [],
        "A3_correct_to_B_synthetic_wrong": [],
        "B_neutral_wrong_to_B_synthetic_correct": [],
        "B_neutral_correct_to_B_synthetic_wrong": [],
    }
    for row in rows:
        a3 = bool(row["A3_top1_correct"])
        synthetic = bool(row["B_SYNTHETIC_TIMING_top1_correct"])
        neutral = bool(row["B_NEUTRAL_TIMING_top1_correct"])
        base = {
            "PuzzleId": row["PuzzleId"],
            "FEN": row["FEN"],
            "rating": row["rating"],
            "MateDepth": row["MateDepth"],
            "previous_move_time_seconds": row["previous_move_time_seconds"],
            "original_move_time_seconds": row["original_move_time_seconds"],
            "target": row["target_uci"],
            "A3_prediction": row["A3_prediction"],
            "A3_rank": row["A3_target_rank"],
            "B_synthetic_prediction": row["B_SYNTHETIC_TIMING_prediction"],
            "B_synthetic_rank": row["B_SYNTHETIC_TIMING_target_rank"],
            "B_neutral_prediction": row["B_NEUTRAL_TIMING_prediction"],
            "B_neutral_rank": row["B_NEUTRAL_TIMING_target_rank"],
        }
        if not a3 and synthetic and len(categories["A3_wrong_to_B_synthetic_correct"]) < max_per_category:
            categories["A3_wrong_to_B_synthetic_correct"].append(base)
        if a3 and not synthetic and len(categories["A3_correct_to_B_synthetic_wrong"]) < max_per_category:
            categories["A3_correct_to_B_synthetic_wrong"].append(base)
        if not neutral and synthetic and len(categories["B_neutral_wrong_to_B_synthetic_correct"]) < max_per_category:
            categories["B_neutral_wrong_to_B_synthetic_correct"].append(base)
        if neutral and not synthetic and len(categories["B_neutral_correct_to_B_synthetic_wrong"]) < max_per_category:
            categories["B_neutral_correct_to_B_synthetic_wrong"].append(base)
    return categories


def _parity_diagnostics(actual, reference=MODEL_B_REFERENCE, tolerances=B_PARITY_TOLERANCES):
    """Return per-field Model B terminal parity diagnostics."""

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


def model_b_parity_status(actual):
    """Compare B synthetic timing metrics to the official terminal test."""

    diagnostics = _parity_diagnostics(actual)
    passed = all(item["result"] == "PASS" for item in diagnostics)
    return {
        "model": "MODEL_B_TIMING_LEGAL_MOVE_SCORER",
        "status": "PASS" if passed else "FAIL",
        "reference": dict(MODEL_B_REFERENCE),
        "actual": {key: actual.get(key) for key in MODEL_B_REFERENCE},
        "diagnostics": diagnostics,
    }


def build_shared_comparison_frame(a3_metrics, b_metrics):
    """Build A3 vs B rows with explicit source validation.

    The source fields prevent the reporting bug where the same Model B metrics
    dictionary is accidentally reused for the A3 column.
    """

    if a3_metrics.get("source") != "OFFICIAL_A3_REFERENCE":
        raise ValueError("A3 metrics must come from OFFICIAL_A3_REFERENCE.")
    if b_metrics.get("source") != "B_SYNTHETIC_TIMING":
        raise ValueError("B metrics must come from B_SYNTHETIC_TIMING.")
    rows = []
    metric_map = [
        ("CE/NLL", "loss", False),
        ("Top1", "top1", True),
        ("Top3", "top3", True),
        ("Top5", "top5", True),
        ("mean legal target rank", "mean_legal_target_rank", False),
        ("median legal target rank", "median_legal_target_rank", False),
        ("illegal Top1", "illegal_top1_rate", True),
        ("N", "n", False),
    ]
    for label, key, is_rate in metric_map:
        a3_value = a3_metrics[key]
        b_value = b_metrics[key]
        delta = b_value - a3_value
        row = {
            "metric": label,
            "a3": a3_value,
            "b": b_value,
            "delta_b_minus_a3": delta,
            "a3_source": a3_metrics["source"],
            "b_source": b_metrics["source"],
        }
        if is_rate:
            row["delta_b_minus_a3_pp"] = delta * 100
        rows.append(row)
    return rows


def run_timing_ablation(config):
    """Run the full read-only A3 vs Model B timing ablation."""

    device = torch.device(config.device)
    timing_dataset = ShardedPyGDataset(root=config.timing_dataset_root, split="test")
    no_timing_dataset = ShardedPyGDataset(root=config.no_timing_dataset_root, split="test")
    dataframe = pd.read_csv(TEST_CSV)
    if len(timing_dataset) != SHARED_TEST_N or len(no_timing_dataset) != SHARED_TEST_N:
        raise ValueError(
            f"Shared-test alignment failed: timing={len(timing_dataset)}, "
            f"no_timing={len(no_timing_dataset)}, expected={SHARED_TEST_N}"
        )

    model_a3_path = Path(config.model_a3_checkpoint)
    model_b_path = Path(config.model_b_checkpoint)
    if not model_a3_path.exists():
        raise FileNotFoundError(f"A3 checkpoint not found: {model_a3_path}")
    if not model_b_path.exists():
        raise FileNotFoundError(f"Model B checkpoint not found: {model_b_path}")

    model_a3, checkpoint_a3 = load_eval_model_a3(model_a3_path, device)
    model_b, checkpoint_b = load_eval_model_b(model_b_path, device)

    a3_eval, a3_rows = evaluate_model_on_dataset(
        model_a3, no_timing_dataset, dataframe, config, "A3", device
    )
    synthetic_metrics, synthetic_rows = evaluate_model_on_dataset(
        model_b, timing_dataset, dataframe, config, "B_SYNTHETIC_TIMING", device
    )
    alignment = _validate_puzzle_alignment(a3_rows, synthetic_rows)
    neutral_metrics, neutral_rows = evaluate_model_on_dataset(
        model_b, timing_dataset, dataframe, config, "B_NEUTRAL_TIMING", device
    )
    zero_metrics = None
    zero_rows = None
    if config.include_zero_timing_and_flag:
        zero_metrics, zero_rows = evaluate_model_on_dataset(
            model_b, timing_dataset, dataframe, config, "B_ZERO_TIMING_AND_FLAG", device
        )

    paired_rows = _merge_per_puzzle(a3_rows, synthetic_rows, neutral_rows, zero_rows)
    if len(paired_rows) != SHARED_TEST_N:
        raise ValueError(f"Paired rows mismatch: {len(paired_rows)} != {SHARED_TEST_N}")

    a3_official = canonical_a3_metrics()
    shared_frame = build_shared_comparison_frame(a3_official, synthetic_metrics)
    mate_depth = _bucket_breakdown(paired_rows, lambda row: f"mateIn{row['MateDepth']}")
    rating = _bucket_breakdown(paired_rows, lambda row: _rating_bucket_object(row["rating"]))

    summary = {
        "status": "MODEL_B_TIMING_ABLATION_COMPLETE",
        "reporting_bug_root_cause": (
            "The Shared Comparison Frame must use independent metric sources. "
            "A3 is populated from OFFICIAL_A3_REFERENCE; B is populated from "
            "B_SYNTHETIC_TIMING. Source validation raises if the same B "
            "dictionary is reused for A3."
        ),
        "shared_test_n": SHARED_TEST_N,
        "a3_checkpoint": str(model_a3_path),
        "model_b_checkpoint": str(model_b_path),
        "model_a3_checkpoint_epoch": checkpoint_a3.get("epoch") if isinstance(checkpoint_a3, dict) else None,
        "model_b_checkpoint_epoch": checkpoint_b.get("epoch") if isinstance(checkpoint_b, dict) else None,
        "shared_test_alignment": alignment,
        "official_model_performance": {
            "A3": a3_official,
            "B_SYNTHETIC_TIMING": synthetic_metrics,
            "shared_comparison_frame": shared_frame,
        },
        "post_hoc_diagnostic_ablations": {
            "B_NEUTRAL_TIMING": neutral_metrics,
            "B_ZERO_TIMING_AND_FLAG": zero_metrics,
        },
        "model_b_terminal_parity": model_b_parity_status(synthetic_metrics),
        "a3_recomputed_for_paired_analysis": a3_eval,
        "paired_rows_path": None,
        "transitions": {
            "A3_vs_B_SYNTHETIC_TIMING": _transition_counts(
                paired_rows, "A3", "B_SYNTHETIC_TIMING"
            ),
            "B_NEUTRAL_TIMING_vs_B_SYNTHETIC_TIMING": _transition_counts(
                paired_rows, "B_NEUTRAL_TIMING", "B_SYNTHETIC_TIMING"
            ),
        },
        "rank_delta": {
            "B_SYNTHETIC_TIMING_minus_A3": _rank_delta(
                paired_rows, "A3", "B_SYNTHETIC_TIMING"
            ),
            "B_SYNTHETIC_TIMING_minus_B_NEUTRAL_TIMING": _rank_delta(
                paired_rows, "B_NEUTRAL_TIMING", "B_SYNTHETIC_TIMING"
            ),
        },
        "mcnemar": {
            "A3_vs_B_SYNTHETIC_TIMING": _mcnemar(
                paired_rows, "A3", "B_SYNTHETIC_TIMING"
            ),
            "B_NEUTRAL_TIMING_vs_B_SYNTHETIC_TIMING": _mcnemar(
                paired_rows, "B_NEUTRAL_TIMING", "B_SYNTHETIC_TIMING"
            ),
        },
        "mate_depth": mate_depth,
        "rating": rating,
        "timing_quantiles": {
            "original_move_time_seconds": _timing_quantile_breakdown(
                paired_rows, "original_move_time_seconds"
            ),
            "previous_move_time_seconds": _timing_quantile_breakdown(
                paired_rows, "previous_move_time_seconds"
            ),
        },
        "timing_distribution_vs_real_games": _timing_distribution(paired_rows),
        "rating_shortcut_analysis": _rating_shortcut(paired_rows),
        "error_examples": _error_examples(
            paired_rows, config.max_error_examples_per_category
        ),
        "interpretation_guardrails": [
            "B_NEUTRAL_TIMING is not A3.",
            "B has a timing encoder, candidate dim 412, extra parameters, and weights learned with timing.",
            "A3 vs B_NEUTRAL is not a pure architecture effect.",
            "B_SYNTHETIC_TIMING vs B_NEUTRAL_TIMING is a post-hoc sensitivity ablation of the same checkpoint.",
            "MateDepth, rating, and timing buckets are descriptive post-hoc analyses only.",
        ],
    }
    return summary, paired_rows


def _rating_bucket_object(rating):
    """Return official rating bucket from an integer rating."""

    class Row:
        Rating = rating

    return _rating_bucket(Row)


def write_outputs(summary, paired_rows, output_dir=OUTPUT_DIR):
    """Write summary, paired rows, and Markdown report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = output_dir / "paired_test_rows.json"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary = dict(summary)
    summary["paired_rows_path"] = str(paired_path)
    paired_path.write_text(json.dumps(paired_rows, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "paired_rows_json": str(paired_path),
        "report_md": str(report_path),
    }


def update_timing_distribution_only(output_dir=OUTPUT_DIR):
    """Refresh real-vs-synthetic timing stats from existing ablation artifacts.

    Parameters:
        output_dir: Directory containing summary.json and paired_test_rows.json.
    Returns:
        Dict of written output paths.
    Side effects:
        Updates summary.json and report.md. It does not run model inference and
        does not alter paired_test_rows.json.
    """

    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.json"
    paired_path = output_dir / "paired_test_rows.json"
    report_path = output_dir / "report.md"
    if not summary_path.exists() or not paired_path.exists():
        raise FileNotFoundError(
            "Timing-distribution-only mode requires existing "
            f"{summary_path} and {paired_path}."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    paired_rows = json.loads(paired_path.read_text(encoding="utf-8"))
    preserved_keys = {
        "official_model_performance": summary.get("official_model_performance"),
        "post_hoc_diagnostic_ablations": summary.get("post_hoc_diagnostic_ablations"),
        "transitions": summary.get("transitions"),
        "mate_depth": summary.get("mate_depth"),
        "rating": summary.get("rating"),
        "mcnemar": summary.get("mcnemar"),
        "rank_delta": summary.get("rank_delta"),
    }
    summary["timing_distribution_vs_real_games"] = _timing_distribution(paired_rows)
    for key, value in preserved_keys.items():
        if summary.get(key) != value:
            raise RuntimeError(f"Unexpected mutation outside timing distribution: {key}")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        report = render_report(summary)
    except KeyError:
        report = _render_timing_distribution_only_report(summary)
    report_path.write_text(report, encoding="utf-8")
    return {
        "summary_json": str(summary_path),
        "paired_rows_json": str(paired_path),
        "report_md": str(report_path),
        "timing_distribution_status": summary["timing_distribution_vs_real_games"]["status"],
        "real_valid_n": summary["timing_distribution_vs_real_games"]["source"]["valid_timing_values"],
    }


def _render_timing_distribution_only_report(summary):
    """Render a compact report when only timing-distribution fields exist."""

    return "\n".join(
        [
            "# Model B Timing Distribution Update",
            "",
            "This report was generated in timing-distribution-only mode.",
            "",
            "## Synthetic Vs Real Timing Distribution",
            "",
            "```json",
            json.dumps(summary["timing_distribution_vs_real_games"], indent=2),
            "```",
            "",
        ]
    )


def _format_value(value):
    """Render numeric values for compact Markdown tables."""

    if isinstance(value, float):
        return f"{value:.12f}"
    return str(value)


def _render_shared_frame(rows):
    """Render A3 vs B official comparison rows."""

    lines = [
        "| Metric | A3 | B | Delta B-A3 | Delta pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta_pp = row.get("delta_b_minus_a3_pp")
        lines.append(
            "| {metric} | {a3} | {b} | {delta} | {delta_pp} |".format(
                metric=row["metric"],
                a3=_format_value(row["a3"]),
                b=_format_value(row["b"]),
                delta=_format_value(row["delta_b_minus_a3"]),
                delta_pp="" if delta_pp is None else f"{delta_pp:.6f}",
            )
        )
    return lines


def _render_parity(parity):
    """Render parity diagnostics table."""

    lines = [
        "| Metric | Expected | Actual | Abs Diff | Tolerance | Result |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in parity["diagnostics"]:
        lines.append(
            f"| {item['metric']} | {item['expected']} | {item['actual']} | "
            f"{item['abs_diff']} | {item['tolerance']} | {item['result']} |"
        )
    return lines


def render_report(summary):
    """Render the official Model B timing ablation report."""

    official = summary["official_model_performance"]
    neutral = summary["post_hoc_diagnostic_ablations"]["B_NEUTRAL_TIMING"]
    zero = summary["post_hoc_diagnostic_ablations"].get("B_ZERO_TIMING_AND_FLAG")
    lines = [
        "# Model B Timing Ablation",
        "",
        "## Evaluation Integrity",
        "",
        f"- Shared test N: `{summary['shared_test_n']}`",
        f"- A3 checkpoint: `{summary['a3_checkpoint']}`",
        f"- Model B checkpoint: `{summary['model_b_checkpoint']}`",
        "- Scope: post-hoc evaluation only; no retraining; no checkpoint edits; no dataset edits.",
        "- A3 official aggregates are preserved from the canonical no-timing baseline.",
        "- B_SYNTHETIC_TIMING uses the stored synthetic timing fields exactly as trained.",
        "- B_NEUTRAL_TIMING sets normalized previous/current timing to `0` and keeps `time_is_synthetic=1`.",
        "- B_ZERO_TIMING_AND_FLAG sets normalized timing to `0` and `time_is_synthetic=0` as secondary diagnostic.",
        "",
        "## Reporting Bug Root Cause",
        "",
        summary["reporting_bug_root_cause"],
        "",
        "## Official Model Performance",
        "",
        *_render_shared_frame(official["shared_comparison_frame"]),
        "",
        "## Model B Terminal Parity",
        "",
        f"MODEL_B_TERMINAL_PARITY: `{summary['model_b_terminal_parity']['status']}`",
        "",
        *_render_parity(summary["model_b_terminal_parity"]),
        "",
        "## Post-Hoc Diagnostic Ablations",
        "",
        "B_NEUTRAL_TIMING is not A3. It uses the same B checkpoint and B architecture, but neutralized timing values.",
        "",
        "```json",
        json.dumps({"B_NEUTRAL_TIMING": neutral, "B_ZERO_TIMING_AND_FLAG": zero}, indent=2),
        "```",
        "",
        "## Transition Counts",
        "",
        "```json",
        json.dumps(summary["transitions"], indent=2),
        "```",
        "",
        "## MateDepth Breakdown",
        "",
        "```json",
        json.dumps(summary["mate_depth"], indent=2),
        "```",
        "",
        "## Rating Breakdown",
        "",
        "```json",
        json.dumps(summary["rating"], indent=2),
        "```",
        "",
        "## Timing Quantile Analysis",
        "",
        "```json",
        json.dumps(summary["timing_quantiles"], indent=2),
        "```",
        "",
        "## Target Rank Delta",
        "",
        "```json",
        json.dumps(summary["rank_delta"], indent=2),
        "```",
        "",
        "## McNemar Paired Tests",
        "",
        "```json",
        json.dumps(summary["mcnemar"], indent=2),
        "```",
        "",
        "## Synthetic Vs Real Timing Distribution",
        "",
        "```json",
        json.dumps(summary["timing_distribution_vs_real_games"], indent=2),
        "```",
        "",
        "## Rating Shortcut Analysis",
        "",
        "```json",
        json.dumps(summary["rating_shortcut_analysis"], indent=2),
        "```",
        "",
        "## Error Examples",
        "",
        "```json",
        json.dumps(summary["error_examples"], indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "Model B is slightly worse than A3 on the official shared test metrics. "
        "The neutral timing result should be interpreted only as sensitivity of the same trained B checkpoint. "
        "It must not be described as A3, because architecture and training history differ.",
        "",
    ]
    return "\n".join(lines)

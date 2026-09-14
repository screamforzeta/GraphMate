"""Generate deterministic synthetic timing fields for puzzle PyG graphs.

Purpose:
    Add Model B timing attributes to the existing sharded puzzle PyG dataset
    without rebuilding graph topology or target labels.
Input:
    data/pyg/{train,val,test}/shard_*.pt plus data/pyg/manifest.json.
Output:
    data/pyg_puzzles_timing/ with copied graph shards, timing attributes, and
    a manifest containing generator parameters and audit statistics.
Run:
    python3 -m src.cli.data.generate_puzzle_timing_dataset --overwrite
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from statistics import mean
from statistics import median
import argparse
import ast
import hashlib
import json
import math
import shutil
import time

import chess
import pandas as pd
import torch


SOURCE_ROOT = Path("data/pyg")
OUTPUT_ROOT = Path("data/pyg_puzzles_timing")
BUILD_ROOT = Path("data/pyg_puzzles_timing_building")
PUZZLE_CSVS = {
    "train": Path("data/final/puzzles/train.csv"),
    "val": Path("data/final/puzzles/val.csv"),
    "test": Path("data/final/puzzles/test.csv"),
}
GAMES_CSVS = {
    "train": Path("data/final/games/games_train.csv"),
    "val": Path("data/final/games/games_val.csv"),
    "test": Path("data/final/games/games_test.csv"),
}
MANIFEST_NAME = "manifest.json"
GENERATOR_VERSION = "puzzle_timing_lognormal_v1"


@dataclass(frozen=True)
class TimingGeneratorConfig:
    """Configuration for deterministic synthetic puzzle timings."""

    global_seed: int = 42
    base_median_seconds: float = 12.0
    rating_alpha: float = 0.55
    sigma: float = 0.45
    previous_sigma: float = 0.35
    previous_correlation_weight: float = 0.70
    minimum_seconds: float = 0.5
    maximum_seconds: float = 120.0
    rating_min: float = 600.0
    rating_max: float = 2400.0


@dataclass(frozen=True)
class NormalizationStats:
    """Train-only log1p z-score parameters."""

    mean: float
    std: float


def stable_unit_interval(puzzle_id, global_seed, stream):
    """Return a deterministic pseudo-random value in (0,1) for one puzzle."""

    key = f"{global_seed}|{stream}|{puzzle_id}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False)
    return (integer + 0.5) / (2**64)


def normal_from_unit(unit_value):
    """Map a deterministic unit value to a standard normal quantile."""

    tensor = torch.tensor(float(unit_value), dtype=torch.float64)
    return float(torch.distributions.Normal(0.0, 1.0).icdf(tensor).item())


def difficulty_from_rating(rating, config):
    """Return clipped rating difficulty in [0,1]."""

    rating = float(rating)
    span = config.rating_max - config.rating_min
    if span <= 0:
        raise ValueError("rating_max must be greater than rating_min.")
    return min(1.0, max(0.0, (rating - config.rating_min) / span))


def synthetic_raw_times(puzzle_id, rating, config):
    """Generate deterministic raw previous/current move times in seconds.

    The current decision time follows a rating-conditioned LogNormal:
        T_current ~ LogNormal(log(median_time), sigma)
        median_time = base_median_seconds * (1 + rating_alpha * difficulty)

    The previous-move time is a correlated log-space sample around the same
    timing context, not a copy of current time.
    """

    difficulty = difficulty_from_rating(rating, config)
    median_time = config.base_median_seconds * (1.0 + config.rating_alpha * difficulty)
    current_z = normal_from_unit(
        stable_unit_interval(puzzle_id, config.global_seed, "current")
    )
    previous_z = normal_from_unit(
        stable_unit_interval(puzzle_id, config.global_seed, "previous")
    )
    current_log = math.log(median_time) + config.sigma * current_z
    previous_center = (
        config.previous_correlation_weight * current_log
        + (1.0 - config.previous_correlation_weight) * math.log(median_time)
    )
    previous_log = previous_center + config.previous_sigma * previous_z
    current = math.exp(current_log)
    previous = math.exp(previous_log)
    return (
        min(config.maximum_seconds, max(config.minimum_seconds, previous)),
        min(config.maximum_seconds, max(config.minimum_seconds, current)),
    )


def normalize_seconds(seconds, stats):
    """Apply train-only log1p z-score normalization."""

    return (math.log1p(float(seconds)) - stats.mean) / stats.std


def fit_train_normalization(source_root, config):
    """Fit normalization stats using only raw synthetic times from train."""

    values = []
    for graph in iter_split_graphs(source_root, "train"):
        previous, current = synthetic_raw_times(graph.puzzle_id, graph.rating, config)
        values.extend([math.log1p(previous), math.log1p(current)])
    if not values:
        raise ValueError("Cannot fit timing normalization on an empty train split.")
    train_mean = sum(values) / len(values)
    variance = sum((value - train_mean) ** 2 for value in values) / len(values)
    train_std = math.sqrt(variance)
    if train_std <= 0:
        train_std = 1.0
    return NormalizationStats(mean=train_mean, std=train_std)


def iter_split_graphs(source_root, split):
    """Yield all graphs from a sharded PyG split in manifest order."""

    manifest = read_json(Path(source_root) / MANIFEST_NAME)
    for shard in manifest["splits"][split]["shards"]:
        graphs = torch.load(Path(source_root) / shard["file"], map_location="cpu", weights_only=False)
        for graph in graphs:
            yield graph


def add_timing_to_graph(graph, config, normalization):
    """Attach normalized and raw timing attributes to one Data graph."""

    previous_seconds, current_seconds = synthetic_raw_times(
        graph.puzzle_id,
        graph.rating,
        config,
    )
    graph.previous_move_time = torch.tensor(
        [[normalize_seconds(previous_seconds, normalization)]],
        dtype=torch.float,
    )
    graph.original_move_time = torch.tensor(
        [[normalize_seconds(current_seconds, normalization)]],
        dtype=torch.float,
    )
    graph.time_is_synthetic = torch.tensor([[1.0]], dtype=torch.float)
    graph.previous_move_time_seconds = torch.tensor([[previous_seconds]], dtype=torch.float)
    graph.original_move_time_seconds = torch.tensor([[current_seconds]], dtype=torch.float)
    return graph


def read_json(path):
    """Read a JSON file."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, payload):
    """Write stable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_shard_atomic(graphs, path):
    """Write one shard through a temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(graphs, tmp_path)
    tmp_path.replace(path)


def summarize(values):
    """Return descriptive statistics for a numeric list."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0}

    def percentile(percent):
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * percent
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[int(position)]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    avg = mean(ordered)
    variance = sum((value - avg) ** 2 for value in ordered) / len(ordered)
    return {
        "n": len(ordered),
        "mean": avg,
        "median": median(ordered),
        "std": math.sqrt(variance),
        "min": ordered[0],
        "max": ordered[-1],
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
    }


def pearson(xs, ys):
    """Return Pearson correlation or None when undefined."""

    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def ranks(values):
    """Return average ranks for Spearman correlation."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end + 1) / 2.0
        for sorted_index in range(index, end):
            result[indexed[sorted_index][0]] = rank
        index = end
    return result


def spearman(xs, ys):
    """Return Spearman rank correlation or None when undefined."""

    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def legal_move_count(fen):
    """Return the number of legal moves for a valid FEN."""

    return len(list(chess.Board(str(fen)).legal_moves))


def split_rating_stats():
    """Return rating summaries for available puzzle CSV splits."""

    stats = {}
    for split, path in PUZZLE_CSVS.items():
        frame = pd.read_csv(path, usecols=["Rating"])
        stats[split] = summarize(frame["Rating"].tolist())
    return stats


def parse_move_times(value):
    """Parse serialized MoveTimes lists and keep valid positive seconds."""

    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    result = []
    for item in parsed:
        if item is None:
            continue
        try:
            seconds = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(seconds) and seconds > 0:
            result.append(seconds)
    return result


def real_game_timing_audit():
    """Summarize usable real move times already parsed from games CSVs."""

    audit = {"splits": {}, "available": "NO", "total_games": 0, "usable_move_times": 0}
    all_times = []
    for split, path in GAMES_CSVS.items():
        if not path.exists():
            audit["splits"][split] = {"exists": False}
            continue
        frame = pd.read_csv(path, usecols=["MoveTimes", "AverageElo", "NumMoves"])
        times = []
        usable_games = 0
        for raw in frame["MoveTimes"]:
            parsed = parse_move_times(raw)
            if parsed:
                usable_games += 1
                times.extend(parsed)
        all_times.extend(times)
        audit["splits"][split] = {
            "exists": True,
            "games": len(frame),
            "usable_games": usable_games,
            "usable_move_times": len(times),
            "seconds": summarize(times),
        }
        audit["total_games"] += len(frame)
    audit["usable_move_times"] = len(all_times)
    audit["seconds"] = summarize(all_times)
    if len(all_times) >= 1000:
        audit["available"] = "PARTIAL"
    elif all_times:
        audit["available"] = "PARTIAL"
    return audit


def build_timing_dataset(
    source_root=SOURCE_ROOT,
    output_root=OUTPUT_ROOT,
    build_root=BUILD_ROOT,
    config=TimingGeneratorConfig(),
    overwrite=False,
):
    """Create a sharded puzzle timing dataset with train-only normalization."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    build_root = Path(build_root)
    if output_root.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True)

    source_manifest = read_json(source_root / MANIFEST_NAME)
    normalization = fit_train_normalization(source_root, config)
    split_audits = {}
    split_manifests = {}
    started = time.perf_counter()

    for split, split_info in source_manifest["splits"].items():
        split_dir = build_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        previous_raw = []
        current_raw = []
        previous_norm = []
        current_norm = []
        ratings = []
        mate_depths = []
        legal_counts = []
        edge_counts = []
        shards = []
        for shard_index, shard in enumerate(split_info["shards"]):
            graphs = torch.load(source_root / shard["file"], map_location="cpu", weights_only=False)
            timed_graphs = []
            for graph in graphs:
                timed = add_timing_to_graph(graph, config, normalization)
                previous_raw.append(float(timed.previous_move_time_seconds.item()))
                current_raw.append(float(timed.original_move_time_seconds.item()))
                previous_norm.append(float(timed.previous_move_time.item()))
                current_norm.append(float(timed.original_move_time.item()))
                ratings.append(float(timed.rating.item() if torch.is_tensor(timed.rating) else timed.rating))
                mate_depths.append(float(timed.mate_depth.item() if torch.is_tensor(timed.mate_depth) else timed.mate_depth))
                legal_counts.append(float(legal_move_count(timed.fen)))
                edge_counts.append(float(timed.edge_index.shape[1]))
                timed_graphs.append(timed)
            file_name = f"shard_{shard_index:05d}.pt"
            save_shard_atomic(timed_graphs, split_dir / file_name)
            shards.append({"file": f"{split}/{file_name}", "num_graphs": len(timed_graphs)})

        split_audits[split] = {
            "num_graphs": len(current_raw),
            "synthetic_count": len(current_raw),
            "real_count": 0,
            "previous_move_time_seconds": summarize(previous_raw),
            "original_move_time_seconds": summarize(current_raw),
            "previous_move_time": summarize(previous_norm),
            "original_move_time": summarize(current_norm),
            "correlations": {
                "original_time_vs_rating": {
                    "pearson": pearson(current_raw, ratings),
                    "spearman": spearman(current_raw, ratings),
                },
                "original_time_vs_mate_depth": {
                    "pearson": pearson(current_raw, mate_depths),
                    "spearman": spearman(current_raw, mate_depths),
                },
                "original_time_vs_legal_candidate_count": {
                    "pearson": pearson(current_raw, legal_counts),
                    "spearman": spearman(current_raw, legal_counts),
                },
                "original_time_vs_edge_count": {
                    "pearson": pearson(current_raw, edge_counts),
                    "spearman": spearman(current_raw, edge_counts),
                },
                "previous_vs_original_time": {
                    "pearson": pearson(previous_raw, current_raw),
                    "spearman": spearman(previous_raw, current_raw),
                },
            },
        }
        split_manifests[split] = {
            "num_graphs": len(current_raw),
            "num_shards": len(shards),
            "shards": shards,
            "stats": {
                "synthetic_count": len(current_raw),
                "real_count": 0,
            },
        }

    manifest = {
        "format_version": 1,
        "source_dataset": str(source_root),
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "global_seed": config.global_seed,
        "distribution_type": "rating_conditioned_lognormal",
        "distribution_parameters": asdict(config),
        "timing_fields": timing_field_semantics(),
        "normalization": {
            "method": "log1p_zscore",
            "fit_split": "train",
            "mean": normalization.mean,
            "std": normalization.std,
        },
        "leakage_policy": {
            "uses_rating": True,
            "potential_shortcut": True,
            "uses_target_move": False,
            "uses_solution_continuation": False,
            "uses_mate_depth_for_generation": False,
            "uses_themes": False,
            "uses_test_labels": False,
        },
        "feature_schema": source_manifest.get("feature_schema", {}),
        "splits": split_manifests,
        "audit": {
            "duration_seconds": time.perf_counter() - started,
            "split_statistics": split_audits,
            "rating_statistics": split_rating_stats(),
            "real_game_timing": real_game_timing_audit(),
        },
    }
    write_json(build_root / MANIFEST_NAME, manifest)
    if output_root.exists():
        shutil.rmtree(output_root)
    build_root.replace(output_root)
    return manifest


def timing_field_semantics():
    """Return formal semantics for Model B timing fields."""

    return {
        "previous_move_time": {
            "semantic_meaning": "Synthetic time context for the move immediately before the solver position.",
            "unit": "unitless log1p seconds z-score using train-only mean/std",
            "raw_seconds_field": "previous_move_time_seconds",
            "source": "deterministic rating-conditioned synthetic generator",
            "normalization": "log1p_zscore_fit_on_train",
            "available_at_inference": True,
        },
        "original_move_time": {
            "semantic_meaning": "Synthetic decision time associated with the current puzzle move to solve.",
            "unit": "unitless log1p seconds z-score using train-only mean/std",
            "raw_seconds_field": "original_move_time_seconds",
            "source": "deterministic rating-conditioned synthetic generator",
            "normalization": "log1p_zscore_fit_on_train",
            "available_at_inference": True,
        },
        "time_is_synthetic": {
            "semantic_meaning": "Indicator for synthetic versus real timing. Current puzzle dataset is fully synthetic.",
            "unit": "binary flag, 1.0 synthetic and 0.0 real",
            "source": "dataset construction policy",
            "normalization": "none",
            "available_at_inference": True,
        },
    }


def parse_args():
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic timing fields for puzzle PyG graphs."
    )
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--build-root", type=Path, default=BUILD_ROOT)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Generate the timing dataset and print a compact audit summary."""

    args = parse_args()
    manifest = build_timing_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        build_root=args.build_root,
        config=TimingGeneratorConfig(global_seed=args.global_seed),
        overwrite=args.overwrite,
    )
    print(f"TIMING_GENERATOR_FOUND = NO")
    print(f"TIMING_DATASET = {args.output_root}")
    print(f"REAL_GAME_TIMING_AVAILABLE = {manifest['audit']['real_game_timing']['available']}")
    for split, stats in manifest["audit"]["split_statistics"].items():
        current = stats["original_move_time_seconds"]
        print(
            f"{split}: n={current['n']} mean={current['mean']:.4f} "
            f"median={current['median']:.4f} p95={current['p95']:.4f}"
        )


if __name__ == "__main__":
    main()

"""Diagnose progressive subset shard locality without scientific training.

Purpose:
    Simulate the naive pathological access pattern analytically, then measure
    the fixed shard-aware pattern with a real DataLoader traversal.
Input:
    data/pyg sharded train split and its manifest.
Output:
    JSON and Markdown reports under artifacts/benchmarks/.
Run:
    python3 -m src.benchmark_progressive_subset_loading --device cuda
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import time

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.graph.pyg_dataset import (
    ShardedPyGDataset,
    access_pattern_stats,
    make_shard_aware_sampler,
)
from src.training.progressive_controller import deterministic_indices


REPORT_JSON = Path("artifacts/benchmarks/progressive_subset_loading.json")
REPORT_MD = Path("artifacts/benchmarks/progressive_subset_loading.md")


@dataclass(frozen=True)
class TraversalCase:
    """One subset/full traversal to diagnose."""

    name: str
    dataset: object
    naive_order: list[int]


def parse_args():
    """Parse diagnostic benchmark CLI options."""

    parser = argparse.ArgumentParser(
        description="Diagnose shard locality for progressive training subsets."
    )
    parser.add_argument("--root", default="data/pyg")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-graphs", type=int, default=12000)
    parser.add_argument("--confirmation-graphs", type=int, default=30000)
    parser.add_argument(
        "--max-full-graphs",
        type=int,
        default=None,
        help="Optional local-only cap for the full traversal diagnostic.",
    )
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--non-blocking", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Record AMP runtime intent; no forward/backward is run.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--pattern-only",
        action="store_true",
        help="Skip DataLoader traversal and report structural locality only.",
    )
    return parser.parse_args()


def resolve_device(choice):
    """Resolve cpu/cuda/auto for optional batch transfers."""

    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(choice)


def clear_dataset_cache(dataset):
    """Clear shard cache for a full or subset sharded dataset."""

    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(target, "clear_cache"):
        target.clear_cache(reset_stats=True)


def dataset_cache_stats(dataset):
    """Return cache counters for a full or subset sharded dataset."""

    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(target, "cache_stats"):
        return target.cache_stats()
    return {}


def make_cases(base_dataset, args):
    """Build full, pilot-like, and confirmation-like traversal cases."""

    confirmation_indices = deterministic_indices(
        len(base_dataset),
        min(args.confirmation_graphs, len(base_dataset)),
        args.seed,
    )
    pilot_indices = confirmation_indices[: min(args.pilot_graphs, len(confirmation_indices))]
    return [
        TraversalCase(
            "pilot_subset",
            Subset(base_dataset, pilot_indices),
            list(range(len(pilot_indices))),
        ),
        TraversalCase(
            "confirmation_subset",
            Subset(base_dataset, confirmation_indices),
            list(range(len(confirmation_indices))),
        ),
        TraversalCase(
            "full",
            (
                Subset(base_dataset, range(min(args.max_full_graphs, len(base_dataset))))
                if args.max_full_graphs is not None
                else base_dataset
            ),
            list(
                range(
                    min(args.max_full_graphs, len(base_dataset))
                    if args.max_full_graphs is not None
                    else len(base_dataset)
                )
            ),
        ),
    ]


def sampler_order(dataset, seed, epoch, shuffle):
    """Materialize one deterministic sampler order for diagnostics."""

    sampler = make_shard_aware_sampler(
        dataset,
        shuffle=shuffle,
        seed=seed,
    )
    sampler.set_epoch(epoch)
    return list(iter(sampler))


def analytical_before(dataset, naive_order, cache_size):
    """Simulate the old traversal without loading any graph objects."""

    stats = access_pattern_stats(dataset, naive_order, cache_size)
    return {
        "mode": "ANALYTICAL_PATTERN_SIMULATION",
        "number_of_samples": stats["samples"],
        "unique_shards": stats["unique_shards"],
        "shard_transitions": stats["shard_transitions"],
        "estimated_shard_loads": stats["simulated_shard_loads"],
        "average_consecutive_samples_per_shard": (
            stats["average_consecutive_samples_per_shard"]
        ),
        "simulated_cache_hits": stats["simulated_cache_hits"],
        "simulated_cache_misses": stats["simulated_cache_misses"],
        "simulated_cache_hit_rate": stats["simulated_cache_hit_rate"],
    }


def traverse_after(dataset, sampler, order, args, device):
    """Measure the shard-aware traversal and collect actual cache counters."""

    clear_dataset_cache(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    start = time.perf_counter()
    graphs = 0
    batches = 0
    for batch in loader:
        if device.type == "cuda":
            batch = batch.to(device, non_blocking=args.non_blocking)
            torch.cuda.synchronize(device)
        graphs += int(batch.y.numel())
        batches += 1
    seconds = time.perf_counter() - start
    stats = dataset_cache_stats(dataset)
    pattern = access_pattern_stats(dataset, order, args.cache_size)
    return {
        "mode": "REAL_SERVER_TRAVERSAL",
        "note": "Data-loading traversal only; no forward/backward or AMP compute.",
        "graphs": graphs,
        "batches": batches,
        "elapsed_seconds": seconds,
        "graphs_per_second": graphs / seconds if seconds > 0 else None,
        "batches_per_second": batches / seconds if seconds > 0 else None,
        "shard_transitions": pattern["shard_transitions"],
        "unique_shards": pattern["unique_shards"],
        "average_consecutive_samples_per_shard": (
            pattern["average_consecutive_samples_per_shard"]
        ),
        **stats,
    }


def compare_case(case, args, device):
    """Compare naive and shard-aware access for one traversal case."""

    naive_order = case.naive_order
    after_sampler = make_shard_aware_sampler(
        case.dataset,
        shuffle=True,
        seed=args.seed,
    )
    after_sampler.set_epoch(args.epoch)
    fixed_order = list(iter(after_sampler))
    before = analytical_before(case.dataset, naive_order, args.cache_size)
    if args.pattern_only:
        after_preview = analytical_before(case.dataset, fixed_order, args.cache_size)
        after = {
            **after_preview,
            "mode": "PATTERN_ONLY_PREVIEW",
        }
    else:
        after = traverse_after(case.dataset, after_sampler, fixed_order, args, device)
    return {
        "name": case.name,
        "graphs": len(case.dataset),
        "before": before,
        "after": after,
        "membership_preserved": set(naive_order) == set(fixed_order),
        "order_changed": naive_order != fixed_order,
    }


def write_reports(payload):
    """Write JSON and Markdown diagnostic reports."""

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Progressive Subset Loading Diagnostic",
        "",
        "LOCAL_VM_DIAGNOSTIC_ONLY unless executed on the training server.",
        "",
    ]
    for result in payload["results"]:
        before = result["before"]
        after = result["after"]
        lines.extend(
            [
                f"## {result['name']}",
                "",
                f"- Graphs: `{result['graphs']}`",
                f"- Membership preserved: `{result['membership_preserved']}`",
                f"- BEFORE mode: `{before['mode']}`",
                f"- BEFORE shard transitions: `{before['shard_transitions']}`",
                f"- BEFORE estimated shard loads: `{before['estimated_shard_loads']}`",
                f"- BEFORE simulated cache hit rate: `{before['simulated_cache_hit_rate']}`",
                f"- AFTER mode: `{after['mode']}`",
                f"- AFTER elapsed seconds: `{after.get('elapsed_seconds')}`",
                f"- AFTER graphs/sec: `{after.get('graphs_per_second')}`",
                f"- AFTER shard transitions: `{after['shard_transitions']}`",
                f"- AFTER shard load count: `{after.get('shard_load_count', after.get('estimated_shard_loads'))}`",
                f"- AFTER cache hit rate: `{after.get('cache_hit_rate', after.get('simulated_cache_hit_rate'))}`",
                "",
            ]
        )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Run local or server-side subset loading diagnostics."""

    args = parse_args()
    device = resolve_device(args.device)
    dataset = ShardedPyGDataset(
        root=args.root,
        split=args.split,
        cache_size=args.cache_size,
    )
    results = []
    cases = make_cases(dataset, args)
    labels = {
        "pilot_subset": "PILOT",
        "confirmation_subset": "CONFIRMATION",
        "full": "FULL",
    }
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {labels.get(case.name, case.name.upper())}", flush=True)
        result = compare_case(case, args, device)
        results.append(result)
        before = result["before"]
        after = result["after"]
        print(
            f"{case.name}: before_loads={before['estimated_shard_loads']} "
            f"before_transitions={before['shard_transitions']} "
            f"after_loads={after.get('shard_load_count', after.get('estimated_shard_loads'))} "
            f"after_seconds={after.get('elapsed_seconds')}",
            flush=True,
        )
    payload = {
        "device": str(device),
        "settings": vars(args),
        "note": (
            "This benchmark diagnoses data access order and is not a scientific "
            "training run. BEFORE is analytical only. AFTER is a real DataLoader "
            "traversal unless --pattern-only is set."
        ),
        "results": results,
    }
    write_reports(payload)
    print(json.dumps(payload, indent=2))
    print(f"reports: {REPORT_JSON}, {REPORT_MD}")


if __name__ == "__main__":
    main()

"""Diagnose progressive subset shard locality without scientific training.

Purpose:
    Compare naive subset traversal with deterministic shard-aware traversal for
    pilot-like, confirmation-like, and full train access patterns.
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


def traverse(dataset, order, args, device):
    """Iterate a dataset order and collect actual cache counters."""

    clear_dataset_cache(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=order,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    start = time.perf_counter()
    graphs = 0
    batches = 0
    for batch in loader:
        if device.type == "cuda":
            batch = batch.to(device, non_blocking=args.pin_memory)
            torch.cuda.synchronize(device)
        graphs += int(batch.y.numel())
        batches += 1
    seconds = time.perf_counter() - start
    stats = dataset_cache_stats(dataset)
    return {
        "graphs": graphs,
        "batches": batches,
        "seconds": seconds,
        "graphs_per_second": graphs / seconds if seconds > 0 else None,
        "batches_per_second": batches / seconds if seconds > 0 else None,
        **stats,
    }


def compare_case(case, args, device):
    """Compare naive and shard-aware access for one traversal case."""

    naive_order = case.naive_order
    fixed_order = sampler_order(
        case.dataset,
        seed=args.seed,
        epoch=args.epoch,
        shuffle=case.name != "full",
    )
    old_pattern = access_pattern_stats(case.dataset, naive_order, args.cache_size)
    new_pattern = access_pattern_stats(case.dataset, fixed_order, args.cache_size)
    if args.pattern_only:
        old_access = old_pattern
        new_access = new_pattern
    else:
        old_access = {
            **old_pattern,
            **traverse(case.dataset, naive_order, args, device),
        }
        new_access = {
            **new_pattern,
            **traverse(case.dataset, fixed_order, args, device),
        }
    return {
        "name": case.name,
        "graphs": len(case.dataset),
        "old_access_pattern": old_access,
        "new_access_pattern": new_access,
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
        old = result["old_access_pattern"]
        new = result["new_access_pattern"]
        lines.extend(
            [
                f"## {result['name']}",
                "",
                f"- Graphs: `{result['graphs']}`",
                f"- Membership preserved: `{result['membership_preserved']}`",
                f"- Old transitions: `{old['shard_transitions']}`",
                f"- New transitions: `{new['shard_transitions']}`",
                f"- Old shard loads: `{old.get('shard_load_count', old.get('simulated_shard_loads'))}`",
                f"- New shard loads: `{new.get('shard_load_count', new.get('simulated_shard_loads'))}`",
                f"- Old cache hit rate: `{old.get('cache_hit_rate', old.get('simulated_cache_hit_rate'))}`",
                f"- New cache hit rate: `{new.get('cache_hit_rate', new.get('simulated_cache_hit_rate'))}`",
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
    results = [
        compare_case(case, args, device)
        for case in make_cases(dataset, args)
    ]
    payload = {
        "device": str(device),
        "settings": vars(args),
        "note": (
            "This benchmark diagnoses data access order. It is not a scientific "
            "training run and VM wall-clock numbers are not server validation."
        ),
        "results": results,
    }
    write_reports(payload)
    print(json.dumps(payload, indent=2))
    print(f"reports: {REPORT_JSON}, {REPORT_MD}")


if __name__ == "__main__":
    main()

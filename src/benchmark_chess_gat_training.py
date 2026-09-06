"""Benchmark ChessGATNoTiming training throughput on the sharded train split.

Purpose:
    Measure DataLoader, shard-cache, transfer, forward/backward, optimizer, AMP,
    and batch-size runtime costs without running convergence training.
Input:
    data/pyg sharded train split and artifacts/move_to_idx.json.
Output:
    Terminal summary plus artifacts/benchmarks JSON and Markdown reports.
Run:
    python3 -m src.benchmark_chess_gat_training
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import platform
import time
import traceback

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from src.graph.pyg_dataset import (
    ShardAwareShuffleSampler,
    ShardedPyGDataset,
)
from src.models import count_trainable_parameters
from src.train_chess_gat import (
    load_move_vocab_size,
    resolve_device,
)
from src.training.chess_gat_trainer import (
    build_model,
    make_grad_scaler,
    set_seed,
)


BENCHMARK_DIR = Path("artifacts/benchmarks")
BENCHMARK_JSON = BENCHMARK_DIR / "chess_gat_training_benchmark.json"
BENCHMARK_MD = BENCHMARK_DIR / "chess_gat_training_benchmark.md"


@dataclass(frozen=True)
class BenchmarkConfig:
    """One runtime configuration to benchmark."""

    batch_size: int
    num_workers: int
    amp: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    non_blocking: bool
    sampler: str = "shard_aware"


def parse_int_list(value):
    """Parse a comma-separated integer list."""

    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def parse_bool_auto(value):
    """Parse true/false/auto CLI options."""

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    if normalized == "auto":
        return "auto"
    raise argparse.ArgumentTypeError(
        f"Expected true, false, or auto; got {value!r}."
    )


def parse_args():
    """Parse benchmark command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark ChessGATNoTiming training runtime."
    )
    parser.add_argument("--batch-sizes", default="32,64,128,256")
    parser.add_argument("--num-workers", default="0,2,4")
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--benchmark-batches", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pin-memory", type=parse_bool_auto, default="auto")
    parser.add_argument("--persistent-workers", type=parse_bool_auto, default="auto")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--non-blocking", type=parse_bool_auto, default="auto")
    parser.add_argument(
        "--amp-modes",
        default="false,true",
        help="Comma-separated AMP modes: false,true",
    )
    parser.add_argument(
        "--sampler-modes",
        default="shard_aware",
        help="Comma-separated sampler modes: shard_aware,standard",
    )
    parser.add_argument(
        "--limit-train-graphs",
        type=int,
        default=None,
        help="Optional first-N train subset for smoke benchmarks.",
    )
    return parser.parse_args()


def _parse_amp_modes(value):
    """Parse comma-separated AMP booleans."""

    return [
        bool(parse_bool_auto(item))
        for item in value.split(",")
        if item.strip()
    ]


def resolve_runtime_options(args, device, num_workers):
    """Resolve auto DataLoader and transfer options for one worker count."""

    is_cuda = torch.device(device).type == "cuda"
    pin_memory = is_cuda if args.pin_memory == "auto" else bool(args.pin_memory)
    persistent_workers = (
        num_workers > 0
        if args.persistent_workers == "auto"
        else bool(args.persistent_workers)
    )
    non_blocking = (
        bool(is_cuda and pin_memory)
        if args.non_blocking == "auto"
        else bool(args.non_blocking)
    )
    prefetch_factor = args.prefetch_factor if num_workers > 0 else None

    return pin_memory, persistent_workers, prefetch_factor, non_blocking


def make_benchmark_configs(args, device):
    """Build the runtime matrix from CLI options."""

    configs = []
    sampler_modes = [
        item.strip()
        for item in args.sampler_modes.split(",")
        if item.strip()
    ]
    for batch_size in parse_int_list(args.batch_sizes):
        for num_workers in parse_int_list(args.num_workers):
            pin_memory, persistent_workers, prefetch_factor, non_blocking = (
                resolve_runtime_options(args, device, num_workers)
            )
            for amp in _parse_amp_modes(args.amp_modes):
                for sampler in sampler_modes:
                    configs.append(
                        BenchmarkConfig(
                            batch_size=batch_size,
                            num_workers=num_workers,
                            amp=amp,
                            pin_memory=pin_memory,
                            persistent_workers=persistent_workers,
                            prefetch_factor=prefetch_factor,
                            non_blocking=non_blocking,
                            sampler=sampler,
                        )
                    )
    return configs


def make_loader(dataset, config):
    """Create a train-only DataLoader for one benchmark config."""

    if config.persistent_workers and config.num_workers <= 0:
        raise ValueError("persistent_workers=True requires num_workers > 0.")
    if config.prefetch_factor is not None and config.num_workers <= 0:
        raise ValueError("prefetch_factor requires num_workers > 0.")

    sampler = None
    shuffle = True
    if config.sampler == "shard_aware":
        sampler = ShardAwareShuffleSampler(dataset)
        shuffle = False
    elif config.sampler != "standard":
        raise ValueError(f"Unknown sampler mode: {config.sampler}")

    kwargs = {
        "batch_size": config.batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": (
            config.persistent_workers
            if config.num_workers > 0
            else False
        ),
    }
    if config.num_workers > 0:
        kwargs["prefetch_factor"] = config.prefetch_factor

    return DataLoader(dataset, **kwargs)


def synchronize_if_cuda(device):
    """Synchronize CUDA work so timing is not affected by async execution."""

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _move_batch(batch, device, non_blocking):
    """Move a PyG batch to the selected device."""

    return batch.to(
        device,
        non_blocking=bool(non_blocking and torch.device(device).type == "cuda"),
    )


def run_training_steps(
    model,
    loader,
    criterion,
    optimizer,
    device,
    config,
    num_steps,
    measure=False,
):
    """Run a fixed number of training batches and optionally time them."""

    use_amp = bool(config.amp and torch.device(device).type == "cuda")
    scaler = make_grad_scaler(
        device,
        amp=use_amp,
    )
    iterator = iter(loader)
    data_seconds = 0.0
    compute_seconds = 0.0
    graphs_processed = 0
    batches_processed = 0
    last_loss = None

    for _ in range(num_steps):
        data_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_elapsed = time.perf_counter() - data_start

        synchronize_if_cuda(device)
        compute_start = time.perf_counter()
        batch = _move_batch(
            batch,
            device,
            config.non_blocking,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=torch.device(device).type,
            enabled=use_amp,
        ):
            logits = model(batch)
            loss = criterion(logits, batch.y)

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite benchmark loss encountered.")

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        synchronize_if_cuda(device)
        compute_elapsed = time.perf_counter() - compute_start

        if measure:
            data_seconds += data_elapsed
            compute_seconds += compute_elapsed
            graphs_processed += int(batch.y.numel())
            batches_processed += 1
            last_loss = float(loss.detach().cpu().item())

    return {
        "data_seconds": data_seconds,
        "compute_seconds": compute_seconds,
        "graphs_processed": graphs_processed,
        "batches_processed": batches_processed,
        "last_loss": last_loss,
    }


def get_cache_stats(dataset, num_workers):
    """Return cache stats when they are meaningful in the main process."""

    if num_workers != 0:
        return {
            "cache_stats_note": (
                "Unavailable for num_workers > 0 because each worker owns "
                "its own dataset cache."
            )
        }

    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(target, "cache_stats"):
        return target.cache_stats()
    return {}


def reset_cache_stats(dataset):
    """Reset cache counters when the dataset exposes them."""

    target = dataset.dataset if isinstance(dataset, Subset) else dataset
    if hasattr(target, "reset_cache_stats"):
        target.reset_cache_stats()


def estimate_epoch_times(train_graphs, graphs_per_second):
    """Estimate one epoch, 30 epochs, and 120 epochs from throughput."""

    if graphs_per_second <= 0:
        return {
            "estimated_full_train_epoch_seconds": None,
            "estimated_epoch_minutes": None,
            "estimated_30_epochs_hours": None,
            "estimated_120_epochs_hours": None,
        }

    epoch_seconds = train_graphs / graphs_per_second
    return {
        "estimated_full_train_epoch_seconds": epoch_seconds,
        "estimated_epoch_minutes": epoch_seconds / 60,
        "estimated_30_epochs_hours": epoch_seconds * 30 / 3600,
        "estimated_120_epochs_hours": epoch_seconds * 120 / 3600,
    }


def empty_cuda_cache():
    """Release CUDA cache after an OOM or finished config."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_benchmark(
    dataset,
    train_graph_count,
    num_classes,
    device,
    config,
    warmup_batches,
    benchmark_batches,
    seed,
):
    """Benchmark one runtime configuration and return a result dictionary."""

    if config.amp and torch.device(device).type != "cuda":
        return {
            **asdict(config),
            "status": "AMP_NOT_APPLICABLE",
            "message": "CUDA AMP requested on a non-CUDA device.",
        }

    set_seed(seed)
    reset_cache_stats(dataset)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    try:
        loader = make_loader(dataset, config)
        model = build_model(num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        run_training_steps(
            model,
            loader,
            criterion,
            optimizer,
            device,
            config,
            warmup_batches,
            measure=False,
        )

        reset_cache_stats(dataset)
        elapsed_start = time.perf_counter()
        measured = run_training_steps(
            model,
            loader,
            criterion,
            optimizer,
            device,
            config,
            benchmark_batches,
            measure=True,
        )
        elapsed_seconds = time.perf_counter() - elapsed_start

        graphs = measured["graphs_processed"]
        batches = measured["batches_processed"]
        graphs_per_second = graphs / elapsed_seconds if elapsed_seconds else 0.0
        batches_per_second = batches / elapsed_seconds if elapsed_seconds else 0.0
        mean_seconds_per_batch = elapsed_seconds / batches if batches else 0.0

        result = {
            **asdict(config),
            "status": "OK",
            "graphs_processed": graphs,
            "batches_processed": batches,
            "elapsed_seconds": elapsed_seconds,
            "graphs_per_second": graphs_per_second,
            "batches_per_second": batches_per_second,
            "mean_seconds_per_batch": mean_seconds_per_batch,
            "data_time_seconds": measured["data_seconds"],
            "compute_time_seconds": measured["compute_seconds"],
            "mean_data_seconds_per_batch": (
                measured["data_seconds"] / batches
                if batches
                else 0.0
            ),
            "mean_compute_seconds_per_batch": (
                measured["compute_seconds"] / batches
                if batches
                else 0.0
            ),
            "last_loss": measured["last_loss"],
            **estimate_epoch_times(train_graph_count, graphs_per_second),
            **get_cache_stats(dataset, config.num_workers),
        }

        if torch.device(device).type == "cuda":
            result["peak_gpu_memory_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            )
            result["peak_gpu_memory_reserved_mb"] = (
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            )
        else:
            result["peak_gpu_memory_allocated_mb"] = None
            result["peak_gpu_memory_reserved_mb"] = None

        return result

    except RuntimeError as error:
        message = str(error)
        status = "OOM" if "out of memory" in message.lower() else "ERROR"
        if status == "OOM":
            empty_cuda_cache()
        return {
            **asdict(config),
            "status": status,
            "message": message,
            "traceback": traceback.format_exc() if status == "ERROR" else None,
        }
    finally:
        empty_cuda_cache()


def classify_bottleneck(results):
    """Classify the likely bottleneck from successful benchmark results."""

    ok_results = [
        result for result in results
        if result.get("status") == "OK"
    ]
    if not ok_results:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "No successful benchmark configuration.",
        }

    best = max(
        ok_results,
        key=lambda item: item["graphs_per_second"],
    )
    data = best["mean_data_seconds_per_batch"]
    compute = best["mean_compute_seconds_per_batch"]

    same_batch = [
        result for result in ok_results
        if result["batch_size"] == best["batch_size"]
        and result["amp"] == best["amp"]
        and result["sampler"] == best["sampler"]
    ]
    worker_gain = 0.0
    if same_batch:
        workers_0 = [
            result for result in same_batch
            if result["num_workers"] == 0
        ]
        if workers_0:
            baseline = workers_0[0]["graphs_per_second"]
            worker_gain = (
                best["graphs_per_second"] / baseline - 1.0
                if baseline
                else 0.0
            )

    if data > compute * 0.5 and worker_gain > 0.2:
        verdict = "DATALOADER_BOUND"
        reason = "Data wait is large and workers improve throughput materially."
    elif data > compute * 0.5:
        verdict = "SHARD_IO_BOUND"
        reason = "Data wait is large without clear worker improvement."
    elif compute > data * 3:
        verdict = "GPU_COMPUTE_BOUND" if torch.cuda.is_available() else "MIXED"
        reason = "Compute time dominates measured batch time."
    else:
        verdict = "MIXED"
        reason = "Data and compute costs are both visible."

    return {
        "verdict": verdict,
        "reason": reason,
        "best_config": best,
        "worker_gain_vs_workers0": worker_gain,
    }


def select_recommended_config(results):
    """Choose a conservative runtime recommendation from successful configs."""

    ok_results = [
        result for result in results
        if result.get("status") == "OK"
    ]
    if not ok_results:
        return None

    best = max(
        ok_results,
        key=lambda item: item["graphs_per_second"],
    )
    threshold = best["graphs_per_second"] * 0.95
    candidates = [
        result for result in ok_results
        if result["graphs_per_second"] >= threshold
    ]

    return min(
        candidates,
        key=lambda item: (
            item.get("peak_gpu_memory_reserved_mb") or 0.0,
            item["num_workers"],
            item["batch_size"],
            int(item["amp"]),
        ),
    )


def collect_environment(device, dataset_size, num_classes, model):
    """Collect hardware/runtime context for the report."""

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "train_graphs": dataset_size,
        "num_classes": num_classes,
        "trainable_parameters": count_trainable_parameters(model),
        "gpu_utilization_note": (
            "GPU utilization is not sampled internally. Observe nvidia-smi "
            "externally during the benchmark if CUDA is used."
        ),
    }


def write_reports(payload):
    """Write benchmark JSON and Markdown reports."""

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_JSON, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    lines = [
        "# Chess GAT Training Benchmark",
        "",
        "## Environment",
        "",
    ]
    for key, value in payload["environment"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            "| Batch | Workers | AMP | Sampler | Graph/s | Data ms | Compute ms | VRAM MB | Epoch min | Status |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in payload["results"]:
        data_ms = result.get("mean_data_seconds_per_batch")
        compute_ms = result.get("mean_compute_seconds_per_batch")
        vram = result.get("peak_gpu_memory_reserved_mb")
        epoch_min = result.get("estimated_epoch_minutes")
        lines.append(
            f"| {result['batch_size']} | {result['num_workers']} | "
            f"{'yes' if result['amp'] else 'no'} | {result['sampler']} | "
            f"{result.get('graphs_per_second', 0.0):.2f} | "
            f"{(data_ms * 1000) if data_ms is not None else 0.0:.2f} | "
            f"{(compute_ms * 1000) if compute_ms is not None else 0.0:.2f} | "
            f"{vram if vram is not None else 'n/a'} | "
            f"{epoch_min if epoch_min is not None else 'n/a'} | "
            f"{result['status']} |"
        )

    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- Verdict: `{payload['bottleneck']['verdict']}`",
            f"- Reason: {payload['bottleneck']['reason']}",
            "",
            "## Recommendation",
            "",
            f"```json\n{json.dumps(payload['recommended_runtime_config'], indent=2)}\n```",
            "",
            "## Warnings",
            "",
        ]
    )
    for warning in payload["warnings"]:
        lines.append(f"- {warning}")

    BENCHMARK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_header(environment):
    """Print benchmark context."""

    print("=" * 50)
    print("Chess GAT Training Benchmark")
    print("=" * 50)
    print(f"Device: {environment['device']}")
    print(f"GPU: {environment['gpu_name']}")
    print(f"Train graphs: {environment['train_graphs']:,}")
    print(f"Trainable parameters: {environment['trainable_parameters']:,}")
    print(f"CPU cores: {environment['cpu_count']}")
    print(f"Torch threads: {environment['torch_threads']}")


def print_config_result(index, result):
    """Print one benchmark result."""

    print("\n" + "-" * 50)
    print(f"CONFIG {index}")
    print("-" * 50)
    print(
        f"batch_size={result['batch_size']} "
        f"workers={result['num_workers']} "
        f"amp={result['amp']} "
        f"sampler={result['sampler']}"
    )
    print(f"status: {result['status']}")
    if result["status"] == "OK":
        print(f"graphs/sec: {result['graphs_per_second']:.2f}")
        print(f"data_time/batch: {result['mean_data_seconds_per_batch']:.4f}s")
        print(f"compute_time/batch: {result['mean_compute_seconds_per_batch']:.4f}s")
        print(f"estimated epoch: {result['estimated_epoch_minutes']:.2f} min")
        print(f"peak VRAM reserved: {result['peak_gpu_memory_reserved_mb']}")
    else:
        print(f"message: {result.get('message')}")


def print_summary(results):
    """Print the compact terminal summary table."""

    print("\n" + "=" * 50)
    print("SUMMARY TABLE")
    print("=" * 50)
    print("Batch | Workers | AMP | Sampler | Graph/s | Data ms | Compute ms | VRAM MB | Epoch min | Status")
    for result in results:
        data_ms = result.get("mean_data_seconds_per_batch")
        compute_ms = result.get("mean_compute_seconds_per_batch")
        vram = result.get("peak_gpu_memory_reserved_mb")
        epoch_min = result.get("estimated_epoch_minutes")
        print(
            f"{result['batch_size']} | {result['num_workers']} | "
            f"{'yes' if result['amp'] else 'no'} | {result['sampler']} | "
            f"{result.get('graphs_per_second', 0.0):.2f} | "
            f"{(data_ms * 1000) if data_ms is not None else 0.0:.2f} | "
            f"{(compute_ms * 1000) if compute_ms is not None else 0.0:.2f} | "
            f"{vram if vram is not None else 'n/a'} | "
            f"{epoch_min if epoch_min is not None else 'n/a'} | "
            f"{result['status']}"
        )


def main():
    """Run the benchmark matrix."""

    args = parse_args()
    device = resolve_device(args.device)
    num_classes = load_move_vocab_size()
    dataset = ShardedPyGDataset(split="train")
    full_train_graphs = len(dataset)
    if args.limit_train_graphs is not None:
        dataset = Subset(
            dataset,
            range(min(args.limit_train_graphs, len(dataset))),
        )

    set_seed(args.seed)
    model_for_context = build_model(num_classes)
    environment = collect_environment(
        device,
        full_train_graphs,
        num_classes,
        model_for_context,
    )
    print_header(environment)

    configs = make_benchmark_configs(args, device)
    results = []
    for index, config in enumerate(configs, start=1):
        result = run_single_benchmark(
            dataset,
            full_train_graphs,
            num_classes,
            device,
            config,
            args.warmup_batches,
            args.benchmark_batches,
            args.seed,
        )
        results.append(result)
        print_config_result(index, result)

    bottleneck = classify_bottleneck(results)
    recommendation = select_recommended_config(results)
    warnings = [
        "Benchmark uses train only and does not select scientific hyperparameters.",
        "CUDA, AMP, and num_workers > 0 can change exact numerical reproducibility.",
    ]
    if not torch.cuda.is_available():
        warnings.append(
            "CUDA is unavailable in this environment; CPU smoke results must "
            "not be used to choose the RTX A2000 runtime config."
        )

    payload = {
        "environment": environment,
        "benchmark_settings": {
            "warmup_batches": args.warmup_batches,
            "benchmark_batches": args.benchmark_batches,
            "limit_train_graphs": args.limit_train_graphs,
        },
        "runtime_config_original": {
            "batch_size": 32,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "non_blocking": False,
            "amp": False,
            "sampler": "shard_aware",
        },
        "results": results,
        "bottleneck": bottleneck,
        "recommended_runtime_config": recommendation,
        "warnings": warnings,
    }
    write_reports(payload)
    print_summary(results)
    print("\nBOTTLENECK CLASSIFICATION")
    print(f"{bottleneck['verdict']}: {bottleneck['reason']}")
    print("\nRECOMMENDED_RUNTIME_CONFIG")
    print(json.dumps(recommendation, indent=2))
    print(f"\nreports: {BENCHMARK_JSON}, {BENCHMARK_MD}")


if __name__ == "__main__":
    main()

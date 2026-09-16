"""Benchmark utilities for MODEL A4 post-move reranking."""

from __future__ import annotations

from pathlib import Path
import json
import time

import torch

from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    build_model_a4,
    evaluate_a4,
    make_a4_loaders,
)


def benchmark_cached_a4_validation(config, device, max_batches=10):
    """Measure cached A4 validation throughput without scientific training."""

    device = torch.device(device)
    model = build_model_a4(config).to(device)
    _, val_loader, manifest = make_a4_loaders(config)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    graphs = 0
    candidates = 0
    started = time.perf_counter()
    use_amp = bool(config.amp and device.type == "cuda")
    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            if batch_index >= max_batches:
                break
            batch = {
                **batch,
                "postmove_batch": batch["postmove_batch"].to(device),
                "a3_candidate_features": batch["a3_candidate_features"].to(device),
                "score_features": batch["score_features"].to(device),
                "candidate_ptr": batch["candidate_ptr"].to(device),
                "target_indices": batch["target_indices"].to(device),
            }
            with torch.amp.autocast(device.type, enabled=use_amp):
                scores = model(
                    batch["a3_candidate_features"],
                    batch["postmove_batch"],
                    batch["score_features"],
                )
            candidates += int(scores.numel())
            graphs += int(batch["target_indices"].numel())
    elapsed = max(time.perf_counter() - started, 1e-12)
    return {
        "cached_validation_batches": min(max_batches, len(val_loader)),
        "graphs": graphs,
        "candidates": candidates,
        "graphs_per_second": graphs / elapsed,
        "candidate_graphs_per_second": candidates / elapsed,
        "elapsed_seconds": elapsed,
        "gpu_peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cache_manifest": manifest,
    }


def write_benchmark(path, payload):
    """Write benchmark JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

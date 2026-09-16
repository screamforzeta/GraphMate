"""Validation-only evaluation helpers for MODEL A4.

Purpose:
    Evaluate A4 on cached validation examples without touching the official
    test split unless a separate CLI explicitly requests it.
Input:
    A4 checkpoint and A4 train/validation cache.
Output:
    Validation metrics including A3 recall@5, A4 conditional Top1, and paired
    A3/A4 transitions.
Run:
    Imported by A4 training, benchmark, and future evaluation CLIs.
"""

from __future__ import annotations

from pathlib import Path
import json

import torch

from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    build_model_a4,
    evaluate_a4,
    make_a4_loaders,
)


def load_a4_checkpoint(checkpoint_path, device):
    """Load A4 model checkpoint for evaluation."""

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config_dict = checkpoint.get("config", {})
    config = ModelA4PostMoveConfig(**{
        key: value
        for key, value in config_dict.items()
        if key in ModelA4PostMoveConfig.__dataclass_fields__
    })
    model = build_model_a4(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, checkpoint


def evaluate_a4_validation(checkpoint_path, device="cpu", batch_size=128, amp=False, non_blocking=False):
    """Evaluate A4 on validation cache only.

    Parameters:
        checkpoint_path: A4 checkpoint path.
        device: Torch device.
        batch_size: Evaluation batch size.
        amp: Enable CUDA AMP.
        non_blocking: Enable non-blocking host-to-device copies.
    Returns:
        Dict with validation-only metrics.
    Side effects:
        Reads checkpoint and cache files.
    """

    device = torch.device(device)
    model, config, checkpoint = load_a4_checkpoint(checkpoint_path, device)
    config.batch_size = batch_size
    config.amp = amp
    config.non_blocking = non_blocking
    _, val_loader, manifest = make_a4_loaders(config)
    metrics = evaluate_a4(model, val_loader, device, config)
    return {
        "split": "val",
        "metrics": metrics,
        "cache_manifest": manifest,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "TEST_SET_EVALUATED": False,
    }


def write_a4_validation_report(path, summary):
    """Write a compact Markdown validation report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary["metrics"]
    text = "\n".join(
        [
            "# Model A4 Post-Move Validation Evaluation",
            "",
            "TEST_SET_EVALUATED = NO",
            "",
            "## Validation Metrics",
            "",
            f"- A3_VAL_TOP5_RECALL: `{metrics['a3_recall_at_5']}`",
            f"- A4_VAL_CONDITIONAL_TOP1: `{metrics['a4_conditional_top1']}`",
            f"- A4_VAL_END_TO_END_TOP1: `{metrics['a4_end_to_end_top1']}`",
            f"- Validation loss: `{metrics['loss']}`",
            "",
            "## Paired Transitions",
            "",
            "```json",
            json.dumps(metrics["paired_transitions"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")

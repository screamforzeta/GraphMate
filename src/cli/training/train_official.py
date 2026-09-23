"""Unified entry point for official GraphMate training configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.graph.pyg_dataset import ShardedPyGDataset
from src.graph.pyg_dataset import load_pyg_dataset
from src.training.official_config import OfficialConfigError
from src.training.official_config import config_kwargs
from src.training.official_config import default_output_dir
from src.training.official_config import ensure_safe_output_dir
from src.training.official_config import load_official_config
from src.training.official_config import missing_required_paths
from src.training.official_config import print_dry_run
from src.training.official_config import resolved_payload
from src.training.official_config import validate_dependency_hashes
from src.training.official_config import write_json


def parse_args():
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Train or inspect GraphMate official model configurations."
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--model", choices=("a", "a2", "a3", "b", "a4"))
    selector.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_timing_dataset(root, split):
    """Load Model B timing split using existing semantics."""

    path = Path(root) / f"{split}_graphs.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    return ShardedPyGDataset(root=root, split=split)


def _require_data(config):
    """Fail with a useful message when required files are absent."""

    missing = missing_required_paths(config)
    if missing:
        lines = ["Missing required files for official training:"]
        lines.extend(f"- {path}" for path in missing)
        lines.append("Run the documented data preparation step before training, or inspect with --dry-run.")
        raise FileNotFoundError("\n".join(lines))


def _train_model_a(config, output_dir, device):
    """Invoke existing Model A convergence training."""

    from src.cli.training.train_model_a import load_graphs
    from src.cli.training.train_model_a import load_move_vocab_size
    from src.training.model_a.convergence_run import ConvergenceRunConfig
    from src.training.model_a.convergence_run import run_convergence_training

    kwargs = config_kwargs(config)
    kwargs["max_runtime_hours"] = None
    kwargs["output_root"] = str(output_dir)
    train = load_graphs("train")
    val = load_graphs("val")
    test = load_graphs("test")
    num_classes = load_move_vocab_size()
    return run_convergence_training(
        train,
        val,
        test,
        num_classes,
        ConvergenceRunConfig(**kwargs),
        device,
        resume=False,
        extend_max_epochs=None,
    )


def _train_model_a2(config, output_dir, device):
    """Invoke existing Model A2 trainer."""

    from src.training.model_a.model_a2_legal_mask import ModelA2LegalMaskConfig
    from src.training.model_a.model_a2_legal_mask import train_model_a2

    kwargs = config_kwargs(config)
    kwargs["output_root"] = str(output_dir)
    return train_model_a2(
        load_pyg_dataset("train"),
        load_pyg_dataset("val"),
        load_pyg_dataset("test"),
        ModelA2LegalMaskConfig(**kwargs),
        device,
        resume=False,
    )


def _train_model_a3(config, output_dir, device):
    """Invoke existing Model A3 trainer."""

    from src.training.model_a.model_a3_legal_scorer import ModelA3LegalScorerConfig
    from src.training.model_a.model_a3_legal_scorer import train_model_a3

    kwargs = config_kwargs(config)
    kwargs["scorer_hidden_dim"] = config["architecture"]["scorer_hidden_dim"]
    kwargs["promotion_embedding_dim"] = config["architecture"]["promotion_embedding_dim"]
    kwargs["output_root"] = str(output_dir)
    return train_model_a3(
        load_pyg_dataset("train"),
        load_pyg_dataset("val"),
        load_pyg_dataset("test"),
        ModelA3LegalScorerConfig(**kwargs),
        device,
        resume=False,
    )


def _train_model_b(config, output_dir, device):
    """Invoke existing Model B trainer."""

    from src.training.model_b.model_b_timing_legal_scorer import (
        ModelBTimingLegalScorerConfig,
    )
    from src.training.model_b.model_b_timing_legal_scorer import train_model_b

    kwargs = config_kwargs(config)
    kwargs["scorer_hidden_dim"] = config["architecture"]["scorer_hidden_dim"]
    kwargs["promotion_embedding_dim"] = config["architecture"]["promotion_embedding_dim"]
    kwargs["timing_hidden_dim"] = config["architecture"]["timing_hidden_dim"]
    kwargs["output_root"] = str(output_dir)
    root = config["data"]["pyg_root"]
    return train_model_b(
        _load_timing_dataset(root, "train"),
        _load_timing_dataset(root, "val"),
        _load_timing_dataset(root, "test"),
        ModelBTimingLegalScorerConfig(**kwargs),
        device,
        resume=False,
    )


def _train_model_a4(config, output_dir, device):
    """Invoke existing Model A4 trainer."""

    from src.training.model_a.model_a4_postmove_reranker import ModelA4PostMoveConfig
    from src.training.model_a.model_a4_postmove_reranker import train_model_a4

    kwargs = config_kwargs(config)
    kwargs["scorer_hidden_dim"] = config["architecture"]["scorer_hidden_dim"]
    kwargs["top_k"] = config["retrieval"]["top_k"]
    kwargs["cache_root"] = config["retrieval"]["cache_root"]
    kwargs["a3_checkpoint"] = config["retrieval"]["a3_checkpoint"]
    kwargs["output_root"] = str(output_dir)
    return train_model_a4(ModelA4PostMoveConfig(**kwargs), device, smoke=False)


TRAINERS = {
    "a": _train_model_a,
    "a2": _train_model_a2,
    "a3": _train_model_a3,
    "b": _train_model_b,
    "a4": _train_model_a4,
}


def main():
    """Run the official training wrapper."""

    args = parse_args()
    config, config_path = load_official_config(model=args.model, config_path=args.config)
    model_id = config["model_id"]
    output_dir = ensure_safe_output_dir(args.output_dir or default_output_dir(model_id))

    if args.dry_run:
        print_dry_run(config, config_path, output_dir)
        return

    _require_data(config)
    validate_dependency_hashes(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = resolved_payload(config, config_path, output_dir, official_config=True)
    write_json(output_dir / "resolved_config.json", payload)
    write_json(output_dir / "run_metadata.json", payload)
    print(f"OFFICIAL_CONFIG_OUTPUT_DIR={output_dir}")
    TRAINERS[model_id](config, output_dir, torch.device(args.device))


if __name__ == "__main__":
    try:
        main()
    except OfficialConfigError as exc:
        raise SystemExit(str(exc)) from exc

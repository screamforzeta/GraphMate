"""Incremental result writing and safe resume checks."""

from __future__ import annotations

import json
from pathlib import Path


def config_identity(config):
    """Return fields that must match for a safe resume."""

    keys = [
        "run_id",
        "model_id",
        "ollama_model",
        "dataset",
        "protocol",
        "dataset_fingerprint",
        "prompt_hash",
        "parser_version",
        "generation_options",
        "thinking_enabled",
    ]
    return {key: config.get(key) for key in keys}


def assert_resume_config_matches(existing_config, new_config):
    """Raise when resume would mix incompatible benchmark configurations."""

    if config_identity(existing_config) != config_identity(new_config):
        raise ValueError("Resume config mismatch; refusing unsafe resume.")
    return True


def append_jsonl(path, record):
    """Append one JSON record to a JSONL file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_completed_keys(path):
    """Load stable completed example keys from a JSONL result file."""

    path = Path(path)
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add((row.get("run_id"), row.get("model_id"), row.get("puzzle_id"), row.get("protocol")))
    return completed

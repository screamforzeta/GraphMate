"""Ollama model discovery and canonical registry mapping."""

from __future__ import annotations


EXPECTED_MODELS = {
    "qwen_3_5_4b": ["qwen", "4b"],
    "qwen_3_5_9b": ["qwen", "9b"],
    "gpt_oss_20b": ["gpt", "oss", "20b"],
}


def normalize_model_name(name):
    """Normalize one Ollama model name for fuzzy local discovery."""

    return str(name).lower().replace("-", "_").replace(":", "_").replace(".", "_")


def discover_registry(ollama_models):
    """Build a registry only from actually discovered Ollama model tags."""

    registry = {}
    for model in ollama_models:
        name = model.get("name") or model.get("model") or ""
        normalized = normalize_model_name(name)
        for registry_id, required_parts in EXPECTED_MODELS.items():
            if registry_id in registry:
                continue
            if all(part in normalized for part in required_parts):
                registry[registry_id] = {
                    "ollama_model": name,
                    "size": model.get("size"),
                    "digest": model.get("digest"),
                    "details": model.get("details"),
                }
    return registry


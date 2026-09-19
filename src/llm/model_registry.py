"""Frozen Ollama model registry and runtime verification helpers."""

from __future__ import annotations


LLM_BENCHMARK_MODELS = {
    "qwen_3_5_4b": {
        "benchmark_id": "qwen_3_5_4b",
        "ollama_model": "qwen3.5:4b",
        "digest": "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd",
        "parameter_size": "4.7B",
        "quantization": "Q4_K_M",
        "size": 3389983735,
        "context_length": 262144,
        "embedding_length": 2560,
        "capabilities": ["vision", "completion", "tools", "thinking"],
        "generation_config": {
            "temperature": 0,
            "num_predict": 16,
            "thinking_mode": "disabled",
            "think": False,
            "status": "RUNTIME_VALIDATED",
        },
    },
    "qwen_3_5_9b": {
        "benchmark_id": "qwen_3_5_9b",
        "ollama_model": "qwen3.5:9b",
        "digest": "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7",
        "parameter_size": "9.7B",
        "quantization": "Q4_K_M",
        "size": 6594474711,
        "context_length": 262144,
        "embedding_length": 4096,
        "capabilities": ["vision", "completion", "tools", "thinking"],
        "generation_config": {
            "temperature": 0,
            "num_predict": 16,
            "thinking_mode": "disabled",
            "think": False,
            "status": "RUNTIME_VALIDATED",
        },
    },
    "gpt_oss_20b": {
        "benchmark_id": "gpt_oss_20b",
        "ollama_model": "gpt-oss:20b",
        "digest": "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7",
        "parameter_size": "20.9B",
        "quantization": "MXFP4",
        "size": 13793441244,
        "context_length": 131072,
        "embedding_length": 2880,
        "capabilities": ["completion", "tools", "thinking"],
        "generation_config": {
            "temperature": 0,
            "num_predict": None,
            "thinking_mode": "low",
            "think": "low",
            "status": "PENDING_RUNTIME_CALIBRATION",
        },
    },
}

EXPECTED_MODELS = LLM_BENCHMARK_MODELS


def normalize_model_name(name):
    """Normalize one Ollama model name for fuzzy local discovery."""

    return str(name).lower().replace("-", "_").replace(":", "_").replace(".", "_")


def discover_registry(ollama_models):
    """Return frozen registry entries whose exact Ollama tags are present."""

    registry = {}
    for model in ollama_models:
        name = model.get("name") or model.get("model") or ""
        for registry_id, expected in LLM_BENCHMARK_MODELS.items():
            if name == expected["ollama_model"]:
                entry = expected.copy()
                entry["runtime_size"] = model.get("size")
                entry["runtime_digest"] = model.get("digest")
                entry["runtime_details"] = model.get("details")
                entry["digest_match"] = model.get("digest") == expected["digest"]
                registry[registry_id] = entry
    return registry


def verify_runtime_registry(ollama_models):
    """Compare /api/tags output against the frozen benchmark registry."""

    discovered = discover_registry(ollama_models)
    return {
        model_id: {
            "found": model_id in discovered,
            "digest_match": bool(discovered.get(model_id, {}).get("digest_match")),
            "expected_tag": expected["ollama_model"],
            "expected_digest": expected["digest"],
            "actual_digest": discovered.get(model_id, {}).get("runtime_digest"),
        }
        for model_id, expected in LLM_BENCHMARK_MODELS.items()
    }


def generation_config_for_model(model_id, num_predict_override=None):
    """Return the frozen or candidate generation config for one model."""

    config = LLM_BENCHMARK_MODELS[model_id]["generation_config"].copy()
    if num_predict_override is not None:
        config["num_predict"] = int(num_predict_override)
    options = {"temperature": config["temperature"]}
    if config.get("num_predict") is not None:
        options["num_predict"] = int(config["num_predict"])
    return {
        "options": options,
        "think": config["think"],
        "thinking_mode": config["thinking_mode"],
        "status": config["status"],
    }

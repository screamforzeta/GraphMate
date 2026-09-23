"""Resolve and validate official GraphMate training configurations."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC
from datetime import datetime
from pathlib import Path
import hashlib
import json
import subprocess
import sys


CONFIG_DIR = Path("configs")
OFFICIAL_MODELS = ("a", "a2", "a3", "b", "a4")
MODEL_CONFIGS = {
    "a": CONFIG_DIR / "model_a.json",
    "a2": CONFIG_DIR / "model_a2.json",
    "a3": CONFIG_DIR / "model_a3.json",
    "b": CONFIG_DIR / "model_b.json",
    "a4": CONFIG_DIR / "model_a4.json",
}
CANONICAL_CHECKPOINT_DIRS = {
    Path("checkpoints/model_a"),
    Path("checkpoints/model_a2"),
    Path("checkpoints/model_a3"),
    Path("checkpoints/model_b"),
    Path("checkpoints/model_a4"),
}
CHECKPOINT_HASHES = {
    "a": "1a72d0b6f675b50c829d76c63343e7016a569c00f993f063b0d0b75061de9b70",
    "a2": "20a76ae5648ded7f8fe3eb835eddd5ec0a87c9043989c37a65b4e8b9dc6901c1",
    "a3": "4efec653a451da7585f3663847c8dc5caaa8ebffadea677617e2f496e4253b80",
    "b": "b8bbad2420ce301582ad08377b1c80f56609513a8dae201600557e255b87d26f",
    "a4": "0cb73acf70487efa5c93f715a5a60c0aa09d64792d894301efaaaf47b2801e99",
}


class OfficialConfigError(ValueError):
    """Raised when an official training config is invalid."""


def read_json(path):
    """Read JSON from path."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    """Write stable JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path):
    """Return SHA256 for a file."""

    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_path(model=None, config_path=None):
    """Resolve an official model name or explicit config path."""

    if bool(model) == bool(config_path):
        raise OfficialConfigError("Provide exactly one of --model or --config.")
    if model:
        key = str(model).lower()
        if key == "a1":
            raise OfficialConfigError("A1 has no training config; it is an inference-only mode over Model A.")
        try:
            return MODEL_CONFIGS[key]
        except KeyError as exc:
            raise OfficialConfigError(
                f"Unknown official model '{model}'. Expected one of: {', '.join(OFFICIAL_MODELS)}."
            ) from exc
    return Path(config_path)


def load_official_config(model=None, config_path=None):
    """Load and validate an official config."""

    path = resolve_config_path(model=model, config_path=config_path)
    if not path.exists():
        raise OfficialConfigError(f"Config file not found: {path}")
    config = read_json(path)
    validate_config(config, path)
    return config, path


def validate_config(config, path=None):
    """Validate the small official training-config schema."""

    required = {
        "schema_version",
        "model_id",
        "run_id",
        "role",
        "official_checkpoint",
        "timing",
        "data",
        "architecture",
        "training",
        "scheduler",
        "provenance",
    }
    missing = sorted(required - set(config))
    if missing:
        raise OfficialConfigError(f"Config missing required keys: {missing}")
    model_id = str(config["model_id"]).lower()
    if model_id == "a1":
        raise OfficialConfigError("A1 must not have an official training config.")
    if model_id not in OFFICIAL_MODELS:
        raise OfficialConfigError(f"Unsupported model_id: {model_id}")
    checkpoint = config["official_checkpoint"]
    if checkpoint.get("sha256") != CHECKPOINT_HASHES[model_id]:
        raise OfficialConfigError(f"Canonical checkpoint hash mismatch for model {model_id}.")
    training = config["training"]
    for key in ("seed", "learning_rate", "weight_decay", "batch_size", "max_epochs"):
        if key not in training:
            raise OfficialConfigError(f"Config training section missing {key}.")
    scheduler = config["scheduler"]
    if scheduler.get("type") != "ReduceLROnPlateau":
        raise OfficialConfigError("Official configs require ReduceLROnPlateau scheduler.")
    if model_id == "a4":
        deps = config.get("dependencies", [])
        if not any(dep.get("model_id") == "a3" for dep in deps):
            raise OfficialConfigError("A4 config must declare its frozen A3 dependency.")
    if path and Path(path).name == "model_a1.json":
        raise OfficialConfigError("A1 must not have an official training config.")
    return True


def default_output_dir(model_id, now=None):
    """Return ignored safe retraining output directory."""

    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts") / "retraining" / str(model_id).lower() / stamp


def ensure_safe_output_dir(output_dir):
    """Reject canonical checkpoint directories as training output."""

    output = Path(output_dir)
    resolved = output.resolve()
    for canonical in CANONICAL_CHECKPOINT_DIRS:
        canonical_resolved = canonical.resolve()
        if resolved == canonical_resolved or canonical_resolved in resolved.parents:
            raise OfficialConfigError(
                f"Refusing to write retraining output under canonical checkpoint directory: {canonical}"
            )
    return output


def required_paths(config):
    """Return required local paths for a config."""

    paths = list(config.get("data", {}).get("required_paths", []))
    for dep in config.get("dependencies", []):
        if dep.get("checkpoint_path"):
            paths.append(dep["checkpoint_path"])
    return [Path(path) for path in dict.fromkeys(paths)]


def missing_required_paths(config):
    """Return required paths that are absent."""

    return [path for path in required_paths(config) if not path.exists()]


def validate_dependency_hashes(config):
    """Validate available dependency checkpoint hashes."""

    results = []
    for dep in config.get("dependencies", []):
        path = dep.get("checkpoint_path")
        expected = dep.get("checkpoint_sha256")
        if not path or not expected or not Path(path).exists():
            continue
        actual = sha256_file(path)
        ok = actual == expected
        results.append({"path": path, "expected_sha256": expected, "actual_sha256": actual, "ok": ok})
        if not ok:
            raise OfficialConfigError(f"Dependency checkpoint hash mismatch: {path}")
    return results


def _git_commit():
    """Return git commit if available without requiring network."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _package_versions():
    """Return relevant local package versions where importable."""

    versions = {"python": sys.version.split()[0]}
    for name, module_name in (
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
    ):
        try:
            module = __import__(module_name)
        except Exception:
            versions[name] = None
        else:
            versions[name] = getattr(module, "__version__", None)
    return versions


def resolved_payload(config, config_path, output_dir, official_config=True):
    """Build a serializable resolved-run payload."""

    payload = {
        "model_id": config["model_id"],
        "run_id": config["run_id"],
        "config_path": str(config_path),
        "official_config": bool(official_config),
        "output_dir": str(output_dir),
        "resolved_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "required_paths": [str(path) for path in required_paths(config)],
        "config": config,
    }
    deps = validate_dependency_hashes(config)
    if deps:
        payload["dependency_checkpoint_hashes"] = deps
    return payload


def print_dry_run(config, config_path, output_dir):
    """Print config inspection output and missing-path diagnostics."""

    payload = resolved_payload(config, config_path, output_dir)
    missing = [str(path) for path in missing_required_paths(config)]
    payload["missing_required_paths"] = missing
    payload["will_train"] = False
    print(json.dumps(payload, indent=2, sort_keys=True))
    if missing:
        print("\nMissing required files:")
        for path in missing:
            print(f"- {path}")
        print("Prepare the required datasets/resources before launching training.")
    return payload


def config_kwargs(config):
    """Return common dataclass keyword arguments from config JSON."""

    training = config["training"]
    scheduler = config["scheduler"]
    architecture = config["architecture"]
    kwargs = {
        "seed": training["seed"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "dropout": architecture["dropout"],
        "batch_size": training["batch_size"],
        "max_epochs": training["max_epochs"],
        "early_stopping_patience": training["early_stopping_patience"],
        "min_delta": training["min_delta"],
        "lr_scheduler_factor": scheduler["factor"],
        "lr_scheduler_patience": scheduler["patience"],
        "min_learning_rate": scheduler["min_lr"],
        "num_workers": training["num_workers"],
        "pin_memory": training["pin_memory"],
        "non_blocking": training["non_blocking"],
        "amp": training["amp"],
    }
    if "persistent_workers" in training:
        kwargs["persistent_workers"] = training["persistent_workers"]
    if "prefetch_factor" in training:
        kwargs["prefetch_factor"] = training["prefetch_factor"]
    return kwargs


def dataclass_dict(instance):
    """Return a dataclass instance as a plain dict."""

    return asdict(instance)

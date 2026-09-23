import json
from pathlib import Path

import pytest

from src.cli.training import train_official
from src.training.official_config import CHECKPOINT_HASHES
from src.training.official_config import MODEL_CONFIGS
from src.training.official_config import OfficialConfigError
from src.training.official_config import ensure_safe_output_dir
from src.training.official_config import load_official_config
from src.training.official_config import missing_required_paths
from src.training.official_config import print_dry_run
from src.training.official_config import validate_config


def test_every_official_config_parses_and_hashes_match():
    for model_id, path in MODEL_CONFIGS.items():
        config, resolved = load_official_config(config_path=path)
        assert resolved == path
        assert config["model_id"] == model_id
        assert config["official_checkpoint"]["sha256"] == CHECKPOINT_HASHES[model_id]


def test_a1_has_no_training_config():
    assert not Path("configs/model_a1.json").exists()
    with pytest.raises(OfficialConfigError, match="A1 has no training config"):
        load_official_config(model="a1")


def test_model_ids_resolve_to_expected_configs():
    for model_id, path in MODEL_CONFIGS.items():
        config, resolved = load_official_config(model=model_id)
        assert resolved == path
        assert config["model_id"] == model_id


def test_a4_dependency_resolves_to_canonical_a3():
    config, _ = load_official_config(model="a4")
    dependency = config["dependencies"][0]
    assert dependency["model_id"] == "a3"
    assert dependency["checkpoint_path"] == "checkpoints/model_a3/best.pt"
    assert dependency["checkpoint_sha256"] == CHECKPOINT_HASHES["a3"]
    assert config["retrieval"]["top_k"] == 5


def test_canonical_checkpoint_dirs_cannot_be_outputs():
    with pytest.raises(OfficialConfigError, match="canonical checkpoint directory"):
        ensure_safe_output_dir("checkpoints/model_a3")
    with pytest.raises(OfficialConfigError, match="canonical checkpoint directory"):
        ensure_safe_output_dir("checkpoints/model_a3/retrain")


def test_schema_validation_catches_malformed_config():
    config, _ = load_official_config(model="a3")
    malformed = dict(config)
    malformed.pop("training")
    with pytest.raises(OfficialConfigError, match="missing required"):
        validate_config(malformed)


def test_missing_data_reports_useful_paths(tmp_path):
    config, _ = load_official_config(model="a3")
    isolated = json.loads(json.dumps(config))
    isolated["data"]["required_paths"] = [str(tmp_path / "missing_manifest.json")]
    missing = missing_required_paths(isolated)
    assert missing == [tmp_path / "missing_manifest.json"]


def test_dry_run_does_not_train(monkeypatch, tmp_path, capsys):
    config, path = load_official_config(model="a3")
    monkeypatch.setattr(
        "src.training.official_config._git_commit",
        lambda: "test-commit",
    )
    monkeypatch.setattr(
        "src.training.official_config._package_versions",
        lambda: {"python": "test"},
    )
    payload = print_dry_run(config, path, tmp_path / "out")
    captured = capsys.readouterr()
    assert payload["will_train"] is False
    assert '"will_train": false' in captured.out
    assert not (tmp_path / "out").exists()


def test_train_official_main_dry_run_skips_trainer(monkeypatch, tmp_path):
    called = []

    def fail_if_called(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("trainer should not be called during dry-run")

    monkeypatch.setattr(train_official, "TRAINERS", {"a3": fail_if_called})
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_official",
            "--model",
            "a3",
            "--dry-run",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    train_official.main()
    assert called == []

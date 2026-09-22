from pathlib import Path

import pytest

from src.cli.data.reset_data import (
    confirm_reset,
    get_reset_targets,
    reset_project_data,
    resolve_reset_path,
)


def test_reset_removes_existing_file_and_directory(tmp_path):
    file_path = tmp_path / "artifacts" / "generated_marker.json"
    directory_path = tmp_path / "data" / "raw"

    file_path.parent.mkdir()
    file_path.write_text("{}", encoding="utf-8")
    directory_path.mkdir(parents=True)
    (directory_path / "sample.pgn").write_text(
        "game",
        encoding="utf-8"
    )

    results = reset_project_data(
        paths=[
            Path("artifacts/generated_marker.json"),
            Path("data/raw"),
        ],
        project_root=tmp_path,
        skip_confirmation=True,
    )

    assert not file_path.exists()
    assert not directory_path.exists()
    assert results[file_path.resolve()] == "removed"
    assert results[directory_path.resolve()] == "removed"


def test_default_reset_paths_do_not_delete_canonical_resources():
    from src.cli.data.reset_data import RESET_PATHS

    assert all(
        not str(path).startswith("resources/move_encoder")
        for path in RESET_PATHS
    )


def test_reset_ignores_missing_path(tmp_path):
    missing_path = tmp_path / "data" / "pyg"

    results = reset_project_data(
        paths=[
            Path("data/pyg"),
        ],
        project_root=tmp_path,
        skip_confirmation=True,
    )

    assert results[missing_path.resolve()] == "not found"


def test_reset_rejects_path_outside_project_root(tmp_path):
    with pytest.raises(ValueError):
        resolve_reset_path(
            Path("../outside"),
            tmp_path
        )


def test_get_reset_targets_rejects_mixed_unsafe_paths(tmp_path):
    with pytest.raises(ValueError):
        get_reset_targets(
            [
                Path("data/raw"),
                Path("../outside"),
            ],
            tmp_path
        )


def test_yes_option_equivalent_does_not_request_input(tmp_path):
    marker_path = tmp_path / "data" / "raw"
    marker_path.mkdir(parents=True)

    def fail_input(prompt):
        raise AssertionError("input should not be called")

    reset_project_data(
        paths=[
            Path("data/raw"),
        ],
        project_root=tmp_path,
        skip_confirmation=True,
        input_func=fail_input,
    )

    assert not marker_path.exists()


def test_negative_confirmation_does_not_remove_anything(tmp_path):
    marker_path = tmp_path / "data" / "raw"
    marker_path.mkdir(parents=True)

    results = reset_project_data(
        paths=[
            Path("data/raw"),
        ],
        project_root=tmp_path,
        skip_confirmation=False,
        input_func=lambda prompt: "no",
    )

    assert results == {}
    assert marker_path.exists()


@pytest.mark.parametrize(
    "response",
    [
        "y",
        "yes",
        "Y",
        " YES ",
    ],
)
def test_confirm_reset_accepts_only_yes_values(response):
    assert confirm_reset(
        input_func=lambda prompt: response
    )

"""Small schema helpers for final experiment analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


NA = "N/A"


@dataclass(frozen=True)
class SourceRecord:
    model_id: str
    benchmark: str
    protocol: str
    run_dir: str
    source_files: list[str]
    official: bool
    completion_state: str
    validity_state: str
    scientific_run_id: str | None = None
    execution_attempt: int | None = None
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    benchmark_version: str | None = None
    freeze_fingerprint: str | None = None
    prompt_version: str | None = None
    parser_version: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    notes: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "benchmark": self.benchmark,
            "protocol": self.protocol,
            "scientific_run_id": self.scientific_run_id,
            "execution_attempt": self.execution_attempt,
            "run_dir": self.run_dir,
            "source_files": self.source_files,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "benchmark_version": self.benchmark_version,
            "freeze_fingerprint": self.freeze_fingerprint,
            "prompt_version": self.prompt_version,
            "parser_version": self.parser_version,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "official": self.official,
            "completion_state": self.completion_state,
            "validity_state": self.validity_state,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class DiscoveredArtifacts:
    root: Path
    included: list[SourceRecord]
    excluded: list[SourceRecord]
    paths: dict[str, Path]


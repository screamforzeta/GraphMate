import json
from types import SimpleNamespace

import torch
from torch.utils.data import Subset
from torch_geometric.data import Data

from src.benchmarks.progressive_subset_loading import (
    TraversalCase,
    analytical_before,
    compare_case,
    make_cases,
    write_reports,
)
from src.graph.pyg_dataset import (
    MANIFEST_NAME,
    ShardedPyGDataset,
    build_manifest,
    make_shard_aware_sampler,
)


def _graph(index):
    return Data(
        x=torch.zeros((64, 15), dtype=torch.float),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 5), dtype=torch.float),
        global_features=torch.zeros((1, 4), dtype=torch.float),
        y=torch.tensor(index % 2, dtype=torch.long),
        graph_marker=torch.tensor(index, dtype=torch.long),
    )


def _build_dataset(tmp_path, shards=4, per_shard=3):
    root = tmp_path / "pyg"
    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    manifest_shards = []
    for shard_id in range(shards):
        file_name = f"train/shard_{shard_id:05d}.pt"
        graphs = [
            _graph(shard_id * per_shard + local_index)
            for local_index in range(per_shard)
        ]
        torch.save(graphs, root / file_name)
        manifest_shards.append(
            {
                "file": file_name,
                "num_graphs": per_shard,
                "start_index": shard_id * per_shard,
                "end_index": (shard_id + 1) * per_shard,
            }
        )
    manifest = build_manifest(
        graphs_per_shard=per_shard,
        splits={
            "train": {
                "num_graphs": shards * per_shard,
                "num_shards": shards,
                "shards": manifest_shards,
            }
        },
    )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return ShardedPyGDataset(root=root, split="train", cache_size=1)


def _args(**overrides):
    values = {
        "batch_size": 2,
        "cache_size": 1,
        "seed": 7,
        "pilot_graphs": 4,
        "confirmation_graphs": 8,
        "max_full_graphs": None,
        "epoch": 1,
        "num_workers": 0,
        "pin_memory": False,
        "non_blocking": False,
        "amp": False,
        "pattern_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_before_analytical_does_not_load_shards(tmp_path, monkeypatch):
    dataset = _build_dataset(tmp_path)

    def fail_if_loaded(_):
        raise AssertionError("BEFORE must not load shard files")

    monkeypatch.setattr(dataset, "_load_shard", fail_if_loaded)

    result = analytical_before(
        dataset,
        [0, 3, 6, 9, 1, 4, 7, 10],
        cache_size=1,
    )

    assert result["mode"] == "ANALYTICAL_PATTERN_SIMULATION"
    assert result["number_of_samples"] == 8
    assert result["unique_shards"] == 4
    assert result["shard_transitions"] == 7
    assert result["estimated_shard_loads"] == 8


def test_compare_case_after_uses_shard_aware_sampler(tmp_path, monkeypatch):
    dataset = _build_dataset(tmp_path)
    used = {"sampler": False}

    def tracking_sampler(*args, **kwargs):
        used["sampler"] = True
        return make_shard_aware_sampler(*args, **kwargs)

    monkeypatch.setattr(
        "src.benchmarks.progressive_subset_loading.make_shard_aware_sampler",
        tracking_sampler,
    )

    case = TraversalCase(
        "pilot_subset",
        Subset(dataset, [0, 3, 6, 9, 1, 4, 7, 10]),
        list(range(8)),
    )
    result = compare_case(case, _args(), torch.device("cpu"))

    assert used["sampler"]
    assert result["before"]["mode"] == "ANALYTICAL_PATTERN_SIMULATION"
    assert result["after"]["mode"] == "REAL_SERVER_TRAVERSAL"
    assert result["after"]["graphs"] == 8
    assert result["membership_preserved"]


def test_make_cases_builds_expected_progressive_membership(tmp_path):
    dataset = _build_dataset(tmp_path, shards=5, per_shard=4)
    cases = make_cases(dataset, _args(pilot_graphs=5, confirmation_graphs=11))

    pilot = cases[0]
    confirmation = cases[1]
    full = cases[2]

    assert pilot.name == "pilot_subset"
    assert confirmation.name == "confirmation_subset"
    assert full.name == "full"
    assert len(pilot.dataset) == 5
    assert len(confirmation.dataset) == 11
    assert len(full.dataset) == len(dataset)
    assert set(pilot.dataset.indices).issubset(set(confirmation.dataset.indices))


def test_write_reports_creates_json_and_markdown(tmp_path, monkeypatch):
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    monkeypatch.setattr(
        "src.benchmarks.progressive_subset_loading.REPORT_JSON",
        json_path,
    )
    monkeypatch.setattr(
        "src.benchmarks.progressive_subset_loading.REPORT_MD",
        md_path,
    )
    payload = {
        "results": [
            {
                "name": "pilot_subset",
                "graphs": 8,
                "membership_preserved": True,
                "before": {
                    "mode": "ANALYTICAL_PATTERN_SIMULATION",
                    "shard_transitions": 7,
                    "estimated_shard_loads": 8,
                    "simulated_cache_hit_rate": 0.0,
                },
                "after": {
                    "mode": "REAL_SERVER_TRAVERSAL",
                    "elapsed_seconds": 0.1,
                    "graphs_per_second": 80.0,
                    "shard_transitions": 3,
                    "shard_load_count": 4,
                    "cache_hit_rate": 0.5,
                },
            }
        ]
    }

    write_reports(payload)

    assert json_path.exists()
    assert md_path.exists()
    assert "ANALYTICAL_PATTERN_SIMULATION" in md_path.read_text(encoding="utf-8")
    assert "REAL_SERVER_TRAVERSAL" in md_path.read_text(encoding="utf-8")

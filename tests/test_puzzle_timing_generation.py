import json

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data

from src.data.timing.puzzle_timing import NormalizationStats
from src.data.timing.puzzle_timing import TimingGeneratorConfig
from src.data.timing.puzzle_timing import add_timing_to_graph
from src.data.timing.puzzle_timing import build_timing_dataset
from src.data.timing.puzzle_timing import synthetic_raw_times


def _graph(puzzle_id, rating=1500.0, mate_depth=3):
    return Data(
        x=torch.randn(64, 15),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 5), dtype=torch.float),
        global_features=torch.zeros(1, 4),
        y=torch.tensor(0, dtype=torch.long),
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        target_move="a1a2",
        puzzle_id=puzzle_id,
        rating=torch.tensor(float(rating)),
        mate_depth=torch.tensor(int(mate_depth)),
    )


def _write_source_dataset(root):
    source = root / "source"
    for split in ("train", "val", "test"):
        (source / split).mkdir(parents=True)
    train_graphs = [_graph("train-a", 1000), _graph("train-b", 2000)]
    val_graphs = [_graph("val-a", 1200)]
    test_graphs = [_graph("test-a", 1800)]
    torch.save(train_graphs, source / "train" / "shard_00000.pt")
    torch.save(val_graphs, source / "val" / "shard_00000.pt")
    torch.save(test_graphs, source / "test" / "shard_00000.pt")
    manifest = {
        "feature_schema": {"node_dim": 15, "edge_dim": 5, "global_dim": 4},
        "splits": {
            split: {
                "num_graphs": 2 if split == "train" else 1,
                "num_shards": 1,
                "shards": [{"file": f"{split}/shard_00000.pt", "num_graphs": 2 if split == "train" else 1}],
            }
            for split in ("train", "val", "test")
        },
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_synthetic_timing_is_deterministic_by_puzzle_id_and_seed():
    config = TimingGeneratorConfig(global_seed=123)
    first = synthetic_raw_times("abc", 1500, config)
    second = synthetic_raw_times("abc", 1500, config)
    other_seed = synthetic_raw_times("abc", 1500, TimingGeneratorConfig(global_seed=124))

    assert first == second
    assert first != other_seed
    assert first[0] > 0
    assert first[1] > 0


def test_add_timing_fields_are_finite_and_batch_to_expected_shape():
    graph = add_timing_to_graph(
        _graph("shape-test"),
        TimingGeneratorConfig(),
        NormalizationStats(mean=2.0, std=0.5),
    )
    batch = Batch.from_data_list([graph, graph.clone()])

    assert batch.previous_move_time.shape == (2, 1)
    assert batch.original_move_time.shape == (2, 1)
    assert batch.time_is_synthetic.shape == (2, 1)
    assert torch.isfinite(batch.previous_move_time).all()
    assert torch.isfinite(batch.original_move_time).all()
    assert torch.all(batch.previous_move_time_seconds > 0)
    assert torch.all(batch.original_move_time_seconds > 0)


def test_timing_dataset_manifest_is_train_only_and_leakage_safe(tmp_path):
    source = _write_source_dataset(tmp_path)
    output = tmp_path / "timing"
    manifest = build_timing_dataset(
        source_root=source,
        output_root=output,
        build_root=tmp_path / "building",
        config=TimingGeneratorConfig(global_seed=7),
        overwrite=True,
    )

    assert manifest["normalization"]["fit_split"] == "train"
    assert manifest["leakage_policy"]["uses_target_move"] is False
    assert manifest["leakage_policy"]["uses_mate_depth_for_generation"] is False
    assert manifest["leakage_policy"]["uses_rating"] is True
    assert manifest["leakage_policy"]["potential_shortcut"] is True
    assert manifest["splits"]["train"]["num_graphs"] == 2
    assert manifest["splits"]["val"]["num_graphs"] == 1
    assert (output / "manifest.json").exists()


def test_same_sample_timing_independent_from_loader_order(tmp_path):
    source = _write_source_dataset(tmp_path)
    output = tmp_path / "timing"
    build_timing_dataset(
        source_root=source,
        output_root=output,
        build_root=tmp_path / "building",
        config=TimingGeneratorConfig(global_seed=9),
        overwrite=True,
    )
    graphs = torch.load(
        output / "train" / "shard_00000.pt",
        map_location="cpu",
        weights_only=False,
    )
    by_id = {
        graph.puzzle_id: (
            float(graph.previous_move_time.item()),
            float(graph.original_move_time.item()),
        )
        for graph in graphs
    }
    reversed_by_id = {
        graph.puzzle_id: (
            float(graph.previous_move_time.item()),
            float(graph.original_move_time.item()),
        )
        for graph in reversed(graphs)
    }

    assert by_id == reversed_by_id

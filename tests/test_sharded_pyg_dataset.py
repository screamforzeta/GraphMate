import json

import pandas as pd
import torch
import pytest
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.graph.pyg_dataset import (
    MANIFEST_NAME,
    ShardAwareSampler,
    ShardedPyGDataset,
    access_pattern_stats,
    build_manifest,
    build_pyg_split,
    make_shard_aware_sampler,
)


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _row(puzzle_id, fen, target_move):
    return {
        "PuzzleId": puzzle_id,
        "OriginalFEN": fen,
        "FEN": fen,
        "Moves": target_move,
        "TargetMove": target_move,
        "Rating": 1500,
        "Themes": "mate mateIn1",
        "MateDepth": 1,
    }


def _build_dataset(tmp_path, split="train"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        _row("p0", "8/8/8/8/8/8/4K3/7k w - - 0 1", "e2e1"),
        _row("p1", "8/8/8/8/8/8/4K3/7k w - - 0 1", "e2d1"),
        _row("p2", "8/8/8/8/8/8/4K3/7k w - - 0 1", "e2f1"),
        _row("p3", "8/8/8/8/8/8/4K3/7k w - - 0 1", "e2d2"),
        _row("p4", "8/8/8/8/8/8/4K3/7k w - - 0 1", "e2f2"),
    ]
    csv_path = tmp_path / f"{split}.csv"
    root = tmp_path / "pyg"
    move_to_idx = {
        "e2e1": 0,
        "e2d1": 1,
        "e2f1": 2,
        "e2d2": 3,
        "e2f2": 4,
    }
    _write_csv(csv_path, rows)
    split_manifest, stats = build_pyg_split(
        csv_path,
        split,
        root,
        move_to_idx,
        graphs_per_shard=2,
    )
    manifest = build_manifest(
        graphs_per_shard=2,
        splits={split: split_manifest},
    )
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return root, stats


def _tiny_graph(index):
    return Data(
        x=torch.zeros((64, 15), dtype=torch.float),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.zeros((1, 5), dtype=torch.float),
        global_features=torch.zeros((1, 4), dtype=torch.float),
        y=torch.tensor(index % 3, dtype=torch.long),
        source_row_index=torch.tensor(index, dtype=torch.long),
        graph_marker=torch.tensor(index, dtype=torch.long),
    )


def _build_synthetic_sharded_dataset(tmp_path, split="train", shards=6, per_shard=4):
    root = tmp_path / "synthetic_pyg"
    split_dir = root / split
    split_dir.mkdir(parents=True)
    manifest_shards = []
    for shard_id in range(shards):
        graphs = [
            _tiny_graph(shard_id * per_shard + local_index)
            for local_index in range(per_shard)
        ]
        file_name = f"{split}/shard_{shard_id:05d}.pt"
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
            split: {
                "num_graphs": shards * per_shard,
                "num_shards": shards,
                "shards": manifest_shards,
            }
        },
    )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_sharded_generation_writes_manifest_and_partial_last_shard(tmp_path):
    root, stats = _build_dataset(tmp_path)
    manifest = json.loads(
        (root / MANIFEST_NAME).read_text(encoding="utf-8")
    )

    assert stats.csv_rows == 5
    assert stats.graphs_generated == 5
    assert stats.graphs_skipped == 0
    assert stats.num_shards == 3
    assert manifest["graphs_per_shard"] == 2
    assert manifest["feature_schema"] == {
        "node_dim": 15,
        "edge_dim": 5,
        "global_dim": 4,
    }
    assert manifest["splits"]["train"]["shards"][-1]["num_graphs"] == 1


def test_sharded_dataset_indexes_across_shards_and_bounds(tmp_path):
    root, _ = _build_dataset(tmp_path)
    dataset = ShardedPyGDataset(root=root, split="train", cache_size=2)

    assert len(dataset) == 5
    assert dataset[0].puzzle_id == "p0"
    assert dataset[1].puzzle_id == "p1"
    assert dataset[2].puzzle_id == "p2"
    assert dataset[-1].puzzle_id == "p4"
    assert int(dataset[2].source_row_index.item()) == 2

    with pytest.raises(IndexError):
        _ = dataset[5]


def test_sharded_dataset_dataloader_shuffle_and_cache(tmp_path):
    root, _ = _build_dataset(tmp_path)
    dataset = ShardedPyGDataset(root=root, split="train", cache_size=2)

    _ = dataset[0]
    assert list(dataset._cache) == [0]
    _ = dataset[2]
    assert list(dataset._cache) == [0, 1]
    _ = dataset[4]
    assert list(dataset._cache) == [1, 2]

    batch = next(
        iter(
            DataLoader(
                dataset,
                batch_size=3,
                shuffle=True,
            )
        )
    )

    assert batch.x.shape[1] == 15
    assert batch.edge_index.shape[0] == 2
    assert batch.edge_attr.shape[1] == 5
    assert batch.global_features.shape == (3, 4)
    assert batch.y.shape == (3,)
    assert batch.source_row_index.shape == (3,)


def test_sharded_dataset_keeps_split_isolation(tmp_path):
    train_root, _ = _build_dataset(tmp_path / "train_case", split="train")
    val_root, _ = _build_dataset(tmp_path / "val_case", split="val")

    train_dataset = ShardedPyGDataset(root=train_root, split="train")
    val_dataset = ShardedPyGDataset(root=val_root, split="val")

    assert train_dataset.split == "train"
    assert val_dataset.split == "val"

    with pytest.raises(ValueError):
        ShardedPyGDataset(root=train_root, split="val")


def test_shard_aware_subset_sampler_preserves_membership_and_determinism(tmp_path):
    root = _build_synthetic_sharded_dataset(tmp_path)
    dataset = ShardedPyGDataset(root=root, split="train", cache_size=2)
    selected = [0, 8, 1, 9, 16, 17, 4, 12, 5, 13]
    subset = Subset(dataset, selected)

    sampler_a = make_shard_aware_sampler(subset, shuffle=True, seed=123)
    sampler_b = make_shard_aware_sampler(subset, shuffle=True, seed=123)
    sampler_c = make_shard_aware_sampler(subset, shuffle=True, seed=123)
    sampler_a.set_epoch(1)
    sampler_b.set_epoch(1)
    sampler_c.set_epoch(2)

    order_a = list(iter(sampler_a))
    order_b = list(iter(sampler_b))
    order_c = list(iter(sampler_c))

    assert order_a == order_b
    assert order_a != order_c
    assert sorted(order_a) == list(range(len(subset)))
    assert sorted(order_c) == list(range(len(subset)))
    assert [selected[index] for index in order_a]


def test_shard_aware_subset_sampler_improves_locality_without_timing(tmp_path):
    root = _build_synthetic_sharded_dataset(tmp_path)
    dataset = ShardedPyGDataset(root=root, split="train", cache_size=1)
    selected = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]
    subset = Subset(dataset, selected)
    naive_order = list(range(len(subset)))
    sampler = ShardAwareSampler(subset, shuffle=False)
    shard_order = list(iter(sampler))

    naive = access_pattern_stats(subset, naive_order, cache_size=1)
    fixed = access_pattern_stats(subset, shard_order, cache_size=1)

    assert set(naive_order) == set(shard_order)
    assert fixed["shard_transitions"] < naive["shard_transitions"]
    assert fixed["simulated_shard_loads"] < naive["simulated_shard_loads"]
    assert fixed["simulated_cache_hit_rate"] > naive["simulated_cache_hit_rate"]


def test_shard_aware_subset_sampler_integrates_with_pyg_dataloader(tmp_path):
    root = _build_synthetic_sharded_dataset(tmp_path)
    dataset = ShardedPyGDataset(root=root, split="train", cache_size=2)
    selected = [0, 8, 1, 9, 16, 17, 4, 12, 5, 13]
    subset = Subset(dataset, selected)
    sampler = make_shard_aware_sampler(subset, shuffle=False)

    visited = []
    for batch in DataLoader(subset, batch_size=3, sampler=sampler):
        visited.extend(int(value) for value in batch.graph_marker.tolist())

    assert sorted(visited) == sorted(selected)
    assert len(visited) == len(set(visited)) == len(selected)

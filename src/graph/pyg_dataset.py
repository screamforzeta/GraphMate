"""Generate and load sharded PyTorch Geometric puzzle datasets.

Purpose:
    Convert final puzzle CSV splits into memory-safe PyG shards and expose a
    dataset object that loads individual graphs during training.
Input:
    data/final/puzzles/{train,val,test}.csv and artifacts/move_to_idx.json.
Output:
    data/pyg/{train,val,test}/shard_XXXXX.pt plus data/pyg/manifest.json.
Run:
    python3 -m src.graph.pyg_dataset --graphs-per-shard 1000 --overwrite
"""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import shutil
import time

import chess
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import Subset
from tqdm import tqdm

from src.graph.graph_builder import build_graph


TRAIN_CSV = Path("data/final/puzzles/train.csv")
VAL_CSV = Path("data/final/puzzles/val.csv")
TEST_CSV = Path("data/final/puzzles/test.csv")
MOVE_ENCODER_PATH = Path("artifacts/move_to_idx.json")
OUTPUT_DIR = Path("data/pyg")
BUILDING_DIR = Path("data/pyg_building")
MANIFEST_NAME = "manifest.json"
DEFAULT_GRAPHS_PER_SHARD = 1000
DEFAULT_CACHE_SIZE = 2
SKIP_OOV_TARGETS = True

SPLIT_CSVS = {
    "train": TRAIN_CSV,
    "val": VAL_CSV,
    "test": TEST_CSV,
}

LEGACY_SPLIT_FILES = {
    "train": OUTPUT_DIR / "train_graphs.pt",
    "val": OUTPUT_DIR / "val_graphs.pt",
    "test": OUTPUT_DIR / "test_graphs.pt",
}


@dataclass
class SplitGenerationStats:
    """Generation counters for one CSV split."""

    csv_rows: int
    graphs_generated: int
    graphs_skipped: int
    invalid_fen: int
    invalid_lichess_sequence: int
    oov_targets: int
    other_errors: int
    num_shards: int
    duration_seconds: float

    def to_dict(self):
        """Return a JSON-serializable stats dictionary."""

        return {
            "csv_rows": self.csv_rows,
            "graphs_generated": self.graphs_generated,
            "graphs_skipped": self.graphs_skipped,
            "invalid_fen": self.invalid_fen,
            "invalid_lichess_sequence": self.invalid_lichess_sequence,
            "oov_targets": self.oov_targets,
            "other_errors": self.other_errors,
            "num_shards": self.num_shards,
            "duration_seconds": self.duration_seconds,
        }


class ShardedPyGDataset(Dataset):
    """Index a split stored as PyG shards.

    Parameters:
        root: Directory containing manifest.json and split shard folders.
        split: Split name, usually train, val, or test.
        cache_size: Number of shards kept in a per-process LRU cache.
    Returns:
        Dataset where each item is one torch_geometric.data.Data graph.
    Side effects:
        Lazily reads shard files in __getitem__. With multiple DataLoader
        workers, each worker owns its own shard cache.
    """

    def __init__(self, root=OUTPUT_DIR, split="train", cache_size=DEFAULT_CACHE_SIZE):
        self.root = Path(root)
        self.split = split
        self.cache_size = max(1, int(cache_size))
        self.manifest_path = self.root / MANIFEST_NAME

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Sharded PyG manifest not found: {self.manifest_path}"
            )

        with open(self.manifest_path, "r", encoding="utf-8") as file:
            self.manifest = json.load(file)

        try:
            split_info = self.manifest["splits"][split]
        except KeyError as exc:
            raise ValueError(f"Split not found in manifest: {split}") from exc

        self.shards = split_info["shards"]
        self.num_graphs = int(split_info["num_graphs"])
        self.offsets = []
        offset = 0
        for shard in self.shards:
            self.offsets.append(offset)
            offset += int(shard["num_graphs"])

        if offset != self.num_graphs:
            raise ValueError(
                f"Manifest count mismatch for {split}: offsets={offset}, "
                f"num_graphs={self.num_graphs}"
            )

        self._cache = OrderedDict()

    def __len__(self):
        """Return the number of graphs in this split."""

        return self.num_graphs

    def __getitem__(self, index):
        """Return a graph by global split index."""

        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for split {self.split}")

        shard_index = bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[shard_index]
        shard_graphs = self._load_shard(shard_index)
        return shard_graphs[local_index]

    def shard_index_for_global_index(self, index):
        """Return the shard index that contains a global graph index."""

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for split {self.split}")
        return bisect_right(self.offsets, index) - 1

    def _load_shard(self, shard_index):
        """Load one shard using the small per-process LRU cache."""

        if shard_index in self._cache:
            self._cache.move_to_end(shard_index)
            return self._cache[shard_index]

        shard_path = self.root / self.shards[shard_index]["file"]
        graphs = torch.load(shard_path, weights_only=False)
        self._cache[shard_index] = graphs
        self._cache.move_to_end(shard_index)

        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

        return graphs


class ShardAwareShuffleSampler(Sampler):
    """Shuffle a sharded dataset while keeping shard-local access batches.

    Parameters:
        dataset: ShardedPyGDataset or a Subset backed by ShardedPyGDataset.
        generator: Optional torch.Generator used by DataLoader.
    Returns:
        Iterator over dataset indices.
    Side effects:
        None.
    """

    def __init__(self, dataset, generator=None):
        self.dataset = dataset
        self.generator = generator
        self._groups = self._build_groups(dataset)

    def __len__(self):
        """Return the number of sampled indices."""

        return len(self.dataset)

    def __iter__(self):
        """Yield shuffled indices grouped by shuffled shard order."""

        shard_ids = list(self._groups)
        shard_order = torch.randperm(
            len(shard_ids),
            generator=self.generator,
        ).tolist()

        for order_index in shard_order:
            indices = self._groups[shard_ids[order_index]]
            local_order = torch.randperm(
                len(indices),
                generator=self.generator,
            ).tolist()
            for local_index in local_order:
                yield indices[local_index]

    @staticmethod
    def _build_groups(dataset):
        """Group visible dataset indices by underlying shard id."""

        if isinstance(dataset, ShardedPyGDataset):
            groups = OrderedDict()
            for index in range(len(dataset)):
                shard_index = dataset.shard_index_for_global_index(index)
                groups.setdefault(shard_index, []).append(index)
            return groups

        if (
            isinstance(dataset, Subset)
            and isinstance(dataset.dataset, ShardedPyGDataset)
        ):
            groups = OrderedDict()
            for visible_index, source_index in enumerate(dataset.indices):
                shard_index = dataset.dataset.shard_index_for_global_index(
                    int(source_index)
                )
                groups.setdefault(shard_index, []).append(visible_index)
            return groups

        raise TypeError(
            "ShardAwareShuffleSampler requires ShardedPyGDataset or a Subset "
            "backed by ShardedPyGDataset."
        )


def is_sharded_dataset(dataset):
    """Return True for sharded datasets and Subset-wrapped sharded datasets."""

    return isinstance(dataset, ShardedPyGDataset) or (
        isinstance(dataset, Subset)
        and isinstance(dataset.dataset, ShardedPyGDataset)
    )


def parse_args():
    """Parse dataset generation CLI options."""

    parser = argparse.ArgumentParser(
        description="Generate sharded PyG chess puzzle datasets."
    )
    parser.add_argument(
        "--graphs-per-shard",
        type=int,
        default=DEFAULT_GRAPHS_PER_SHARD,
        help="Maximum number of graphs saved in each shard.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Final sharded dataset directory.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=BUILDING_DIR,
        help="Temporary generation directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory after a complete build.",
    )
    return parser.parse_args()


def load_move_encoder(path=MOVE_ENCODER_PATH):
    """Load the train-only move vocabulary."""

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _atomic_save(obj, path):
    """Save a torch object through a temporary file and atomic rename."""

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _write_json(path, payload):
    """Write JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _save_shard(buffer, split_dir, split_name, shard_index):
    """Write the current shard buffer and return manifest metadata."""

    file_name = f"shard_{shard_index:05d}.pt"
    shard_path = split_dir / file_name
    _atomic_save(buffer, shard_path)
    return {
        "file": f"{split_name}/{file_name}",
        "num_graphs": len(buffer),
    }


def _classify_graph_error(error):
    """Map an expected conversion error to a stats counter name."""

    message = str(error)
    if "Unknown target move" in message:
        return "oov_targets"
    if "illegal" in message.lower():
        return "invalid_lichess_sequence"
    return "other_errors"


def build_pyg_split(csv_path, split_name, output_root, move_to_idx, graphs_per_shard):
    """Generate one split as a sequence of shard files.

    Parameters:
        csv_path: Final puzzle CSV path for the split.
        split_name: Split folder name.
        output_root: Temporary root receiving generated shards.
        move_to_idx: Train-only move vocabulary.
        graphs_per_shard: Maximum graphs held in memory before flushing.
    Returns:
        Tuple of split manifest dictionary and SplitGenerationStats.
    Side effects:
        Reads csv_path and writes shard files under output_root/split_name.
    """

    if graphs_per_shard <= 0:
        raise ValueError("graphs_per_shard must be positive.")

    print("\n" + "=" * 50)
    print(f"PROCESSING: {split_name} ({csv_path})")
    print("=" * 50)

    df = pd.read_csv(csv_path)
    split_dir = Path(output_root) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    error_counts = {
        "invalid_fen": 0,
        "invalid_lichess_sequence": 0,
        "oov_targets": 0,
        "other_errors": 0,
    }
    shards = []
    buffer = []
    graphs_generated = 0
    start = time.perf_counter()

    for source_row_index, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name,
    ):
        try:
            target_move = str(row.TargetMove)
            if target_move not in move_to_idx:
                error_counts["oov_targets"] += 1
                if SKIP_OOV_TARGETS:
                    continue

            try:
                chess.Board(row.FEN)
            except Exception:
                error_counts["invalid_fen"] += 1
                continue

            graph = build_graph(
                fen=row.FEN,
                target_move=target_move,
                move_to_idx=move_to_idx,
            )
            graph.puzzle_id = str(row.PuzzleId)
            graph.rating = torch.tensor(row.Rating, dtype=torch.float)
            graph.mate_depth = torch.tensor(row.MateDepth, dtype=torch.long)
            graph.source_row_index = torch.tensor(
                int(source_row_index),
                dtype=torch.long,
            )

            buffer.append(graph)
            graphs_generated += 1

            if len(buffer) >= graphs_per_shard:
                shards.append(
                    _save_shard(buffer, split_dir, split_name, len(shards))
                )
                buffer = []

        except Exception as error:
            error_counts[_classify_graph_error(error)] += 1
            print(f"[WARNING] Failed graph for row {source_row_index}: {error}")

    if buffer:
        shards.append(_save_shard(buffer, split_dir, split_name, len(shards)))

    duration = time.perf_counter() - start
    graphs_skipped = sum(error_counts.values())
    stats = SplitGenerationStats(
        csv_rows=len(df),
        graphs_generated=graphs_generated,
        graphs_skipped=graphs_skipped,
        invalid_fen=error_counts["invalid_fen"],
        invalid_lichess_sequence=error_counts["invalid_lichess_sequence"],
        oov_targets=error_counts["oov_targets"],
        other_errors=error_counts["other_errors"],
        num_shards=len(shards),
        duration_seconds=duration,
    )

    print_split_summary(split_name, stats)
    return {
        "num_graphs": graphs_generated,
        "num_shards": len(shards),
        "shards": shards,
        "stats": stats.to_dict(),
    }, stats


def print_split_summary(split_name, stats):
    """Print generation counts for one split."""

    print(f"\n{split_name.upper()}")
    print(f"CSV rows: {stats.csv_rows:,}")
    print(f"Graphs generated: {stats.graphs_generated:,}")
    print(f"Skipped: {stats.graphs_skipped:,}")
    print(f"OOV: {stats.oov_targets:,}")
    print(f"Invalid FEN: {stats.invalid_fen:,}")
    print(f"Invalid Lichess sequences: {stats.invalid_lichess_sequence:,}")
    print(f"Other errors: {stats.other_errors:,}")
    print(f"Shards: {stats.num_shards:,}")
    print(f"Duration: {stats.duration_seconds:.2f}s")


def build_manifest(graphs_per_shard, splits):
    """Create the top-level manifest payload."""

    return {
        "format_version": 1,
        "graphs_per_shard": graphs_per_shard,
        "feature_schema": {
            "node_dim": 15,
            "edge_dim": 5,
            "global_dim": 4,
        },
        "splits": splits,
    }


def generate_sharded_datasets(
    graphs_per_shard=DEFAULT_GRAPHS_PER_SHARD,
    output_dir=OUTPUT_DIR,
    build_dir=BUILDING_DIR,
    overwrite=False,
):
    """Generate all CSV splits into a sharded PyG dataset.

    Parameters:
        graphs_per_shard: Maximum graphs stored in memory per shard buffer.
        output_dir: Final data/pyg-like directory.
        build_dir: Temporary directory used for safe generation.
        overwrite: Whether an existing output_dir may be replaced.
    Returns:
        Manifest dictionary written to disk.
    Side effects:
        Writes build_dir, then replaces output_dir only after all splits finish.
    """

    output_dir = Path(output_dir)
    build_dir = Path(build_dir)

    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Re-run with --overwrite to replace it."
        )
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)

    move_to_idx = load_move_encoder()
    print("=" * 50)
    print("PYG SHARDED DATASET GENERATION")
    print("=" * 50)
    print(f"move vocabulary: {len(move_to_idx):,}")
    print(f"graphs_per_shard: {graphs_per_shard:,}")
    print(f"max graphs held in buffer: {graphs_per_shard:,}")

    splits = {}
    try:
        for split_name, csv_path in SPLIT_CSVS.items():
            split_manifest, _ = build_pyg_split(
                csv_path,
                split_name,
                build_dir,
                move_to_idx,
                graphs_per_shard,
            )
            splits[split_name] = split_manifest

        manifest = build_manifest(graphs_per_shard, splits)
        _write_json(build_dir / MANIFEST_NAME, manifest)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        build_dir.replace(output_dir)
    except Exception:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        raise

    print("\n" + "=" * 50)
    print("ALL SHARDED DATASETS GENERATED")
    print("=" * 50)
    print(f"manifest: {output_dir / MANIFEST_NAME}")
    return manifest


def legacy_graph_path(split, root=OUTPUT_DIR):
    """Return the old monolithic graph path for a split."""

    return Path(root) / f"{split}_graphs.pt"


def load_pyg_dataset(root=OUTPUT_DIR, split="train", cache_size=DEFAULT_CACHE_SIZE):
    """Load the preferred sharded split, falling back to legacy .pt files.

    Parameters:
        root: PyG dataset root.
        split: Split name.
        cache_size: Shard cache size for the preferred loader.
    Returns:
        ShardedPyGDataset or a legacy list of Data objects.
    Side effects:
        Reads manifest metadata or the legacy split file.
    """

    root = Path(root)
    if (root / MANIFEST_NAME).exists():
        return ShardedPyGDataset(
            root=root,
            split=split,
            cache_size=cache_size,
        )

    path = legacy_graph_path(split, root)
    if not path.exists():
        raise FileNotFoundError(
            f"No sharded manifest or legacy graph file found for {split} in {root}"
        )

    return torch.load(path, weights_only=False)


def main():
    """Run sharded dataset generation from the command line."""

    args = parse_args()
    generate_sharded_datasets(
        graphs_per_shard=args.graphs_per_shard,
        output_dir=args.output_dir,
        build_dir=args.build_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

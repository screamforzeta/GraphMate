"""Generate serialized PyTorch Geometric datasets from puzzle CSV splits.

Purpose:
    Convert train/validation/test puzzle CSV rows into lists of PyG Data
    objects that can be loaded during GNN training.
It:
    - loads train/val/test CSVs
    - builds graph objects
    - serializes PyG datasets to disk

The resulting files can later be loaded directly
during GNN training.

This is the final preprocessing step before training.
Input:
    CSV datasets:
    - train.csv
    - val.csv
    - test.csv

    Move encoder:
    - move_to_idx.json
Output:
    Serialized PyTorch datasets:
    - train_graphs.pt
    - val_graphs.pt
    - test_graphs.pt

Each dataset contains:
    List[torch_geometric.data.Data]
Run:
    python3 src/graph/pyg_dataset.py
"""

from pathlib import Path
import json

import pandas as pd
import torch
from tqdm import tqdm

from src.graph.graph_builder import (
    build_graph,
)

# =========================================================
# CONFIG
# =========================================================

TRAIN_CSV = Path(
    "data/final/puzzles/train.csv"
)

VAL_CSV = Path(
    "data/final/puzzles/val.csv"
)

TEST_CSV = Path(
    "data/final/puzzles/test.csv"
)

MOVE_ENCODER_PATH = Path(
    "artifacts/move_to_idx.json"
)

OUTPUT_DIR = Path(
    "data/pyg"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_OUTPUT = (
    OUTPUT_DIR / "train_graphs.pt"
)

VAL_OUTPUT = (
    OUTPUT_DIR / "val_graphs.pt"
)

TEST_OUTPUT = (
    OUTPUT_DIR / "test_graphs.pt"
)

# =========================================================
# SETTINGS
# =========================================================

"""
For debugging:
    set to small number

For full dataset:
    use None
"""

MAX_SAMPLES = 5000


# =========================================================
# LOAD MOVE ENCODER
# =========================================================

def load_move_encoder():
    """
    Load move vocabulary.

    Returns
    -------
    dict

    Side effects
    ------------
    Reads MOVE_ENCODER_PATH from disk.
    """

    with open(
        MOVE_ENCODER_PATH,
        "r"
    ) as f:

        move_to_idx = json.load(f)

    return move_to_idx


# =========================================================
# DATASET CREATION
# =========================================================

def build_pyg_dataset(
    csv_path,
    output_path,
    move_to_idx,
):
    """
    Build serialized PyG dataset.

    Parameters
    ----------
    csv_path : Path
    output_path : Path
    move_to_idx : dict

    Returns
    -------
    None

    Side effects
    ------------
    Reads csv_path, builds graph objects, writes output_path with torch.save,
    and prints graph generation warnings/statistics.
    """

    print("\n" + "=" * 50)
    print(f"PROCESSING: {csv_path.name}")
    print("=" * 50)

    # =============================================
    # LOAD CSV
    # =============================================

    df = pd.read_csv(csv_path)

    print(
        f"[INFO] Loaded "
        f"{len(df)} samples"
    )

    # =============================================
    # OPTIONAL DEBUG LIMIT
    # =============================================

    if MAX_SAMPLES is not None:

        df = df.iloc[:MAX_SAMPLES]

        print(
            f"[INFO] Using first "
            f"{len(df)} samples"
        )

    # =============================================
    # GRAPH CONTAINER
    # =============================================

    graphs = []

    failed_graphs = 0

    # =============================================
    # BUILD GRAPHS
    # =============================================

    print("[INFO] Building graphs...")

    for row in tqdm(
        df.itertuples(),
        total=len(df),
    ):

        try:

            graph = build_graph(
                fen=row.FEN,
                target_move=row.TargetMove,
                move_to_idx=move_to_idx,
            )

            # =====================================
            # ADD EXTRA METADATA
            # =====================================

            graph.puzzle_id = row.PuzzleId

            graph.rating = torch.tensor(
                row.Rating,
                dtype=torch.float
            )

            graph.mate_depth = torch.tensor(
                row.MateDepth,
                dtype=torch.long
            )

            graphs.append(graph)

        except Exception as e:

            failed_graphs += 1

            print(
                f"[WARNING] Failed graph: "
                f"{row.PuzzleId}"
            )

            print(f"[WARNING] {e}")

    # =============================================
    # SAVE DATASET
    # =============================================

    print("\n[INFO] Saving dataset...")

    torch.save(
        graphs,
        output_path
    )

    # =============================================
    # FINAL LOGS
    # =============================================

    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(
        f"[INFO] Successful graphs: "
        f"{len(graphs)}"
    )

    print(
        f"[INFO] Failed graphs: "
        f"{failed_graphs}"
    )

    if len(graphs) > 0:

        sample_graph = graphs[0]

        print("\n[INFO] Sample graph:")

        print(sample_graph)

        print("\n[INFO] Node feature shape:")

        print(sample_graph.x.shape)

        print("\n[INFO] Edge index shape:")

        print(
            sample_graph.edge_index.shape
        )

        print("\n[INFO] Edge attr shape:")

        print(
            sample_graph.edge_attr.shape
        )

    print(f"\n[INFO] Saved to:")
    print(f"[INFO] {output_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    """
    Main dataset generation pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side effects
    ------------
    Reads the move encoder and writes train/val/test PyG dataset files.
    """

    print("=" * 50)
    print("PYG DATASET GENERATION")
    print("=" * 50)

    # =============================================
    # LOAD MOVE ENCODER
    # =============================================

    print("[INFO] Loading move encoder...")

    move_to_idx = load_move_encoder()

    print(
        f"[INFO] Loaded "
        f"{len(move_to_idx)} moves"
    )

    # =============================================
    # BUILD DATASETS
    # =============================================

    build_pyg_dataset(
        TRAIN_CSV,
        TRAIN_OUTPUT,
        move_to_idx,
    )

    build_pyg_dataset(
        VAL_CSV,
        VAL_OUTPUT,
        move_to_idx,
    )

    build_pyg_dataset(
        TEST_CSV,
        TEST_OUTPUT,
        move_to_idx,
    )

    print("\n" + "=" * 50)
    print("ALL DATASETS GENERATED")
    print("=" * 50)


if __name__ == "__main__":
    main()

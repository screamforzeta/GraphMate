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

import chess
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

# OOV targets are counted explicitly and excluded from current class-index
# datasets, because the move vocabulary is intentionally train-only.
SKIP_OOV_TARGETS = True


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
    dict
        Summary counts for generated graphs and skipped/error rows.

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

    error_counts = {
        "invalid_fen": 0,
        "invalid_lichess_sequence": 0,
        "oov_targets": 0,
        "other_errors": 0,
    }

    # =============================================
    # BUILD GRAPHS
    # =============================================

    print("[INFO] Building graphs...")

    for row in tqdm(
        df.itertuples(),
        total=len(df),
    ):

        try:
            if row.TargetMove not in move_to_idx:
                error_counts["oov_targets"] += 1
                print(
                    f"[WARNING] OOV target for "
                    f"{row.PuzzleId}: {row.TargetMove}"
                )

                if SKIP_OOV_TARGETS:
                    continue

            try:
                chess.Board(row.FEN)
            except Exception:
                error_counts["invalid_fen"] += 1
                print(
                    f"[WARNING] Invalid FEN for "
                    f"{row.PuzzleId}"
                )
                continue

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

            message = str(e)

            if "Unknown target move" in message:
                error_counts["oov_targets"] += 1
            elif "illegal" in message.lower():
                error_counts["invalid_lichess_sequence"] += 1
            else:
                error_counts["other_errors"] += 1

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

    input_samples = len(df)
    total_errors = sum(error_counts.values())

    print(
        f"[INFO] Input samples: "
        f"{input_samples}"
    )

    print(
        f"[INFO] Graphs generated: "
        f"{len(graphs)}"
    )

    print(
        f"[INFO] Invalid Lichess sequences: "
        f"{error_counts['invalid_lichess_sequence']}"
    )

    print(
        f"[INFO] Invalid FEN: "
        f"{error_counts['invalid_fen']}"
    )

    print(
        f"[INFO] OOV targets: "
        f"{error_counts['oov_targets']}"
    )

    print(
        f"[INFO] Other errors: "
        f"{error_counts['other_errors']}"
    )

    print(
        f"[INFO] Total skipped/errors: "
        f"{total_errors}"
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

    return {
        "input_samples": input_samples,
        "graphs_generated": len(graphs),
        "invalid_lichess_sequences": error_counts[
            "invalid_lichess_sequence"
        ],
        "invalid_fen": error_counts["invalid_fen"],
        "oov_targets": error_counts["oov_targets"],
        "other_errors": error_counts["other_errors"],
    }


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

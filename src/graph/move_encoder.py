"""Build the target-move vocabulary for puzzle classification.

Purpose:
    Create a stable mapping between UCI target moves and integer class labels
    using only the training split.
Input:
    data/final/puzzles/train.csv
Output:
    resources/move_encoder/move_to_idx.json
    resources/move_encoder/idx_to_move.json
    resources/move_encoder/move_encoder_stats.json
Run:
    python3 src/graph/move_encoder.py
"""

from pathlib import Path
import pandas as pd
import json
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = Path(
    "data/final/puzzles/train.csv"
)

VAL_CSV = Path(
    "data/final/puzzles/val.csv"
)

TEST_CSV = Path(
    "data/final/puzzles/test.csv"
)

OUTPUT_DIR = Path("resources/move_encoder")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MOVE_TO_IDX_FILE = (
    OUTPUT_DIR / "move_to_idx.json"
)

IDX_TO_MOVE_FILE = (
    OUTPUT_DIR / "idx_to_move.json"
)

STATS_FILE = (
    OUTPUT_DIR / "move_encoder_stats.json"
)


# =========================================================
# BUILD ENCODER
# =========================================================

def compute_oov_stats(csv_path, move_to_idx):
    """Compute target moves missing from the training vocabulary.

    Parameters:
        csv_path: Puzzle split CSV path.
        move_to_idx: Training-only move vocabulary.
    Returns:
        Dictionary with input count, OOV count, OOV percentage, and examples.
    Side effects:
        Reads csv_path from disk.
    """

    df = pd.read_csv(csv_path)

    targets = (
        df["TargetMove"]
        .dropna()
        .astype(str)
    )

    oov_targets = targets[
        ~targets.isin(move_to_idx)
    ]

    total = len(targets)
    oov_count = len(oov_targets)

    return {
        "split": csv_path.stem,
        "input_targets": total,
        "oov_targets": oov_count,
        "oov_percentage": (
            oov_count / total * 100
            if total
            else 0.0
        ),
        "examples": sorted(
            oov_targets.unique().tolist()
        )[:20],
    }


def build_move_encoder():
    """Build and save move-to-index and index-to-move dictionaries.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Reads the training puzzle CSV, creates resources/move_encoder/, and
        writes the canonical checked-in encoder JSON files plus statistics.
    """

    print("[INFO] Loading training dataset...")

    df = pd.read_csv(INPUT_CSV)

    print(
        f"[INFO] Loaded {len(df)} samples"
    )

    # =====================================================
    # EXTRACT MOVES
    # =====================================================

    print("[INFO] Extracting target moves...")

    moves = (
        df["TargetMove"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(
        f"[INFO] Found {len(moves)} target moves"
    )

    # =====================================================
    # UNIQUE MOVES
    # =====================================================

    unique_moves = sorted(
        list(set(moves))
    )

    print(
        f"[INFO] Unique moves: "
        f"{len(unique_moves)}"
    )

    # =====================================================
    # BUILD VOCABULARY
    # =====================================================

    print("[INFO] Building move vocabulary...")

    move_to_idx = {
        move: idx
        for idx, move
        in enumerate(unique_moves)
    }

    idx_to_move = {
        idx: move
        for move, idx
        in move_to_idx.items()
    }

    # =====================================================
    # MOVE FREQUENCIES
    # =====================================================

    print("[INFO] Computing move frequencies...")

    move_counter = Counter(moves)

    most_common = (
        move_counter.most_common(20)
    )

    # =====================================================
    # SAVE ENCODERS
    # =====================================================

    print("[INFO] Saving encoders...")

    with open(
        MOVE_TO_IDX_FILE,
        "w"
    ) as f:

        json.dump(
            move_to_idx,
            f,
            indent=4
        )

    with open(
        IDX_TO_MOVE_FILE,
        "w"
    ) as f:

        json.dump(
            idx_to_move,
            f,
            indent=4
        )

    # =====================================================
    # STATS
    # =====================================================

    stats = {
        "num_unique_moves":
            len(unique_moves),

        "num_total_samples":
            len(moves),

        "top_20_moves": [
            {
                "move": move,
                "count": count
            }
            for move, count
            in most_common
        ]
    }

    # =====================================================
    # LOGS
    # =====================================================

    split_oov_stats = []

    for split_path in (VAL_CSV, TEST_CSV):
        if split_path.exists():
            split_oov_stats.append(
                compute_oov_stats(
                    split_path,
                    move_to_idx,
                )
            )

    stats["oov_stats"] = split_oov_stats

    with open(
        STATS_FILE,
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    print("\n" + "=" * 50)
    print("MOVE ENCODER SUMMARY")
    print("=" * 50)

    print(
        f"[INFO] Total samples: "
        f"{len(moves)}"
    )

    print(
        f"[INFO] Unique moves: "
        f"{len(unique_moves)}"
    )

    print("\n[INFO] Top 20 most common moves:")

    for move, count in most_common:

        percentage = (
            count / len(moves)
        ) * 100

        print(
            f"{move:<8} "
            f"{count:<8} "
            f"({percentage:.2f}%)"
        )

    if split_oov_stats:
        print("\n[INFO] OOV targets against train vocabulary:")

        for split_stats in split_oov_stats:
            print(
                f"[INFO] {split_stats['split']}: "
                f"{split_stats['oov_targets']}/"
                f"{split_stats['input_targets']} "
                f"({split_stats['oov_percentage']:.2f}%)"
            )

    print("\n[INFO] Files saved:")

    print(
        f"[INFO] {MOVE_TO_IDX_FILE}"
    )

    print(
        f"[INFO] {IDX_TO_MOVE_FILE}"
    )

    print(
        f"[INFO] {STATS_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    """Run move encoder generation as a script entry point.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Delegates to build_move_encoder().
    """

    build_move_encoder()


if __name__ == "__main__":
    main()

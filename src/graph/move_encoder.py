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

OUTPUT_DIR = Path("artifacts")
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

def build_move_encoder():

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

    with open(
        STATS_FILE,
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

    # =====================================================
    # LOGS
    # =====================================================

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

    build_move_encoder()


if __name__ == "__main__":
    main()
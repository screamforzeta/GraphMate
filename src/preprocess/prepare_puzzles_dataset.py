from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = Path(
    "data/processed/puzzles/mate_puzzles_clean.csv"
)

OUTPUT_DIR = Path("data/final/puzzles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VAL_FILE = OUTPUT_DIR / "val.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

# =========================================================
# DATASET SETTINGS
# =========================================================

MAX_SAMPLES_PER_DEPTH = 20000

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================

def extract_target_move(moves_str):

    """
    Extract first move from puzzle solution.
    """

    try:

        moves = str(moves_str).split()

        if len(moves) == 0:
            return None

        return moves[0]

    except Exception:
        return None


def print_dataset_stats(df, name):

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET STATS")
    print(f"{'=' * 50}")

    print(f"[INFO] Total samples: {len(df)}")

    print("\n[INFO] Mate depth distribution:")

    depth_counts = (
        df["MateDepth"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for depth, percentage in depth_counts.items():

        print(
            f"Mate-in-{depth}: "
            f"{percentage:.2f}%"
        )

    print("\n[INFO] Average rating:")

    print(round(df["Rating"].mean(), 2))

    print("\n[INFO] Rating std:")

    print(round(df["Rating"].std(), 2))


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset():

    # =====================================================
    # LOAD
    # =====================================================

    print("[INFO] Loading cleaned puzzle dataset...")

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Loaded {len(df)} puzzles")

    # =====================================================
    # TARGET MOVE
    # =====================================================

    print("[INFO] Extracting target moves...")

    df["TargetMove"] = df["Moves"].apply(
        extract_target_move
    )

    df = df[
        df["TargetMove"].notna()
    ]

    # =====================================================
    # BALANCE DATASET
    # =====================================================

    print(
        "[INFO] Creating balanced dataset..."
    )

    balanced_parts = []

    for depth in sorted(
        df["MateDepth"].unique()
    ):

        subset = df[
            df["MateDepth"] == depth
        ]

        print(
            f"[INFO] Mate-in-{depth}: "
            f"{len(subset)} available"
        )

        sample_size = min(
            MAX_SAMPLES_PER_DEPTH,
            len(subset)
        )

        sampled = subset.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        balanced_parts.append(sampled)

        print(
            f"[INFO] Mate-in-{depth}: "
            f"{sample_size} selected"
        )

    df_balanced = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    # =====================================================
    # KEEP ONLY USEFUL COLUMNS
    # =====================================================

    columns_to_keep = [
        "PuzzleId",
        "FEN",
        "Moves",
        "TargetMove",
        "Rating",
        "Themes",
        "MateDepth",
    ]

    df_balanced = df_balanced[
        columns_to_keep
    ]

    # =====================================================
    # SHUFFLE
    # =====================================================

    print("[INFO] Shuffling dataset...")

    df_balanced = df_balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # =====================================================
    # SPLIT TRAIN/VAL/TEST
    # =====================================================

    print(
        "[INFO] Creating train/val/test split..."
    )

    train_df, temp_df = train_test_split(
        df_balanced,
        test_size=(1 - TRAIN_RATIO),
        stratify=df_balanced["MateDepth"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(
            TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        ),
        stratify=temp_df["MateDepth"],
        random_state=RANDOM_STATE
    )

    # =====================================================
    # SAVE
    # =====================================================

    print("[INFO] Saving datasets...")

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    val_df.to_csv(
        VAL_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    # =====================================================
    # FINAL LOGS
    # =====================================================

    total_samples = len(df_balanced)

    print("\n" + "=" * 50)
    print("FINAL DATASET SUMMARY")
    print("=" * 50)

    print(
        f"[INFO] Total puzzles balanced samples: "
        f"{total_samples}"
    )

    print(
        f"[INFO] Puzzles - Train samples: "
        f"{len(train_df)} "
        f"({len(train_df)/total_samples*100:.2f}%)"
    )

    print(
        f"[INFO] Puzzles - Validation samples: "
        f"{len(val_df)} "
        f"({len(val_df)/total_samples*100:.2f}%)"
    )

    print(
        f"[INFO] Puzzles - Test samples: "
        f"{len(test_df)} "
        f"({len(test_df)/total_samples*100:.2f}%)"
    )

    # =====================================================
    # PER DATASET STATS
    # =====================================================

    print_dataset_stats(
        train_df,
        "train"
    )

    print_dataset_stats(
        val_df,
        "validation"
    )

    print_dataset_stats(
        test_df,
        "test"
    )

    # =====================================================
    # SAVE COMPLETE
    # =====================================================

    print("\n[INFO] Dataset preparation complete.")

    print(f"[INFO] Puzzles - Train: {TRAIN_FILE}")
    print(f"[INFO] Puzzles - Validation: {VAL_FILE}")
    print(f"[INFO] Puzzles - Test: {TEST_FILE}")


# =========================================================
# MAIN
# =========================================================

def main():

    prepare_dataset()


if __name__ == "__main__":
    main()
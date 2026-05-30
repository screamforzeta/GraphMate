from pathlib import Path
import pandas as pd
import chess

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = Path(
    "data/processed/puzzles/mate_puzzles.csv"
)

OUTPUT_DIR = Path("data/processed/puzzles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "mate_puzzles_clean.csv"

MIN_RATING = 400
MAX_RATING = 3500

VALID_MATE_DEPTHS = {1, 2, 3, 4, 5}


# =========================================================
# HELPERS
# =========================================================

def is_valid_fen(fen):

    try:
        chess.Board(fen)
        return True

    except Exception:
        return False


def is_valid_rating(rating):

    try:

        rating = int(rating)

        return MIN_RATING <= rating <= MAX_RATING

    except Exception:
        return False


# =========================================================
# CLEANING
# =========================================================

def clean_puzzles():

    if OUTPUT_CSV.exists():

        print(
            f"[INFO] Clean puzzle dataset already exists:"
        )

        print(f"[INFO] {OUTPUT_CSV}")

        return

    print("[INFO] Loading puzzle dataset...")

    df = pd.read_csv(INPUT_CSV)

    original_size = len(df)

    print(f"[INFO] Original puzzles: {original_size}")

    # =====================================================
    # VALID FEN
    # =====================================================

    print("[INFO] Validating FENs...")

    df = df[
        df["FEN"].apply(is_valid_fen)
    ]

    print(
        f"[INFO] Remaining after FEN validation: "
        f"{len(df)}"
    )

    # =====================================================
    # VALID RATINGS
    # =====================================================

    print("[INFO] Filtering valid ratings...")

    df = df[
        df["Rating"].apply(is_valid_rating)
    ]

    print(
        f"[INFO] Remaining after rating filtering: "
        f"{len(df)}"
    )

    # =====================================================
    # VALID MATE DEPTH
    # =====================================================

    print("[INFO] Filtering mate depths...")

    df = df[
        df["MateDepth"].isin(
            VALID_MATE_DEPTHS
        )
    ]

    print(
        f"[INFO] Remaining after mate depth filtering: "
        f"{len(df)}"
    )

    # =====================================================
    # REMOVE EMPTY MOVES
    # =====================================================

    print("[INFO] Removing empty move sequences...")

    df = df[
        df["Moves"].notna()
    ]

    df = df[
        df["Moves"].astype(str).str.len() > 0
    ]

    print(
        f"[INFO] Remaining after move filtering: "
        f"{len(df)}"
    )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    print("[INFO] Removing duplicates...")

    before_dedup = len(df)

    df = df.drop_duplicates(
        subset=["FEN", "Moves"]
    )

    removed_duplicates = (
        before_dedup - len(df)
    )

    print(
        f"[INFO] Removed duplicates: "
        f"{removed_duplicates}"
    )

    # =====================================================
    # CLEAN TYPES
    # =====================================================

    print("[INFO] Converting column types...")

    df["Rating"] = df["Rating"].astype(int)

    df["MateDepth"] = (
        df["MateDepth"].astype(int)
    )

    # =====================================================
    # FINAL STATS
    # =====================================================

    final_size = len(df)

    removed = original_size - final_size

    print("\n[INFO] Cleaning complete.")

    print(f"[INFO] Final puzzles: {final_size}")

    print(f"[INFO] Removed puzzles: {removed}")

    print("\n[INFO] Mate depth distribution:")

    print(
        df["MateDepth"]
        .value_counts()
        .sort_index()
    )

    print("\n[INFO] Average rating:")

    print(round(df["Rating"].mean(), 2))

    # =====================================================
    # SAVE
    # =====================================================

    print("\n[INFO] Saving cleaned dataset...")

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        "\n[INFO] Saved cleaned puzzle dataset:"
    )

    print(f"[INFO] {OUTPUT_CSV}")


# =========================================================
# MAIN
# =========================================================

def main():

    clean_puzzles()


if __name__ == "__main__":
    main()
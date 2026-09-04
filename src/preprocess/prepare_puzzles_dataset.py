"""Prepare final train/validation/test puzzle CSV files.

Purpose:
    Add the target move, balance mate depths, shuffle rows, and create
    stratified train/validation/test splits for the puzzle task.
Input:
    data/processed/puzzles/mate_puzzles_clean.csv
Output:
    data/final/puzzles/train.csv
    data/final/puzzles/val.csv
    data/final/puzzles/test.csv
Run:
    python3 src/preprocess/prepare_puzzles_dataset.py
"""

from pathlib import Path
import time
import chess
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

PROGRESS_EVERY_ROWS = 25000


# =========================================================
# HELPERS
# =========================================================

def extract_target_move(moves_str):
    """Extract the solver target move from a Lichess puzzle move line.

    Parameters:
        moves_str: Space-separated moves from the Lichess puzzle CSV. The first
            move reaches the puzzle position, and the second move is the first
            move to solve.
    Returns:
        Second move as a string, or None when the input is too short/invalid.
    Side effects:
        None.
    """

    try:

        moves = str(moves_str).split()

        if len(moves) < 2:
            return None

        return moves[1]

    except Exception:
        return None


def format_duration(seconds):
    """Format elapsed seconds as HH:MM:SS.

    Parameters:
        seconds: Duration in seconds.
    Returns:
        Human-readable duration string.
    Side effects:
        None.
    """

    seconds = max(
        0,
        int(seconds)
    )

    hours, remainder = divmod(
        seconds,
        3600
    )
    minutes, secs = divmod(
        remainder,
        60
    )

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def print_transform_progress(processed, total, start_time):
    """Print one progress line for puzzle transformation.

    Parameters:
        processed: Number of rows already processed.
        total: Total rows to process.
        start_time: Monotonic timestamp captured before processing.
    Returns:
        None.
    Side effects:
        Writes progress information to stdout.
    """

    elapsed = time.monotonic() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    percentage = (
        processed / total * 100
        if total
        else 100
    )

    remaining = max(
        total - processed,
        0
    )
    eta = (
        remaining / rate
        if rate > 0
        else 0
    )

    print(
        f"[INFO] Progress: {processed:,} / {total:,} "
        f"({percentage:.2f}%) | {rate:,.0f} puzzles/s | "
        f"elapsed {format_duration(elapsed)} | "
        f"ETA {format_duration(eta)}"
    )


def transform_lichess_puzzle_fields(original_fen, moves_str):
    """Validate and transform one Lichess puzzle from raw fields.

    Parameters:
        original_fen: Raw FEN from the Lichess puzzle CSV.
        moves_str: Space-separated Lichess puzzle move sequence.
    Returns:
        Dict with transformed FEN, OriginalFEN, TargetMove, and error reason.
    Side effects:
        None.
    """

    moves = str(moves_str).split()

    if len(moves) < 2:
        return {
            "OriginalFEN": original_fen,
            "FEN": None,
            "TargetMove": None,
            "InvalidReason": "too_few_moves",
        }

    try:
        board = chess.Board(original_fen)
    except Exception:
        return {
            "OriginalFEN": original_fen,
            "FEN": None,
            "TargetMove": None,
            "InvalidReason": "invalid_fen",
        }

    try:
        setup_move = chess.Move.from_uci(moves[0])
    except Exception:
        return {
            "OriginalFEN": original_fen,
            "FEN": None,
            "TargetMove": None,
            "InvalidReason": "invalid_setup_uci",
        }

    if setup_move not in board.legal_moves:
        return {
            "OriginalFEN": original_fen,
            "FEN": None,
            "TargetMove": None,
            "InvalidReason": "illegal_setup_move",
        }

    board.push(setup_move)

    try:
        target_move = chess.Move.from_uci(moves[1])
    except Exception:
        return {
            "OriginalFEN": original_fen,
            "FEN": board.fen(),
            "TargetMove": None,
            "InvalidReason": "invalid_target_uci",
        }

    if target_move not in board.legal_moves:
        return {
            "OriginalFEN": original_fen,
            "FEN": board.fen(),
            "TargetMove": moves[1],
            "InvalidReason": "illegal_target_move",
        }

    return {
        "OriginalFEN": original_fen,
        "FEN": board.fen(),
        "TargetMove": moves[1],
        "InvalidReason": None,
    }


def transform_lichess_puzzle(row):
    """Validate and transform one Lichess puzzle row.

    Parameters:
        row: Pandas row with FEN and Moves fields.
    Returns:
        Dict with transformed FEN, OriginalFEN, TargetMove, and error reason.
    Side effects:
        None.
    """

    return transform_lichess_puzzle_fields(
        row["FEN"],
        row["Moves"]
    )


def transform_puzzle_dataframe(df):
    """Transform all puzzle rows while reporting progress.

    Parameters:
        df: Cleaned puzzle DataFrame with FEN and Moves columns.
    Returns:
        DataFrame with OriginalFEN, transformed FEN, TargetMove, and
        InvalidReason columns.
    Side effects:
        Prints periodic progress updates to stdout.
    """

    total = len(df)
    print(
        f"[INFO] Transforming {total:,} Lichess puzzle positions..."
    )

    start_time = time.monotonic()
    transformed_rows = []

    for processed, row in enumerate(
        df[["FEN", "Moves"]].itertuples(index=False),
        start=1
    ):
        transformed_rows.append(
            transform_lichess_puzzle_fields(
                row.FEN,
                row.Moves
            )
        )

        if (
            processed % PROGRESS_EVERY_ROWS == 0
            or processed == total
        ):
            print_transform_progress(
                processed,
                total,
                start_time
            )

    transformed = pd.DataFrame(
        transformed_rows,
        index=df.index
    )

    invalid_count = int(
        transformed["InvalidReason"].notna().sum()
    )
    valid_count = total - invalid_count

    print(
        "[INFO] Puzzle transformation completed: "
        f"{valid_count:,} valid, {invalid_count:,} invalid, "
        f"elapsed {format_duration(time.monotonic() - start_time)}"
    )

    return transformed


def print_dataset_stats(df, name):
    """Print basic split statistics for a puzzle DataFrame.

    Parameters:
        df: Puzzle split DataFrame.
        name: Human-readable split name.
    Returns:
        None.
    Side effects:
        Writes summary statistics to stdout.
    """

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
    """Build balanced puzzle splits and save them to final CSV files.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Reads INPUT_CSV and writes TRAIN_FILE, VAL_FILE, and TEST_FILE.
    """

    # =====================================================
    # LOAD
    # =====================================================

    print("[INFO] Loading cleaned puzzle dataset...")

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Loaded {len(df)} puzzles")

    # =====================================================
    # TARGET MOVE
    # =====================================================

    transformed = transform_puzzle_dataframe(df)

    invalid_mask = transformed["InvalidReason"].notna()
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        print("[WARNING] Invalid Lichess puzzle rows:")
        print(transformed["InvalidReason"].value_counts())

    df["OriginalFEN"] = transformed["OriginalFEN"]
    df["FEN"] = transformed["FEN"]
    df["TargetMove"] = transformed["TargetMove"]

    df = df[~invalid_mask].copy()

    print(
        f"[INFO] Valid transformed puzzles: "
        f"{len(df)}"
    )
    print(
        f"[INFO] Invalid transformed puzzles: "
        f"{invalid_count}"
    )

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
        "OriginalFEN",
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
    """Run puzzle dataset preparation as a script entry point.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Delegates to prepare_dataset().
    """

    prepare_dataset()


if __name__ == "__main__":
    main()

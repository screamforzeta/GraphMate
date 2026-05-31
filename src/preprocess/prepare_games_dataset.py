from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = Path(
    "data/processed/games/games_clean.csv"
)

OUTPUT_DIR = Path("data/final/games")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = OUTPUT_DIR / "games_train.csv"
VAL_FILE = OUTPUT_DIR / "games_val.csv"
TEST_FILE = OUTPUT_DIR / "games_test.csv"

# =========================================================
# SETTINGS
# =========================================================

MAX_GAMES_PER_TYPE = 300

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_STATE = 42


# =========================================================
# HELPERS
# =========================================================

def parse_time_control(tc):

    """
    Parse Lichess time control.

    Example:
        180+2
    """

    try:

        base, increment = tc.split("+")

        return int(base), int(increment)

    except Exception:

        return None, None


def compute_average_elo(row):

    return (
        row["WhiteElo"]
        + row["BlackElo"]
    ) / 2


def classify_game_length(num_moves):

    if num_moves < 20:
        return "short"

    if num_moves < 60:
        return "medium"

    return "long"


def print_stats(df, name):

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 50}")

    print(f"[INFO] Samples: {len(df)}")

    print("\n[INFO] Game type distribution:")

    distribution = (
        df["GameType"]
        .value_counts(normalize=True)
        * 100
    )

    for game_type, percentage in distribution.items():

        print(
            f"{game_type}: "
            f"{percentage:.2f}%"
        )

    print("\n[INFO] Average Elo:")

    print(round(df["AverageElo"].mean(), 2))

    print("\n[INFO] Average NumMoves:")

    print(round(df["NumMoves"].mean(), 2))


# =========================================================
# PREPARE
# =========================================================

def prepare_games_dataset():

    print("[INFO] Loading cleaned games...")

    df = pd.read_csv(INPUT_CSV)

    print(f"[INFO] Loaded {len(df)} games")

    # =====================================================
    # TIME CONTROL FEATURES
    # =====================================================

    print("[INFO] Parsing time controls...")

    parsed_tc = df["TimeControl"].apply(
        parse_time_control
    )

    df["BaseTime"] = parsed_tc.apply(
        lambda x: x[0]
    )

    df["Increment"] = parsed_tc.apply(
        lambda x: x[1]
    )

    # =====================================================
    # REMOVE INVALID TIME CONTROLS
    # =====================================================

    df = df[
        df["BaseTime"].notna()
    ]

    # =====================================================
    # EXTRA FEATURES
    # =====================================================

    print("[INFO] Creating derived features...")

    df["AverageElo"] = df.apply(
        compute_average_elo,
        axis=1
    )

    df["GameLengthCategory"] = (
        df["NumMoves"].apply(
            classify_game_length
        )
    )

    # =====================================================
    # BALANCE GAME TYPES
    # =====================================================

    print("[INFO] Balancing game types...")

    balanced_parts = []

    for game_type in sorted(
        df["GameType"].unique()
    ):

        subset = df[
            df["GameType"] == game_type
        ]

        print(
            f"[INFO] {game_type}: "
            f"{len(subset)} available"
        )

        sample_size = min(
            MAX_GAMES_PER_TYPE,
            len(subset)
        )

        sampled = subset.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

        balanced_parts.append(sampled)

        print(
            f"[INFO] {game_type}: "
            f"{sample_size} selected"
        )

    df_balanced = pd.concat(
        balanced_parts,
        ignore_index=True
    )

    # =====================================================
    # SHUFFLE
    # =====================================================

    df_balanced = df_balanced.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    # =====================================================
    # SPLIT
    # =====================================================

    print("[INFO] Creating splits...")

    train_df, temp_df = train_test_split(
        df_balanced,
        test_size=(1 - TRAIN_RATIO),
        stratify=df_balanced["GameType"],
        random_state=RANDOM_STATE
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(
            TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        ),
        stratify=temp_df["GameType"],
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
    # LOGS
    # =====================================================

    total = len(df_balanced)

    print("\n" + "=" * 50)
    print("FINAL GAMES DATASET SUMMARY")
    print("=" * 50)

    print(f"[INFO] Total games: {total}")

    print(
        f"[INFO] Games - Train: "
        f"{len(train_df)} "
        f"({len(train_df)/total*100:.2f}%)"
    )

    print(
        f"[INFO] Games - Validation: "
        f"{len(val_df)} "
        f"({len(val_df)/total*100:.2f}%)"
    )

    print(
        f"[INFO] Games - Test: "
        f"{len(test_df)} "
        f"({len(test_df)/total*100:.2f}%)"
    )

    print_stats(train_df, "train")
    print_stats(val_df, "validation")
    print_stats(test_df, "test")

    print("\n[INFO] Games dataset ready.")

    print(f"[INFO] Games - Train: {TRAIN_FILE}")
    print(f"[INFO] Games - Validation: {VAL_FILE}")
    print(f"[INFO] Games - Test: {TEST_FILE}")


# =========================================================
# MAIN
# =========================================================

def main():

    prepare_games_dataset()


if __name__ == "__main__":
    main()
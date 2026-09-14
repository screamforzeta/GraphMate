"""Validate and clean parsed Lichess game metadata.

Purpose:
    Filter parsed games by result, Elo range, game type, and legal move
    sequence before saving the cleaned game dataset.
Input:
    data/processed/games/games_metadata.csv
Output:
    data/processed/games/games_clean.csv
Run:
    python3 src/preprocess/clean_games.py
"""

from pathlib import Path
import pandas as pd
import chess

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV = Path(
    "data/processed/games/games_metadata.csv"
)

OUTPUT_DIR = Path("data/processed/games")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "games_clean.csv"

MIN_MOVES = 10

VALID_RESULTS = {
    "1-0",
    "0-1",
    "1/2-1/2"
}

VALID_TIME_CONTROLS = {
    "Bullet",
    "Blitz",
    "Rapid"
}


# =========================================================
# HELPERS
# =========================================================

def is_valid_elo(value):
    """Check whether a value is a plausible chess Elo rating.

    Parameters:
        value: Raw Elo value from the parsed CSV.
    Returns:
        True if value can be parsed as an int in [400, 3500], else False.
    Side effects:
        None.
    """

    try:
        elo = int(value)

        return 400 <= elo <= 3500

    except Exception:
        return False


def is_valid_moves(moves_str):
    """Validate a UCI move sequence from the initial chess position.

    Parameters:
        moves_str: Space-separated UCI move string.
    Returns:
        True if the sequence has enough moves and every move is legal.
    Side effects:
        None.
    """

    if not isinstance(moves_str, str):
        return False

    moves = moves_str.strip().split()

    if len(moves) < MIN_MOVES:
        return False

    board = chess.Board()

    try:

        for move_uci in moves:

            move = chess.Move.from_uci(move_uci)

            if move not in board.legal_moves:
                return False

            board.push(move)

        return True

    except Exception:
        return False


def extract_game_type(event):
    """Extract the supported Lichess speed category from an event name.

    Parameters:
        event: PGN Event header value.
    Returns:
        "Bullet", "Blitz", "Rapid", or None when no supported type is found.
    Side effects:
        None.
    """

    if not isinstance(event, str):
        return None

    for game_type in VALID_TIME_CONTROLS:

        if game_type.lower() in event.lower():
            return game_type

    return None


# =========================================================
# CLEANING
# =========================================================

def clean_games():
    """Clean parsed game metadata and save the validated game CSV.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Reads INPUT_CSV, validates rows with python-chess, and writes OUTPUT_CSV.
    """

    if OUTPUT_CSV.exists():
        print(f"[INFO] Clean dataset already exists: {OUTPUT_CSV}")
        return

    print("[INFO] Loading games metadata...")

    df = pd.read_csv(INPUT_CSV)

    original_size = len(df)

    print(f"[INFO] Original games: {original_size}")

    # =====================================================
    # REMOVE EMPTY MOVES
    # =====================================================

    print("[INFO] Removing empty games...")

    df = df[
        df["MovesUCI"].notna()
    ]

    # =====================================================
    # VALID RESULTS
    # =====================================================

    print("[INFO] Filtering valid results...")

    df = df[
        df["Result"].isin(VALID_RESULTS)
    ]

    # =====================================================
    # VALID ELO
    # =====================================================

    print("[INFO] Filtering valid Elo ratings...")

    df = df[
        df["WhiteElo"].apply(is_valid_elo)
    ]

    df = df[
        df["BlackElo"].apply(is_valid_elo)
    ]

    # =====================================================
    # GAME TYPE
    # =====================================================

    print("[INFO] Extracting game types...")

    df["GameType"] = df["Event"].apply(
        extract_game_type
    )

    df = df[
        df["GameType"].notna()
    ]

    # =====================================================
    # VALID MOVES
    # =====================================================

    print("[INFO] Validating legal moves...")

    valid_mask = []

    for idx, moves in enumerate(df["MovesUCI"]):

        if idx % 100 == 0:
            print(
                f"[INFO] Validating game "
                f"{idx}/{len(df)}"
            )

        valid_mask.append(
            is_valid_moves(moves)
        )

    df = df[valid_mask]

    # =====================================================
    # CLEAN TYPES
    # =====================================================

    df["WhiteElo"] = df["WhiteElo"].astype(int)
    df["BlackElo"] = df["BlackElo"].astype(int)
    df["NumMoves"] = df["NumMoves"].astype(int)

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    print("[INFO] Removing duplicate games...")

    df = df.drop_duplicates(
        subset=["MovesUCI"]
    )

    # =====================================================
    # FINAL STATS
    # =====================================================

    final_size = len(df)

    removed = original_size - final_size

    print("[INFO] Cleaning complete.")
    print(f"[INFO] Final games: {final_size}")
    print(f"[INFO] Removed games: {removed}")

    print("\n[INFO] Game type distribution:")

    print(
        df["GameType"]
        .value_counts()
    )

    print("\n[INFO] Average Elo:")

    avg_elo = (
        df["WhiteElo"].mean()
        + df["BlackElo"].mean()
    ) / 2

    print(round(avg_elo, 2))

    # =====================================================
    # SAVE
    # =====================================================

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(f"\n[INFO] Saved cleaned dataset:")
    print(f"[INFO] {OUTPUT_CSV}")


# =========================================================
# MAIN
# =========================================================

def main():
    """Run game cleaning as a script entry point.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Delegates to clean_games().
    """

    clean_games()


if __name__ == "__main__":
    main()

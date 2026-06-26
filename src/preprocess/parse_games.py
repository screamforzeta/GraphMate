"""Parse sampled Lichess PGN games into tabular metadata.

Purpose:
    Read a sampled PGN file, extract game metadata, UCI move sequences, clock
    values, and derived move-time estimates.
Input:
    data/raw/games/sample_1000_games.pgn
Output:
    data/processed/games/games_metadata.csv
Run:
    python3 src/preprocess/parse_games.py
"""

from pathlib import Path
import chess.pgn
import pandas as pd
from tqdm import tqdm

#Config
INPUT_PGN = Path("data/raw/games/sample_1000_games.pgn")
OUTPUT_DIR = Path("data/processed/games")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "games_metadata.csv"

MAX_GAMES = None #Parse all games

# Helpers

def extract_clock_times(node_comment: str):
    """Extract remaining clock time from a PGN node comment.

    Example:
        [%clk 0:04:32]
    Parameters:
        node_comment: PGN comment string attached to a move node.
    Returns:
        Remaining seconds as float, or None if no valid clock is present.
    Side effects:
        None.
    """

    if "[%clk " not in node_comment:
        return None

    try:

        start = node_comment.index("[%clk ") + len("[%clk ")
        end = node_comment.index("]", start)

        clock_str = node_comment[start:end]

        parts = clock_str.split(":")

        if len(parts) == 3:

            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])

            total_seconds = (
                hours * 3600
                + minutes * 60
                + seconds
            )

            return total_seconds

    except Exception:
        return None

    return None


def compute_move_times(clock_values):
    """Convert remaining clock values into per-move think times.

    Example:
        move_time[i] = clock[i - 1] - clock[i]
    Parameters:
        clock_values: List of remaining clock values in seconds or None.
    Returns:
        List of think times in seconds, using None for missing/invalid values.
    Side effects:
        None.
    """

    move_times = []

    for i in range(1, len(clock_values)):

        prev_clock = clock_values[i - 1]
        curr_clock = clock_values[i]

        if prev_clock is None or curr_clock is None:
            move_times.append(None)
            continue

        think_time = prev_clock - curr_clock

        if think_time < 0:
            move_times.append(None)
        else:
            move_times.append(think_time)

    return move_times


# =========================================================
# PARSE SINGLE GAME
# =========================================================

def parse_single_game(game):
    """Extract metadata and move-level fields from one PGN game.

    Parameters:
        game: chess.pgn.Game instance returned by python-chess.
    Returns:
        Dictionary with headers, UCI moves, clock values, move times, and
        average move time.
    Side effects:
        None.
    """

    headers = game.headers

    white_elo = headers.get("WhiteElo")
    black_elo = headers.get("BlackElo")
    time_control = headers.get("TimeControl")
    event = headers.get("Event")
    result = headers.get("Result")

    board = game.board()

    moves_uci = []
    clock_values = []

    node = game

    while node.variations:

        next_node = node.variation(0)

        move = next_node.move

        moves_uci.append(move.uci())

        # Extract optional Lichess clock annotations from the move comment.
        comment = next_node.comment

        clock_sec = extract_clock_times(comment)

        clock_values.append(clock_sec)

        board.push(move)

        node = next_node

    # =========================================================
    # THINK TIMES
    # =========================================================

    move_times = compute_move_times(clock_values)

    valid_move_times = [
        t for t in move_times
        if t is not None
    ]

    avg_move_time = (
        sum(valid_move_times) / len(valid_move_times)
        if valid_move_times
        else None
    )

    return {
        "Event": event,
        "Result": result,
        "WhiteElo": white_elo,
        "BlackElo": black_elo,
        "TimeControl": time_control,
        "NumMoves": len(moves_uci),
        "MovesUCI": " ".join(moves_uci),
        "ClockValues": str(clock_values),
        "MoveTimes": str(move_times),
        "AvgMoveTime": avg_move_time,
    }


# =========================================================
# MAIN PARSER
# =========================================================

def parse_games():
    """Parse all sampled PGN games and save the metadata CSV.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Reads INPUT_PGN and writes OUTPUT_CSV unless the output already exists.
    """

    if OUTPUT_CSV.exists():
        print(f"[INFO] Metadata already exists: {OUTPUT_CSV}")
        return

    print("[INFO] Starting PGN parsing...")

    parsed_games = []

    with open(INPUT_PGN, encoding="utf-8") as pgn_file:

        game_counter = 0

        while True:

            game = chess.pgn.read_game(pgn_file)

            if game is None:
                break

            try:

                parsed = parse_single_game(game)

                parsed_games.append(parsed)

                game_counter += 1

                if game_counter % 100 == 0:
                    print(
                        f"[INFO] Parsed {game_counter} games"
                    )

                if (
                    MAX_GAMES is not None
                    and game_counter >= MAX_GAMES
                ):
                    break

            except Exception as e:

                print(
                    f"[WARNING] Failed to parse game "
                    f"{game_counter}: {e}"
                )

    # =========================================================
    # SAVE
    # =========================================================

    print("[INFO] Creating DataFrame...")

    df = pd.DataFrame(parsed_games)

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"[INFO] Saved metadata for "
        f"{len(df)} games"
    )

    print(f"[INFO] Output: {OUTPUT_CSV}")


# =========================================================
# MAIN
# =========================================================

def main():
    """Run PGN parsing as a script entry point.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Delegates to parse_games().
    """

    parse_games()


if __name__ == "__main__":
    main()

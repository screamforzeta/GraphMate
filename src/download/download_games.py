"""Stream a small PGN sample from the Lichess game archive.

Purpose:
    Read the compressed Lichess monthly PGN archive as a stream and extract a
    limited number of games without downloading the full archive to disk.
Input:
    Remote .pgn.zst file from the Lichess standard rated database.
Output:
    data/raw/games/sample_1000_games.pgn
Run:
    python3 src/download/download_games.py
"""

from pathlib import Path
import requests
import zstandard as zstd
import io

# =========================================================
# CONFIG
# =========================================================

INPUT_URL = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_2025-01.pgn.zst"
)

OUTPUT_DIR = Path("data/raw/games")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PGN = OUTPUT_DIR / "sample_1000_games.pgn"

MAX_GAMES = 1000
CHUNK_SIZE = 1024 * 64


# =========================================================
# STREAM DOWNLOAD + EXTRACT
# =========================================================

def stream_extract_games():
    """Download, decompress, and save the configured PGN sample.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Opens a streaming HTTP response, writes OUTPUT_PGN, and logs progress.
    """

    if OUTPUT_PGN.exists():
        print(f"[INFO] Sample already exists: {OUTPUT_PGN}")
        return

    print("[INFO] Starting streaming download + extraction...")

    response = requests.get(INPUT_URL, stream=True)
    response.raise_for_status()

    dctx = zstd.ZstdDecompressor()

    compressed_stream = response.raw

    with dctx.stream_reader(compressed_stream) as reader:

        text_stream = io.TextIOWrapper(
            reader,
            encoding="utf-8",
            errors="ignore"
        )

        game_count = 0
        current_game = []

        with open(OUTPUT_PGN, "w", encoding="utf-8") as out_file:

            for line in text_stream:

                current_game.append(line)

                stripped = line.strip()

                # Detect real game boundaries from terminal PGN result tokens.
                if (
                    stripped.endswith("1-0")
                    or stripped.endswith("0-1")
                    or stripped.endswith("1/2-1/2")
                ):

                    game_text = "".join(current_game)

                    # Keep only chunks that look like complete PGN games.
                    if (
                        "[Event " in game_text
                        and "[Site " in game_text
                        and "1." in game_text
                    ):

                        out_file.write(game_text)
                        out_file.write("\n\n")

                        game_count += 1

                        if game_count % 100 == 0:
                            print(
                                f"[INFO] Extracted "
                                f"{game_count} games"
                            )

                        if game_count >= MAX_GAMES:

                            print(
                                f"[INFO] Reached target "
                                f"of {MAX_GAMES} games"
                            )

                            return

                    current_game = []


# =========================================================
# MAIN
# =========================================================

def main():
    """Run the PGN streaming extraction step.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Creates data/raw/games/ and writes the sampled PGN file if missing.
    """

    stream_extract_games()

    print(f"[INFO] Sample saved to: {OUTPUT_PGN}")


if __name__ == "__main__":
    main()

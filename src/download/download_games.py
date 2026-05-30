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

                if line.strip() == "":

                    game_text = "".join(current_game)

                    if "[Event " in game_text and "1." in game_text:

                        out_file.write(game_text)
                        out_file.write("\n")

                        game_count += 1

                        if game_count % 100 == 0:
                            print(f"[INFO] Extracted {game_count} games")

                        if game_count >= MAX_GAMES:

                            print(
                                f"[INFO] Reached target of "
                                f"{MAX_GAMES} games"
                            )

                            return

                    current_game = []


# =========================================================
# MAIN
# =========================================================

def main():

    stream_extract_games()

    print(f"[INFO] Sample saved to: {OUTPUT_PGN}")


if __name__ == "__main__":
    main()
from pathlib import Path
import io
import zstandard as zstd
import pandas as pd
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================

INPUT_FILE = Path(
    "data/raw/puzzles/lichess_puzzles.csv.zst"
)

OUTPUT_DIR = Path("data/processed/puzzles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "mate_puzzles.csv"

VALID_THEMES = {
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
}

CHUNK_SIZE = 100_000


# =========================================================
# HELPERS
# =========================================================

def extract_mate_theme(themes: str):

    for theme in VALID_THEMES:
        if theme in themes:
            return theme

    return None


# =========================================================
# PREPROCESS
# =========================================================

def preprocess_puzzles():

    if OUTPUT_FILE.exists():
        print(f"[INFO] Processed file already exists: {OUTPUT_FILE}")
        return

    print("[INFO] Starting puzzle preprocessing...")

    dctx = zstd.ZstdDecompressor()

    filtered_chunks = []

    with open(INPUT_FILE, "rb") as compressed:

        with dctx.stream_reader(compressed) as reader:

            text_stream = io.TextIOWrapper(
                reader,
                encoding="utf-8"
            )

            csv_iter = pd.read_csv(
                text_stream,
                chunksize=CHUNK_SIZE
            )

            for chunk_idx, chunk in enumerate(csv_iter):

                print(f"[INFO] Processing chunk {chunk_idx}")

                # =================================================
                # FILTER MATE PUZZLES
                # =================================================

                filtered = chunk[
                    chunk["Themes"].astype(str).apply(
                        lambda x: any(
                            theme in x
                            for theme in VALID_THEMES
                        )
                    )
                ].copy()

                # =================================================
                # EXTRACT MATE DEPTH
                # =================================================

                filtered["MateTheme"] = filtered["Themes"].apply(
                    extract_mate_theme
                )

                filtered["MateDepth"] = (
                    filtered["MateTheme"]
                    .str.replace("mateIn", "", regex=False)
                    .astype(int)
                )

                # =================================================
                # KEEP ONLY USEFUL COLUMNS
                # =================================================

                columns_to_keep = [
                    "PuzzleId",
                    "FEN",
                    "Moves",
                    "Rating",
                    "Themes",
                    "MateTheme",
                    "MateDepth",
                ]

                filtered = filtered[columns_to_keep]

                filtered_chunks.append(filtered)

    # =========================================================
    # CONCAT
    # =========================================================

    print("[INFO] Concatenating filtered chunks...")

    final_df = pd.concat(
        filtered_chunks,
        ignore_index=True
    )

    # =========================================================
    # SAVE
    # =========================================================

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"[INFO] Saved {len(final_df)} mate puzzles to:"
    )
    print(f"[INFO] {OUTPUT_FILE}")


# =========================================================
# MAIN
# =========================================================

def main():

    preprocess_puzzles()


if __name__ == "__main__":
    main()
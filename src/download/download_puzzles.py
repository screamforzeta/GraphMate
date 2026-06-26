"""Download the Lichess puzzle database archive.

Purpose:
    Fetch the compressed Lichess puzzle CSV archive used by the puzzle
    preprocessing pipeline.
Input:
    Remote .csv.zst file from the Lichess database endpoint.
Output:
    data/raw/puzzles/lichess_puzzles.csv.zst
Run:
    python3 src/download/download_puzzles.py
"""

from pathlib import Path
import requests
from tqdm import tqdm

PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

OUTPUT_DIR = Path("data/raw/puzzles")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "lichess_puzzles.csv.zst"


def download_file(url: str, output_path: Path):
    """Stream a remote file to disk with a tqdm progress bar.

    Parameters:
        url: Source URL to download.
        output_path: Destination file path.
    Returns:
        None.
    Side effects:
        Performs an HTTP GET request and writes bytes to output_path.
    """

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:

        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def main():
    """Download the puzzle archive unless it already exists.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Creates data/raw/puzzles/ and writes the compressed puzzle archive.
    """

    if OUTPUT_FILE.exists():
        print(f"[INFO] File already exists: {OUTPUT_FILE}")
        return

    print("[INFO] Downloading Lichess puzzle database...")
    download_file(PUZZLE_URL, OUTPUT_FILE)

    print("[INFO] Download complete.")


if __name__ == "__main__":
    main()

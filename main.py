"""Run the project data and graph generation pipeline.

Purpose:
    Download Lichess source data, preprocess chess puzzles and games, build
    move encoders, extract graph features, and serialize PyG datasets.
Input:
    Remote Lichess archives and intermediate CSV files under data/.
Output:
    Processed CSV datasets, move vocabulary artifacts, and PyG graph files.
Run:
    python3 main.py
"""

# Download raw data.
from src.data.download import download_puzzles
from src.data.download import download_games

# Preprocess raw puzzle and PGN data.
from src.data.preprocess import preprocess_puzzles
from src.data.preprocess import parse_games

# Validate and clean preprocessed data.
from src.data.preprocess import clean_games
from src.data.preprocess import clean_puzzles

# Partition and prepare final CSV datasets.
from src.data.preprocess import prepare_puzzles_dataset
from src.data.preprocess import prepare_games_dataset

# Graph construction and encoding.
from src.graph import move_encoder
from src.graph import node_features
from src.graph import edge_features
from src.graph import graph_builder
from src.graph import pyg_dataset

# The Streamlit debugger is a standalone app. Run it separately with:
# streamlit run src/graph/debug/streamlit_graph_debugger.py
# Do not import it here, because Streamlit executes UI code at import time.


def main():
    """Execute every pipeline stage in the required order.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Downloads source data if missing, writes CSV datasets, writes move
        encoder JSON files, and writes serialized PyG graph datasets.
    """

    # Data section.
    download_puzzles.main()
    download_games.main()
    preprocess_puzzles.main()
    parse_games.main()
    clean_games.main()
    clean_puzzles.main()
    prepare_puzzles_dataset.main()
    prepare_games_dataset.main()

    # Graph section.
    move_encoder.main()
    pyg_dataset.generate_sharded_datasets(
        overwrite=True,
    )

    # Manual feature/debug tests and Streamlit visualization are excluded.


if __name__ == "__main__":
    main()

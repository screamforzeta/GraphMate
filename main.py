#Data pipeline for downloading, preprocessing, validating, and preparing chess puzzles and games datasets for training a chess engine.

#Download RAW data
from src.download import download_puzzles
from src.download import download_games
#Preprocess RAW data
from src.preprocess import preprocess_puzzles
from src.preprocess import parse_games
#Validate and clean preprocessed data
from src.preprocess import clean_games
from src.preprocess import clean_puzzles
#Partition and prepare final datasets
from src.preprocess import prepare_puzzles_dataset
from src.preprocess import prepare_games_dataset

#Graph construction and encoding

#Node feature construction for chess positions, used as input to a graph neural network.
from src.graph import move_encoder
#Node feature construction for chess positions, used as input to a graph neural network.
from src.graph import node_features
#Edge feature construction for chess positions, used as input to a graph neural network.
from src.graph import edge_features
#Graph building utilities to construct graph data structures from chess positions and moves.
from src.graph import graph_builder
#PyTorch Geometric dataset construction for chess graph data, preparing it for training GNN models.
from src.graph import pyg_dataset

#Testing and visualization scripts for inspecting graph structures and features.
from src.graph.debug import streamlit_graph_debugger as visualize_graph


def main():

    #Data section
    download_puzzles.main()
    download_games.main()
    preprocess_puzzles.main()
    parse_games.main()
    clean_games.main()
    clean_puzzles.main()
    prepare_puzzles_dataset.main()
    prepare_games_dataset.main()

    #Graph section
    move_encoder.main()
    node_features.main()
    edge_features.main()
    graph_builder.main()
    pyg_dataset.main()

    #Testing and visualization section
    #visualize_graph.main()


if __name__ == "__main__":
    main()
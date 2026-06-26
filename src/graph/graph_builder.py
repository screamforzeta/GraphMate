"""Build complete PyTorch Geometric graph objects for chess puzzles.

Purpose:
    Convert one puzzle position and target move into a PyG Data object.
It combines:
    - node features
    - edge features
    - graph-level features
    - target encoding

The resulting graph is represented as:
    torch_geometric.data.Data
and will later be used directly by the GNN/GAT model.

This module acts as the bridge between:
    CSV datasets
and
    Graph Neural Networks.
Input:
    FEN string, target UCI move, and move encoder dictionary.
Output:
    torch_geometric.data.Data containing:
    x
    edge_index
    edge_attr
    y
    global_features
Run:
    python3 src/graph/graph_builder.py
"""

import json
from pathlib import Path

import chess
import torch
from torch_geometric.data import Data

from src.graph.node_features import (
    extract_node_features,
)

from src.graph.edge_features import (
    extract_edge_features,
)

# =========================================================
# GLOBAL FEATURE DIMENSION
# =========================================================

"""
Graph-level feature layout:

[0] side_to_move
[1] is_check
[2] fullmove_number_normalized
[3] halfmove_clock_normalized

TOTAL = 4
"""

GLOBAL_FEATURE_DIM = 4


# =========================================================
# GLOBAL FEATURES
# =========================================================

def extract_global_features(board):
    """
    Extract graph-level chess features.

    Parameters
    ----------
    board : chess.Board

    Returns
    -------
    torch.FloatTensor
        shape = [GLOBAL_FEATURE_DIM]
    """

    # =============================================
    # SIDE TO MOVE
    # =============================================

    side_to_move = (
        1.0 if board.turn == chess.WHITE
        else 0.0
    )

    # =============================================
    # CHECK STATUS
    # =============================================

    is_check = (
        1.0 if board.is_check()
        else 0.0
    )

    # =============================================
    # MOVE COUNTERS
    # =============================================

    fullmove_number = (
        min(board.fullmove_number, 200)
        / 200.0
    )

    halfmove_clock = (
        min(board.halfmove_clock, 100)
        / 100.0
    )

    features = torch.tensor(
        [
            side_to_move,
            is_check,
            fullmove_number,
            halfmove_clock,
        ],
        dtype=torch.float
    )

    return features


# =========================================================
# TARGET ENCODING
# =========================================================

def encode_target_move(
    target_move,
    move_to_idx,
):
    """
    Convert move string into target index.

    Parameters
    ----------
    target_move : str
    move_to_idx : dict

    Returns
    -------
    torch.LongTensor

    Side effects
    ------------
    Raises ValueError when target_move is not in move_to_idx.
    """

    if target_move not in move_to_idx:

        raise ValueError(
            f"Unknown target move: "
            f"{target_move}"
        )

    move_idx = move_to_idx[target_move]

    y = torch.tensor(
        move_idx,
        dtype=torch.long
    )

    return y


# =========================================================
# BUILD GRAPH
# =========================================================

def build_graph(
    fen,
    target_move,
    move_to_idx,
):
    """
    Build complete PyG graph object.

    Parameters
    ----------
    fen : str
    target_move : str
    move_to_idx : dict

    Returns
    -------
    torch_geometric.data.Data

    Side effects
    ------------
    Raises errors from chess parsing, feature extraction, or target encoding.
    """

    # =============================================
    # BOARD
    # =============================================

    board = chess.Board(fen)

    # =============================================
    # NODE FEATURES
    # =============================================

    x = extract_node_features(fen)

    # =============================================
    # EDGE FEATURES
    # =============================================

    edge_index, edge_attr = (
        extract_edge_features(fen)
    )

    # =============================================
    # GLOBAL FEATURES
    # =============================================

    global_features = (
        extract_global_features(board)
    )

    # =============================================
    # TARGET
    # =============================================

    y = encode_target_move(
        target_move,
        move_to_idx,
    )

    # =============================================
    # CREATE PYG DATA OBJECT
    # =============================================

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    # =============================================
    # ADD CUSTOM ATTRIBUTES
    # =============================================

    graph.global_features = (
        global_features
    )

    graph.fen = fen

    graph.target_move = target_move

    return graph


# =========================================================
# LOAD MOVE ENCODER
# =========================================================

def load_move_encoder(
    path="artifacts/move_to_idx.json"
):
    """
    Load move encoder vocabulary.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict

    Side effects
    ------------
    Reads the JSON encoder file from disk.
    """

    with open(path, "r") as f:

        move_to_idx = json.load(f)

    return move_to_idx


# =========================================================
# TEST
# =========================================================

def main():
    """
    Run a simple local graph construction test.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side effects
    ------------
    Loads the move encoder JSON and prints a sample graph summary.
    """

    # =============================================
    # TEST POSITION
    # =============================================

    test_fen = (
        "r1bqkbnr/pppp1ppp/2n5/4p3/"
        "4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )

    test_move = "f3e5"

    # =============================================
    # LOAD ENCODER
    # =============================================

    move_to_idx = (
        load_move_encoder()
    )

    # =============================================
    # BUILD GRAPH
    # =============================================

    graph = build_graph(
        test_fen,
        test_move,
        move_to_idx,
    )

    # =============================================
    # LOGS
    # =============================================

    print("=" * 50)
    print("GRAPH BUILDER TEST")
    print("=" * 50)

    print("\n[INFO] Graph summary:")
    print(graph)

    print("\n[INFO] Node feature shape:")
    print(graph.x.shape)

    print("\n[INFO] Edge index shape:")
    print(graph.edge_index.shape)

    print("\n[INFO] Edge attr shape:")
    print(graph.edge_attr.shape)

    print("\n[INFO] Target:")
    print(graph.y)

    print("\n[INFO] Global features:")
    print(graph.global_features)

    print("\n[INFO] FEN:")
    print(graph.fen)

    print("\n[INFO] Target move:")
    print(graph.target_move)


if __name__ == "__main__":
    main()

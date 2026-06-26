"""Interactive Streamlit debugger for chess graph samples.

Purpose:
    Visualize one generated puzzle graph together with its chessboard,
    edge overlays, node metadata, and raw puzzle fields.
Input:
    data/final/puzzles/train.csv and artifacts/move_to_idx.json.
Output:
    Streamlit UI with graph/chessboard visualization; no pipeline files are
    written by this module.
Run:
    streamlit run src/graph/debug/streamlit_graph_debugger.py

Notes:
    This is a standalone app and must not be imported by main.py.
"""

# =========================================================
# PROJECT ROOT FIX
# =========================================================

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# =========================================================
# IMPORTS
# =========================================================

import json
import random

import chess
import chess.svg

import pandas as pd
import streamlit as st

from streamlit.components.v1 import html

from streamlit_agraph import (
    agraph,
    Node,
    Edge,
    Config,
)

from src.graph.graph_builder import (
    build_graph,
)

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = Path(
    "data/final/puzzles/train.csv"
)

MOVE_ENCODER_PATH = Path(
    "artifacts/move_to_idx.json"
)

BOARD_SIZE = 700

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Chess GNN Debugger",
    layout="wide",
)

# =========================================================
# EDGE FEATURES
# =========================================================

EDGE_FEATURE_NAMES = [
    "legal_move",
    "attack",
    "defend",
    "pin",
    "check_line",
]

# =========================================================
# SOFT COLORS
# =========================================================

EDGE_COLORS = {
    "legal_move": "#5fa8ff88",
    "attack": "#ff666688",
    "defend": "#66ff9988",
    "pin": "#ffd16688",
    "check_line": "#c77dff88",
}

GRAPH_EDGE_COLORS = {
    "legal_move": "#5fa8ff",
    "attack": "#ff6666",
    "defend": "#66ff99",
    "pin": "#ffd166",
    "check_line": "#c77dff",
}

# =========================================================
# PIECE SVG URLs
# =========================================================

PIECE_IMAGES = {

    "P": "https://upload.wikimedia.org/wikipedia/commons/4/45/Chess_plt45.svg",
    "N": "https://upload.wikimedia.org/wikipedia/commons/7/70/Chess_nlt45.svg",
    "B": "https://upload.wikimedia.org/wikipedia/commons/b/b1/Chess_blt45.svg",
    "R": "https://upload.wikimedia.org/wikipedia/commons/7/72/Chess_rlt45.svg",
    "Q": "https://upload.wikimedia.org/wikipedia/commons/1/15/Chess_qlt45.svg",
    "K": "https://upload.wikimedia.org/wikipedia/commons/4/42/Chess_klt45.svg",

    "p": "https://upload.wikimedia.org/wikipedia/commons/c/c7/Chess_pdt45.svg",
    "n": "https://upload.wikimedia.org/wikipedia/commons/e/ef/Chess_ndt45.svg",
    "b": "https://upload.wikimedia.org/wikipedia/commons/9/98/Chess_bdt45.svg",
    "r": "https://upload.wikimedia.org/wikipedia/commons/f/ff/Chess_rdt45.svg",
    "q": "https://upload.wikimedia.org/wikipedia/commons/4/47/Chess_qdt45.svg",
    "k": "https://upload.wikimedia.org/wikipedia/commons/f/f0/Chess_kdt45.svg",
}

# =========================================================
# HELPERS
# =========================================================

@st.cache_data
def load_dataset():
    """Load the puzzle split used by the debugger.

    Parameters:
        None.
    Returns:
        Pandas DataFrame loaded from DATASET_PATH.
    Side effects:
        Reads the CSV file from disk and caches the result in Streamlit.
    """

    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_move_encoder():
    """Load the move vocabulary used to encode selected puzzle targets.

    Parameters:
        None.
    Returns:
        Dictionary mapping UCI moves to class indices.
    Side effects:
        Reads MOVE_ENCODER_PATH from disk and caches the result in Streamlit.
    """

    with open(
        MOVE_ENCODER_PATH,
        "r"
    ) as f:

        return json.load(f)


def edge_type_from_features(features):
    """Return the first active edge type name from an edge feature vector.

    Parameters:
        features: Edge feature vector ordered like EDGE_FEATURE_NAMES.
    Returns:
        Edge type name, or "unknown" when no known feature is active.
    Side effects:
        None.
    """

    for idx, name in enumerate(
        EDGE_FEATURE_NAMES
    ):

        if features[idx] == 1:
            return name

    return "unknown"


def build_svg_arrows(
    graph,
    show_legal,
    show_attack,
    show_defend,
    show_pin,
    show_check,
):
    """Create chess.svg arrows for enabled edge types.

    Parameters:
        graph: PyG Data object with edge_index and edge_attr.
        show_legal: Whether legal-move edges are displayed.
        show_attack: Whether attack edges are displayed.
        show_defend: Whether defend edges are displayed.
        show_pin: Whether pin edges are displayed.
        show_check: Whether check-line edges are displayed.
    Returns:
        List of chess.svg.Arrow objects.
    Side effects:
        None.
    """

    arrows = []

    num_edges = (
        graph.edge_index.shape[1]
    )

    for edge_idx in range(num_edges):

        src = (
            graph.edge_index[0][edge_idx]
            .item()
        )

        dst = (
            graph.edge_index[1][edge_idx]
            .item()
        )

        features = (
            graph.edge_attr[edge_idx]
            .tolist()
        )

        edge_type = edge_type_from_features(
            features
        )

        # =================================================
        # FILTERS
        # =================================================

        if (
            edge_type == "legal_move"
            and not show_legal
        ):
            continue

        if (
            edge_type == "attack"
            and not show_attack
        ):
            continue

        if (
            edge_type == "defend"
            and not show_defend
        ):
            continue

        if (
            edge_type == "pin"
            and not show_pin
        ):
            continue

        if (
            edge_type == "check_line"
            and not show_check
        ):
            continue

        arrows.append(

            chess.svg.Arrow(
                tail=src,
                head=dst,
                color=EDGE_COLORS[
                    edge_type
                ],
            )
        )

    return arrows

# =========================================================
# LOAD DATA
# =========================================================

df = load_dataset()

move_to_idx = load_move_encoder()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Chess GNN Debugger"
)

# =========================================================
# RANDOM PUZZLE
# =========================================================

if st.sidebar.button(
    "Load Random Puzzle"
):

    random_idx = random.randint(
        0,
        len(df) - 1
    )

    st.session_state[
        "current_idx"
    ] = random_idx

if "current_idx" not in st.session_state:

    st.session_state[
        "current_idx"
    ] = 0

# =========================================================
# PUZZLE SELECTOR
# =========================================================

current_idx = st.sidebar.slider(
    "Puzzle Index",
    0,
    len(df) - 1,
    st.session_state["current_idx"]
)

sample = df.iloc[current_idx]

# =========================================================
# VIEW MODE
# =========================================================

st.sidebar.subheader(
    "Visualization Mode"
)

view_mode = st.sidebar.radio(
    "Select View",
    [
        "Chessboard",
        "Graph",
        "Both",
    ]
)

# =========================================================
# EDGE FILTERS
# =========================================================

st.sidebar.subheader(
    "Edge Filters"
)

show_legal = st.sidebar.checkbox(
    "Legal Moves",
    value=True
)

show_attack = st.sidebar.checkbox(
    "Attack Edges",
    value=False
)

show_defend = st.sidebar.checkbox(
    "Defend Edges",
    value=False
)

show_pin = st.sidebar.checkbox(
    "Pin Edges",
    value=True
)

show_check = st.sidebar.checkbox(
    "Check Lines",
    value=True
)

# =========================================================
# TITLE
# =========================================================

st.title(
    "Chess GNN Graph Debugger"
)

# =========================================================
# INFO
# =========================================================

info1, info2, info3 = st.columns(3)

info1.metric(
    "Puzzle ID",
    sample.PuzzleId
)

info2.metric(
    "Mate Depth",
    sample.MateDepth
)

info3.metric(
    "Target Move",
    sample.TargetMove
)

# =========================================================
# BUILD GRAPH
# =========================================================

graph = build_graph(
    fen=sample.FEN,
    target_move=sample.TargetMove,
    move_to_idx=move_to_idx,
)

board = chess.Board(sample.FEN)

# =========================================================
# BUILD ARROWS
# =========================================================

board_arrows = build_svg_arrows(
    graph=graph,

    show_legal=show_legal,
    show_attack=show_attack,
    show_defend=show_defend,
    show_pin=show_pin,
    show_check=show_check,
)

# =========================================================
# STATS
# =========================================================

stats1, stats2, stats3 = st.columns(3)

stats1.metric(
    "Nodes",
    graph.x.shape[0]
)

stats2.metric(
    "Edges",
    graph.edge_index.shape[1]
)

stats3.metric(
    "Node Features",
    graph.x.shape[1]
)

# =========================================================
# CHESSBOARD VIEW
# =========================================================

if view_mode in ["Chessboard", "Both"]:

    st.subheader(
        "Chessboard View"
    )

    board_svg = chess.svg.board(
        board=board,
        size=BOARD_SIZE,
        arrows=board_arrows,
        coordinates=True,
    )

    html(
        board_svg,
        height=BOARD_SIZE + 50,
    )

# =========================================================
# GRAPH VIEW
# =========================================================

if view_mode in ["Graph", "Both"]:

    st.subheader(
        "Graph View"
    )

    nodes = []

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        square_name = chess.square_name(
            square
        )

        row = (
            7
            - chess.square_rank(square)
        )

        col = chess.square_file(square)

        # =================================================
        # PIECE IMAGE
        # =================================================

        if piece is not None:

            piece_symbol = piece.symbol()

            image_url = PIECE_IMAGES[
                piece_symbol
            ]

            nodes.append(

                Node(
                    id=square,

                    label=square_name,

                    shape="image",

                    image=image_url,

                    size=35,

                    font={
                        "size": 18,
                        "color": "#ffffff",
                        "strokeWidth": 4,
                    },

                    x=col * 140,

                    y=row * 140,

                    physics=False,
                )
            )

        else:

            nodes.append(

                Node(
                    id=square,

                    label=square_name,

                    shape="dot",

                    size=6,

                    color="#888888",

                    font={
                        "size": 16,
                        "color": "#cccccc",
                    },

                    x=col * 140,

                    y=row * 140,

                    physics=False,
                )
            )

    # =====================================================
    # EDGES
    # =====================================================

    edges = []

    num_edges = (
        graph.edge_index.shape[1]
    )

    for edge_idx in range(num_edges):

        src = (
            graph.edge_index[0][edge_idx]
            .item()
        )

        dst = (
            graph.edge_index[1][edge_idx]
            .item()
        )

        features = (
            graph.edge_attr[edge_idx]
            .tolist()
        )

        edge_type = edge_type_from_features(
            features
        )

        # =================================================
        # FILTERS
        # =================================================

        if (
            edge_type == "legal_move"
            and not show_legal
        ):
            continue

        if (
            edge_type == "attack"
            and not show_attack
        ):
            continue

        if (
            edge_type == "defend"
            and not show_defend
        ):
            continue

        if (
            edge_type == "pin"
            and not show_pin
        ):
            continue

        if (
            edge_type == "check_line"
            and not show_check
        ):
            continue

        edges.append(

            Edge(
                source=src,
                target=dst,

                color=GRAPH_EDGE_COLORS[
                    edge_type
                ],

                title=edge_type,

                width=1.5,

                smooth=False,

                arrows="to",
            )
        )

    # =====================================================
    # CONFIG
    # =====================================================

    config = Config(

        width="100%",
        height=1100,

        directed=True,

        physics=False,

        hierarchical=False,

        nodeHighlightBehavior=True,

        highlightColor="#ff0000",

        collapsible=False,
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    agraph(
        nodes=nodes,
        edges=edges,
        config=config,
    )

# =========================================================
# RAW DATA
# =========================================================

with st.expander(
    "Show Raw Puzzle Data"
):

    st.write(sample)

# =========================================================
# FEN
# =========================================================

with st.expander(
    "Show FEN"
):

    st.code(sample.FEN)

# =========================================================
# GRAPH SUMMARY
# =========================================================

with st.expander(
    "Graph Summary"
):

    st.write(graph)

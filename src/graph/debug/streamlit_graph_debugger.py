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
    load_move_encoder as load_graph_move_encoder,
)
from src.inference.model_a.chess_gat_inference import (
    EXPECTED_CHECKPOINT_PATH,
    MODEL_A_NAME,
    check_user_move,
    evaluate_mate_in_one_dataframe,
    run_error_analysis,
    target_rank_explanation,
    target_row_from_logits,
    write_error_analysis_report,
    is_legal_uci,
    is_mate_in_one_row,
    load_model_bundle,
    move_to_san,
    puzzle_row_to_graph,
    run_single_inference,
    verify_target_checkmate,
)
from src.inference.model_a.no_timing_multimodel import (
    MODEL_A2_CHECKPOINT_PATH,
    MODEL_A3_CHECKPOINT_PATH,
    MODEL_OPTIONS,
    error_analysis_record,
    evaluate_mate_in_one_modes,
    load_model_a2_bundle,
    load_model_a3_bundle,
    run_model_mode_inference,
)

# =========================================================
# CONFIG
# =========================================================

DATASET_PATHS = {
    "train": Path("data/final/puzzles/train.csv"),
    "validation": Path("data/final/puzzles/val.csv"),
    "test": Path("data/final/puzzles/test.csv"),
}

MOVE_ENCODER_PATH = Path(
    "artifacts/move_to_idx.json"
)

BOARD_SIZE = 700
USER_MOVE_COLOR = "#f59e0b"
MODEL_TOP1_COLOR = "#38bdf8"
GROUND_TRUTH_COLOR = "#22c55e"

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
def load_dataset(split):
    """Load the puzzle split used by the debugger.

    Parameters:
        None.
    Returns:
        Pandas DataFrame loaded from the selected split CSV.
    Side effects:
        Reads the CSV file from disk and caches the result in Streamlit.
    """

    return pd.read_csv(DATASET_PATHS[split])


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

    return load_graph_move_encoder(MOVE_ENCODER_PATH)


def edge_types_from_features(features):
    """Return all active edge type names from an edge feature vector.

    Parameters:
        features: Edge feature vector ordered like EDGE_FEATURE_NAMES.
    Returns:
        Active edge type names, or ["unknown"] when no known feature is active.
    Side effects:
        None.
    """

    edge_types = []

    for idx, name in enumerate(
        EDGE_FEATURE_NAMES
    ):

        if features[idx] == 1:
            edge_types.append(name)

    return edge_types or ["unknown"]


def edge_type_from_features(features):
    """Return the first active edge type for color compatibility.

    Parameters:
        features: Edge feature vector ordered like EDGE_FEATURE_NAMES.
    Returns:
        First active edge type name, or "unknown".
    Side effects:
        None.
    """

    return edge_types_from_features(features)[0]


def should_show_edge(
    edge_types,
    show_legal,
    show_attack,
    show_defend,
    show_pin,
    show_check,
):
    """Check whether any active edge type is enabled in the UI filters.

    Parameters:
        edge_types: Active edge type names for one graph edge.
        show_legal: Whether legal-move edges are displayed.
        show_attack: Whether attack edges are displayed.
        show_defend: Whether defend edges are displayed.
        show_pin: Whether pin edges are displayed.
        show_check: Whether check-line edges are displayed.
    Returns:
        True when at least one active edge type is enabled.
    Side effects:
        None.
    """

    enabled = {
        "legal_move": show_legal,
        "attack": show_attack,
        "defend": show_defend,
        "pin": show_pin,
        "check_line": show_check,
    }

    return any(
        enabled.get(edge_type, False)
        for edge_type in edge_types
    )


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

        edge_types = edge_types_from_features(
            features
        )

        if not should_show_edge(
            edge_types,
            show_legal,
            show_attack,
            show_defend,
            show_pin,
            show_check,
        ):
            continue

        edge_type = edge_types[0]

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


def build_move_arrow(move_uci, color):
    """Create one chess.svg arrow for a UCI move.

    Parameters:
        move_uci: Move in UCI notation.
        color: SVG-compatible arrow color.
    Returns:
        chess.svg.Arrow, or None when move_uci is invalid.
    Side effects:
        None.
    """

    try:
        move = chess.Move.from_uci(str(move_uci))
    except ValueError:
        return None
    return chess.svg.Arrow(
        tail=move.from_square,
        head=move.to_square,
        color=color,
    )


@st.cache_resource
def load_model_a_resource():
    """Load the frozen Model A bundle once per Streamlit process.

    Parameters:
        None.
    Returns:
        Tuple (bundle, error_message). Only one item is non-None.
    Side effects:
        Reads vocabulary and checkpoint files if available.
    """

    try:
        return load_model_bundle(), None
    except Exception as error:
        return None, str(error)


@st.cache_resource
def load_model_a2_resource():
    """Load the official Model A2 bundle once per Streamlit process."""

    try:
        return load_model_a2_bundle(), None
    except Exception as error:
        return None, str(error)


@st.cache_resource
def load_model_a3_resource():
    """Load the official Model A3 bundle once per Streamlit process."""

    try:
        return load_model_a3_bundle(), None
    except Exception as error:
        return None, str(error)


def reset_puzzle_state():
    """Clear per-puzzle user and inference state."""

    st.session_state["user_move"] = ""
    st.session_state["last_user_result"] = None
    st.session_state["revealed_solution"] = False
    st.session_state["model_result"] = None


def apply_filtered_index(new_index):
    """Set a new puzzle index and reset per-puzzle UI state."""

    st.session_state["current_idx"] = int(new_index)
    reset_puzzle_state()

# =========================================================
# LOAD DATA / SESSION DEFAULTS
# =========================================================

if "current_idx" not in st.session_state:
    st.session_state["current_idx"] = 0

if "selected_split" not in st.session_state:
    st.session_state["selected_split"] = "test"

if "session_counters" not in st.session_state:
    st.session_state["session_counters"] = {
        "attempted": 0,
        "human_correct": 0,
        "model_top1_correct": 0,
        "model_top3_correct": 0,
        "model_top5_correct": 0,
    }

for key, default in [
    ("user_move", ""),
    ("revealed_solution", False),
    ("model_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

selected_split = st.sidebar.selectbox(
    "Dataset Split",
    list(DATASET_PATHS),
    index=list(DATASET_PATHS).index(st.session_state["selected_split"]),
)

if selected_split != st.session_state["selected_split"]:
    st.session_state["selected_split"] = selected_split
    apply_filtered_index(0)

df_full = load_dataset(selected_split)

move_to_idx = load_move_encoder()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Chess GNN Debugger"
)

st.sidebar.caption(
    "Model A UI is diagnostic only. Do not use test-set browsing for future model selection."
)

# =========================================================
# FILTERS
# =========================================================

st.sidebar.subheader(
    "Puzzle Filters"
)

mate_in_one_only = st.sidebar.checkbox(
    "Mate-in-1 only",
    value=False,
)

theme_options = sorted(
    {
        theme
        for themes in df_full.get("Themes", pd.Series(dtype=str)).dropna()
        for theme in str(themes).split()
    }
)

selected_theme = st.sidebar.selectbox(
    "Theme",
    ["All"] + theme_options,
)

rating_min = int(df_full["Rating"].min()) if "Rating" in df_full else 0
rating_max = int(df_full["Rating"].max()) if "Rating" in df_full else 3000
rating_range = st.sidebar.slider(
    "Rating",
    min_value=rating_min,
    max_value=rating_max,
    value=(rating_min, rating_max),
)

df = df_full.copy()
if mate_in_one_only:
    df = df[
        df.apply(
            is_mate_in_one_row,
            axis=1,
        )
    ]
if selected_theme != "All":
    df = df[
        df["Themes"].fillna("").str.split().apply(
            lambda themes: selected_theme in themes
        )
    ]
if "Rating" in df:
    df = df[
        df["Rating"].between(
            rating_range[0],
            rating_range[1],
        )
    ]

df = df.reset_index(drop=False).rename(
    columns={"index": "source_row_index"}
)

if df.empty:
    st.warning("No puzzles match the selected filters.")
    st.stop()

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

    apply_filtered_index(random_idx)

nav_previous, nav_next = st.sidebar.columns(2)
if nav_previous.button("Previous"):
    apply_filtered_index(max(0, st.session_state["current_idx"] - 1))
if nav_next.button("Next"):
    apply_filtered_index(min(len(df) - 1, st.session_state["current_idx"] + 1))

# =========================================================
# PUZZLE SELECTOR
# =========================================================

current_idx = st.sidebar.slider(
    "Puzzle Index",
    0,
    len(df) - 1,
    min(st.session_state["current_idx"], len(df) - 1)
)

if current_idx != st.session_state["current_idx"]:
    apply_filtered_index(current_idx)

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

st.caption(
    f"Split: {selected_split} | Source row: {sample.source_row_index} | "
    "Displayed FEN is the transformed solver position used by Model A."
)

# =========================================================
# BUILD GRAPH
# =========================================================

graph = puzzle_row_to_graph(
    row=sample,
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

if st.session_state.get("user_move"):
    user_arrow = build_move_arrow(
        st.session_state["user_move"],
        USER_MOVE_COLOR,
    )
    if user_arrow is not None:
        board_arrows.append(user_arrow)

model_result = st.session_state.get("model_result")
if model_result and model_result.get("topk"):
    model_arrow = build_move_arrow(
        model_result["topk"][0]["move"],
        MODEL_TOP1_COLOR,
    )
    if model_arrow is not None:
        board_arrows.append(model_arrow)

if st.session_state.get("revealed_solution"):
    target_arrow = build_move_arrow(
        sample.TargetMove,
        GROUND_TRUTH_COLOR,
    )
    if target_arrow is not None:
        board_arrows.append(target_arrow)

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

        edge_types = edge_types_from_features(
            features
        )

        if not should_show_edge(
            edge_types,
            show_legal,
            show_attack,
            show_defend,
            show_pin,
            show_check,
        ):
            continue

        edge_type = edge_types[0]

        edges.append(

            Edge(
                source=src,
                target=dst,

                color=GRAPH_EDGE_COLORS[
                    edge_type
                ],

                title=", ".join(edge_types),

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
# MODEL A PUZZLE TRAINING AND VERIFICATION
# =========================================================

st.divider()
st.header("Puzzle Training & Model Verification")

bundle, model_error = load_model_a_resource()
bundle_a2, model_a2_error = load_model_a2_resource()
bundle_a3, model_a3_error = load_model_a3_resource()

status_cols = st.columns(4)
status_cols[0].metric("Primary Model", "MODEL_A3_NO_TIMING")
status_cols[1].metric(
    "Device",
    str(bundle_a3.device).upper() if bundle_a3 else "UNAVAILABLE",
)
status_cols[2].metric(
    "Checkpoint",
    str(bundle_a3.checkpoint_path) if bundle_a3 else "unavailable",
)
status_cols[3].metric(
    "Vocabulary",
    f"{len(move_to_idx):,}",
)

if model_error:
    st.warning(
        "Checkpoint unavailable. Expected path: "
        f"`{EXPECTED_CHECKPOINT_PATH}`. "
        f"Loader message: {model_error}"
    )
elif bundle and bundle.checkpoint_path != EXPECTED_CHECKPOINT_PATH:
    st.warning(
        "FALLBACK / NON-OFFICIAL CHECKPOINT loaded for local single-puzzle "
        "debug only: "
        f"`{bundle.checkpoint_path}`. The frozen convergence checkpoint is "
        f"expected at `{EXPECTED_CHECKPOINT_PATH}`. Official Model A Error "
        "Analysis is disabled until the frozen checkpoint is available."
    )
elif bundle:
    st.success("Checkpoint status: OFFICIAL FROZEN CHECKPOINT")

st.info(
    "No-timing phase is frozen. Model A3 is the official no-timing baseline. "
    "A3 shared-test metrics: Top1 67.56%, Top3 85.66%, Top5 91.86%, "
    "mean legal rank 2.2243. Model A and A2 remain available for diagnostics."
)
if model_a2_error:
    st.caption(f"Model A2 unavailable at `{MODEL_A2_CHECKPOINT_PATH}`: {model_a2_error}")
if model_a3_error:
    st.warning(f"Model A3 unavailable at `{MODEL_A3_CHECKPOINT_PATH}`: {model_a3_error}")
elif bundle_a3:
    st.success("Model A3 checkpoint status: OFFICIAL NO-TIMING BASELINE")

puzzle_tab, model_tab, mate_tab, error_tab = st.tabs(
    [
        "Human Puzzle Training",
        "Model A3 Inference",
        "Mate-in-1 Evaluation",
        "Error Analysis",
    ]
)

with puzzle_tab:
    st.subheader("Try the puzzle")
    st.write(
        "Ground truth is hidden until reveal. Use UCI notation, for example `g5f7`."
    )
    move_input = st.text_input(
        "Your move",
        value=st.session_state.get("user_move", ""),
        key="user_move_input",
    )
    check_col, reveal_col, reset_col = st.columns(3)
    if check_col.button("Check move"):
        result = check_user_move(
            board=board,
            user_move=move_input,
            target_move=sample.TargetMove,
        )
        st.session_state["user_move"] = result["move"]
        st.session_state["last_user_result"] = result
        counters = st.session_state["session_counters"]
        counters["attempted"] += 1
        counters["human_correct"] += int(result["status"] == "CORRECT")
    if reveal_col.button("Reveal solution"):
        st.session_state["revealed_solution"] = True
    if reset_col.button("Reset answer"):
        reset_puzzle_state()

    result = st.session_state.get("last_user_result")
    if result:
        st.write(f"Your move: `{result['move']}`")
        if st.session_state.get("revealed_solution"):
            st.write(f"Expected: `{sample.TargetMove}`")
            san = move_to_san(board, sample.TargetMove)
            if san:
                st.write(f"Expected SAN: `{san}`")
        st.write(f"Result: `{result['status']}`")
        st.caption(result["message"])

    if st.session_state.get("revealed_solution"):
        st.success(f"Solution: `{sample.TargetMove}`")
        continuation = str(sample.Moves).split()[1:]
        if continuation:
            st.write("Solution line:")
            st.code(" ".join(continuation))

    counters = st.session_state["session_counters"]
    st.write("Session counters")
    counter_cols = st.columns(4)
    counter_cols[0].metric("Attempted", counters["attempted"])
    counter_cols[1].metric("Human correct", counters["human_correct"])
    counter_cols[2].metric("Model Top1", counters["model_top1_correct"])
    counter_cols[3].metric("Model Top5", counters["model_top5_correct"])

with model_tab:
    st.subheader("Model A3 legal-candidate predictions")
    st.caption(
        "A3 scores only legal moves from the current position. It never emits "
        "global 1,786-class logits."
    )
    top_k = st.selectbox(
        "Top-K",
        [5, 10, 20],
        index=0,
    )
    if st.button("Ask Model A3", disabled=bundle_a3 is None):
        with st.spinner("Running Model A3 legal-candidate inference..."):
            try:
                result = run_model_mode_inference(
                    "Model A3 Legal Scorer",
                    row=sample,
                    bundle_a3=bundle_a3,
                    top_k=top_k,
                )
                st.session_state["model_result"] = result
                counters = st.session_state["session_counters"]
                counters["model_top1_correct"] += int(result["top1_hit"])
                counters["model_top3_correct"] += int(result["top3_hit"])
                counters["model_top5_correct"] += int(result["top5_hit"])
            except Exception as error:
                st.error(f"Model inference failed: {error}")

    result = st.session_state.get("model_result")
    if result:
        top1 = result["topk"][0] if result["topk"] else None
        result_cols = st.columns(4)
        result_cols[0].metric("Top-1 target match", result["top1_label"])
        result_cols[1].metric("Top-3 contains target", result["top3_label"])
        result_cols[2].metric("Top-5 contains target", result["top5_label"])
        result_cols[3].metric(
            "Target Legal Rank",
            "unavailable — target OOV"
            if result["target_oov"]
            else f"#{result['target_rank']} / {result.get('legal_candidate_count', 'legal')}",
        )
        st.write(f"Inference: `{result['inference_status']}`")
        st.write("Illegal Top1 rate: `0.0` by construction.")
        st.write(f"Legal candidates: `{result.get('legal_candidate_count')}`")
        if top1:
            st.write(f"A3 Top-1: `{top1['move']}`")
            st.write(f"Top-1 score: `{top1.get('score')}`")
        if result["target_rank"] and result["target_rank"] > top_k:
            st.info(f"Ground truth appears at rank #{result['target_rank']}.")
        st.dataframe(
            pd.DataFrame(result["topk"]).rename(
                columns={
                    "rank": "Rank",
                    "move": "Move",
                    "san": "SAN",
                    "score": "Score",
                    "legal": "Legal",
                    "ground_truth": "Ground Truth",
                }
            ),
            use_container_width=True,
        )

with mate_tab:
    st.subheader("Mate-in-1 diagnostic subgroup")
    mode_selection = st.selectbox("Model", MODEL_OPTIONS, index=3)
    compare_all = st.checkbox("Compare all", value=True)
    if is_mate_in_one_row(sample):
        is_mate = verify_target_checkmate(
            fen=sample.FEN,
            target_move=sample.TargetMove,
        )
        st.write("Can the selected no-timing model find the mate?")
        st.write(f"Expected mating move: `{sample.TargetMove}`")
        st.write(f"Target checkmates: `{is_mate}`")
        if not is_mate:
            st.warning("Metadata says mateIn1, but target checkmate verification failed.")
        result = st.session_state.get("model_result")
        if result and result["topk"]:
            top1 = result["topk"][0]
            st.write(f"Last inference Top-1: `{top1['move']}`")
            st.write("Result: `MATE FOUND`" if result["top1_hit"] else "Result: `MATE MISSED`")
            st.write(f"Target rank: `{result['target_rank']}`")
            st.write(f"Top3 hit: `{result['top3_hit']}`")
            st.write(f"Top5 hit: `{result['top5_hit']}`")
            st.write(f"Prediction legal: `{top1['legal']}`")
        else:
            st.caption("Run Ask Model A3 to fill single-puzzle mate diagnostics.")
    else:
        st.info("Current puzzle is not marked as mateIn1.")

    st.caption(
        "Batch evaluation is a diagnostic subgroup evaluation, not a new "
        "official Model A test score."
    )
    eval_limit = st.number_input(
        "Optional evaluation limit",
        min_value=0,
        value=0,
        step=50,
    )
    selected_modes = MODEL_OPTIONS if compare_all else [mode_selection]
    available_bundles = {
        "Model A Raw": bundle,
        "Model A Best-Legal": bundle,
        "Model A2 Masked": bundle_a2,
        "Model A3 Legal Scorer": bundle_a3,
    }
    disabled_modes = [mode for mode in selected_modes if available_bundles.get(mode) is None]
    if disabled_modes:
        st.warning(f"Unavailable model resources: {', '.join(disabled_modes)}")
    if st.button("Evaluate Mate-in-1 puzzles", disabled=bool(disabled_modes)):
        with st.spinner("Running batched Mate-in-1 diagnostics..."):
            try:
                metrics = evaluate_mate_in_one_modes(
                    dataframe=df_full,
                    modes=selected_modes,
                    bundles=available_bundles,
                    limit=int(eval_limit) or None,
                )
                st.session_state["mate_eval_metrics"] = metrics
            except Exception as error:
                st.error(f"Mate-in-1 evaluation failed: {error}")

    metrics = st.session_state.get("mate_eval_metrics")
    if metrics:
        table = pd.DataFrame(
            [
                {
                    "Model": mode,
                    "N": values["N"],
                    "Top1": values["top1"],
                    "Top3": values["top3"],
                    "Top5": values["top5"],
                    "Mean rank": values["mean_legal_target_rank"],
                    "Median rank": values["median_legal_target_rank"],
                    "Illegal Top1": values["illegal_top1_rate"],
                }
                for mode, values in metrics.items()
            ]
        )
        st.dataframe(table, use_container_width=True)

with error_tab:
    st.subheader("Error Analysis")
    st.caption(
        "DIAGNOSTIC POST-HOC ANALYSIS. This tab interprets the already frozen "
        "test result; it must not be used for Model A selection or tuning."
    )
    analysis_mode = st.selectbox("Analysis model", MODEL_OPTIONS, index=3)
    analysis_bundle_available = {
        "Model A Raw": bundle is not None,
        "Model A Best-Legal": bundle is not None,
        "Model A2 Masked": bundle_a2 is not None,
        "Model A3 Legal Scorer": bundle_a3 is not None,
    }[analysis_mode]
    official_ready = analysis_bundle_available
    if not official_ready:
        st.warning(
            "Required checkpoint for the selected model is not loaded. "
            "Batch Error Analysis is disabled."
        )
    min_theme_samples = st.number_input(
        "Minimum theme sample",
        min_value=1,
        value=30,
        step=5,
    )
    run_disabled = not official_ready or selected_split != "test"
    if selected_split != "test":
        st.info("Metric parity is defined for the official test split only.")
    run_col, clear_col = st.columns(2)
    if run_col.button("Run Error Analysis", disabled=run_disabled):
        progress = st.progress(0)
        status = st.empty()

        def update_progress(processed, total):
            status.write(f"Processed {processed:,} / {total:,} examples")
            progress.progress(processed / total if total else 1.0)

        try:
            rows = list(df_full.itertuples(index=False))
            records = []
            for processed, row in enumerate(rows, 1):
                prediction = run_model_mode_inference(
                    analysis_mode,
                    row,
                    bundle_a=bundle,
                    bundle_a2=bundle_a2,
                    bundle_a3=bundle_a3,
                    top_k=5,
                )
                records.append(error_analysis_record(row, prediction))
                if processed % 50 == 0:
                    update_progress(processed, len(rows))
            update_progress(len(rows), len(rows))
            st.session_state["model_a_error_analysis"] = {
                "model": analysis_mode,
                "records": records,
            }
            status.write("Analysis complete.")
        except Exception as error:
            st.error(f"Error Analysis failed: {error}")
    if clear_col.button("Clear analysis results"):
        st.session_state["model_a_error_analysis"] = None

    analysis = st.session_state.get("model_a_error_analysis")
    if analysis:
        if "records" in analysis:
            st.write(f"Model: `{analysis['model']}`")
            st.dataframe(pd.DataFrame(analysis["records"]), use_container_width=True)
        else:
            overall = analysis["overall"]
            rates = overall["rates"]
            parity = overall["parity"]
            st.metric("OFFICIAL METRIC PARITY", parity["status"])
            if parity["status"] != "PASS":
                st.warning(
                    "Inference parity with the frozen test evaluation was not achieved."
                )
                st.json(parity)
            dash = st.columns(4)
            dash[0].metric("Test N", overall["evaluable"])
            dash[1].metric("Top1", f"{rates['top1']:.2%}")
            dash[2].metric("Top3", f"{rates['top3']:.2%}")
            dash[3].metric("Top5", f"{rates['top5']:.2%}")

            st.write("Error structure")
            error_df = pd.DataFrame(
                [
                    {"Category": key, "Count": value}
                    for key, value in overall["error_categories"].items()
                ]
            )
            st.bar_chart(error_df, x="Category", y="Count")
            st.dataframe(error_df, use_container_width=True)

            st.write("Rank recovery")
            st.json(overall["rank_recovery"])
            rank_df = pd.DataFrame(
                [
                    {"Rank bucket": key, "Count": value}
                    for key, value in overall["rank_buckets"].items()
                ]
            )
            st.bar_chart(rank_df, x="Rank bucket", y="Count")
            st.dataframe(rank_df, use_container_width=True)

            st.write("Raw vs legal diagnostic")
            st.metric(
                "Diagnostic best-legal Top1",
                f"{rates['best_legal_top1']:.2%}",
                delta=f"{(rates['best_legal_top1'] - rates['top1']) * 100:.2f} pp",
            )

            st.write("Mate depth analysis")
            st.dataframe(
                pd.DataFrame(analysis["mate_depth"].values()),
                use_container_width=True,
            )
            if "mateIn1" in analysis["mate_depth"]:
                st.write("Mate-in-1 focus")
                st.json(analysis["mate_depth"]["mateIn1"])

            st.write("Theme analysis")
            st.caption("Theme metrics are overlapping subgroup analyses.")
            st.dataframe(
                pd.DataFrame(analysis["theme"].values()),
                use_container_width=True,
            )

            st.write("Rating analysis")
            st.dataframe(
                pd.DataFrame(analysis["rating"].values()),
                use_container_width=True,
            )

            st.write("Explore mistakes")
            mistake_filter = st.selectbox(
                "Mistake filter",
                [
                    "Top1 incorrect",
                    "Illegal Top1",
                    "Legal wrong Top1",
                    "Target rank >5",
                    "Mate-in-1 missed",
                ],
            )
            examples = overall["illegal_examples"] if mistake_filter == "Illegal Top1" else overall["mistake_examples"]
            st.dataframe(pd.DataFrame(examples), use_container_width=True)
            if analysis.get("artifact_paths"):
                st.write("Report artifacts")
                st.json(analysis["artifact_paths"])

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

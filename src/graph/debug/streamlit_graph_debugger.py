"""Interactive Streamlit debugger for chess graph samples.

Purpose:
    Visualize one generated puzzle graph together with its chessboard,
    edge overlays, node metadata, and raw puzzle fields.
Input:
    data/final/puzzles/train.csv and resources/move_encoder/move_to_idx.json.
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
    move_to_san as inference_move_to_san,
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
from src.inference.model_a.postmove_reranker import predict_a4_from_fen
from src.streamlit_app.graph_metadata import graph_representation_metadata
from src.streamlit_app.model_registry import model_table_rows
from src.streamlit_app.puzzle_sequence import (
    MATE_DEPTH_OPTIONS,
    PuzzleSession,
    filter_by_mate_depth,
    legal_move_options,
    move_history_rows,
    move_to_san,
    move_to_san_label,
    solution_moves_from_lichess_moves,
    solution_san_sequence,
)
from src.streamlit_app.puzzle_inference import (
    apply_pending_widget_resets,
    clear_model_rollout,
    clear_position_dependent_predictions,
    clear_revealed_hint,
    current_solver_step,
    current_solver_target,
    human_attempt_summary,
    invalidate_prediction_if_fen_changed,
    remaining_solution_from_session,
    revealed_next_move_for_session,
    request_widget_reset,
    row_for_current_fen,
    run_reference_line_rollout,
)
from src.streamlit_app.results_loader import (
    canonical_result_snapshot,
    canonical_mate_depth_rows,
    canonical_rating_rows,
    format_metric_rows,
    load_frozen_results,
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
    "resources/move_encoder/move_to_idx.json"
)

BOARD_SIZE = 700
USER_MOVE_COLOR = "#f59e0b"
MODEL_TOP1_COLOR = "#38bdf8"
GROUND_TRUTH_COLOR = "#22c55e"

READABLE_MODEL_OPTIONS = {
    "A - Move Classifier": "Model A Raw",
    "A1 - A + Best Legal Move": "Model A Best-Legal",
    "A2 - Legal-Masked Classifier": "Model A2 Masked",
    "A3 - Legal Move Scorer": "Model A3 Legal Scorer",
}

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


def reset_puzzle_state(clear_rollout=False):
    """Clear per-puzzle user and inference state."""

    st.session_state["user_move"] = ""
    st.session_state["last_user_result"] = None
    clear_position_dependent_predictions(st.session_state)
    clear_revealed_hint(st.session_state)
    if clear_rollout:
        clear_model_rollout(st.session_state)
    st.session_state["puzzle_session"] = None
    st.session_state["playback_ply"] = 0
    request_widget_reset(st.session_state, "user_move_input")


def apply_filtered_index(new_index):
    """Set a new puzzle index and reset per-puzzle UI state."""

    st.session_state["current_idx"] = int(new_index)
    reset_puzzle_state(clear_rollout=True)


def predict_next_move_for_fen(
    model_name,
    fen,
    sample_row,
    bundle_a,
    bundle_a2,
    bundle_a3,
):
    """Predict one move for the selected Streamlit model at a given FEN."""

    if model_name.startswith("A4"):
        result = predict_a4_from_fen(
            fen,
            a3_checkpoint=MODEL_A3_CHECKPOINT_PATH,
            a4_checkpoint="checkpoints/model_a4/best.pt",
            top_k=5,
            amp=False,
        )
        return result["a4_top1"], result
    mode = READABLE_MODEL_OPTIONS[model_name]
    row = row_for_current_fen(sample_row, fen)
    result = run_model_mode_inference(
        mode,
        row=row,
        bundle_a=bundle_a,
        bundle_a2=bundle_a2,
        bundle_a3=bundle_a3,
        top_k=5,
    )
    predicted = result["topk"][0]["move"] if result.get("topk") else None
    return predicted, result

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
    ("revealed_next_move", None),
    ("model_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

apply_pending_widget_resets(
    st.session_state,
    defaults={
        "user_move_input": "",
    },
)

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
    "Chess Puzzle GNN"
)

st.sidebar.caption(
    "Browse puzzles, try moves, and inspect frozen model predictions."
)

# =========================================================
# FILTERS
# =========================================================

st.sidebar.subheader(
    "Puzzle Filters"
)

mate_depth_filter = st.sidebar.selectbox(
    "Mate depth",
    MATE_DEPTH_OPTIONS,
    index=0,
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
df = filter_by_mate_depth(df, mate_depth_filter)
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
solution_moves = solution_moves_from_lichess_moves(sample.Moves)
if st.session_state.get("puzzle_session") is None:
    st.session_state["puzzle_session"] = PuzzleSession.create(sample.FEN, solution_moves)
puzzle_session = st.session_state["puzzle_session"]
current_fen = puzzle_session.current_fen
current_target_move = current_solver_target(puzzle_session)
inference_row = row_for_current_fen(
    sample,
    current_fen,
    target_move=current_target_move or sample.TargetMove,
)
invalidate_prediction_if_fen_changed(st.session_state, current_fen)

# =========================================================
# VIEW MODE
# =========================================================

st.sidebar.subheader(
    "Advanced Visualization"
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
    "Chess Puzzle GNN Trainer"
)

# =========================================================
# INFO
# =========================================================

solver_board_for_header = chess.Board(current_fen)
side_to_move = "WHITE TO MOVE" if solver_board_for_header.turn == chess.WHITE else "BLACK TO MOVE"

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "Puzzle ID",
    sample.PuzzleId
)

info2.metric(
    "Mate",
    f"in {int(sample.MateDepth)}"
)

info3.metric(
    "Side to move",
    side_to_move
)

info4.metric(
    "Rating",
    int(sample.Rating) if "Rating" in sample else "n/a"
)

st.caption(
    f"Split: {selected_split} | Puzzle {current_idx + 1:,} of {len(df):,}. "
    "The reference answer stays hidden until you reveal it."
)

# =========================================================
# BUILD GRAPH
# =========================================================

graph = puzzle_row_to_graph(
    row=inference_row,
    move_to_idx=move_to_idx,
)

board = chess.Board(current_fen)

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

revealed_hint = st.session_state.get("revealed_next_move")
if revealed_hint and revealed_hint.get("fen") == puzzle_session.current_fen:
    target_arrow = build_move_arrow(
        revealed_hint["move"],
        GROUND_TRUTH_COLOR,
    )
    if target_arrow is not None:
        board_arrows.append(target_arrow)

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
st.header("Play, Predict, Compare")

bundle, model_error = load_model_a_resource()
bundle_a2, model_a2_error = load_model_a2_resource()
bundle_a3, model_a3_error = load_model_a3_resource()

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

puzzle_tab, model_tab, performance_tab, advanced_tab = st.tabs(
    [
        "Puzzle",
        "Models",
        "Performance",
        "Advanced",
    ]
)

with puzzle_tab:
    st.subheader("Try the puzzle")
    top_left, top_right = st.columns(2)
    top_left.metric("Mate", f"in {int(sample.MateDepth)}")
    top_right.metric("Turn", side_to_move)
    html(
        chess.svg.board(
            board=chess.Board(puzzle_session.current_fen),
            size=520,
            coordinates=True,
        ),
        height=570,
    )
    if puzzle_session.is_complete_solution():
        st.success("Puzzle solved!")
    else:
        st.info(f"Your turn: {side_to_move}. Moves are shown in standard chess notation.")
    st.markdown("#### Your move")
    legal_options = legal_move_options(puzzle_session.current_fen)
    if legal_options and not puzzle_session.is_complete_solution():
        option_labels = [option["label"] for option in legal_options]
        selected_label = st.selectbox("Your move", option_labels)
        selected_legal_move = legal_options[option_labels.index(selected_label)]["uci"]
    else:
        selected_legal_move = None
        if not puzzle_session.is_complete_solution():
            st.info("No legal moves in current position.")
    with st.expander("Advanced UCI move entry", expanded=False):
        move_input = st.text_input(
            "Raw UCI move",
            key="user_move_input",
        )
    play_col, undo_col, reset_col, reveal_col = st.columns(4)
    if play_col.button("Play move", disabled=selected_legal_move is None):
        result = puzzle_session.play_solver_move_with_reference_reply(selected_legal_move)
        st.session_state["user_move"] = result["move"]
        st.session_state["last_user_result"] = result
        clear_position_dependent_predictions(st.session_state)
        if result["status"] in {"CORRECT", "COMPLETE"}:
            clear_revealed_hint(st.session_state)
        request_widget_reset(st.session_state, "user_move_input")
        counters = st.session_state["session_counters"]
        counters["attempted"] += 1
        counters["human_correct"] += int(result["status"] == "COMPLETE")
        st.rerun()
    if undo_col.button("Undo"):
        puzzle_session.undo()
        st.session_state["last_user_result"] = None
        clear_position_dependent_predictions(st.session_state)
        clear_revealed_hint(st.session_state)
        request_widget_reset(st.session_state, "user_move_input")
        st.rerun()
    if reveal_col.button("Reveal next move"):
        next_move = revealed_next_move_for_session(puzzle_session)
        if next_move:
            st.session_state["revealed_next_move"] = {
                "fen": puzzle_session.current_fen,
                "solver_step": current_solver_step(puzzle_session),
                "move": next_move,
                "san": move_to_san(puzzle_session.current_fen, next_move),
            }
    if reset_col.button("Reset answer"):
        reset_puzzle_state()
        st.rerun()

    with st.expander("Advanced UCI controls", expanded=False):
        if st.button("Play raw UCI"):
            move_to_check = st.session_state.get("user_move_input", "")
            try:
                result = puzzle_session.play_solver_move_with_reference_reply(move_to_check)
            except ValueError as error:
                result = {"move": move_to_check, "status": "ILLEGAL", "message": str(error), "auto_reply": None}
            st.session_state["user_move"] = result["move"]
            st.session_state["last_user_result"] = result
            clear_position_dependent_predictions(st.session_state)
            if result["status"] in {"CORRECT", "COMPLETE"}:
                clear_revealed_hint(st.session_state)
            request_widget_reset(st.session_state, "user_move_input")
            counters = st.session_state["session_counters"]
            counters["attempted"] += 1
            counters["human_correct"] += int(result["status"] == "COMPLETE")
            st.rerun()

    result = st.session_state.get("last_user_result")
    if result:
        if result["status"] == "COMPLETE":
            st.success(result["message"])
        elif result["status"] == "CORRECT":
            st.success(result["message"])
        elif result["status"] == "INCORRECT":
            st.error(result["message"])
        else:
            st.warning(result["message"])
        st.write(f"You played: `{result.get('san') or result.get('move')}`")
        if result.get("auto_reply"):
            st.write(f"Reference reply: `{result.get('auto_reply_san') or result['auto_reply']}`")
        st.caption(result["message"])
    st.write("Move history")
    history_rows = move_history_rows(
        puzzle_session.start_fen,
        puzzle_session.human_moves,
        puzzle_session.auto_reply_indices,
    )
    if history_rows:
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True)
    else:
        st.caption("No moves played yet.")

    revealed_hint = st.session_state.get("revealed_next_move")
    if revealed_hint and revealed_hint.get("fen") == puzzle_session.current_fen:
        st.info(f"Hint for this position: `{revealed_hint.get('san') or revealed_hint['move']}`")

    st.divider()
    st.markdown("#### Need help?")
    model_choice_col, ask_col, attempt_col = st.columns([2, 1, 1])
    puzzle_model = model_choice_col.selectbox(
        "Model",
        list(READABLE_MODEL_OPTIONS) + ["A4 - Post-Move Reranker"],
        index=4,
        key="puzzle_model_choice",
    )
    if ask_col.button("Suggest a move"):
        st.session_state["model_prediction_fen"] = puzzle_session.current_fen
        try:
            if puzzle_model.startswith("A4"):
                st.session_state["a4_result"] = predict_a4_from_fen(
                    puzzle_session.current_fen,
                    a3_checkpoint=MODEL_A3_CHECKPOINT_PATH,
                    a4_checkpoint="checkpoints/model_a4/best.pt",
                    top_k=5,
                    amp=False,
                )
                st.session_state["model_result"] = None
            else:
                mode = READABLE_MODEL_OPTIONS[puzzle_model]
                row = row_for_current_fen(
                    sample,
                    puzzle_session.current_fen,
                    target_move=current_solver_target(puzzle_session) or sample.TargetMove,
                )
                st.session_state["model_result"] = run_model_mode_inference(
                    mode,
                    row=row,
                    bundle_a=bundle,
                    bundle_a2=bundle_a2,
                    bundle_a3=bundle_a3,
                    top_k=5,
                )
                st.session_state["a4_result"] = None
        except Exception as error:
            st.error(f"Model inference failed: {error}")

    if attempt_col.button(
        "Try to solve the full puzzle",
        help=(
            "The model chooses each move for the solving side. Opponent replies "
            "follow the official Lichess puzzle line."
        ),
    ):
        human_fen_before_rollout = puzzle_session.current_fen
        remaining_solution = remaining_solution_from_session(puzzle_session)

        def rollout_predict(fen):
            predicted, _ = predict_next_move_for_fen(
                puzzle_model,
                fen,
                sample,
                bundle,
                bundle_a2,
                bundle_a3,
            )
            return predicted

        try:
            st.session_state["reference_line_rollout"] = run_reference_line_rollout(
                puzzle_session.current_fen,
                remaining_solution,
                rollout_predict,
            )
            st.session_state["reference_line_rollout_model"] = puzzle_model
            st.session_state["reference_line_rollout_start_fen"] = puzzle_session.current_fen
            st.session_state["reference_line_rollout_start_step"] = current_solver_step(puzzle_session)
            assert puzzle_session.current_fen == human_fen_before_rollout
        except Exception as error:
            st.error(f"Full-puzzle attempt failed: {error}")

    prediction_matches_current = (
        st.session_state.get("model_prediction_fen") == puzzle_session.current_fen
    )
    if prediction_matches_current:
        st.markdown("#### Model result")
        if st.session_state.get("a4_result"):
            predicted = st.session_state["a4_result"]["a4_top1"]
            st.metric(f"{puzzle_model} suggests", move_to_san(puzzle_session.current_fen, predicted) or predicted)
        elif st.session_state.get("model_result") and st.session_state["model_result"].get("topk"):
            predicted = st.session_state["model_result"]["topk"][0]["move"]
            st.metric(f"{puzzle_model} suggests", move_to_san(puzzle_session.current_fen, predicted) or predicted)

    rollout = st.session_state.get("reference_line_rollout")
    if rollout:
        rollout_model = st.session_state.get("reference_line_rollout_model", puzzle_model)
        rollout_start_step = st.session_state.get("reference_line_rollout_start_step", 1)
        summary = human_attempt_summary(rollout_model, sample.MateDepth, rollout)
        st.markdown("#### Full puzzle attempt")
        st.caption(
            f"{rollout_model} attempted this puzzle from move {rollout_start_step} "
            f"of {int(sample.MateDepth)}."
        )
        st.write(summary["title"])
        if summary["solved"]:
            st.success(summary["result"])
        else:
            st.warning(summary["result"])
        st.caption(
            f"Mate in {int(sample.MateDepth)} means {int(sample.MateDepth)} moves by the solving side."
        )
        for row in rollout["rows"]:
            if row["correct"]:
                st.write(f"✓ Move {row['solver_ply']}: {row['predicted_san']}")
                if row.get("reference_reply_san"):
                    st.write(f"  Opponent: {row['reference_reply_san']}")
            else:
                st.write(f"✗ Move {row['solver_ply']}")
                st.write(f"  {rollout_model} chose: {row['predicted_san'] or row['predicted_uci']}")
                st.write(f"  Solution: {row['expected_san'] or row['expected_uci']}")
        with st.expander("Show technical rollout details", expanded=False):
            st.dataframe(pd.DataFrame(rollout["rows"]), use_container_width=True)
        if st.button("Clear model attempt"):
            clear_model_rollout(st.session_state)
            st.rerun()

    with st.expander("Give up / show full solution", expanded=False):
        if solution_moves:
            st.write("Full solution")
            st.dataframe(pd.DataFrame(solution_san_sequence(sample.FEN, solution_moves)), use_container_width=True)
            playback_max = len(solution_moves)
            playback_cols = st.columns(4)
            if playback_cols[0].button("Start"):
                st.session_state["playback_ply"] = 0
            if playback_cols[1].button("Previous"):
                st.session_state["playback_ply"] = max(0, st.session_state.get("playback_ply", 0) - 1)
            if playback_cols[2].button("Next"):
                st.session_state["playback_ply"] = min(playback_max, st.session_state.get("playback_ply", 0) + 1)
            if playback_cols[3].button("End"):
                st.session_state["playback_ply"] = playback_max
            playback_board = chess.Board(sample.FEN)
            for move_uci in solution_moves[: st.session_state.get("playback_ply", 0)]:
                move = chess.Move.from_uci(move_uci)
                if move in playback_board.legal_moves:
                    playback_board.push(move)
            st.caption(f"Playback ply {st.session_state.get('playback_ply', 0)} / {playback_max}")
            html(chess.svg.board(board=playback_board, size=420, coordinates=True), height=470)

    counters = st.session_state["session_counters"]
    st.write("Session counters")
    counter_cols = st.columns(4)
    counter_cols[0].metric("Attempted", counters["attempted"])
    counter_cols[1].metric("Human correct", counters["human_correct"])
    counter_cols[2].metric("Model Top1", counters["model_top1_correct"])
    counter_cols[3].metric("Model Top5", counters["model_top5_correct"])

with model_tab:
    st.subheader("Ask a frozen model")
    st.caption("Prediction position: Current board. Models predict the next move only.")
    readable_names = list(READABLE_MODEL_OPTIONS) + [
        "A4 - Post-Move Reranker",
        "B - Timing-Aware Legal Move Scorer",
    ]
    selected_readable_model = st.selectbox("Model", readable_names, index=3)
    selected_model_mode = READABLE_MODEL_OPTIONS.get(selected_readable_model)
    top_k = st.selectbox(
        "Top-K",
        [5, 10, 20],
        index=0,
    )
    show_ground_truth = st.checkbox(
        "Show evaluation against reference answer",
        value=bool(
            st.session_state.get("revealed_next_move")
            and st.session_state["revealed_next_move"].get("fen") == puzzle_session.current_fen
        ),
    )
    selected_bundle_available = False
    if selected_model_mode:
        selected_bundle_available = {
            "Model A Raw": bundle is not None,
            "Model A Best-Legal": bundle is not None,
            "Model A2 Masked": bundle_a2 is not None,
            "Model A3 Legal Scorer": bundle_a3 is not None,
        }[selected_model_mode]
    elif selected_readable_model.startswith("A4"):
        selected_bundle_available = Path("checkpoints/model_a4/best.pt").exists()
    else:
        st.info("Model B is shown in Performance because its frozen timing ablation is not an interactive puzzle solver in this UI.")

    if st.button("Ask selected model", disabled=not selected_bundle_available):
        with st.spinner("Running frozen next-move inference..."):
            try:
                if selected_readable_model.startswith("A4"):
                    result = predict_a4_from_fen(
                        puzzle_session.current_fen,
                        a3_checkpoint=MODEL_A3_CHECKPOINT_PATH,
                        a4_checkpoint="checkpoints/model_a4/best.pt",
                        top_k=5,
                        amp=False,
                    )
                    st.session_state["a4_result"] = result
                else:
                    result = run_model_mode_inference(
                        selected_model_mode,
                        row=row_for_current_fen(
                            sample,
                            puzzle_session.current_fen,
                            target_move=current_solver_target(puzzle_session) or sample.TargetMove,
                        ),
                        bundle_a=bundle,
                        bundle_a2=bundle_a2,
                        bundle_a3=bundle_a3,
                        top_k=top_k,
                    )
                    st.session_state["model_result"] = result
                    counters = st.session_state["session_counters"]
                    counters["model_top1_correct"] += int(result["top1_hit"])
                    counters["model_top3_correct"] += int(result["top3_hit"])
                    counters["model_top5_correct"] += int(result["top5_hit"])
                st.session_state["model_prediction_fen"] = puzzle_session.current_fen
            except Exception as error:
                st.error(f"Model inference failed: {error}")

    prediction_matches_current = (
        st.session_state.get("model_prediction_fen") == puzzle_session.current_fen
    )
    if selected_readable_model.startswith("A4") and st.session_state.get("a4_result") and prediction_matches_current:
        a4_result = st.session_state["a4_result"]
        st.metric("Predicted move", move_to_san(puzzle_session.current_fen, a4_result["a4_top1"]) or a4_result["a4_top1"])
        col_a3, col_a4 = st.columns(2)
        with col_a3:
            st.write("A3 candidates sent to A4")
            a3_rows = pd.DataFrame(a4_result["a3_topk"])
            move_col = "uci" if "uci" in a3_rows else "move"
            if move_col in a3_rows:
                a3_rows["Move"] = [move_to_san(puzzle_session.current_fen, move) or move for move in a3_rows[move_col]]
            st.dataframe(a3_rows, use_container_width=True)
        with col_a4:
            st.write("A4 ranking")
            a4_rows = pd.DataFrame(a4_result["a4_ranking"])
            move_col = "uci" if "uci" in a4_rows else "move"
            if move_col in a4_rows:
                a4_rows["Move"] = [move_to_san(puzzle_session.current_fen, move) or move for move in a4_rows[move_col]]
            st.dataframe(a4_rows, use_container_width=True)
    elif selected_model_mode and st.session_state.get("model_result") and prediction_matches_current:
        result = st.session_state["model_result"]
        top1 = result["topk"][0] if result["topk"] else None
        if top1:
            st.metric("Predicted move", move_to_san(puzzle_session.current_fen, top1["move"]) or top1["move"])
        if show_ground_truth:
            result_cols = st.columns(4)
            result_cols[0].metric("Top-1 target match", result["top1_label"])
            result_cols[1].metric("Top-3 contains target", result["top3_label"])
            result_cols[2].metric("Top-5 contains target", result["top5_label"])
            result_cols[3].metric(
                "Target legal rank",
                "unavailable - target OOV"
                if result["target_oov"]
                else f"#{result['target_rank']} / {result.get('legal_candidate_count', 'legal')}",
            )
        st.write(f"Legal candidates: `{result.get('legal_candidate_count')}`")
        if show_ground_truth and result["target_rank"] and result["target_rank"] > top_k:
            st.info(f"Ground truth appears at rank #{result['target_rank']}.")
        topk_df = pd.DataFrame(result["topk"]).rename(
            columns={
                "rank": "Rank",
                "move": "Move",
                "san": "SAN",
                "score": "Score",
                "legal": "Legal",
                "ground_truth": "Ground Truth",
            }
        )
        if not show_ground_truth and "Ground Truth" in topk_df:
            topk_df = topk_df.drop(columns=["Ground Truth"])
        st.dataframe(
            topk_df,
            use_container_width=True,
        )
    with st.expander("How these models work", expanded=False):
        st.dataframe(pd.DataFrame(model_table_rows()), use_container_width=True)
        st.info("Model B uses synthetic timing features. Frozen results show that this timing representation did not improve over A3.")

with performance_tab:
    st.subheader("Current model performance")
    st.caption("Frozen documented results only. This page does not run official test evaluation.")
    frozen_results = load_frozen_results()
    snapshot = canonical_result_snapshot()
    headline_rows = [
        {"Model": "A3 - Legal Move Scorer", "Top-1": "67.56%", "Role": "official no-timing baseline"},
        {"Model": "A4 - Post-Move Reranker", "Top-1": "85.41%", "Role": "best frozen GNN result"},
        {"Model": "B - Timing-Aware Legal Move Scorer", "Top-1": "65.98%", "Role": "synthetic timing ablation"},
    ]
    st.dataframe(pd.DataFrame(headline_rows), use_container_width=True)
    st.metric("A4 improvement over A3", "+17.85 percentage points")
    st.info("Synthetic timing did not improve over A3 in the implemented Model B ablation.")

    st.write("Performance by puzzle depth")
    mate_depth_rows = canonical_mate_depth_rows()
    st.dataframe(pd.DataFrame(format_metric_rows(mate_depth_rows)), use_container_width=True)
    chart_depth = pd.DataFrame(mate_depth_rows).set_index("Puzzle depth")[["A3", "A4"]]
    st.bar_chart(chart_depth)

    st.write("Performance by rating")
    rating_rows = canonical_rating_rows()
    st.dataframe(pd.DataFrame(format_metric_rows(rating_rows)), use_container_width=True)
    chart_rating = pd.DataFrame(rating_rows).set_index("Rating")[["A3", "A4"]]
    st.bar_chart(chart_rating)
    if frozen_results.get("a4_terminal") is None:
        st.caption(
            "Detailed frozen A4 artifact is not present locally; displayed values use the documented canonical frozen summary."
        )

    with st.expander("Raw artifact availability", expanded=False):
        st.json({key: value is not None for key, value in frozen_results.items()})
        st.json(snapshot)

    st.subheader("Optional local depth diagnostic")
    mode_selection = st.selectbox("Model", MODEL_OPTIONS, index=3)
    compare_all = st.checkbox("Compare all", value=True)
    if is_mate_in_one_row(sample):
        is_mate = verify_target_checkmate(
            fen=sample.FEN,
            target_move=sample.TargetMove,
        )
        st.write("Can the selected no-timing model find this mate-in-1 puzzle?")
        if (
            st.session_state.get("revealed_next_move")
            and st.session_state["revealed_next_move"].get("fen") == puzzle_session.current_fen
        ):
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
    if st.button("Run local Mate-in-1 diagnostic", disabled=bool(disabled_modes)):
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

with advanced_tab:
    st.subheader("Technical model status")
    status_cols = st.columns(4)
    status_cols[0].metric("Primary model", "MODEL_A3_NO_TIMING")
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

    st.write("Graph internals")
    graph_cols = st.columns(3)
    graph_cols[0].metric("Nodes", graph.x.shape[0])
    graph_cols[1].metric("Edges", graph.edge_index.shape[1])
    graph_cols[2].metric("Node features", graph.x.shape[1])
    with st.expander("Graph representation used by frozen models", expanded=False):
        st.json(graph_representation_metadata())

    st.subheader("Model mistakes")
    st.caption(
        "Diagnostic post-hoc analysis. Use it to inspect which puzzle types a model struggles with, not to tune frozen results."
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
        "Only show puzzle themes with at least this many examples",
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
    "Show Puzzle Metadata"
):
    safe_fields = {
        "PuzzleId": sample.PuzzleId,
        "MateDepth": int(sample.MateDepth),
        "Rating": int(sample.Rating) if "Rating" in sample else None,
        "Themes": sample.Themes if "Themes" in sample else None,
        "Split": selected_split,
        "Source row": int(sample.source_row_index),
    }
    st.write(safe_fields)
    if st.session_state.get("revealed_next_move"):
        st.warning("Solution fields are visible because a hint has been revealed.")
        st.write(sample)
    else:
        st.caption("Solution and target fields are hidden until Reveal next move is pressed.")

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

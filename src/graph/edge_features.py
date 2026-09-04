"""Extract sparse graph edges and edge features from a chess position.

Purpose:
    Convert chess relationships in a FEN position into PyTorch Geometric
    edge_index and edge_attr tensors.
The graph representation uses:
    - 64 nodes (board squares)
    - directed edges between squares

Edges represent chess relationships such as:
    - legal moves
    - attacks
    - defenses
    - pins
    - check lines

Each square pair has one edge with a multilabel feature vector.

The output is compatible with PyTorch Geometric:
    - edge_index
    - edge_attr
Input:
    FEN string.
Output:
    edge_index as torch.LongTensor [2, num_edges] and edge_attr as
    torch.FloatTensor [num_edges, EDGE_FEATURE_DIM].
Run:
    python3 src/graph/edge_features.py
"""

import chess
import torch

# =========================================================
# EDGE FEATURE DIMENSION
# =========================================================

"""
Edge feature layout:

[0] legal_move
[1] attack
[2] defend
[3] pin
[4] check_line

TOTAL = 5 edge features
"""

EDGE_FEATURE_DIM = 5


# =========================================================
# EDGE BUILDING HELPERS
# =========================================================

def create_edge_feature(
    legal_move=0.0,
    attack=0.0,
    defend=0.0,
    pin=0.0,
    check_line=0.0,
):
    """
    Create edge feature vector.

    Parameters
    ----------
    legal_move : float
    attack : float
    defend : float
    pin : float
    check_line : float

    Returns
    -------
    list[float]
        Length = EDGE_FEATURE_DIM
    """

    return [
        legal_move,
        attack,
        defend,
        pin,
        check_line,
    ]


def add_edge(
    edge_index,
    edge_attr,
    from_square,
    to_square,
    features,
):
    """
    Add edge to graph containers.

    Parameters
    ----------
    edge_index : dict or list
    edge_attr : list or None
    from_square : int
    to_square : int
    features : list

    Returns
    -------
    None

    Side effects
    ------------
    Aggregates features for one edge key when edge_index is a dict. Appends to
    list containers for backward-compatible manual use.
    """

    if isinstance(edge_index, dict):
        key = (from_square, to_square)

        if key not in edge_index:
            edge_index[key] = [0.0] * EDGE_FEATURE_DIM

        edge_index[key] = [
            max(existing, new)
            for existing, new
            in zip(edge_index[key], features)
        ]
        return

    edge_index.append(
        [from_square, to_square]
    )

    edge_attr.append(features)


# =========================================================
# LEGAL MOVE EDGES
# =========================================================

def extract_legal_move_edges(
    board,
    edge_index,
    edge_attr,
):
    """
    Create edges for all legal moves.

    Edge:
        from_square -> to_square

    Feature:
        legal_move = 1

    Parameters
    ----------
    board : chess.Board
    edge_index : list
    edge_attr : list

    Returns
    -------
    None

    Side effects
    ------------
    Appends legal-move edges to edge_index and edge_attr.
    """

    for move in board.legal_moves:

        from_sq = move.from_square
        to_sq = move.to_square

        features = create_edge_feature(
            legal_move=1.0
        )

        add_edge(
            edge_index,
            edge_attr,
            from_sq,
            to_sq,
            features,
        )


# =========================================================
# ATTACK / DEFEND EDGES
# =========================================================

def extract_attack_defend_edges(
    board,
    edge_index,
    edge_attr,
):
    """
    Create attack and defend edges.

    Attack:
        piece attacks opponent piece

    Defend:
        piece attacks friendly piece

    Parameters
    ----------
    board : chess.Board
    edge_index : list
    edge_attr : list

    Returns
    -------
    None

    Side effects
    ------------
    Appends attack and defend edges to edge_index and edge_attr.
    """

    for from_square in chess.SQUARES:

        piece = board.piece_at(from_square)

        if piece is None:
            continue

        attacked_squares = board.attacks(
            from_square
        )

        for to_square in attacked_squares:

            target_piece = board.piece_at(
                to_square
            )

            # =============================================
            # EMPTY SQUARE
            # =============================================

            if target_piece is None:
                continue

            # =============================================
            # ATTACK EDGE
            # =============================================

            if (
                piece.color
                != target_piece.color
            ):

                features = create_edge_feature(
                    attack=1.0
                )

            # =============================================
            # DEFEND EDGE
            # =============================================

            else:

                features = create_edge_feature(
                    defend=1.0
                )

            add_edge(
                edge_index,
                edge_attr,
                from_square,
                to_square,
                features,
            )


# =========================================================
# PIN EDGES
# =========================================================

def extract_pin_edges(
    board,
    edge_index,
    edge_attr,
):
    """
    Create pin edges.

    Edge:
        attacking_piece -> pinned_piece

    Feature:
        pin = 1

    Parameters
    ----------
    board : chess.Board
    edge_index : list
    edge_attr : list

    Returns
    -------
    None

    Side effects
    ------------
    Appends pin edges to edge_index and edge_attr.
    """

    for king_square in chess.SQUARES:

        king = board.piece_at(king_square)

        if (
            king is None
            or king.piece_type != chess.KING
        ):
            continue

        king_color = king.color

        for direction in (
            1,
            -1,
            8,
            -8,
            9,
            -9,
            7,
            -7,
        ):

            pinned_square = None
            current = king_square

            while True:

                current = current + direction

                if not _is_step_on_board(
                    current - direction,
                    current,
                    direction,
                ):
                    break

                piece = board.piece_at(current)

                if piece is None:
                    continue

                if pinned_square is None:

                    if piece.color == king_color:
                        pinned_square = current
                        continue

                    break

                if (
                    piece.color != king_color
                    and _is_slider_for_direction(
                        piece,
                        direction,
                    )
                ):

                    features = create_edge_feature(
                        pin=1.0
                    )

                    add_edge(
                        edge_index,
                        edge_attr,
                        current,
                        pinned_square,
                        features,
                    )

                break


def _is_step_on_board(previous_square, current_square, direction):
    """
    Check whether a ray step stays on the chessboard.

    Parameters
    ----------
    previous_square : int
    current_square : int
    direction : int

    Returns
    -------
    bool
    """

    if current_square < 0 or current_square >= 64:
        return False

    previous_file = chess.square_file(previous_square)
    current_file = chess.square_file(current_square)

    if direction in (1, -1):
        return abs(current_file - previous_file) == 1

    if direction in (9, -7):
        return current_file - previous_file == 1

    if direction in (7, -9):
        return previous_file - current_file == 1

    return True


def _is_slider_for_direction(piece, direction):
    """
    Check whether a piece can pin along a ray direction.

    Parameters
    ----------
    piece : chess.Piece
    direction : int

    Returns
    -------
    bool
    """

    if direction in (1, -1, 8, -8):
        return piece.piece_type in (
            chess.ROOK,
            chess.QUEEN,
        )

    return piece.piece_type in (
        chess.BISHOP,
        chess.QUEEN,
    )


# =========================================================
# CHECK LINE EDGES
# =========================================================

def extract_check_line_edges(
    board,
    edge_index,
    edge_attr,
):
    """
    Create check-line edges.

    If a king is in check:
        attacker -> king

    Feature:
        check_line = 1

    Parameters
    ----------
    board : chess.Board
    edge_index : list
    edge_attr : list

    Returns
    -------
    None

    Side effects
    ------------
    Appends checker-to-king edges when the side to move is in check. This
    feature does not enumerate every square on the geometric check line.
    """

    if not board.is_check():
        return

    king_square = board.king(
        board.turn
    )

    attackers = board.attackers(
        not board.turn,
        king_square
    )

    for attacker_square in attackers:

        features = create_edge_feature(
            check_line=1.0
        )

        add_edge(
            edge_index,
            edge_attr,
            attacker_square,
            king_square,
            features,
        )


# =========================================================
# MAIN EDGE EXTRACTION
# =========================================================

def extract_edge_features(fen, board=None):
    """
    Extract graph edges and edge features.

    Parameters
    ----------
    fen : str
    board : chess.Board or None

    Returns
    -------
    edge_index : torch.LongTensor
    edge_attr : torch.FloatTensor
    """

    if board is None:
        board = chess.Board(fen)

    edge_features = {}

    # =====================================================
    # EXTRACT EDGE TYPES
    # =====================================================

    extract_legal_move_edges(
        board,
        edge_features,
        None,
    )

    extract_attack_defend_edges(
        board,
        edge_features,
        None,
    )

    extract_pin_edges(
        board,
        edge_features,
        None,
    )

    extract_check_line_edges(
        board,
        edge_features,
        None,
    )

    # =====================================================
    # CONVERT TO TENSORS
    # =====================================================

    sorted_edges = sorted(edge_features.items())

    if not sorted_edges:
        return (
            torch.empty(
                (2, 0),
                dtype=torch.long,
            ),
            torch.empty(
                (0, EDGE_FEATURE_DIM),
                dtype=torch.float,
            ),
        )

    edge_index = torch.tensor(
        [
            [from_square, to_square]
            for (from_square, to_square), _
            in sorted_edges
        ],
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        [
            features
            for _, features
            in sorted_edges
        ],
        dtype=torch.float
    )

    return edge_index, edge_attr


# =========================================================
# TEST
# =========================================================

def main():
    """
    Run a simple local edge feature extraction test.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side effects
    ------------
    Prints edge tensor shapes and a small edge sample.
    """

    test_fen = (
        "r1bqkbnr/pppp1ppp/2n5/4p3/"
        "4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )

    edge_index, edge_attr = (
        extract_edge_features(
            test_fen
        )
    )

    print("=" * 50)
    print("EDGE FEATURE TEST")
    print("=" * 50)

    print(
        f"edge_index shape: "
        f"{edge_index.shape}"
    )

    print(
        f"edge_attr shape: "
        f"{edge_attr.shape}"
    )

    print("\nFirst 10 edges:")

    for i in range(
        min(10, edge_attr.shape[0])
    ):

        src = edge_index[0][i].item()
        dst = edge_index[1][i].item()

        feat = edge_attr[i].tolist()

        print(
            f"{src} -> {dst} : {feat}"
        )


if __name__ == "__main__":
    main()

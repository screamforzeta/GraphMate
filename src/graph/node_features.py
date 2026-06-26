"""Extract node features from a chess position.

Purpose:
    Convert a FEN position into one feature vector per chessboard square.
Each chessboard position is converted into:
    - 64 nodes (one per square)
For every node/square we compute:
    - piece type
    - piece color
    - occupied flag
    - board coordinates
    - attacked by white
    - attacked by black
    - legal mobility
    - pinned status
    - piece value

The output is a PyTorch tensor:
    x.shape = [64, NUM_FEATURES]
This tensor will later be used as:
    Data.x
Input:
    FEN string.
Output:
    torch.FloatTensor with shape [64, NODE_FEATURE_DIM].
Run:
    python3 src/graph/node_features.py
"""

from pathlib import Path
import chess
import torch

# =========================================================
# PIECE ENCODING
# =========================================================

"""
Piece encoding used for one-hot representation.

Example:
    Pawn   -> [1,0,0,0,0,0]
    Knight -> [0,1,0,0,0,0]
"""

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

NUM_PIECE_TYPES = 6

# =========================================================
# PIECE VALUES
# =========================================================

"""
Classical chess material values.
"""

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

# =========================================================
# FEATURE DIMENSION
# =========================================================

"""
Feature layout per node:

[0:6]   piece one-hot
[6]     color
[7]     occupied
[8]     row
[9]     col
[10]    attacked_by_white
[11]    attacked_by_black
[12]    legal_mobility
[13]    is_pinned
[14]    piece_value

TOTAL = 15 features
"""

NODE_FEATURE_DIM = 15


# =========================================================
# HELPERS
# =========================================================

def get_piece_one_hot(piece):
    """
    Convert chess piece into one-hot encoding.

    Parameters
    ----------
    piece : chess.Piece or None

    Returns
    -------
    list[float]
        Length = 6
    """

    one_hot = [0.0] * NUM_PIECE_TYPES

    if piece is None:
        return one_hot

    piece_idx = PIECE_TO_INDEX[piece.piece_type]

    one_hot[piece_idx] = 1.0

    return one_hot


def get_piece_color(piece):
    """
    Encode piece color.

    Parameters
    ----------
    piece : chess.Piece or None

    Returns
    -------
    float
        1.0 -> white
        0.0 -> black or empty
    """

    if piece is None:
        return 0.0

    return 1.0 if piece.color == chess.WHITE else 0.0


def get_piece_value(piece):
    """
    Return material value of piece.

    Parameters
    ----------
    piece : chess.Piece or None

    Returns
    -------
    float
    """

    if piece is None:
        return 0.0

    return float(
        PIECE_VALUES[piece.piece_type]
    )


def get_square_coordinates(square):
    """
    Convert square index into normalized coordinates.

    Parameters
    ----------
    square : int

    Returns
    -------
    tuple(float, float)

    Notes
    -----
    Coordinates normalized in [0,1]
    """

    row = chess.square_rank(square) / 7.0
    col = chess.square_file(square) / 7.0

    return row, col


def count_attackers(board, square, color):
    """
    Count number of attacking pieces.

    Parameters
    ----------
    board : chess.Board
    square : int
    color : bool

    Returns
    -------
    float
    """

    attackers = board.attackers(
        color,
        square
    )

    return float(len(attackers))


def compute_legal_mobility(board, square):
    """
    Count legal moves originating from square.

    Parameters
    ----------
    board : chess.Board
    square : int

    Returns
    -------
    float
    """

    mobility = 0

    for move in board.legal_moves:

        if move.from_square == square:
            mobility += 1

    return float(mobility)


def is_piece_pinned(board, square):
    """
    Check if piece is pinned.

    Parameters
    ----------
    board : chess.Board
    square : int

    Returns
    -------
    float
        1.0 -> pinned
        0.0 -> not pinned
    """

    piece = board.piece_at(square)

    if piece is None:
        return 0.0

    return float(
        board.is_pinned(
            piece.color,
            square
        )
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_node_features(fen):
    """
    Extract node feature tensor from FEN.

    Parameters
    ----------
    fen : str

    Returns
    -------
    torch.FloatTensor
        shape = [64, NODE_FEATURE_DIM]
    """

    board = chess.Board(fen)

    node_features = []

    # =====================================================
    # ITERATE OVER ALL 64 SQUARES
    # =====================================================

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        # =================================================
        # BASIC FEATURES
        # =================================================

        piece_one_hot = get_piece_one_hot(
            piece
        )

        color = get_piece_color(piece)

        occupied = (
            1.0 if piece is not None
            else 0.0
        )

        row, col = get_square_coordinates(
            square
        )

        # =================================================
        # TACTICAL FEATURES
        # =================================================

        attacked_by_white = count_attackers(
            board,
            square,
            chess.WHITE
        )

        attacked_by_black = count_attackers(
            board,
            square,
            chess.BLACK
        )

        legal_mobility = compute_legal_mobility(
            board,
            square
        )

        pinned = is_piece_pinned(
            board,
            square
        )

        piece_value = get_piece_value(
            piece
        )

        # =================================================
        # FINAL FEATURE VECTOR
        # =================================================

        features = (
            piece_one_hot
            + [color]
            + [occupied]
            + [row]
            + [col]
            + [attacked_by_white]
            + [attacked_by_black]
            + [legal_mobility]
            + [pinned]
            + [piece_value]
        )

        node_features.append(features)

    # =====================================================
    # CONVERT TO TENSOR
    # =====================================================

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    return x


# =========================================================
# TEST
# =========================================================

def main():
    """
    Run a simple local node feature extraction test.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Side effects
    ------------
    Prints tensor shape and one sample feature vector.
    """

    test_fen = (
        "r1bqkbnr/pppp1ppp/2n5/4p3/"
        "4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )

    x = extract_node_features(
        test_fen
    )

    print("=" * 50)
    print("NODE FEATURE TEST")
    print("=" * 50)

    print(f"Tensor shape: {x.shape}")

    print("\nFirst node features:")
    print(x[0])


if __name__ == "__main__":
    main()

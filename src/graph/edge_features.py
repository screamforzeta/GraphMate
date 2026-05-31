"""
=========================================================
edge_features.py
=========================================================

DESCRIPTION:
------------
This module extracts graph edges and edge features from
a chess position represented in FEN notation.

The graph representation uses:
    - 64 nodes (board squares)
    - directed edges between squares

Edges represent chess relationships such as:
    - legal moves
    - attacks
    - defenses
    - pins
    - check lines

Each edge has a feature vector describing its type.

The output is compatible with PyTorch Geometric:
    - edge_index
    - edge_attr

INPUT:
------
- FEN string

OUTPUT:
-------
edge_index : torch.LongTensor
    shape = [2, num_edges]

edge_attr : torch.FloatTensor
    shape = [num_edges, EDGE_FEATURE_DIM]

=========================================================
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
    edge_index : list
    edge_attr : list
    from_square : int
    to_square : int
    features : list
    """

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
    """

    for square in chess.SQUARES:

        piece = board.piece_at(square)

        if piece is None:
            continue

        if not board.is_pinned(
            piece.color,
            square
        ):
            continue

        king_square = board.king(
            piece.color
        )

        # =============================================
        # FIND ATTACKERS OF PINNED PIECE
        # =============================================

        attackers = board.attackers(
            not piece.color,
            square
        )

        for attacker_square in attackers:

            attacker_piece = board.piece_at(
                attacker_square
            )

            if attacker_piece is None:
                continue

            # =========================================
            # CHECK IF KING IS ON SAME LINE
            # =========================================

            if chess.square_distance(
                attacker_square,
                king_square
            ) >= 1:

                features = create_edge_feature(
                    pin=1.0
                )

                add_edge(
                    edge_index,
                    edge_attr,
                    attacker_square,
                    square,
                    features,
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

def extract_edge_features(fen):
    """
    Extract graph edges and edge features.

    Parameters
    ----------
    fen : str

    Returns
    -------
    edge_index : torch.LongTensor
    edge_attr : torch.FloatTensor
    """

    board = chess.Board(fen)

    edge_index = []
    edge_attr = []

    # =====================================================
    # EXTRACT EDGE TYPES
    # =====================================================

    extract_legal_move_edges(
        board,
        edge_index,
        edge_attr,
    )

    extract_attack_defend_edges(
        board,
        edge_index,
        edge_attr,
    )

    extract_pin_edges(
        board,
        edge_index,
        edge_attr,
    )

    extract_check_line_edges(
        board,
        edge_index,
        edge_attr,
    )

    # =====================================================
    # CONVERT TO TENSORS
    # =====================================================

    edge_index = torch.tensor(
        edge_index,
        dtype=torch.long
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_attr,
        dtype=torch.float
    )

    return edge_index, edge_attr


# =========================================================
# TEST
# =========================================================

def main():
    """
    Simple local test.
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
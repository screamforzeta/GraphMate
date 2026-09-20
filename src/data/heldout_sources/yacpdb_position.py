"""Orthodox YACPDB algebraic-position normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chess


PIECE_TYPES = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "S": chess.KNIGHT,
    "N": chess.KNIGHT,
    "P": chess.PAWN,
}
PIECE_RE = re.compile(r"^(?P<piece>[KQRBSPN])(?P<square>[a-h][1-8])$")


@dataclass(frozen=True)
class NormalizedPosition:
    """Normalized orthodox chess position from YACPDB algebraic data."""

    fen: str
    side_to_move: str
    side_to_move_source: str
    warnings: list[str]


def side_to_move_for_stipulation(stipulation: str) -> tuple[bool | None, str | None]:
    """Return solving side for supported stipulations.

    Orthodox directmates use White as the solving side in YACPDB/Popeye
    convention. Other problem types are rejected before normalization.
    """

    if re.fullmatch(r"#(10|[1-9])", str(stipulation).strip()):
        return chess.WHITE, None
    return None, "SIDE_TO_MOVE_UNKNOWN"


def normalize_algebraic_position(record: dict[str, Any]) -> tuple[NormalizedPosition | None, str | None]:
    """Convert real YACPDB `algebraic` piece lists to a conservative FEN."""

    algebraic = record.get("algebraic")
    if not isinstance(algebraic, dict):
        return None, "UNSUPPORTED_ALGEBRAIC_POSITION"
    if "neutral" in algebraic and algebraic.get("neutral"):
        return None, "FAIRY_PIECE"
    turn, turn_error = side_to_move_for_stipulation(str(record.get("stipulation") or ""))
    if turn_error or turn is None:
        return None, turn_error or "SIDE_TO_MOVE_UNKNOWN"

    board = chess.Board.empty()
    occupied: set[chess.Square] = set()
    king_counts = {chess.WHITE: 0, chess.BLACK: 0}
    for color_name, color in (("white", chess.WHITE), ("black", chess.BLACK)):
        pieces = algebraic.get(color_name, [])
        if pieces is None:
            continue
        if not isinstance(pieces, list):
            return None, "UNSUPPORTED_ALGEBRAIC_POSITION"
        for raw_piece in pieces:
            if not isinstance(raw_piece, str):
                return None, "UNSUPPORTED_ALGEBRAIC_POSITION"
            token = raw_piece.strip()
            match = PIECE_RE.fullmatch(token)
            if not match:
                return None, "FAIRY_PIECE" if re.search(r"[A-Z][A-Za-z]*\s", token) else "UNSUPPORTED_ALGEBRAIC_POSITION"
            square = chess.parse_square(match.group("square"))
            if square in occupied:
                return None, "DUPLICATE_SQUARE"
            occupied.add(square)
            piece_type = PIECE_TYPES[match.group("piece")]
            if piece_type == chess.KING:
                king_counts[color] += 1
            board.set_piece_at(square, chess.Piece(piece_type, color))

    if king_counts[chess.WHITE] != 1 or king_counts[chess.BLACK] != 1:
        return None, "MISSING_KING"
    board.turn = turn
    board.castling_rights = chess.BB_EMPTY
    board.ep_square = None
    board.halfmove_clock = 0
    board.fullmove_number = 1
    status = board.status()
    fatal_status = status & ~chess.STATUS_OPPOSITE_CHECK
    if fatal_status:
        return None, "MALFORMED_POSITION"
    return (
        NormalizedPosition(
            fen=board.fen(),
            side_to_move="white" if turn == chess.WHITE else "black",
            side_to_move_source="directmate_stipulation_white_to_move",
            warnings=["CASTLING_RIGHTS_NOT_IN_SOURCE_SET_NONE", "EN_PASSANT_NOT_IN_SOURCE_SET_NONE"],
        ),
        None,
    )


import chess

from src.data.heldout_sources.yacpdb_position import normalize_algebraic_position


def record(algebraic, stipulation="#2"):
    return {"id": 1, "stipulation": stipulation, "algebraic": algebraic, "solution": "1.Qa1-a8 !"}


def test_real_observed_algebraic_syntax_to_fen():
    normalized, reason = normalize_algebraic_position(
        record({"white": ["Kc6", "Qf3", "Sb5", "Pa3"], "black": ["Kc4", "Se4", "Pd4"]})
    )

    assert reason is None
    assert normalized.fen == "8/8/2K5/1N6/2kpn3/P4Q2/8/8 w - - 0 1"
    assert normalized.side_to_move == "white"
    assert "CASTLING_RIGHTS_NOT_IN_SOURCE_SET_NONE" in normalized.warnings


def test_all_standard_piece_types_and_multiple_pieces():
    normalized, reason = normalize_algebraic_position(
        record(
            {
                "white": ["Ke1", "Qa1", "Rh1", "Bc1", "Sg1", "Pa2"],
                "black": ["Ke8", "Qd8", "Ra8", "Bf8", "Sg8", "Pa7"],
            }
        )
    )

    assert reason is None
    board = chess.Board(normalized.fen)
    assert board.piece_at(chess.G1).symbol() == "N"
    assert board.castling_rights == chess.BB_EMPTY
    assert board.ep_square is None


def test_position_rejects_malformed_square_duplicate_and_unsupported_piece():
    assert normalize_algebraic_position(record({"white": ["Ke1"], "black": ["Ke9"]}))[1] == "UNSUPPORTED_ALGEBRAIC_POSITION"
    assert normalize_algebraic_position(record({"white": ["Ke1", "Qa1"], "black": ["Ka1"]}))[1] == "DUPLICATE_SQUARE"
    assert normalize_algebraic_position(record({"white": ["Ke1", "Gg2"], "black": ["Ke8"]}))[1] == "UNSUPPORTED_ALGEBRAIC_POSITION"


def test_position_rejects_fairy_neutral_and_missing_king():
    assert normalize_algebraic_position(record({"white": ["Ke1", "Royal Zf3"], "black": ["Ke8"]}))[1] == "FAIRY_PIECE"
    assert normalize_algebraic_position(record({"white": ["Ke1"], "black": ["Ke8"], "neutral": ["Qa4"]}))[1] == "FAIRY_PIECE"
    assert normalize_algebraic_position(record({"white": ["Qa1"], "black": ["Ke8"]}))[1] == "MISSING_KING"


def test_side_to_move_unknown_for_non_directmate():
    assert normalize_algebraic_position(record({"white": ["Ke1"], "black": ["Ke8"]}, stipulation="h#2"))[1] == "SIDE_TO_MOVE_UNKNOWN"


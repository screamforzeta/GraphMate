import chess

from src.data.heldout_sources.yacpdb_solution import extract_structural_key, structurally_identify_key


MATE1_BOARD = chess.Board("7k/6Q1/6K1/8/8/8/8/8 w - - 0 1")
CAPTURE_BOARD = chess.Board("4k3/8/8/8/8/5r2/8/4K2Q w - - 0 1")
PROMOTION_BOARD = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
CASTLE_BOARD = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")


def test_simple_actual_key_and_annotations():
    key, reason = extract_structural_key("1.Qg7-f8 ! threat: 2.Qf8#", MATE1_BOARD)

    assert reason is None
    assert key.key_token == "Qg7-f8"
    assert key.key_move_uci == "g7f8"


def test_id_26026_setplay_regression_does_not_select_first_line():
    solution = """1...d4-d3 2.Qf3-f7 #
1...Se4-f2 {(S~)} 2.Sb5-d6 # {(A)}
 but 1...Se4-c3 !

1.Qf3-d1 ! zugzwang.
   1...d4-d3  2.Qd1-a4 #
   1...Se4-f2  {(S~)} 2.Qd1-c2 #
   1...Se4-c3 2.Sb5-d6 # {(A)}"""
    board = chess.Board("8/8/2K5/1N6/2kpn3/P4Q2/8/8 w - - 0 1")

    key, reason = extract_structural_key(solution, board)

    assert reason is None
    assert key.key_token == "Qf3-d1"
    assert key.key_move_uci == "f3d1"
    assert key.skipped_setplay_count >= 2


def test_tries_before_actual_key_are_skipped():
    solution = """1.Qa1-b1 ? threat: 2.Sd2-c4 # but 1...Sd4-c2 !
1.Qa1-e1 ? threat: 2.Sd2-c4 # but 1...Sd4-e2 !

1.Qa1-a8 ! threat:
2.Sd2-c4 #"""
    board = chess.Board("1n6/2pR3b/4p3/4k2N/1R1nP1pp/8/1K1N4/Q7 w - - 0 1")

    key, reason = extract_structural_key(solution, board)

    assert reason is None
    assert key.key_move_uci == "a1a8"
    assert key.skipped_try_count == 2


def test_threats_branches_comments_check_and_mate_notation():
    board = chess.Board("6R1/2K5/8/8/5p2/6p1/1Q3Nk1/8 w - - 0 1")
    key, reason = extract_structural_key("1.Sf2-g4 + !\n1...Kg2-h1\n2.Qb2-h2 + !", board)

    assert reason is None
    assert key.key_move_uci == "f2g4"


def test_capture_promotion_castling_and_san():
    assert extract_structural_key("1.Qh1*f3 #", CAPTURE_BOARD)[0].key_move_uci == "h1f3"
    assert extract_structural_key("1.a7-a8=Q #", PROMOTION_BOARD)[0].key_move_uci == "a7a8q"
    assert extract_structural_key("1.O-O !", CASTLE_BOARD)[0].key_move_uci == "e1g1"
    assert extract_structural_key("1.Qf8#", MATE1_BOARD)[0].key_move_uci == "g7f8"


def test_capture_marker_mismatch_and_illegal_key():
    assert extract_structural_key("1.Qg7xf8 #", MATE1_BOARD)[1] == "CAPTURE_MARKER_MISMATCH"
    assert extract_structural_key("1.Qg7-g1 #", MATE1_BOARD)[1] == "KEY_NOT_LEGAL"


def test_ambiguous_no_key_malformed_and_unsupported():
    assert structurally_identify_key("1.Qg7-f8 !\n1.Kg6-h6 !")[1] == "AMBIGUOUS_KEY"
    assert structurally_identify_key("1...Kh8-g8 2.Qg7-f8 #")[1] == "NO_ACTUAL_KEY"
    assert extract_structural_key("", MATE1_BOARD)[1] == "NO_SOLUTION"
    assert extract_structural_key("1.NotAMove !", MATE1_BOARD)[1] == "UNSUPPORTED_MOVE_NOTATION"


def test_castling_without_source_rights_is_rejected():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")

    assert extract_structural_key("1.O-O !", board)[1] == "UNKNOWN_CASTLING_RIGHTS"

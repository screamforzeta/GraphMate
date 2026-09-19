import json

import pytest

from src.cli.evaluation.evaluate_llm_relaxed import evaluate_predictions
from src.llm.relaxed_parsing import RELAXED_PARSER_STATUS, RELAXED_PARSER_VERSION, relaxed_parse_move


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
CAPTURE_FEN = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
CHECKMATE_FEN = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
PROMOTION_FEN = "k7/4P3/8/8/8/8/8/7K w - - 0 1"
PROMOTION_AMBIGUOUS_FEN = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
DISAMBIGUATED_KNIGHT_FEN = "4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1"
ROOK_B8_FEN = "1R2k3/8/8/8/8/8/8/4K3 w - - 0 1"


def assert_parsed(raw, fen, uci, method=None):
    parsed = relaxed_parse_move(raw, fen)

    assert parsed.parse_status == "PARSED"
    assert parsed.parsed_uci == uci
    assert parsed.is_legal is True
    if method:
        assert parsed.parse_method == method
    return parsed


def test_relaxed_exact_uci_legal_and_promotion_uci():
    assert RELAXED_PARSER_STATUS == "FROZEN"
    assert_parsed("e2e4", START_FEN, "e2e4", "EXACT_UCI")
    assert_parsed("e7e8q", PROMOTION_FEN, "e7e8q", "EXACT_UCI")


def test_relaxed_exact_uci_illegal_is_not_reinterpreted():
    parsed = relaxed_parse_move("e2e5", START_FEN)

    assert parsed.parse_status == "ILLEGAL_MOVE"
    assert parsed.parse_method == "EXACT_UCI"
    assert parsed.parsed_uci == "e2e5"
    assert parsed.is_legal is False


def test_relaxed_exact_san_pawn_piece_capture_checkmate_promotion_castling():
    assert_parsed("e4", START_FEN, "e2e4", "EXACT_SAN_RECHECK")
    assert_parsed("Nf3", START_FEN, "g1f3", "EXACT_SAN_RECHECK")
    assert_parsed("exd5", CAPTURE_FEN, "e4d5", "EXACT_SAN_RECHECK")
    assert_parsed("Qh4#", CHECKMATE_FEN, "d8h4", "EXACT_SAN_RECHECK")
    assert_parsed("e8=Q+", PROMOTION_FEN, "e7e8q", "EXACT_SAN_RECHECK")
    assert_parsed("O-O", CASTLE_FEN, "e1g1", "EXACT_SAN_RECHECK")


def test_relaxed_castling_queenside_and_no_san_repair():
    assert_parsed("O-O-O", CASTLE_FEN, "e1c1", "EXACT_SAN_RECHECK")

    assert relaxed_parse_move("Nf3!", START_FEN).parse_status == "UNRECOVERABLE"
    assert relaxed_parse_move("Nf3.", START_FEN).parse_status == "UNRECOVERABLE"
    assert relaxed_parse_move("0-0", CASTLE_FEN).parse_status == "UNRECOVERABLE"


def test_relaxed_piece_source_destination_rule():
    assert_parsed("Nb8-c6", BLACK_START_FEN, "b8c6", "PIECE_SOURCE_DESTINATION")
    assert_parsed("Nb8c6", BLACK_START_FEN, "b8c6", "PIECE_SOURCE_DESTINATION")
    assert_parsed("Rb8-b7", ROOK_B8_FEN, "b8b7", "PIECE_SOURCE_DESTINATION")


def test_relaxed_piece_source_destination_rejects_inconsistency():
    assert relaxed_parse_move("Rb8-c6", BLACK_START_FEN).parse_status == "UNRECOVERABLE"
    assert relaxed_parse_move("Nb7-c5", BLACK_START_FEN).parse_status == "UNRECOVERABLE"
    assert relaxed_parse_move("Nb8-a8", BLACK_START_FEN).parse_status == "UNRECOVERABLE"


def test_relaxed_source_separator_destination_rule():
    quiet = assert_parsed("e2-e4", START_FEN, "e2e4", "SOURCE_SEPARATOR_DESTINATION")
    capture = assert_parsed("e4xd5", CAPTURE_FEN, "e4d5", "SOURCE_SEPARATOR_DESTINATION")

    assert quiet.diagnostics["separator"] == "-"
    assert capture.diagnostics["separator"] == "x"


def test_relaxed_source_separator_destination_rejects_bad_capture_semantics():
    dash_capture = relaxed_parse_move("e4-d5", CAPTURE_FEN)
    x_non_capture = relaxed_parse_move("e2xe4", START_FEN)
    illegal = relaxed_parse_move("e2-e5", START_FEN)
    ambiguous_promotion = relaxed_parse_move("a7-a8", PROMOTION_AMBIGUOUS_FEN)

    assert dash_capture.parse_status == "UNRECOVERABLE"
    assert dash_capture.failure_reason == "dash_separator_used_for_capture"
    assert x_non_capture.parse_status == "UNRECOVERABLE"
    assert x_non_capture.failure_reason == "capture_separator_without_capture"
    assert illegal.parse_status == "UNRECOVERABLE"
    assert ambiguous_promotion.parse_status == "AMBIGUOUS"


def test_relaxed_permissive_noncanonical_san_remains_rejected():
    false_check = relaxed_parse_move("Nf3+", START_FEN)
    missing_check = relaxed_parse_move("Qf8#", "5k2/8/8/8/8/8/8/5Q1K w - - 0 1")
    omitted_capture = relaxed_parse_move("Qf6+", "4k3/8/5r2/8/8/8/8/4K2Q w - - 0 1")

    assert false_check.parse_status == "UNRECOVERABLE"
    assert missing_check.parse_status == "UNRECOVERABLE"
    assert omitted_capture.parse_status == "UNRECOVERABLE"


def test_relaxed_rejects_substrings_prose_and_multiple_options():
    for raw in ["garbageNf3garbage", "abcRa8xyz", "bRc8", "qf5", "The best move is Nf3", "Nf3 or Qh5"]:
        parsed = relaxed_parse_move(raw, START_FEN)
        assert parsed.parse_status == "UNRECOVERABLE"
        assert parsed.parsed_uci is None


def test_relaxed_malformed_unrecoverable_and_empty():
    malformed = relaxed_parse_move("cxbp", START_FEN)
    empty = relaxed_parse_move("```   ```", START_FEN)

    assert malformed.parse_status == "UNRECOVERABLE"
    assert empty.parse_status == "EMPTY"


def test_relaxed_black_to_move_and_position_dependence():
    assert_parsed("e5", BLACK_START_FEN, "e7e5", "EXACT_SAN_RECHECK")
    white = relaxed_parse_move("e5", START_FEN)

    assert white.parse_status == "UNRECOVERABLE"


def test_relaxed_disambiguated_san():
    assert_parsed("Nbd2", DISAMBIGUATED_KNIGHT_FEN, "b1d2", "EXACT_SAN_RECHECK")


def test_relaxed_previous_unsafe_regressions_do_not_extract_inner_tokens():
    knight = relaxed_parse_move("Nb8-c6", START_FEN)
    rook = relaxed_parse_move("Rb8-b7", "4k2r/8/8/8/8/8/8/4K3 w - - 0 1")

    assert knight.parse_status == "UNRECOVERABLE"
    assert knight.parsed_uci != "c7c6"
    assert rook.parse_status == "UNRECOVERABLE"
    assert rook.parsed_uci != "h8b8"


def test_relaxed_target_independence_by_signature_and_behavior():
    parsed_a = relaxed_parse_move("Nf3", START_FEN)
    parsed_b = relaxed_parse_move("Nf3", START_FEN)

    assert parsed_a.parsed_uci == parsed_b.parsed_uci == "g1f3"
    with pytest.raises(TypeError):
        relaxed_parse_move("Nf3", START_FEN, target_move="e2e4")


def test_relaxed_evaluator_writes_separate_artifacts_and_summary(tmp_path):
    predictions = tmp_path / "official" / "next_move" / "qwen_3_5_4b" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    rows = [
        {
            "puzzle_id": "p1",
            "fen": START_FEN,
            "target_move": "e2e4",
            "outcome": "PARSE_ERROR",
            "raw_final_content": "e4",
            "mate_depth": 1,
            "rating": 1500,
        },
        {
            "puzzle_id": "p2",
            "fen": START_FEN,
            "target_move": "e2e4",
            "outcome": "CORRECT",
            "raw_final_content": "e2e4",
            "mate_depth": 1,
            "rating": 1500,
        },
        {
            "puzzle_id": "p3",
            "fen": START_FEN,
            "target_move": "e2e4",
            "outcome": "PARSE_ERROR",
            "raw_final_content": "Nf3 or e4",
            "mate_depth": 2,
            "rating": 1700,
        },
    ]
    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    output_dir = tmp_path / "analysis"

    result = evaluate_predictions(predictions, output_dir)
    relaxed_path = output_dir / "predictions_relaxed.jsonl"
    relaxed_rows = [
        json.loads(line)
        for line in relaxed_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["summary"]["parser_version"] == RELAXED_PARSER_VERSION
    assert result["summary"]["N"] == 3
    assert result["summary"]["strict"]["parse_errors"] == 2
    assert result["summary"]["relaxed"]["correct"] == 2
    assert result["summary"]["relaxed"]["ambiguous"] == 0
    assert result["summary"]["relaxed"]["parse_errors"] == 1
    assert result["summary"]["recovered_correct_from_strict_parse_error"] == 1
    assert result["parser_coverage"]["categorized_records"] == 3
    assert result["parser_coverage"]["categorized_rate"] == 1.0
    assert relaxed_rows[0]["strict_outcome"] == "PARSE_ERROR"
    assert relaxed_rows[0]["relaxed_outcome"] == "CORRECT"
    assert relaxed_rows[0]["relaxed_parse_method"] == "EXACT_SAN_RECHECK"


def test_relaxed_evaluator_refuses_to_overwrite_existing_analysis(tmp_path):
    predictions = tmp_path / "official" / "next_move" / "qwen_3_5_4b" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    predictions.write_text(
        json.dumps(
            {
                "puzzle_id": "p1",
                "fen": START_FEN,
                "target_move": "e2e4",
                "outcome": "CORRECT",
                "raw_final_content": "e2e4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "analysis"
    evaluate_predictions(predictions, output_dir)

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        evaluate_predictions(predictions, output_dir)

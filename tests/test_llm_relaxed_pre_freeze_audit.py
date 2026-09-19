import json

from src.cli.evaluation.audit_llm_relaxed_pre_freeze import run_audit
from src.llm.relaxed_pre_freeze_audit import analyze_deterministic_recoverability


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
ROOK_FEN = "4k3/R7/8/8/8/8/8/4K3 w - - 0 1"
ROOK_CAPTURE_FEN = "4k3/R7/r7/8/8/8/8/4K3 w - - 0 1"
KNIGHT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"
CAPTURE_FEN = "4k3/4p3/4P3/8/8/8/8/4K3 w - - 0 1"
PROMOTION_FEN = "k7/4P3/8/8/8/8/8/7K w - - 0 1"


def test_piece_source_destination_recoverability():
    result = analyze_deterministic_recoverability("Ra7a6", ROOK_FEN)

    assert result.family == "piece_src_dst"
    assert result.recoverable is True
    assert result.candidate_uci == "a7a6"
    assert result.proposed_rule == "piece_source_destination"


def test_piece_source_separator_capture_requires_real_capture():
    legal_capture = analyze_deterministic_recoverability("Ra7xa6", ROOK_CAPTURE_FEN)
    non_capture = analyze_deterministic_recoverability("Ra7xa6", ROOK_FEN)

    assert legal_capture.recoverable is True
    assert legal_capture.candidate_uci == "a7a6"
    assert non_capture.recoverable is False
    assert non_capture.impossible is True


def test_source_separator_and_promotion_variants():
    separated = analyze_deterministic_recoverability("e2-e4", START_FEN)
    promotion = analyze_deterministic_recoverability("e7e8Q", PROMOTION_FEN)

    assert separated.recoverable is True
    assert separated.candidate_uci == "e2e4"
    assert promotion.recoverable is True
    assert promotion.candidate_uci == "e7e8q"
    assert promotion.family == "promotion_coordinate"


def test_annotation_inconsistency_is_not_safe_recovery():
    result = analyze_deterministic_recoverability("e2e4+", START_FEN)

    assert result.family == "uci_with_check_suffix"
    assert result.recoverable is False
    assert result.annotation_inconsistent is True


def test_san_like_and_non_chess_classification():
    san_like = analyze_deterministic_recoverability("Qe5", START_FEN)
    non_chess = analyze_deterministic_recoverability("I resign", START_FEN)

    assert san_like.family == "san_like"
    assert san_like.recoverable is False
    assert non_chess.family == "non_chess"


def test_audit_cli_outputs_artifacts_without_target_accuracy(tmp_path):
    predictions = tmp_path / "official" / "next_move" / "qwen_3_5_4b" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    rows = [
        {
            "puzzle_id": "p1",
            "fen": ROOK_FEN,
            "target_move": "a7a8",
            "raw_final_content": "Ra7a6",
            "outcome": "PARSE_ERROR",
        },
        {
            "puzzle_id": "p2",
            "fen": START_FEN,
            "target_move": "e2e4",
            "raw_final_content": "abcNf3xyz",
            "outcome": "PARSE_ERROR",
        },
    ]
    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = run_audit([str(predictions)], tmp_path / "audit")
    model_audit = result["model_audits"][0]
    proposed = result["proposed_recovery_rules"]

    assert model_audit["unrecoverable_count"] == 1
    assert model_audit["deterministically_recoverable_count"] == 1
    assert (tmp_path / "audit" / "qwen_3_5_4b_audit.json").exists()
    assert (tmp_path / "audit" / "text_wrapper_audit.json").exists()
    assert (tmp_path / "audit" / "proposed_recovery_rules.json").exists()
    assert (tmp_path / "audit" / "pre_freeze_summary.md").exists()
    assert not any("accuracy" in key for key in model_audit)
    assert any(rule["rule_id"] == "piece_source_destination" for rule in proposed)

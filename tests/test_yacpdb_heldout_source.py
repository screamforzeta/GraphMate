import json

from src.data.heldout_sources.yacpdb import (
    build_yacpdb_dataset,
    classify_stipulation,
    extract_key_uci,
    normalize_yacpdb_record,
    scan_availability,
)


MATE1_FEN = "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"
CASTLE_FEN = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
PROMOTION_FEN = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
CAPTURE_FEN = "4k3/8/8/8/8/5r2/8/4K2Q w - - 0 1"


def yacpdb_record(**overrides):
    record = {
        "id": "100",
        "stipulation": "#1",
        "fen": MATE1_FEN,
        "solution": "1.Qg7-f8 #",
        "composer": "Composer",
        "source": "Source",
        "license": "unknown",
    }
    record.update(overrides)
    return record


def test_classify_valid_orthodox_depths_and_reject_ranges():
    assert classify_stipulation({"stipulation": "#1"}) == (1, None)
    assert classify_stipulation({"stipulation": "#5"}) == (5, None)
    assert classify_stipulation({"stipulation": "#10"}) == (10, None)
    assert classify_stipulation({"stipulation": "#0"})[1] == "UNSUPPORTED_STIPULATION"
    assert classify_stipulation({"stipulation": "#11"})[1] == "UNSUPPORTED_STIPULATION"


def test_reject_non_directmate_fairy_twin_and_condition():
    assert classify_stipulation({"stipulation": "h#2"})[1] == "UNSUPPORTED_NON_DIRECTMATE"
    assert classify_stipulation({"stipulation": "s#2"})[1] == "UNSUPPORTED_NON_DIRECTMATE"
    assert classify_stipulation({"stipulation": "#2", "conditions": "Circe"})[1] == "UNSUPPORTED_FAIRY_CONDITION"
    assert classify_stipulation({"stipulation": "#2", "twins": "b) Ka1->a2"})[1] == "UNSUPPORTED_TWIN"


def test_extract_keys_capture_promotion_castling_and_annotations():
    assert extract_key_uci(yacpdb_record(solution="1.Qg7-f8 ! threat: 2.Qf8#"), MATE1_FEN) == ("g7f8", None)
    assert extract_key_uci(yacpdb_record(solution="1.Qh1*f3 #"), CAPTURE_FEN) == ("h1f3", None)
    assert extract_key_uci(yacpdb_record(solution="1.a7-a8=Q #"), PROMOTION_FEN) == ("a7a8q", None)
    assert extract_key_uci(yacpdb_record(solution="1.O-O !"), CASTLE_FEN) == ("e1g1", None)


def test_reject_try_ambiguous_malformed_unsupported_and_illegal_key():
    assert extract_key_uci(yacpdb_record(solution="1.Qg7-f8?"), MATE1_FEN)[1] == "TRY_ONLY"
    assert extract_key_uci(yacpdb_record(solution=""), MATE1_FEN)[1] == "NO_SOLUTION"
    assert extract_key_uci(yacpdb_record(solution="1.NotAMove"), MATE1_FEN)[1] == "UNSUPPORTED_MOVE_NOTATION"
    assert extract_key_uci(yacpdb_record(solution="1.Qg7-g1"), MATE1_FEN)[1] == "KEY_NOT_LEGAL"


def test_normalize_yacpdb_record_validates_key():
    candidate, reason = normalize_yacpdb_record(yacpdb_record(id="abc"))

    assert reason is None
    assert candidate.source_problem_id == "abc"
    assert candidate.mate_depth == 1
    assert candidate.key_move_uci == "g7f8"
    assert candidate.forced_mate_verification_status == "NOT_VERIFIED_ENGINE_NOT_USED"


def test_availability_scan_counts_rejections():
    payload = scan_availability(
        [
            yacpdb_record(id="ok"),
            yacpdb_record(id="bad-stip", stipulation="h#2"),
            yacpdb_record(id="bad-key", solution="1.Qg7-g1"),
        ]
    )

    assert payload["eligible_count"] == 1
    assert payload["availability_by_depth"]["MateIn1"]["eligible"] == 1
    assert payload["rejection_reasons"]["UNSUPPORTED_NON_DIRECTMATE"] == 1
    assert payload["rejection_reasons"]["KEY_NOT_LEGAL"] == 1


def test_build_yacpdb_dataset_validated_not_frozen(monkeypatch, tmp_path):
    records = [
        yacpdb_record(id="ok1"),
        yacpdb_record(id="ok2", fen=CASTLE_FEN, solution="1.O-O !"),
        yacpdb_record(id="dupe", fen=MATE1_FEN, solution="1.Qg7-f8 #"),
    ]
    monkeypatch.setattr(
        "src.data.heldout_sources.yacpdb.load_lichess_overlap_keys",
        lambda: {
            "train": {"exact": set(), "normalized": set(), "original_exact": set()},
            "val": {"exact": set(), "normalized": set(), "original_exact": set()},
            "test": {"exact": set(), "normalized": set(), "original_exact": set()},
        },
    )

    manifest = build_yacpdb_dataset(records, tmp_path / "heldout", "yacpdb_test", per_depth=None, seed=42)

    assert manifest["lifecycle_status"] == "VALIDATED"
    assert manifest["lifecycle_status"] != "FROZEN"
    assert manifest["selected_count"] == 2
    assert manifest["forced_mate_verified"] is False
    assert manifest["GNN_inference_performed"] is False
    assert manifest["LLM_inference_performed"] is False
    assert (tmp_path / "heldout" / "processed" / "yacpdb_manual_review.json").exists()
    assert (tmp_path / "heldout" / "final" / "manifest.json").exists()

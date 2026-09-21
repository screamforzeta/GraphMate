import json
from pathlib import Path

import chess
import pytest

from src.data.heldout_sources import yacpdb_availability as availability
from src.data.heldout_sources.yacpdb_client import YacpdbResponse


MATE1_FEN = "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"


def page(entries, count=None):
    return YacpdbResponse(
        method="GET",
        url="https://yacpdb.org/gateway/ql",
        status=200,
        content_type="application/json",
        body=json.dumps({"success": True, "result": {"entries": entries, "count": len(entries) if count is None else count}}).encode(),
    )


def directmate_record(record_id, fen=MATE1_FEN, solution="1.Qg7-f8 !"):
    return {
        "id": record_id,
        "stipulation": "#1",
        "fen": fen,
        "solution": solution,
        "authors": ["Composer"],
        "source": {"name": "Source", "date": {"year": 1900}},
    }


def test_query_for_each_depth_and_fingerprint_determinism():
    queries = {depth: availability.query_for_depth(depth) for depth in range(1, 11)}

    assert queries[1] == 'Stip("^#1$")'
    assert queries[10] == 'Stip("^#10$")'
    assert availability.scan_fingerprint({"queries": queries}) == availability.scan_fingerprint({"queries": dict(reversed(list(queries.items())))})


def test_per_depth_cap_pagination_and_duplicate_ids(monkeypatch, tmp_path):
    responses = {
        (1, 1): page([directmate_record(1), directmate_record(2)], count=4),
        (1, 2): page([directmate_record(2), directmate_record(3)], count=4),
    }

    def fake_load_or_fetch(client, depth_dir, depth, page_no, query):
        return responses[(depth, page_no)], False

    monkeypatch.setattr(availability, "load_or_fetch_page", fake_load_or_fetch)
    stats, review, completed = availability.scan_depth(
        client=object(),
        output_root=tmp_path,
        depth=1,
        per_depth_cap=3,
        lichess_keys={"train": {"exact": set(), "normalized": set(), "original_exact": set()}},
    )

    assert completed is True
    assert stats["source_reported_query_count"] == 4
    assert stats["records_inspected"] == 3
    assert stats["duplicate_yacpdb_ids"] == 1
    assert stats["eligible"] == 3
    assert len(review) == 2


def test_funnel_consistency_and_rejection_aggregation(monkeypatch, tmp_path):
    records = [
        directmate_record(1),
        {"id": 2, "stipulation": "h#2", "fen": MATE1_FEN, "solution": "1.Qg7-f8 !"},
        directmate_record(3, solution="1.Qg7-f8?"),
    ]

    def fake_load_or_fetch(client, depth_dir, depth, page_no, query):
        return page(records, count=3), False

    monkeypatch.setattr(availability, "load_or_fetch_page", fake_load_or_fetch)
    stats, _review, _completed = availability.scan_depth(
        client=object(),
        output_root=tmp_path,
        depth=1,
        per_depth_cap=10,
        lichess_keys={"train": {"exact": set(), "normalized": set(), "original_exact": set()}},
    )

    assert stats["records_inspected"] == stats["source_filter_pass"] + stats["source_filter_rejected"]
    assert stats["source_filter_pass"] == stats["position_normalized"] + stats["position_rejected"]
    assert stats["position_normalized"] == stats["key_extracted"] + stats["key_rejected"]
    assert stats["key_extracted"] == stats["key_legal"] + stats["key_illegal"]
    assert stats["rejection_reasons"]["UNSUPPORTED_NON_DIRECTMATE"] == 1
    assert stats["rejection_reasons"]["TRY_ONLY"] == 1


def test_duplicate_position_and_contamination_stats(monkeypatch, tmp_path):
    records = [directmate_record(1), directmate_record(2)]
    normalized = " ".join(chess.Board(MATE1_FEN).fen().split()[:4])

    def fake_load_or_fetch(client, depth_dir, depth, page_no, query):
        return page(records, count=2), False

    monkeypatch.setattr(availability, "load_or_fetch_page", fake_load_or_fetch)
    stats, _review, _completed = availability.scan_depth(
        client=object(),
        output_root=tmp_path,
        depth=1,
        per_depth_cap=10,
        lichess_keys={"train": {"exact": {MATE1_FEN}, "normalized": {normalized}, "original_exact": set()}},
    )

    assert stats["eligible"] == 2
    assert stats["unique_eligible_positions"] == 1
    assert stats["normalized_position_duplicates"] == 1
    assert stats["contaminated"] == 2
    assert stats["clean_unique_eligible"] == 0


def test_resume_rejects_incompatible_fingerprint(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "yacpdb_availability_state.json").write_text(json.dumps({"scan_fingerprint": "old"}))

    with pytest.raises(RuntimeError, match="Incompatible"):
        availability.run_scan(tmp_path, per_depth_cap=1, timeout=1)


def test_availability_artifact_serialization(monkeypatch, tmp_path):
    def fake_scan_depth(client, output_root, depth, per_depth_cap, lichess_keys):
        stats = availability.empty_depth_stats(depth, availability.query_for_depth(depth))
        stats["source_reported_query_count"] = 0
        return stats, [], True

    monkeypatch.setattr(availability, "scan_depth", fake_scan_depth)
    monkeypatch.setattr(availability, "load_lichess_overlap_keys", lambda: {})

    payload = availability.run_scan(tmp_path, per_depth_cap=1, timeout=1)

    assert payload["status"] == "COMPLETE"
    assert (tmp_path / "processed" / "yacpdb_availability.json").exists()
    assert "scan_fingerprint" in payload


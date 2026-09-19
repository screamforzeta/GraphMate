import json

from src.cli.evaluation.audit_llm_san_consistency import run_consistency_audit


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_san_consistency_audit_detects_permissive_python_chess_san(tmp_path):
    predictions = tmp_path / "official" / "next_move" / "qwen_3_5_4b" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True)
    rows = [
        {
            "puzzle_id": "p-san",
            "fen": START_FEN,
            "target_move": "h2h4",
            "raw_final_content": "Nf3+",
            "outcome": "PARSE_ERROR",
        },
        {
            "puzzle_id": "p-src",
            "fen": START_FEN,
            "target_move": "h2h4",
            "raw_final_content": "e2-e4",
            "outcome": "PARSE_ERROR",
        },
    ]
    with predictions.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    result = run_consistency_audit([str(predictions)], tmp_path / "audit")
    model_audit = result["model_audits"][0]
    san_example = model_audit["san_representative_examples"][0]
    source_case = model_audit["source_separator_destination_cases"][0]

    assert model_audit["san_discrepancy_count"] == 1
    assert model_audit["san_reason_counts"] == {"python_chess_permissive_noncanonical_san": 1}
    assert san_example["raw_final_content"] == "Nf3+"
    assert san_example["audit_candidate_uci"] == "g1f3"
    assert san_example["canonical_san"] == "Nf3"
    assert san_example["transformation_documented_in_relaxed_v1"] is False
    assert source_case["safe"] is True
    assert source_case["target_knowledge_used"] is False
    assert result["target_used"] is False
    assert (tmp_path / "audit" / "san_consistency_audit.json").exists()
    assert (tmp_path / "audit" / "san_consistency_summary.md").exists()

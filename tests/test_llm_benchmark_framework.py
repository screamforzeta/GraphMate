from pathlib import Path

import pytest

from src.llm.datasets import BenchmarkPuzzle, validate_heldout_classic_record
from src.llm.model_registry import discover_registry
from src.llm.parsing import parse_uci_response
from src.llm.prompting import build_prompt
from src.llm.resume import append_jsonl, assert_resume_config_matches, load_completed_keys
from src.llm.scoring import evaluate_reference_line, score_next_move, summarize_next_move_records


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PROMOTION_FEN = "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1"


def test_discover_registry_uses_actual_ollama_tags():
    registry = discover_registry(
        [
            {"name": "qwen3.5:4b", "size": 1},
            {"name": "qwen3.5:9b", "size": 2},
            {"name": "gpt-oss:20b", "size": 3},
        ]
    )

    assert registry["qwen_3_5_4b"]["ollama_model"] == "qwen3.5:4b"
    assert registry["qwen_3_5_9b"]["ollama_model"] == "qwen3.5:9b"
    assert registry["gpt_oss_20b"]["ollama_model"] == "gpt-oss:20b"


def test_prompt_has_no_target_or_metadata_leakage():
    prompt = build_prompt(START_FEN)
    payload = f"{prompt['system']}\n{prompt['user']}"

    assert START_FEN in payload
    assert "TargetMove" not in payload
    assert "MateDepth" not in payload
    assert "Rating" not in payload
    assert "Themes" not in payload
    assert "legal moves" not in payload.lower()


@pytest.mark.parametrize(
    ("raw", "accepted"),
    [
        ("e2e4", True),
        (" e2e4 ", True),
        ("e7e8q", True),
        ("The best move is e2e4.", False),
        ("e2-e4", False),
        ("Nf3", False),
        ("e9e4", False),
        ("0000", False),
        ("", False),
        ("e2e4 e7e5", False),
        ("```e2e4```", False),
    ],
)
def test_strict_parser_policy(raw, accepted):
    fen = PROMOTION_FEN if raw.strip() == "e7e8q" else START_FEN
    result = parse_uci_response(raw, fen)

    assert result.parse_success is accepted


def test_legality_and_next_move_scoring():
    correct = score_next_move("e2e4", START_FEN, "e2e4")
    wrong = score_next_move("d2d4", START_FEN, "e2e4")
    illegal = score_next_move("e2e5", START_FEN, "e2e4")

    assert correct["outcome"] == "CORRECT"
    assert wrong["outcome"] == "LEGAL_BUT_WRONG"
    assert illegal["outcome"] == "ILLEGAL_MOVE"


def test_reference_line_stops_on_wrong_or_illegal_and_completes_mate_in_one():
    puzzle = BenchmarkPuzzle("smoke", START_FEN, "e2e4", ["e2e4", "e7e5"], mate_depth=1)

    solved = evaluate_reference_line(puzzle, lambda _fen: "e2e4")
    wrong = evaluate_reference_line(puzzle, lambda _fen: "d2d4")
    illegal = evaluate_reference_line(puzzle, lambda _fen: "e2e5")

    assert solved["solved"] is True
    assert solved["solver_moves_required"] == 1
    assert wrong["solved"] is False
    assert wrong["failure_step"] == 1
    assert illegal["steps"][0]["outcome"] == "ILLEGAL_MOVE"


def test_reference_line_mate_depth_semantics_and_no_chat_history_needed():
    puzzle = BenchmarkPuzzle(
        "mate2_like",
        START_FEN,
        "e2e4",
        ["e2e4", "e7e5", "g1f3"],
        mate_depth=2,
    )
    seen_fens = []

    def predictor(fen):
        seen_fens.append(fen)
        return "e2e4" if len(seen_fens) == 1 else "g1f3"

    result = evaluate_reference_line(puzzle, predictor)

    assert result["solved"] is True
    assert result["solver_moves_required"] == 2
    assert len(seen_fens) == 2
    assert seen_fens[0] != seen_fens[1]


def test_summary_counts_and_resume_guards(tmp_path):
    records = [
        score_next_move("e2e4", START_FEN, "e2e4"),
        score_next_move("d2d4", START_FEN, "e2e4"),
        score_next_move("e2e5", START_FEN, "e2e4"),
    ]
    summary = summarize_next_move_records(records)

    assert summary["N"] == 3
    assert summary["correct"] == 1
    assert summary["illegal_move_count"] == 1

    config = {
        "run_id": "r1",
        "model_id": "m",
        "ollama_model": "tag",
        "dataset": "lichess-test",
        "protocol": "next-move",
        "dataset_fingerprint": "abc",
        "prompt_hash": "prompt",
        "parser_version": "parser",
        "generation_options": {"temperature": 0},
    }
    assert_resume_config_matches(config, dict(config))
    changed = dict(config)
    changed["prompt_hash"] = "other"
    with pytest.raises(ValueError):
        assert_resume_config_matches(config, changed)

    path = tmp_path / "raw_predictions.jsonl"
    append_jsonl(path, {"run_id": "r1", "model_id": "m", "puzzle_id": "p", "protocol": "next-move"})
    assert ("r1", "m", "p", "next-move") in load_completed_keys(path)


def test_heldout_classic_schema():
    assert validate_heldout_classic_record(
        {
            "puzzle_id": "classic-1",
            "source": "fixture",
            "initial_solver_fen": START_FEN,
            "reference_moves_uci": ["e2e4"],
            "mate_depth": 1,
        }
    )
    with pytest.raises(ValueError):
        validate_heldout_classic_record({"puzzle_id": "bad"})

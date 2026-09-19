"""Scoring and rollout logic for LLM chess benchmarks."""

from __future__ import annotations

import chess

from src.llm.parsing import parse_uci_response


def score_next_move(raw_response, fen, target_move):
    """Score one next-move prediction against a canonical target UCI."""

    parsed = parse_uci_response(raw_response, fen)
    correct = parsed.legal and parsed.parsed_uci == str(target_move)
    if not parsed.parse_success:
        outcome = "PARSE_ERROR"
    elif not parsed.legal:
        outcome = "ILLEGAL_MOVE"
    elif correct:
        outcome = "CORRECT"
    else:
        outcome = "LEGAL_BUT_WRONG"
    return {
        "raw_response": parsed.raw_response,
        "parsed_uci": parsed.parsed_uci,
        "parsed_san": parsed.san,
        "parse_success": parsed.parse_success,
        "legal": parsed.legal,
        "target_move": str(target_move),
        "correct": correct,
        "outcome": outcome,
    }


def reference_line_pairs(reference_moves):
    """Return solving-side move and optional opponent reply pairs."""

    return [
        (
            reference_moves[index],
            reference_moves[index + 1] if index + 1 < len(reference_moves) else None,
        )
        for index in range(0, len(reference_moves), 2)
    ]


def evaluate_reference_line(puzzle, predict_raw_move):
    """Evaluate one independent-prompt reference-line attempt."""

    board = chess.Board(puzzle.initial_solver_fen)
    steps = []
    solved = True
    failure_step = None
    for step_number, (expected_solver_move, opponent_reply) in enumerate(
        reference_line_pairs(puzzle.reference_moves_uci),
        1,
    ):
        fen_before = board.fen()
        raw_response = predict_raw_move(fen_before)
        scored = score_next_move(raw_response, fen_before, expected_solver_move)
        step = {
            "solver_step": step_number,
            "fen_before": fen_before,
            "raw_response": raw_response,
            "model_move_uci": scored["parsed_uci"],
            "model_move_san": scored["parsed_san"],
            "reference_move_uci": expected_solver_move,
            "correct": scored["correct"],
            "outcome": scored["outcome"],
            "opponent_reply_uci": opponent_reply,
            "opponent_reply_san": None,
        }
        steps.append(step)
        if not scored["correct"]:
            solved = False
            failure_step = step_number
            break
        move = chess.Move.from_uci(expected_solver_move)
        board.push(move)
        step["fen_after_model_move"] = board.fen()
        if opponent_reply is not None:
            reply = chess.Move.from_uci(opponent_reply)
            if reply not in board.legal_moves:
                solved = False
                failure_step = step_number
                step["outcome"] = "REFERENCE_REPLY_ILLEGAL"
                break
            step["opponent_reply_san"] = board.san(reply)
            board.push(reply)
            step["fen_after_reference_reply"] = board.fen()
    return {
        "puzzle_id": puzzle.puzzle_id,
        "solved": solved,
        "failure_step": failure_step,
        "steps": steps,
        "solver_moves_correct": sum(1 for step in steps if step["correct"]),
        "solver_moves_required": len(reference_line_pairs(puzzle.reference_moves_uci)),
        "final_fen": board.fen(),
    }


def summarize_next_move_records(records):
    """Aggregate next-move records into benchmark metrics."""

    total = len(records)
    correct = sum(1 for row in records if row.get("correct"))
    parse_errors = sum(1 for row in records if row.get("outcome") == "PARSE_ERROR")
    illegal = sum(1 for row in records if row.get("outcome") == "ILLEGAL_MOVE")
    legal_wrong = sum(1 for row in records if row.get("outcome") == "LEGAL_BUT_WRONG")
    return {
        "N": total,
        "correct": correct,
        "top1_accuracy": correct / total if total else None,
        "parse_failure_count": parse_errors,
        "parse_failure_rate": parse_errors / total if total else None,
        "illegal_move_count": illegal,
        "illegal_move_rate": illegal / total if total else None,
        "legal_but_wrong_count": legal_wrong,
        "legal_but_wrong_rate": legal_wrong / total if total else None,
    }


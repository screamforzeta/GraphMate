"""Current-position inference helpers for the Streamlit puzzle UI."""

from __future__ import annotations

from types import SimpleNamespace

import chess

from src.streamlit_app.puzzle_sequence import PuzzleSession


def row_for_current_fen(row, current_fen, target_move=None):
    """Return a row-like copy whose FEN is the current board position.

    Parameters:
        row: Pandas row-like object from a puzzle split.
        current_fen: FEN currently shown in the UI.
        target_move: Optional target move for hidden evaluation at this FEN.
    Returns:
        SimpleNamespace with the original row fields and updated FEN/TargetMove.
    Side effects:
        None.
    """

    payload = row.to_dict() if hasattr(row, "to_dict") else vars(row).copy()
    payload["FEN"] = str(current_fen)
    if target_move is not None:
        payload["TargetMove"] = str(target_move)
    return SimpleNamespace(**payload)


def current_solver_target(session):
    """Return the next solver move expected from the current session state.

    Parameters:
        session: PuzzleSession with human/reference moves already applied.
    Returns:
        UCI move string, or None when the puzzle is complete.
    Side effects:
        None.
    """

    return session.next_expected_move()


def invalidate_prediction_if_fen_changed(session_state, current_fen):
    """Clear cached model predictions when the board position changes.

    Parameters:
        session_state: Mutable Streamlit-like state mapping.
        current_fen: Canonical current board FEN.
    Returns:
        True when a stale prediction was cleared.
    Side effects:
        Mutates session_state keys for model predictions.
    """

    previous_fen = session_state.get("model_prediction_fen")
    if previous_fen is None or previous_fen == current_fen:
        return False
    session_state["model_result"] = None
    session_state["a4_result"] = None
    session_state["model_prediction_fen"] = None
    return True


def reference_line_pairs(solution_moves):
    """Split a Lichess solution line into solver moves and reference replies.

    Parameters:
        solution_moves: UCI moves after the Lichess setup move.
    Returns:
        List of (solver_move, optional_reply_move) tuples.
    Side effects:
        None.
    """

    pairs = []
    for index in range(0, len(solution_moves), 2):
        solver_move = solution_moves[index]
        reply_move = solution_moves[index + 1] if index + 1 < len(solution_moves) else None
        pairs.append((solver_move, reply_move))
    return pairs


def run_reference_line_rollout(initial_solver_fen, solution_moves, predict_next_move):
    """Attempt a puzzle along canonical Lichess reference replies.

    Parameters:
        initial_solver_fen: FEN at the first solver turn.
        solution_moves: UCI solution moves after the setup move.
        predict_next_move: Callable receiving current FEN and returning a UCI move.
    Returns:
        Dictionary with rollout rows, solved flag, failure ply, and final FEN.
    Side effects:
        Calls predict_next_move but does not mutate any PuzzleSession.
    """

    board = chess.Board(initial_solver_fen)
    rows = []
    solved = True
    failure_ply = None
    for solver_index, (expected_solver_move, reference_reply) in enumerate(
        reference_line_pairs(solution_moves),
        1,
    ):
        position_fen = board.fen()
        predicted = predict_next_move(position_fen)
        predicted_move = chess.Move.from_uci(str(predicted)) if predicted else None
        expected_move = chess.Move.from_uci(expected_solver_move)
        predicted_san = board.san(predicted_move) if predicted_move in board.legal_moves else None
        expected_san = board.san(expected_move) if expected_move in board.legal_moves else None
        correct = predicted == expected_solver_move
        rows.append(
            {
                "solver_ply": solver_index,
                "position_fen": position_fen,
                "predicted_uci": predicted,
                "predicted_san": predicted_san,
                "expected_uci": expected_solver_move,
                "expected_san": expected_san,
                "correct": correct,
                "reference_reply_uci": reference_reply,
                "reference_reply_san": None,
            }
        )
        if not correct or expected_move not in board.legal_moves:
            solved = False
            failure_ply = solver_index
            break
        board.push(expected_move)
        if reference_reply is not None:
            reply = chess.Move.from_uci(reference_reply)
            if reply not in board.legal_moves:
                solved = False
                failure_ply = solver_index
                break
            rows[-1]["reference_reply_san"] = board.san(reply)
            board.push(reply)
    return {
        "rows": rows,
        "solved": solved,
        "failure_ply": failure_ply,
        "final_fen": board.fen(),
        "solver_moves_correct": sum(1 for row in rows if row["correct"]),
        "solver_moves_required": len(reference_line_pairs(solution_moves)),
    }


def session_from_initial_solver_fen(initial_solver_fen, solution_moves):
    """Create a puzzle session with explicit initial/current FEN semantics."""

    return PuzzleSession.create(initial_solver_fen, solution_moves)

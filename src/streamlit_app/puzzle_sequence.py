"""Puzzle filtering and move-sequence helpers for Mate-in-N UI workflows."""

from __future__ import annotations

from dataclasses import dataclass

import chess


MATE_DEPTH_OPTIONS = ["All", "Mate in 1", "Mate in 2", "Mate in 3", "Mate in 4", "Mate in 5"]


def mate_depth_from_label(label):
    """Return MateDepth integer for a UI label, or None for All."""

    if label == "All":
        return None
    return int(str(label).rsplit(" ", 1)[1])


def filter_by_mate_depth(dataframe, label):
    """Filter a puzzle DataFrame by MateDepth UI label."""

    depth = mate_depth_from_label(label)
    if depth is None:
        return dataframe
    return dataframe[dataframe["MateDepth"].astype(int) == depth]


def split_moves(moves):
    """Split a Lichess puzzle Moves field into UCI moves."""

    return [move for move in str(moves).split() if move]


def solution_moves_from_lichess_moves(moves):
    """Return solver solution moves, excluding the Lichess setup move."""

    parsed = split_moves(moves)
    return parsed[1:]


def transformed_solver_fen(original_fen, moves):
    """Apply Moves[0] to OriginalFEN and return the solver-position FEN."""

    parsed = split_moves(moves)
    if not parsed:
        raise ValueError("Moves field is empty.")
    board = chess.Board(original_fen)
    setup = chess.Move.from_uci(parsed[0])
    if setup not in board.legal_moves:
        raise ValueError(f"Illegal setup move: {parsed[0]}")
    board.push(setup)
    return board.fen()


def solution_san_sequence(fen, solution_moves):
    """Convert a solution UCI sequence to SAN from the solver position."""

    board = chess.Board(fen)
    rows = []
    for ply, move_uci in enumerate(solution_moves, 1):
        move = chess.Move.from_uci(str(move_uci))
        if move not in board.legal_moves:
            rows.append({"ply": ply, "uci": move_uci, "san": None, "legal": False})
            break
        san = board.san(move)
        rows.append({"ply": ply, "uci": move_uci, "san": san, "legal": True})
        board.push(move)
    return rows


@dataclass
class PuzzleSession:
    """Move-by-move human puzzle state independent of Streamlit."""

    start_fen: str
    solution_moves: list[str]
    human_moves: list[str]
    current_fen: str

    @classmethod
    def create(cls, start_fen, solution_moves):
        """Create a new session at the solver position."""

        return cls(
            start_fen=str(start_fen),
            solution_moves=list(solution_moves),
            human_moves=[],
            current_fen=str(start_fen),
        )

    def legal_moves(self):
        """Return legal UCI moves for the current position."""

        return [move.uci() for move in chess.Board(self.current_fen).legal_moves]

    def play(self, move_uci):
        """Apply one legal human move and advance state."""

        board = chess.Board(self.current_fen)
        move = chess.Move.from_uci(str(move_uci))
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move: {move_uci}")
        board.push(move)
        self.human_moves.append(str(move_uci))
        self.current_fen = board.fen()
        return self

    def undo(self):
        """Undo the last human move by replaying from the start position."""

        if self.human_moves:
            self.human_moves.pop()
        board = chess.Board(self.start_fen)
        replay = []
        for move_uci in self.human_moves:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                replay.append(move_uci)
        self.human_moves = replay
        self.current_fen = board.fen()
        return self

    def reset(self):
        """Reset to the start position."""

        self.human_moves = []
        self.current_fen = self.start_fen
        return self

    def is_exact_solution_prefix(self):
        """Return whether human moves match the solution prefix."""

        return self.human_moves == self.solution_moves[: len(self.human_moves)]

    def is_complete_solution(self):
        """Return whether human moves exactly equal the solution sequence."""

        return self.human_moves == self.solution_moves


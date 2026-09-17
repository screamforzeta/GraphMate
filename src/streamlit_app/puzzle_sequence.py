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


def move_to_san_label(fen, move_uci):
    """Return a SAN-first label for one legal UCI move.

    Parameters:
        fen: Current board FEN.
        move_uci: Candidate move in UCI notation.
    Returns:
        String with SAN notation and UCI in parentheses.
    Side effects:
        None.
    """

    board = chess.Board(fen)
    move = chess.Move.from_uci(str(move_uci))
    san = board.san(move) if move in board.legal_moves else "illegal"
    return f"{san} ({move_uci})"


def legal_move_options(fen):
    """Return legal moves as SAN-first UI options.

    Parameters:
        fen: Current board FEN.
    Returns:
        List of dictionaries containing uci, san, and label.
    Side effects:
        None.
    """

    board = chess.Board(fen)
    options = []
    for move in board.legal_moves:
        san = board.san(move)
        options.append({"uci": move.uci(), "san": san, "label": f"{san} ({move.uci()})"})
    return options


def move_history_rows(start_fen, moves, auto_reply_indices=None):
    """Return SAN move history rows for played user/reference moves.

    Parameters:
        start_fen: Initial solver-position FEN.
        moves: Played move sequence in UCI notation.
        auto_reply_indices: Zero-based move indices inserted as reference replies.
    Returns:
        List of display rows with ply, side, SAN, UCI, and source label.
    Side effects:
        None.
    """

    auto_reply_indices = set(auto_reply_indices or [])
    board = chess.Board(start_fen)
    rows = []
    for index, move_uci in enumerate(moves):
        move = chess.Move.from_uci(str(move_uci))
        if move not in board.legal_moves:
            rows.append(
                {
                    "ply": index + 1,
                    "side": "White" if board.turn == chess.WHITE else "Black",
                    "san": None,
                    "uci": move_uci,
                    "source": "Reference reply" if index in auto_reply_indices else "Your move",
                    "legal": False,
                }
            )
            break
        san = board.san(move)
        rows.append(
            {
                "ply": index + 1,
                "side": "White" if board.turn == chess.WHITE else "Black",
                "san": san,
                "uci": move_uci,
                "source": "Reference reply" if index in auto_reply_indices else "Your move",
                "legal": True,
            }
        )
        board.push(move)
    return rows


@dataclass
class PuzzleSession:
    """Move-by-move human puzzle state independent of Streamlit."""

    start_fen: str
    solution_moves: list[str]
    human_moves: list[str]
    current_fen: str
    auto_reply_indices: list[int] | None = None

    @classmethod
    def create(cls, start_fen, solution_moves):
        """Create a new session at the solver position."""

        return cls(
            start_fen=str(start_fen),
            solution_moves=list(solution_moves),
            human_moves=[],
            current_fen=str(start_fen),
            auto_reply_indices=[],
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
        """Undo the last user move and any trailing reference reply."""

        if self.human_moves:
            self.human_moves.pop()
        if self.human_moves and (len(self.human_moves) - 1) in set(self.auto_reply_indices or []):
            self.human_moves.pop()
        board = chess.Board(self.start_fen)
        replay = []
        replay_auto_indices = []
        for move_uci in self.human_moves:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves:
                board.push(move)
                if len(replay) in set(self.auto_reply_indices or []):
                    replay_auto_indices.append(len(replay))
                replay.append(move_uci)
        self.human_moves = replay
        self.auto_reply_indices = replay_auto_indices
        self.current_fen = board.fen()
        return self

    def reset(self):
        """Reset to the start position."""

        self.human_moves = []
        self.auto_reply_indices = []
        self.current_fen = self.start_fen
        return self

    def is_exact_solution_prefix(self):
        """Return whether human moves match the solution prefix."""

        return self.human_moves == self.solution_moves[: len(self.human_moves)]

    def is_complete_solution(self):
        """Return whether human moves exactly equal the solution sequence."""

        return self.human_moves == self.solution_moves

    def play_solver_move_with_reference_reply(self, move_uci):
        """Play one solver move and auto-play the reference opponent reply.

        Parameters:
            move_uci: Solver move proposed by the user in UCI notation.
        Returns:
            Dictionary with status, message, played move, optional auto reply,
            and completion flag.
        Side effects:
            Mutates this session only when the user move is legal and matches
            the next solution ply.
        """

        board = chess.Board(self.current_fen)
        move = chess.Move.from_uci(str(move_uci))
        if move not in board.legal_moves:
            return {
                "move": str(move_uci),
                "status": "ILLEGAL",
                "message": "Illegal move in the current position.",
                "auto_reply": None,
                "complete": False,
            }

        expected_index = len(self.human_moves)
        expected_move = (
            self.solution_moves[expected_index]
            if expected_index < len(self.solution_moves)
            else None
        )
        if str(move_uci) != expected_move:
            return {
                "move": str(move_uci),
                "status": "INCORRECT",
                "message": "Incorrect move.",
                "auto_reply": None,
                "complete": False,
            }

        self.play(move_uci)
        auto_reply = None
        if len(self.human_moves) < len(self.solution_moves):
            reply_uci = self.solution_moves[len(self.human_moves)]
            reply_board = chess.Board(self.current_fen)
            reply = chess.Move.from_uci(reply_uci)
            if reply in reply_board.legal_moves:
                auto_reply_index = len(self.human_moves)
                self.play(reply_uci)
                self.auto_reply_indices = list(self.auto_reply_indices or [])
                self.auto_reply_indices.append(auto_reply_index)
                auto_reply = reply_uci

        complete = self.is_complete_solution()
        return {
            "move": str(move_uci),
            "status": "COMPLETE" if complete else "CORRECT",
            "message": "Puzzle solved." if complete else "Correct move - keep going.",
            "auto_reply": auto_reply,
            "complete": complete,
        }

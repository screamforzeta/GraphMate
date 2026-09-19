"""Frozen secondary parser for post-hoc LLM chess move analysis.

`relaxed_chess_move_v1` accepts only whole-string, deterministic chess move
notations. It uses raw model output plus the current FEN, never target moves or
benchmark correctness, and never extracts arbitrary inner substrings.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import chess

from src.llm.parsing import UCI_PATTERN


RELAXED_PARSER_VERSION = "relaxed_chess_move_v1"
RELAXED_PARSER_STATUS = "FROZEN"
PIECE_SOURCE_DESTINATION_PATTERN = re.compile(
    r"^(?P<piece>[KQRBN])(?P<src>[a-h][1-8])(?P<sep>[-x]?)(?P<dst>[a-h][1-8])(?P<suffix>[+#]?)$"
)
SOURCE_SEPARATOR_DESTINATION_PATTERN = re.compile(
    r"^(?P<src>[a-h][1-8])(?P<sep>[-x])(?P<dst>[a-h][1-8])$"
)
PIECE_SYMBOLS = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
}


@dataclass(frozen=True)
class RelaxedParseResult:
    """Structured result for one relaxed parser decision."""

    raw_content: str
    normalized_content: str
    parse_status: str
    parse_method: str
    candidate_notation: str | None = None
    parsed_uci: str | None = None
    parsed_san: str | None = None
    is_legal: bool = False
    ambiguity_count: int = 0
    candidate_count: int = 0
    failure_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable representation."""

        return asdict(self)


def normalize_outer_text(raw_content: str | None) -> tuple[str, list[str]]:
    """Normalize only transport-level wrapping, not chess notation semantics."""

    raw = "" if raw_content is None else str(raw_content)
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    diagnostics = ["strip_whitespace"] if text != raw else []
    if text.startswith("```") and text.endswith("```"):
        inner = text[3:-3].strip()
        if "\n" not in inner:
            text = inner
            diagnostics.append("removed_single_token_code_fence")
    elif text.startswith("`") and text.endswith("`") and text.count("`") == 2:
        text = text[1:-1].strip()
        diagnostics.append("removed_single_token_code_span")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
        diagnostics.append("removed_surrounding_quotes")
    return text, diagnostics


def successful_result(
    raw: str,
    normalized: str,
    board: chess.Board,
    move: chess.Move,
    method: str,
    diagnostics: dict | None = None,
) -> RelaxedParseResult:
    """Build a successful parse result for one legal move."""

    return RelaxedParseResult(
        raw_content=raw,
        normalized_content=normalized,
        parse_status="PARSED",
        parse_method=method,
        candidate_notation=normalized,
        parsed_uci=move.uci(),
        parsed_san=board.san(move),
        is_legal=True,
        candidate_count=1,
        diagnostics=diagnostics or {},
    )


def parse_exact_uci(raw: str, normalized: str, board: chess.Board, diagnostics: dict) -> RelaxedParseResult | None:
    """Apply whole-string exact UCI parsing."""

    if not UCI_PATTERN.fullmatch(normalized):
        return None
    try:
        move = chess.Move.from_uci(normalized)
    except ValueError as exc:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "EXACT_UCI",
            candidate_notation=normalized,
            failure_reason=f"invalid_uci:{exc}",
            diagnostics=diagnostics,
        )
    if move not in board.legal_moves:
        return RelaxedParseResult(
            raw,
            normalized,
            "ILLEGAL_MOVE",
            "EXACT_UCI",
            candidate_notation=normalized,
            parsed_uci=normalized,
            failure_reason="syntactically_valid_uci_illegal",
            candidate_count=1,
            diagnostics=diagnostics,
        )
    return successful_result(raw, normalized, board, move, "EXACT_UCI", diagnostics)


def parse_exact_san(raw: str, normalized: str, board: chess.Board, diagnostics: dict) -> RelaxedParseResult | None:
    """Apply whole-string exact SAN parsing via python-chess."""

    try:
        move = board.parse_san(normalized)
    except ValueError:
        return None
    if board.san(move) != normalized:
        return None
    return successful_result(raw, normalized, board, move, "EXACT_SAN_RECHECK", diagnostics)


def parse_piece_source_destination(
    raw: str,
    normalized: str,
    board: chess.Board,
    diagnostics: dict,
) -> RelaxedParseResult | None:
    """Parse whole-string piece + source + optional separator + destination."""

    match = PIECE_SOURCE_DESTINATION_PATTERN.fullmatch(normalized)
    if not match:
        return None
    piece_symbol = match.group("piece")
    source = match.group("src")
    destination = match.group("dst")
    separator = match.group("sep")
    source_square = chess.parse_square(source)
    destination_square = chess.parse_square(destination)
    piece = board.piece_at(source_square)
    rule_diagnostics = diagnostics | {"separator": separator or "", "source": source, "destination": destination}
    if not piece or piece.color != board.turn or piece.piece_type != PIECE_SYMBOLS[piece_symbol]:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "PIECE_SOURCE_DESTINATION",
            candidate_notation=normalized,
            failure_reason="piece_prefix_or_source_mismatch",
            diagnostics=rule_diagnostics,
        )
    move = chess.Move(source_square, destination_square)
    if move not in board.legal_moves:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "PIECE_SOURCE_DESTINATION",
            candidate_notation=normalized,
            parsed_uci=move.uci(),
            failure_reason="explicit_piece_source_destination_illegal",
            candidate_count=1,
            diagnostics=rule_diagnostics,
        )
    if separator == "x" and not board.is_capture(move):
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "PIECE_SOURCE_DESTINATION",
            candidate_notation=normalized,
            parsed_uci=move.uci(),
            failure_reason="capture_separator_without_capture",
            candidate_count=1,
            diagnostics=rule_diagnostics,
        )
    return successful_result(raw, normalized, board, move, "PIECE_SOURCE_DESTINATION", rule_diagnostics)


def parse_source_separator_destination(
    raw: str,
    normalized: str,
    board: chess.Board,
    diagnostics: dict,
) -> RelaxedParseResult | None:
    """Parse whole-string source + '-'/'x' + destination notation."""

    match = SOURCE_SEPARATOR_DESTINATION_PATTERN.fullmatch(normalized)
    if not match:
        return None
    source = match.group("src")
    destination = match.group("dst")
    separator = match.group("sep")
    source_square = chess.parse_square(source)
    destination_square = chess.parse_square(destination)
    rule_diagnostics = diagnostics | {"separator": separator, "source": source, "destination": destination}
    all_source_destination_candidates = [
        move
        for move in board.legal_moves
        if move.from_square == source_square
        and move.to_square == destination_square
    ]
    if len(all_source_destination_candidates) > 1:
        return RelaxedParseResult(
            raw,
            normalized,
            "AMBIGUOUS",
            "SOURCE_SEPARATOR_DESTINATION",
            candidate_notation=normalized,
            ambiguity_count=len(all_source_destination_candidates),
            candidate_count=len(all_source_destination_candidates),
            failure_reason="multiple_legal_moves_match_source_destination",
            diagnostics=rule_diagnostics | {"candidate_uci": [move.uci() for move in all_source_destination_candidates]},
        )
    candidates = [move for move in all_source_destination_candidates if move.promotion is None]
    if len(candidates) > 1:
        return RelaxedParseResult(
            raw,
            normalized,
            "AMBIGUOUS",
            "SOURCE_SEPARATOR_DESTINATION",
            candidate_notation=normalized,
            ambiguity_count=len(candidates),
            candidate_count=len(candidates),
            failure_reason="multiple_legal_moves_match_source_destination",
            diagnostics=rule_diagnostics | {"candidate_uci": [move.uci() for move in candidates]},
        )
    if not candidates:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "SOURCE_SEPARATOR_DESTINATION",
            candidate_notation=normalized,
            failure_reason="no_legal_move_matches_source_destination",
            diagnostics=rule_diagnostics,
        )
    move = candidates[0]
    is_capture = board.is_capture(move)
    if separator == "-" and is_capture:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "SOURCE_SEPARATOR_DESTINATION",
            candidate_notation=normalized,
            parsed_uci=move.uci(),
            candidate_count=1,
            failure_reason="dash_separator_used_for_capture",
            diagnostics=rule_diagnostics,
        )
    if separator == "x" and not is_capture:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "SOURCE_SEPARATOR_DESTINATION",
            candidate_notation=normalized,
            parsed_uci=move.uci(),
            candidate_count=1,
            failure_reason="capture_separator_without_capture",
            diagnostics=rule_diagnostics,
        )
    return successful_result(raw, normalized, board, move, "SOURCE_SEPARATOR_DESTINATION", rule_diagnostics)


def relaxed_parse_move(raw_final_content: str | None, fen: str) -> RelaxedParseResult:
    """Parse a model response as one deterministic legal chess move."""

    raw = "" if raw_final_content is None else str(raw_final_content)
    normalized, steps = normalize_outer_text(raw)
    diagnostics = {
        "normalization_steps": steps,
        "target_used": False,
        "substring_extraction": False,
        "parser_status": RELAXED_PARSER_STATUS,
    }
    if not normalized:
        return RelaxedParseResult(
            raw,
            normalized,
            "EMPTY",
            "EMPTY",
            failure_reason="empty_after_normalization",
            diagnostics=diagnostics,
        )
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return RelaxedParseResult(
            raw,
            normalized,
            "UNRECOVERABLE",
            "UNRECOVERABLE",
            failure_reason=f"invalid_fen:{exc}",
            diagnostics=diagnostics,
        )
    for parser in (
        parse_exact_uci,
        parse_exact_san,
        parse_piece_source_destination,
        parse_source_separator_destination,
    ):
        result = parser(raw, normalized, board, diagnostics)
        if result is not None:
            return result
    return RelaxedParseResult(
        raw,
        normalized,
        "UNRECOVERABLE",
        "UNRECOVERABLE",
        failure_reason="no_whole_string_deterministic_rule",
        diagnostics=diagnostics,
    )

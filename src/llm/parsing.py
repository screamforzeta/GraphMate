"""Strict deterministic parsing for LLM chess move outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess


PARSER_VERSION = "strict_uci_v1"
UCI_PATTERN = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


@dataclass(frozen=True)
class ParseResult:
    """Parsed move result with legality metadata."""

    raw_response: str
    parsed_uci: str | None
    parse_success: bool
    legal: bool
    status: str
    san: str | None = None
    error: str | None = None


def parse_uci_response(raw_response, fen):
    """Parse a raw LLM response as exactly one UCI move.

    The frozen policy accepts only a stripped single-token UCI move such as
    e2e4 or e7e8q. Verbose answers, SAN, coordinate hyphens, code fences, and
    multiple moves are parse errors.
    """

    raw = "" if raw_response is None else str(raw_response)
    text = raw.strip()
    if not UCI_PATTERN.fullmatch(text):
        return ParseResult(raw, None, False, False, "PARSE_ERROR", error="not_single_uci")
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(text)
    except ValueError as error:
        return ParseResult(raw, text, False, False, "PARSE_ERROR", error=str(error))
    if move not in board.legal_moves:
        return ParseResult(raw, text, True, False, "ILLEGAL_MOVE")
    return ParseResult(raw, text, True, True, "LEGAL_MOVE", san=board.san(move))


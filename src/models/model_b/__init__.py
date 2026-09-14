"""Timing-aware Model B architectures."""

from src.models.model_b.chess_timing_legal_scorer import (
    ChessGATTimingLegalMoveScorer,
    ChessGATWithTiming,
)


__all__ = [
    "ChessGATTimingLegalMoveScorer",
    "ChessGATWithTiming",
]

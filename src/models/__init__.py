"""Project neural network model exports.

Purpose:
    Expose chess-specific PyTorch/PyG models used after graph generation.
Input:
    Model modules under src/models/.
Output:
    Public model classes and small model utilities.
Role:
    Keeps training and smoke-test imports stable as more models are added.
"""

from src.models.model_a.chess_gat import (
    ChessGATNoTiming,
    count_trainable_parameters,
)
from src.models.model_a.chess_legal_scorer import (
    ChessGATLegalMoveScorer,
    promotion_id,
)
from src.models.model_b.chess_timing_legal_scorer import (
    ChessGATTimingLegalMoveScorer,
    ChessGATWithTiming,
)


__all__ = [
    "ChessGATNoTiming",
    "ChessGATLegalMoveScorer",
    "ChessGATTimingLegalMoveScorer",
    "ChessGATWithTiming",
    "count_trainable_parameters",
    "promotion_id",
]

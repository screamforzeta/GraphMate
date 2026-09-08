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

from src.models.chess_gat import (
    ChessGATNoTiming,
    count_trainable_parameters,
)
from src.models.chess_legal_scorer import (
    ChessGATLegalMoveScorer,
    promotion_id,
)


__all__ = [
    "ChessGATNoTiming",
    "ChessGATLegalMoveScorer",
    "count_trainable_parameters",
    "promotion_id",
]

"""Training entry points for timing-aware Model B on the A3 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.models.model_b.chess_timing_legal_scorer import (
    ChessGATTimingLegalMoveScorer,
)
from src.training.model_a.model_a3_legal_scorer import (
    ModelA3LegalScorerConfig,
    evaluate_a3,
    load_a3_checkpoint,
    run_a3_epoch,
    save_a3_checkpoint,
    train_model_a3,
)


RUN_ID = "MODEL_B_TIMING_LEGAL_MOVE_SCORER"
OUTPUT_ROOT = Path("artifacts/model_b_timing_legal_move_scorer")


@dataclass
class ModelBTimingLegalScorerConfig(ModelA3LegalScorerConfig):
    """Model B config: same optimization frame as A3 plus timing width."""

    timing_hidden_dim: int = 16
    output_root: str = str(OUTPUT_ROOT)
    run_id: str = RUN_ID
    parent_run_id: str = "MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING"
    model_class: str = "ChessGATTimingLegalMoveScorer"
    config_source: str = "EXTENDS_MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING"


def build_model_b(config):
    """Instantiate Model B as a timing-aware extension of A3."""

    return ChessGATTimingLegalMoveScorer(
        input_dim=15,
        edge_dim=5,
        hidden_per_head=32,
        heads=4,
        num_layers=2,
        global_feature_dim=4,
        scorer_hidden_dim=config.scorer_hidden_dim,
        promotion_embedding_dim=config.promotion_embedding_dim,
        timing_hidden_dim=config.timing_hidden_dim,
        dropout=config.dropout,
    )


def train_model_b(train_graphs, val_graphs, test_graphs, config, device, resume=False):
    """Train Model B with the shared A3 legal-candidate training pipeline."""

    from src.training.model_a import model_a3_legal_scorer as a3

    original_builder = a3.build_model_a3
    try:
        a3.build_model_a3 = build_model_b
        return train_model_a3(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            config=config,
            device=device,
            resume=resume,
        )
    finally:
        a3.build_model_a3 = original_builder


__all__ = [
    "ModelBTimingLegalScorerConfig",
    "build_model_b",
    "evaluate_a3",
    "load_a3_checkpoint",
    "run_a3_epoch",
    "save_a3_checkpoint",
    "train_model_b",
]

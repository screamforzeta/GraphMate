"""Timing-aware Model B training code."""

from src.training.model_b.model_b_timing_legal_scorer import (
    ModelBTimingLegalScorerConfig,
    build_model_b,
    train_model_b,
)


__all__ = [
    "ModelBTimingLegalScorerConfig",
    "build_model_b",
    "train_model_b",
]

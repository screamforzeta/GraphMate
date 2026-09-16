"""Model B timing-aware evaluation code."""

from src.evaluation.model_b.model_b_timing_ablation import (
    MODEL_B_REFERENCE,
    ModelBTimingAblationConfig,
    build_shared_comparison_frame,
    model_b_parity_status,
    run_timing_ablation,
    write_outputs,
)

__all__ = [
    "MODEL_B_REFERENCE",
    "ModelBTimingAblationConfig",
    "build_shared_comparison_frame",
    "model_b_parity_status",
    "run_timing_ablation",
    "write_outputs",
]

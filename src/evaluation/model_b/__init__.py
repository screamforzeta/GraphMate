"""Model B timing-aware evaluation code."""

from src.evaluation.model_b.model_b_timing_ablation import (
    MODEL_B_REFERENCE,
    ModelBTimingAblationConfig,
    build_shared_comparison_frame,
    extract_real_move_times,
    model_b_parity_status,
    run_timing_ablation,
    update_timing_distribution_only,
    write_outputs,
)

__all__ = [
    "MODEL_B_REFERENCE",
    "ModelBTimingAblationConfig",
    "build_shared_comparison_frame",
    "extract_real_move_times",
    "model_b_parity_status",
    "run_timing_ablation",
    "update_timing_distribution_only",
    "write_outputs",
]

"""Classic held-out benchmark evaluation framework."""

from src.evaluation.classic.adapters import (
    A3ClassicAdapter,
    A4ClassicAdapter,
    ClassicModelAdapter,
    ModelBClassicAdapter,
    MockClassicAdapter,
    QwenClassicAdapter,
)
from src.evaluation.classic.core import (
    CLASSIC_EVALUATION_PROTOCOL_VERSION,
    EXPECTED_FREEZE_FINGERPRINT,
    ClassicPredictionRecord,
    ClassicSample,
    aggregate_predictions,
    build_prediction_record,
    load_classic_samples,
    prediction_record_from_json,
    score_classic_prediction,
    verify_frozen_benchmark,
)
from src.evaluation.classic.runner import (
    build_run_identity,
    prepare_official_run_directory,
    write_prediction_records,
)

__all__ = [
    "A3ClassicAdapter",
    "A4ClassicAdapter",
    "CLASSIC_EVALUATION_PROTOCOL_VERSION",
    "EXPECTED_FREEZE_FINGERPRINT",
    "ClassicModelAdapter",
    "ClassicPredictionRecord",
    "ClassicSample",
    "MockClassicAdapter",
    "ModelBClassicAdapter",
    "QwenClassicAdapter",
    "aggregate_predictions",
    "build_prediction_record",
    "build_run_identity",
    "load_classic_samples",
    "prediction_record_from_json",
    "prepare_official_run_directory",
    "score_classic_prediction",
    "verify_frozen_benchmark",
    "write_prediction_records",
]

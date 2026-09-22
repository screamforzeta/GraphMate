"""Frozen model metadata registry used by documentation and Streamlit."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelMetadata:
    """Read-only metadata for one model family member."""

    key: str
    display_name: str
    status: str
    runnable_in_streamlit: bool
    checkpoint_path: str | None
    architecture_summary: str
    timing_used: bool
    output_semantics: str
    notes: str

    def to_dict(self):
        """Return JSON/table friendly metadata."""

        payload = asdict(self)
        payload["checkpoint_exists"] = (
            Path(self.checkpoint_path).exists() if self.checkpoint_path else False
        )
        return payload


MODEL_REGISTRY = {
    "A": ModelMetadata(
        key="A",
        display_name="A - Move Classifier",
        status="frozen historical baseline",
        runnable_in_streamlit=True,
        checkpoint_path="checkpoints/model_a/best.pt",
        architecture_summary="GAT graph encoder with fixed move-vocabulary classifier.",
        timing_used=False,
        output_semantics="Global 1,786-class move logits; may predict illegal moves.",
        notes="Also exposed with best-legal post-filtering as A1/diagnostic best-legal mode.",
    ),
    "A1": ModelMetadata(
        key="A1",
        display_name="A1 - A + Best Legal Move",
        status="historical diagnostic mode, not a separate architecture",
        runnable_in_streamlit=True,
        checkpoint_path="checkpoints/model_a/best.pt",
        architecture_summary="Model A logits filtered to legal moves at inference time.",
        timing_used=False,
        output_semantics="Best legal move from Model A vocabulary scores.",
        notes="Repository evidence treats A1 as a best-legal inference mode rather than a new checkpoint.",
    ),
    "A2": ModelMetadata(
        key="A2",
        display_name="A2 - Legal-Masked Classifier",
        status="frozen",
        runnable_in_streamlit=True,
        checkpoint_path="checkpoints/model_a2/best.pt",
        architecture_summary="Model A architecture trained/evaluated with legal masking.",
        timing_used=False,
        output_semantics="Masked fixed-vocabulary logits over legal moves.",
        notes="Improves legality but remains fixed-vocabulary classification.",
    ),
    "A3": ModelMetadata(
        key="A3",
        display_name="A3 - Legal Move Scorer",
        status="frozen official no-timing baseline",
        runnable_in_streamlit=True,
        checkpoint_path="checkpoints/model_a3/best.pt",
        architecture_summary="GAT encoder plus legal-candidate source/destination scorer.",
        timing_used=False,
        output_semantics="One score per legal move candidate.",
        notes="Official no-timing baseline for A4 and Model B comparisons.",
    ),
    "A4": ModelMetadata(
        key="A4",
        display_name="A4 - Post-Move Reranker",
        status="frozen",
        runnable_in_streamlit=False,
        checkpoint_path="checkpoints/model_a4/best.pt",
        architecture_summary="Frozen A3 Top-5 retrieval plus post-move GAT reranker.",
        timing_used=False,
        output_semantics="Reranked A3 Top-5; end-to-end prediction remains one next move.",
        notes="Inference adapter exists; Streamlit displays frozen results and A4 semantics unless checkpoint is available.",
    ),
    "B": ModelMetadata(
        key="B",
        display_name="B - Timing-Aware Legal Move Scorer",
        status="frozen",
        runnable_in_streamlit=False,
        checkpoint_path="checkpoints/model_b/best.pt",
        architecture_summary="A3 legal scorer with timing encoder appended to graph context.",
        timing_used=True,
        output_semantics="One score per legal move candidate using synthetic timing features.",
        notes="Synthetic timing ablation; not evidence that real human timing is useless.",
    ),
}


def list_models():
    """Return all frozen model metadata in lineage order."""

    return [MODEL_REGISTRY[key] for key in ("A", "A1", "A2", "A3", "A4", "B")]


def model_table_rows():
    """Return registry rows for display and tests."""

    return [model.to_dict() for model in list_models()]

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data

from src.models.model_a.chess_legal_scorer import ChessGATLegalMoveScorer
from src.models.model_b import ChessGATTimingLegalMoveScorer
from src.training.model_a.model_a3_legal_scorer import (
    ModelA3LegalScorerConfig,
    build_candidate_batch,
    grouped_cross_entropy,
    run_a3_epoch,
)
from src.training.model_b import ModelBTimingLegalScorerConfig
from src.training.model_b import build_model_b


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _graph(target_move="e2e4", with_timing=True):
    graph = Data(
        x=torch.randn(64, 15),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 5), dtype=torch.float),
        global_features=torch.zeros(1, 4),
        y=torch.tensor(0, dtype=torch.long),
        fen=START_FEN,
        target_move=target_move,
    )
    if with_timing:
        graph.previous_move_time = torch.tensor([[12.0]], dtype=torch.float)
        graph.original_move_time = torch.tensor([[12.0]], dtype=torch.float)
        graph.time_is_synthetic = torch.tensor([[1]], dtype=torch.long)
    return graph


def test_model_b_scores_only_a3_legal_candidates_with_timing_context():
    model = ChessGATTimingLegalMoveScorer(dropout=0.0)
    batch = Batch.from_data_list(
        [
            _graph("e2e4"),
            _graph("g1f3"),
        ]
    )
    candidate = build_candidate_batch(batch)

    output = model(batch, candidate["candidate_moves"])
    loss = grouped_cross_entropy(
        output["scores"],
        output["candidate_ptr"],
        candidate["target_indices"],
    )

    assert output["candidate_ptr"].tolist() == [0, 20, 40]
    assert output["scores"].shape == (40,)
    assert output["candidate_uci"][:20] == [
        move.uci() for move in candidate["candidate_moves"][0]
    ]
    assert torch.isfinite(loss)


def test_model_b_requires_timing_features_so_it_cannot_silently_become_a3():
    model = ChessGATTimingLegalMoveScorer(dropout=0.0)
    batch = Batch.from_data_list([_graph(with_timing=False)])
    candidate = build_candidate_batch(batch)

    with pytest.raises(ValueError, match="requires timing attributes"):
        model(batch, candidate["candidate_moves"])


def test_model_b_uses_a3_training_epoch_and_has_extra_timing_parameters():
    from torch_geometric.loader import DataLoader

    config = ModelBTimingLegalScorerConfig(
        batch_size=2,
        max_epochs=1,
        amp=False,
        pin_memory=False,
        non_blocking=False,
    )
    model = build_model_b(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = DataLoader([_graph("e2e4"), _graph("g1f3")], batch_size=2)

    metrics = run_a3_epoch(model, loader, optimizer, torch.device("cpu"), config, scaler)

    assert metrics["num_examples"] == 2
    assert model.candidate_dim > ChessGATLegalMoveScorer(dropout=0.0).candidate_dim
    assert isinstance(config, ModelA3LegalScorerConfig)

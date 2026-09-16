import inspect

import chess
import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data

from src.cli.training import train_model_a4_postmove
from src.models.model_a.chess_legal_scorer import ChessGATLegalMoveScorer
from src.models.model_a.chess_postmove_reranker import ChessA4PostMoveReranker
from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    apply_candidate_to_fen,
    assert_a3_frozen,
    build_postmove_graph,
    collate_a4_examples,
    freeze_a3,
    grouped_cross_entropy_from_topk,
    select_topk_a3,
    top1_metrics_from_scores,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _graph(fen=START_FEN, target_move="e2e4"):
    return Data(
        x=torch.randn(64, 15),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros((2, 5), dtype=torch.float),
        global_features=torch.zeros(1, 4),
        y=torch.tensor(0, dtype=torch.long),
        fen=fen,
        target_move=target_move,
        source_row_index=torch.tensor(0),
    )


def test_a4_candidate_dimension_matches_declared_representation():
    model = ChessA4PostMoveReranker(dropout=0.0)

    assert model.candidate_dim == 530


def test_a3_parameters_are_frozen_for_a4():
    a3 = ChessGATLegalMoveScorer(dropout=0.0)
    freeze_a3(a3)
    assert_a3_frozen(a3)

    assert not any(parameter.requires_grad for parameter in a3.parameters())
    assert not a3.training


def test_a3_optimizer_exclusion_is_simple_when_a4_optimizer_uses_a4_parameters():
    a3 = freeze_a3(ChessGATLegalMoveScorer(dropout=0.0))
    a4 = ChessA4PostMoveReranker(dropout=0.0)
    optimizer = torch.optim.Adam(a4.parameters(), lr=1e-3)
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert optimized_ids
    assert all(id(parameter) not in optimized_ids for parameter in a3.parameters())


def test_top5_retrieval_uses_device_side_order_and_detects_rerankable():
    output = {
        "scores": torch.tensor([0.1, 3.0, 2.0, 1.0, 0.5, -1.0]),
        "candidate_ptr": torch.tensor([0, 6]),
        "candidate_uci": ["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "f2f3"],
    }
    rows = select_topk_a3(output, torch.tensor([2]), top_k=5)

    assert rows[0]["uci"] == ["b2b3", "c2c3", "d2d3", "e2e3", "a2a3"]
    assert rows[0]["target_candidate_index"] == 1
    assert rows[0]["rerankable"] is True


def test_target_not_in_top5_is_not_injected():
    output = {
        "scores": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, -10.0]),
        "candidate_ptr": torch.tensor([0, 6]),
        "candidate_uci": ["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "f2f3"],
    }
    rows = select_topk_a3(output, torch.tensor([5]), top_k=5)

    assert "f2f3" not in rows[0]["uci"]
    assert rows[0]["target_candidate_index"] is None
    assert rows[0]["rerankable"] is False


def test_grouped_cross_entropy_from_topk_does_not_mix_graphs():
    scores = torch.tensor([1.0, 2.0, 100.0], requires_grad=True)
    ptr = torch.tensor([0, 2, 3])
    targets = torch.tensor([1, 0])

    loss = grouped_cross_entropy_from_topk(scores, ptr, targets)
    loss.backward()

    expected = (torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0) - 2.0) / 2
    assert torch.allclose(loss.detach(), expected)
    assert scores.grad is not None


def test_candidate_move_application_does_not_mutate_original_board():
    board = chess.Board(START_FEN)
    before = board.fen()
    resulting = apply_candidate_to_fen(START_FEN, "e2e4")

    assert board.fen() == before
    assert chess.Board(resulting).turn == chess.BLACK


def test_postmove_graph_has_resulting_side_to_move_global_feature():
    graph = build_postmove_graph(START_FEN, "e2e4", "e2e4", {"e2e4": 0})

    assert graph.fen == graph.resulting_fen
    assert graph.global_features.shape == (1, 4)
    assert graph.global_features[0, 0].item() == 0.0


def test_terminal_resulting_position_is_representable():
    fen = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"
    graph = build_postmove_graph(fen, "d8h4", "d8h4", {"d8h4": 0})

    assert chess.Board(graph.resulting_fen).is_checkmate()
    assert graph.x.shape == (64, 15)
    assert graph.global_features.shape == (1, 4)


def test_postmove_graph_batching_and_candidate_vector_forward():
    a4 = ChessA4PostMoveReranker(dropout=0.0)
    graphs = [_graph(), _graph()]
    batch = Batch.from_data_list(graphs)
    a3_features = torch.randn(2, 396)
    score_features = torch.randn(2, 2)

    scores = a4(a3_features, batch, score_features)

    assert scores.shape == (2,)


def test_a3_raw_and_centered_score_calculation():
    output = {
        "scores": torch.tensor([1.0, 2.0, 4.0]),
        "candidate_ptr": torch.tensor([0, 3]),
        "candidate_uci": ["a2a3", "b2b3", "c2c3"],
    }
    row = select_topk_a3(output, torch.tensor([1]), top_k=3)[0]

    assert row["raw_scores"] == [4.0, 2.0, 1.0]
    assert pytest.approx(sum(row["centered_scores"])) == 0.0


def test_a4_ranking_metrics_and_paired_transitions():
    scores = torch.tensor([0.0, 2.0, 1.0, 4.0])
    ptr = torch.tensor([0, 2, 4])
    targets = torch.tensor([1, 0])

    metrics = top1_metrics_from_scores(
        scores,
        ptr,
        targets,
        a3_top1=["a", "wrong"],
        target_uci=["a", "target"],
    )

    assert metrics["a4_top1_correct"] == 1
    assert metrics["a3_correct_a4_correct"] == 1
    assert metrics["a3_wrong_a4_wrong"] == 1


def test_training_cli_does_not_load_test_split():
    source = inspect.getsource(train_model_a4_postmove.main)

    assert 'split="test"' not in source
    assert "load_pyg_dataset" not in source

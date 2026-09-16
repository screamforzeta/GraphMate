import inspect

import chess
import pytest
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_geometric.data import Data

from src.cli.training import train_model_a4_postmove
from src.models.model_a.chess_legal_scorer import ChessGATLegalMoveScorer
from src.models.model_a.chess_postmove_reranker import ChessA4PostMoveReranker
from src.training.model_a.model_a4_postmove_reranker import (
    ModelA4PostMoveConfig,
    A4PostMoveCacheDataset,
    apply_candidate_to_fen,
    assert_a3_frozen,
    build_postmove_graph,
    collate_a4_examples,
    freeze_a3,
    grouped_cross_entropy_from_topk,
    move_a4_batch_to_device,
    select_topk_a3,
    top1_metrics_from_scores,
    validate_a4_batch_contract,
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


def _cache_example(candidate_uci, target_index=0, target_move="e2e4"):
    graphs = [
        build_postmove_graph(START_FEN, move, target_move, {target_move: 0})
        for move in candidate_uci
    ]
    count = len(candidate_uci)
    raw = torch.arange(count, dtype=torch.float)
    centered = raw - raw.mean()
    return {
        "puzzle_id": "puzzle",
        "source_row_index": 0,
        "fen": START_FEN,
        "target_move": target_move,
        "candidate_uci": list(candidate_uci),
        "a3_raw_scores": raw.tolist(),
        "a3_centered_scores": centered.tolist(),
        "target_candidate_index": int(target_index),
        "a3_candidate_features": torch.randn(count, 396),
        "score_features": torch.stack([raw, centered], dim=1),
        "postmove_graphs": graphs,
    }


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


def test_cache_record_schema_matches_builder_contract():
    example = _cache_example(["e2e4", "g1f3"], target_index=1)

    assert sorted(example) == [
        "a3_candidate_features",
        "a3_centered_scores",
        "a3_raw_scores",
        "candidate_uci",
        "fen",
        "postmove_graphs",
        "puzzle_id",
        "score_features",
        "source_row_index",
        "target_candidate_index",
        "target_move",
    ]
    assert len(example["postmove_graphs"]) == len(example["candidate_uci"])
    assert example["a3_candidate_features"].shape == (2, 396)
    assert example["score_features"].shape == (2, 2)


def test_collate_creates_canonical_postmove_batch_with_variable_candidate_counts():
    examples = [
        _cache_example(["e2e4", "g1f3", "d2d4", "c2c4", "b1c3"], target_index=2),
        _cache_example(["e2e4", "g1f3", "d2d4"], target_index=1),
    ]

    batch = collate_a4_examples(examples)
    summary = validate_a4_batch_contract(batch)

    assert sorted(batch) == [
        "a3_candidate_features",
        "a3_top1",
        "candidate_counts",
        "candidate_ptr",
        "candidate_uci",
        "postmove_batch",
        "score_features",
        "target_indices",
        "target_uci",
    ]
    assert batch["candidate_counts"].tolist() == [5, 3]
    assert batch["candidate_ptr"].tolist() == [0, 5, 8]
    assert summary["original_positions"] == 2
    assert summary["total_candidates"] == 8
    assert batch["postmove_batch"].num_graphs == 8
    assert tuple(batch["postmove_batch"].global_features.shape) == (8, 4)
    assert len(batch["candidate_uci"]) == 8


def test_torch_dataloader_uses_a4_collate_contract(tmp_path):
    cache_dir = tmp_path / "cache" / "train"
    cache_dir.mkdir(parents=True)
    torch.save([_cache_example(["e2e4", "g1f3"])], cache_dir / "examples.pt")
    dataset = A4PostMoveCacheDataset(tmp_path / "cache", "train")
    loader = TorchDataLoader(dataset, batch_size=1, collate_fn=collate_a4_examples)

    batch = next(iter(loader))

    assert "postmove_batch" in batch
    assert validate_a4_batch_contract(batch)["total_candidates"] == 2


def test_move_to_device_forward_loss_backward_and_gradient_contract():
    a3 = freeze_a3(ChessGATLegalMoveScorer(dropout=0.0))
    a4 = ChessA4PostMoveReranker(dropout=0.0)
    batch = collate_a4_examples(
        [
            _cache_example(["e2e4", "g1f3", "d2d4", "c2c4", "b1c3"], target_index=2),
            _cache_example(["e2e4", "g1f3", "d2d4"], target_index=1),
        ]
    )
    batch = move_a4_batch_to_device(batch, torch.device("cpu"), non_blocking=True)

    scores = a4(
        batch["a3_candidate_features"],
        batch["postmove_batch"],
        batch["score_features"],
    )
    loss = grouped_cross_entropy_from_topk(
        scores,
        batch["candidate_ptr"],
        batch["target_indices"],
    )
    loss.backward()

    assert scores.numel() == 8
    assert torch.isfinite(loss)
    assert not any(parameter.grad is not None for parameter in a3.parameters())
    assert any(parameter.grad is not None for parameter in a4.parameters())


def test_batch_contract_validation_error_is_explicit():
    with pytest.raises(ValueError, match="MODEL_A4_BATCH_CONTRACT_INVALID"):
        validate_a4_batch_contract({"candidate_ptr": torch.tensor([0])})


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

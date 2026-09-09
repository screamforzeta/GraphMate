import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data

from src.models.model_a.chess_legal_scorer import (
    ChessGATLegalMoveScorer,
    promotion_id,
)
from src.training.model_a.model_a3_legal_scorer import (
    ModelA3LegalScorerConfig,
    build_candidate_batch,
    build_model_a3,
    evaluate_a3,
    grouped_cross_entropy,
    legal_candidates_from_fen,
    load_a3_checkpoint,
    run_a3_epoch,
    save_a3_checkpoint,
)


START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
PROMOTION_FEN = "7k/P7/8/8/8/8/8/K7 w - - 0 1"


def _graph(fen=START_FEN, target_move="e2e4", y=0):
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.zeros((2, 5), dtype=torch.float)
    return Data(
        x=torch.randn(64, 15),
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=torch.zeros(1, 4),
        y=torch.tensor([y], dtype=torch.long),
        fen=fen,
        target_move=target_move,
        source_row_index=0,
    )


def test_legal_candidate_enumeration_uses_python_chess_semantics():
    moves = [move.uci() for move in legal_candidates_from_fen(START_FEN)]

    assert "e2e4" in moves
    assert "g1f3" in moves
    assert len(moves) == 20


def test_promotion_candidates_have_distinct_trainable_ids():
    moves = {move.uci(): move for move in legal_candidates_from_fen(PROMOTION_FEN)}

    assert promotion_id(moves["a7a8q"]) != promotion_id(moves["a7a8r"])
    assert promotion_id(moves["a7a8b"]) != promotion_id(moves["a7a8n"])


def test_target_index_is_local_to_legal_candidates_and_vocab_independent():
    batch = Batch.from_data_list([_graph(target_move="g1f3", y=999)])

    candidate = build_candidate_batch(batch)
    legal_uci = [move.uci() for move in candidate["candidate_moves"][0]]

    assert "g1f3" in legal_uci
    assert candidate["target_indices"].item() == legal_uci.index("g1f3")


def test_target_not_legal_raises_hard_alignment_error():
    batch = Batch.from_data_list([_graph(target_move="e1e2")])

    with pytest.raises(ValueError, match="not legal"):
        build_candidate_batch(batch)


def test_grouped_cross_entropy_does_not_mix_graph_candidates():
    scores = torch.tensor([1.0, 2.0, 100.0], requires_grad=True)
    ptr = torch.tensor([0, 2, 3])
    targets = torch.tensor([1, 0])

    loss = grouped_cross_entropy(scores, ptr, targets)
    loss.backward()

    expected = (torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0) - 2.0) / 2
    assert torch.allclose(loss.detach(), expected)
    assert scores.grad is not None


def test_model_scores_variable_candidates_without_fixed_vocab_logits():
    model = ChessGATLegalMoveScorer(dropout=0.0)
    batch = Batch.from_data_list(
        [
            _graph(target_move="e2e4"),
            _graph(target_move="g1f3"),
        ]
    )
    candidate = build_candidate_batch(batch)

    output = model(batch, candidate["candidate_moves"])

    assert output["scores"].ndim == 1
    assert output["candidate_ptr"].tolist() == [0, 20, 40]
    assert output["scores"].shape != (2, 1786)


def test_loss_is_finite_and_backward_reaches_model_parameters():
    model = ChessGATLegalMoveScorer(dropout=0.0)
    batch = Batch.from_data_list([_graph(target_move="e2e4")])
    candidate = build_candidate_batch(batch)
    output = model(batch, candidate["candidate_moves"])

    loss = grouped_cross_entropy(
        output["scores"],
        output["candidate_ptr"],
        candidate["target_indices"],
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_tiny_train_eval_and_checkpoint_resume_smoke(tmp_path):
    from torch_geometric.loader import DataLoader

    config = ModelA3LegalScorerConfig(
        batch_size=2,
        max_epochs=1,
        amp=False,
        pin_memory=False,
        non_blocking=False,
        output_root=str(tmp_path),
    )
    model = build_model_a3(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = DataLoader(
        [
            _graph(target_move="e2e4", y=0),
            _graph(target_move="g1f3", y=1),
        ],
        batch_size=2,
    )

    train_metrics = run_a3_epoch(model, loader, optimizer, torch.device("cpu"), config, scaler)
    eval_metrics = evaluate_a3(model, loader, torch.device("cpu"), config)
    save_a3_checkpoint(
        tmp_path / "last.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        1,
        eval_metrics["loss"],
        1,
        0,
        config,
        [{"epoch": 1}],
    )
    restored = build_model_a3(config)
    checkpoint = load_a3_checkpoint(tmp_path / "last.pt", restored, device="cpu")

    assert train_metrics["num_examples"] == 2
    assert eval_metrics["num_examples"] == 2
    assert checkpoint["epoch"] == 1

import copy

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models import ChessGATNoTiming
from src.training import (
    ChessGATTrainingConfig,
    EarlyStoppingState,
    compute_topk_accuracies,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_graph_splits,
)


def make_training_graph(target=0):
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [2, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.float,
    )
    x = torch.zeros((64, 15), dtype=torch.float)
    x[:, 8] = torch.arange(64, dtype=torch.float) / 63.0
    x[:, 9] = torch.arange(64, dtype=torch.float).flip(0) / 63.0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=torch.tensor(
            [[1.0, 0.0, 0.1, 0.0]],
            dtype=torch.float,
        ),
        y=torch.tensor(target, dtype=torch.long),
    )


def make_loader(num_graphs=4, num_classes=5, shuffle=False):
    graphs = [
        make_training_graph(target=index % num_classes)
        for index in range(num_graphs)
    ]
    return DataLoader(
        graphs,
        batch_size=num_graphs,
        shuffle=shuffle,
    )


def test_topk_accuracy_handles_basic_cases_and_large_k():
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.1, 0.2],
        ]
    )
    targets = torch.tensor([1, 2])

    metrics = compute_topk_accuracies(
        logits,
        targets,
        top_k=(1, 2, 5),
    )

    assert metrics[1] == pytest.approx(0.5)
    assert metrics[2] == pytest.approx(1.0)
    assert metrics[5] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_train_one_epoch_returns_metrics_and_updates_parameters():
    torch.manual_seed(1)
    model = ChessGATNoTiming(
        num_classes=5,
        dropout=0.0,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    loader = make_loader(num_graphs=4, num_classes=5)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    metrics = train_one_epoch(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        top_k=(1,),
    )

    changed = any(
        not torch.allclose(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert 0.0 <= metrics["top1"] <= 1.0
    assert metrics["num_examples"] == 4
    assert changed


def test_evaluate_returns_metrics_without_changing_parameters():
    model = ChessGATNoTiming(
        num_classes=5,
        dropout=0.0,
    )
    before = copy.deepcopy(
        model.state_dict()
    )
    loader = make_loader(num_graphs=4, num_classes=5)

    metrics = evaluate(
        model,
        loader,
        torch.nn.CrossEntropyLoss(),
        torch.device("cpu"),
        top_k=(1, 3, 5),
    )

    for key, value in before.items():
        assert torch.allclose(
            value,
            model.state_dict()[key],
        )
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert set(metrics) >= {"loss", "top1", "top3", "top5", "num_examples"}


def test_checkpoint_save_and_reload_restores_state_and_metadata(tmp_path):
    model = ChessGATNoTiming(
        num_classes=5,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    path = tmp_path / "best.pt"
    config = ChessGATTrainingConfig(
        checkpoint_path=str(path),
        history_path=str(tmp_path / "history.json"),
    )

    save_checkpoint(
        path,
        model,
        optimizer,
        epoch=3,
        best_val_loss=1.23,
        config=config,
        num_classes=5,
        train_metrics={"loss": 1.0},
        val_metrics={"loss": 1.23},
    )

    reloaded = ChessGATNoTiming(
        num_classes=5,
        dropout=0.0,
    )
    checkpoint = load_checkpoint(
        path,
        reloaded,
        device="cpu",
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["best_val_loss"] == 1.23
    assert checkpoint["num_classes"] == 5
    assert checkpoint["move_vocab_size"] == 5
    for key, value in model.state_dict().items():
        assert torch.allclose(
            value,
            reloaded.state_dict()[key],
        )


def test_early_stopping_state_resets_and_stops_after_patience():
    state = EarlyStoppingState(
        patience=2,
        min_delta=0.0,
    )

    assert state.update(5.0)
    assert state.epochs_without_improvement == 0
    assert state.update(4.0)
    assert state.epochs_without_improvement == 0
    assert not state.update(4.5)
    assert state.epochs_without_improvement == 1
    assert not state.should_stop
    assert not state.update(4.6)
    assert state.should_stop


def test_validate_graph_splits_rejects_target_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        validate_graph_splits(
            {
                "train": [make_training_graph(target=99)],
                "val": [make_training_graph(target=0)],
                "test": [make_training_graph(target=0)],
            },
            num_classes=5,
        )

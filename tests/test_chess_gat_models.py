import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models import (
    ChessGATNoTiming,
    count_trainable_parameters,
)


def make_graph(
    num_classes=7,
    global_features=None,
    edge_attr=None,
):
    x = torch.zeros(
        (64, 15),
        dtype=torch.float,
    )
    x[:, 8] = torch.arange(64, dtype=torch.float) / 63.0
    x[:, 9] = torch.arange(64, dtype=torch.float).flip(0) / 63.0

    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [2, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    if edge_attr is None:
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

    if global_features is None:
        global_features = torch.tensor(
            [[1.0, 0.0, 0.1, 0.0]],
            dtype=torch.float,
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=global_features,
        y=torch.tensor(
            0 % num_classes,
            dtype=torch.long,
        ),
    )


def make_batch(graphs):
    return next(
        iter(
            DataLoader(
                graphs,
                batch_size=len(graphs),
                shuffle=False,
            )
        )
    )


def test_chess_gat_instantiates_with_configurable_num_classes():
    model = ChessGATNoTiming(
        num_classes=11,
    )

    assert model.num_classes == 11
    assert count_trainable_parameters(model) > 0


def test_chess_gat_single_graph_forward_shape_and_finite_logits():
    num_classes = 7
    model = ChessGATNoTiming(
        num_classes=num_classes,
    )
    batch = make_batch(
        [
            make_graph(num_classes=num_classes),
        ]
    )

    logits = model(batch)

    assert logits.shape == (1, num_classes)
    assert logits.dtype == torch.float
    assert torch.isfinite(logits).all()


def test_chess_gat_batch_forward_shape_and_global_features_batching():
    num_classes = 9
    model = ChessGATNoTiming(
        num_classes=num_classes,
    )
    graphs = [
        make_graph(num_classes=num_classes)
        for _ in range(4)
    ]
    batch = make_batch(graphs)

    logits = model(batch)

    assert batch.x.shape == (64 * 4, 15)
    assert batch.global_features.shape == (4, 4)
    assert logits.shape == (4, num_classes)
    assert torch.isfinite(logits).all()


def test_chess_gat_requires_no_event_ids_or_timing():
    model = ChessGATNoTiming(
        num_classes=5,
    )
    batch = make_batch(
        [
            make_graph(num_classes=5),
        ]
    )

    assert not hasattr(batch, "event_ids")
    assert not hasattr(batch, "edge_time")
    assert model(batch).shape == (1, 5)


def test_chess_gat_backward_produces_finite_gradients():
    num_classes = 6
    model = ChessGATNoTiming(
        num_classes=num_classes,
    )
    batch = make_batch(
        [
            make_graph(num_classes=num_classes)
            for _ in range(3)
        ]
    )
    batch.y = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    logits = model(batch)
    loss = torch.nn.CrossEntropyLoss()(
        logits,
        batch.y,
    )
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    assert gradients
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_global_features_influence_output():
    model = ChessGATNoTiming(
        num_classes=5,
        dropout=0.0,
    )
    model.eval()

    base = make_batch(
        [
            make_graph(
                num_classes=5,
                global_features=torch.zeros((1, 4)),
            ),
        ]
    )
    changed = make_batch(
        [
            make_graph(
                num_classes=5,
                global_features=torch.ones((1, 4)),
            ),
        ]
    )

    with torch.no_grad():
        base_logits = model(base)
        changed_logits = model(changed)

    assert not torch.allclose(
        base_logits,
        changed_logits,
        atol=1e-6,
    )


def test_multilabel_edge_attr_is_consumed_by_model():
    torch.manual_seed(123)
    model = ChessGATNoTiming(
        num_classes=5,
        dropout=0.0,
    )
    model.eval()

    base_edge_attr = torch.zeros(
        (6, 5),
        dtype=torch.float,
    )
    changed_edge_attr = base_edge_attr.clone()
    changed_edge_attr[0] = torch.tensor(
        [1, 1, 0, 0, 0],
        dtype=torch.float,
    )
    changed_edge_attr[1] = torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.float,
    )

    base = make_batch(
        [
            make_graph(
                num_classes=5,
                edge_attr=base_edge_attr,
            ),
        ]
    )
    changed = make_batch(
        [
            make_graph(
                num_classes=5,
                edge_attr=changed_edge_attr,
            ),
        ]
    )

    with torch.no_grad():
        base_logits = model(base)
        changed_logits = model(changed)

    assert not torch.allclose(
        base_logits,
        changed_logits,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch: setattr(batch, "x", torch.zeros((64, 14))),
            "batch.x feature dimension",
        ),
        (
            lambda batch: setattr(batch, "edge_attr", torch.zeros((6, 4))),
            "batch.edge_attr feature dimension",
        ),
        (
            lambda batch: setattr(
                batch,
                "global_features",
                torch.zeros((1, 3)),
            ),
            "batch.global_features feature dimension",
        ),
        (
            lambda batch: batch.__delattr__("edge_attr"),
            "requires batch.edge_attr",
        ),
    ],
)
def test_chess_gat_validation_errors_are_readable(mutate, message):
    model = ChessGATNoTiming(
        num_classes=5,
    )
    batch = make_batch(
        [
            make_graph(num_classes=5),
        ]
    )
    mutate(batch)

    with pytest.raises(ValueError, match=message):
        model(batch)

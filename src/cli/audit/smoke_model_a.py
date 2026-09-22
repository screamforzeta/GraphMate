"""Run a read-only smoke test for ChessGATNoTiming.

Purpose:
    Verify that generated PyG chess graphs can flow through the no-timing GAT
    model, produce graph-level logits, compute loss, and run one backward pass.
Input:
    data/pyg sharded train split and resources/move_encoder/move_to_idx.json.
Output:
    Printed tensor shapes, parameter count, loss, and pass/fail checks.
Role:
    Confirms representation-to-model compatibility before real training code.
"""

from pathlib import Path
import json

import torch
from torch_geometric.loader import DataLoader

from src.graph.pyg_dataset import load_pyg_dataset
from src.models import (
    ChessGATNoTiming,
    count_trainable_parameters,
)


MOVE_ENCODER_PATH = Path("resources/move_encoder/move_to_idx.json")
SMOKE_BATCH_SIZE = 8


def load_num_classes(path=MOVE_ENCODER_PATH):
    """Load the move vocabulary size.

    Parameters:
        path: Path to move_to_idx.json.
    Returns:
        Number of move classes.
    Side effects:
        Reads the move encoder JSON from disk.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        move_to_idx = json.load(file)

    return len(move_to_idx)


def run_smoke_test():
    """Run one forward/loss/backward check on real generated graphs.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Reads generated graph data and prints a smoke-test report.
    """

    print("ChessGATNoTiming smoke test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    num_classes = load_num_classes()
    dataset = load_pyg_dataset(split="train")
    graphs = [
        dataset[index]
        for index in range(min(SMOKE_BATCH_SIZE, len(dataset)))
    ]

    loader = DataLoader(
        graphs,
        batch_size=len(graphs),
        shuffle=False,
    )
    batch = next(iter(loader)).to(device)

    model = ChessGATNoTiming(
        num_classes=num_classes,
    ).to(device)

    logits = model(batch)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(
        logits,
        batch.y,
    )
    loss.backward()

    finite_logits = bool(torch.isfinite(logits).all())
    finite_gradients = all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    any_gradient = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"graphs loaded: {len(graphs)}")
    print(f"batch size: {len(graphs)}")
    print(f"x: {tuple(batch.x.shape)}")
    print(f"edge_index: {tuple(batch.edge_index.shape)}")
    print(f"edge_attr: {tuple(batch.edge_attr.shape)}")
    print(f"global_features: {tuple(batch.global_features.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"num_classes: {num_classes}")
    print(f"trainable_parameters: {count_trainable_parameters(model):,}")
    print(f"loss: {loss.item():.6f}")
    print(f"finite_logits: {'PASS' if finite_logits else 'FAIL'}")
    print(
        "backward: "
        f"{'PASS' if any_gradient and finite_gradients else 'FAIL'}"
    )

    if not finite_logits or not any_gradient or not finite_gradients:
        raise SystemExit(1)

    print("\nFINAL VERDICT:")
    print("CHESS_GAT_NO_TIMING_SMOKE_TEST_PASS")


def main():
    """Run the smoke test command.

    Parameters:
        None.
    Returns:
        None.
    Side effects:
        Delegates to run_smoke_test().
    """

    run_smoke_test()


if __name__ == "__main__":
    main()

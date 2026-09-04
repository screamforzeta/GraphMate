from collections import Counter

import chess
import pytest
import torch

from src.graph.edge_features import (
    EDGE_FEATURE_DIM,
    extract_edge_features,
)
from src.graph.graph_builder import build_graph
from src.graph.node_features import (
    NODE_FEATURE_DIM,
    compute_legal_mobility_map,
    extract_node_features,
)


def _edge_features_by_pair(fen):
    edge_index, edge_attr = extract_edge_features(fen)
    return {
        (
            edge_index[0, idx].item(),
            edge_index[1, idx].item(),
        ): edge_attr[idx].tolist()
        for idx in range(edge_index.shape[1])
    }


def test_mobility_counts_both_colors_without_mutating_turn():
    board = chess.Board(
        "r3k3/8/8/8/8/8/8/4K2R w - - 0 1"
    )
    original_fen = board.fen()

    mobility = compute_legal_mobility_map(board)

    assert mobility[chess.H1] > 0
    assert mobility[chess.A8] > 0
    assert board.fen() == original_fen


def test_node_feature_shape():
    x = extract_node_features(
        "8/8/8/8/8/8/4p3/4K2R w - - 0 1"
    )

    assert x.shape == (64, NODE_FEATURE_DIM)


def test_edge_multilabel_legal_only_and_legal_attack_and_defend():
    legal_only_fen = (
        "8/8/8/8/8/8/8/4K2k w - - 0 1"
    )
    features = _edge_features_by_pair(legal_only_fen)
    assert features[(chess.E1, chess.D1)] == [1.0, 0.0, 0.0, 0.0, 0.0]

    legal_attack_fen = (
        "8/8/8/8/8/4p3/3P4/4K2k w - - 0 1"
    )
    features = _edge_features_by_pair(legal_attack_fen)
    assert features[(chess.D2, chess.E3)] == [1.0, 1.0, 0.0, 0.0, 0.0]

    defend_fen = (
        "8/8/8/8/8/8/K7/R6k w - - 0 1"
    )
    features = _edge_features_by_pair(defend_fen)
    assert features[(chess.A1, chess.A2)] == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_no_duplicate_edges_after_multilabel_merge():
    edge_index, _ = extract_edge_features(
        "8/8/8/8/8/4p3/3P4/4K2k w - - 0 1"
    )
    pairs = [
        (
            edge_index[0, idx].item(),
            edge_index[1, idx].item(),
        )
        for idx in range(edge_index.shape[1])
    ]

    counts = Counter(pairs)

    assert all(count == 1 for count in counts.values())


@pytest.mark.parametrize(
    ("fen", "pinner", "pinned"),
    [
        (
            "4r3/8/8/8/8/8/4N3/4K3 w - - 0 1",
            chess.E8,
            chess.E2,
        ),
        (
            "8/8/7b/8/8/8/3N4/2K5 w - - 0 1",
            chess.H6,
            chess.D2,
        ),
        (
            "8/8/8/8/8/8/8/KN5q w - - 0 1",
            chess.H1,
            chess.B1,
        ),
    ],
)
def test_pin_edges_identify_slider_pinner(fen, pinner, pinned):
    features = _edge_features_by_pair(fen)

    assert features[(pinner, pinned)][3] == 1.0


def test_attacker_of_piece_that_is_not_pinner_does_not_create_pin_edge():
    fen = "8/8/8/8/8/8/3n4/2K5 w - - 0 1"
    features = _edge_features_by_pair(fen)

    assert (chess.D2, chess.C4) not in features
    assert all(feature[3] == 0.0 for feature in features.values())


def test_check_line_represents_checker_to_king_single_and_double_check():
    single = _edge_features_by_pair(
        "4r3/8/8/8/8/8/8/4K3 w - - 0 1"
    )
    assert single[(chess.E8, chess.E1)][4] == 1.0

    double = _edge_features_by_pair(
        "4r3/8/8/8/1b6/8/8/4K3 w - - 0 1"
    )
    assert double[(chess.E8, chess.E1)][4] == 1.0
    assert double[(chess.B4, chess.E1)][4] == 1.0


def test_graph_shapes_batching_and_serialization(tmp_path):
    pytest.importorskip("torch_geometric")
    from torch_geometric.loader import DataLoader

    move_to_idx = {"e1e2": 0, "h1h2": 1}
    graphs = [
        build_graph(
            "8/8/8/8/8/8/4K3/7k w - - 0 1",
            "e1e2",
            move_to_idx,
        ),
        build_graph(
            "8/8/8/8/8/8/7K/6k1 w - - 0 1",
            "h2h1",
            {"h2h1": 0},
        ),
    ]
    graphs[0].puzzle_id = "p1"
    graphs[1].puzzle_id = "p2"

    batch = next(iter(DataLoader(graphs, batch_size=2)))

    assert batch.x.shape[1] == NODE_FEATURE_DIM
    assert batch.edge_index.shape[0] == 2
    assert batch.edge_attr.shape[1] == EDGE_FEATURE_DIM
    assert batch.y.shape == (2,)
    assert batch.global_features.shape == (2, 4)
    assert batch.fen == [graphs[0].fen, graphs[1].fen]
    assert batch.target_move == [graphs[0].target_move, graphs[1].target_move]
    assert batch.puzzle_id == ["p1", "p2"]

    output_path = tmp_path / "graphs.pt"
    torch.save(graphs, output_path)
    loaded = torch.load(output_path, weights_only=False)

    assert len(loaded) == 2
    assert loaded[0].x.shape == (64, NODE_FEATURE_DIM)

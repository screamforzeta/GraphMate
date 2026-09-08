"""Legal-candidate move scorer for no-timing chess GAT experiments.

Purpose:
    Encode a chess board with the same GAT block used by Model A, then score
    only the legal move candidates available in each position.
Input:
    Batched PyG data with x, edge_index, edge_attr, global_features, batch, and
    a per-graph list of python-chess legal moves.
Output:
    One scalar score per legal candidate, grouped by graph with candidate_ptr.
Role:
    Implements MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING without changing Model A or
    Model A2.
"""

from __future__ import annotations

import chess
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


PROMOTION_TO_ID = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}


def promotion_id(move):
    """Return the compact promotion id used by the candidate scorer.

    Parameters:
        move: python-chess Move object.
    Returns:
        Integer promotion id: 0 none, 1 queen, 2 rook, 3 bishop, 4 knight.
    Side effects:
        None.
    """

    return PROMOTION_TO_ID.get(move.promotion, 0)


class ChessGATLegalMoveScorer(nn.Module):
    """Score legal move candidates with a no-timing GAT board encoder.

    Parameters:
        input_dim: Number of node features per square.
        edge_dim: Number of tactical edge features per edge.
        hidden_per_head: Hidden channels per GAT attention head.
        heads: Number of GAT attention heads.
        num_layers: Number of GAT layers. Only 2 is supported.
        global_feature_dim: Number of graph-level features concatenated after
            pooling.
        scorer_hidden_dim: Hidden dimension of the candidate MLP scorer.
        promotion_embedding_dim: Trainable embedding size for promotion type.
        dropout: Dropout probability after hidden activations.
    Returns:
        None.
    Side effects:
        Initializes trainable PyTorch modules.
    """

    def __init__(
        self,
        input_dim=15,
        edge_dim=5,
        hidden_per_head=32,
        heads=4,
        num_layers=2,
        global_feature_dim=4,
        scorer_hidden_dim=128,
        promotion_embedding_dim=8,
        dropout=0.30,
    ):
        super().__init__()

        if num_layers != 2:
            raise ValueError(
                "ChessGATLegalMoveScorer currently supports only num_layers=2."
            )

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.hidden_per_head = hidden_per_head
        self.heads = heads
        self.num_layers = num_layers
        self.global_feature_dim = global_feature_dim
        self.scorer_hidden_dim = scorer_hidden_dim
        self.promotion_embedding_dim = promotion_embedding_dim
        self.dropout_probability = dropout
        self.node_hidden_dim = hidden_per_head * heads
        self.graph_context_dim = self.node_hidden_dim + global_feature_dim
        self.candidate_dim = (
            self.node_hidden_dim
            + self.node_hidden_dim
            + self.graph_context_dim
            + promotion_embedding_dim
        )

        # Board encoder matches Model A: [N,15] -> [N,128] -> [N,128].
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )
        self.gat2 = GATConv(
            in_channels=self.node_hidden_dim,
            out_channels=hidden_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.promotion_embedding = nn.Embedding(5, promotion_embedding_dim)

        # Small listwise scorer. It never emits fixed vocabulary logits.
        self.scorer = nn.Sequential(
            nn.Linear(self.candidate_dim, scorer_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def _validate_batch(self, batch):
        """Validate required tensors before encoding.

        Parameters:
            batch: Batched PyG data object.
        Returns:
            None.
        Side effects:
            Raises ValueError on missing attributes or shape mismatches.
        """

        for attribute in ("x", "edge_index", "edge_attr", "global_features", "batch"):
            if not hasattr(batch, attribute) or getattr(batch, attribute) is None:
                raise ValueError(f"ChessGATLegalMoveScorer requires batch.{attribute}.")

        if batch.x.ndim != 2 or batch.x.shape[1] != self.input_dim:
            raise ValueError(f"batch.x must have shape [num_nodes,{self.input_dim}].")
        if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
            raise ValueError("batch.edge_index must have shape [2,num_edges].")
        if batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != self.edge_dim:
            raise ValueError(f"batch.edge_attr must have shape [num_edges,{self.edge_dim}].")
        if batch.edge_attr.shape[0] != batch.edge_index.shape[1]:
            raise ValueError("batch.edge_attr rows must match edge_index columns.")
        if batch.global_features.ndim != 2:
            raise ValueError("batch.global_features must have shape [batch_size,global_dim].")
        if batch.global_features.shape[1] != self.global_feature_dim:
            raise ValueError(
                f"batch.global_features must have {self.global_feature_dim} columns."
            )

    def encode_board(self, batch):
        """Return node embeddings and graph context for a PyG batch.

        Parameters:
            batch: Batched PyG data object.
        Returns:
            Tuple (node_embeddings, graph_context). Shapes are [N,128] and
            [B,132] with default dimensions.
        Side effects:
            None.
        """

        self._validate_batch(batch)
        x = self.gat1(batch.x, batch.edge_index, edge_attr=batch.edge_attr)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.gat2(x, batch.edge_index, edge_attr=batch.edge_attr)
        node_embeddings = self.activation(x)
        graph_embedding = global_mean_pool(node_embeddings, batch.batch)
        graph_context = torch.cat([graph_embedding, batch.global_features], dim=1)
        return node_embeddings, graph_context

    def score_candidates(self, batch, candidate_moves):
        """Score variable legal candidates for every graph in a batch.

        Parameters:
            batch: Batched PyG object.
            candidate_moves: List of per-graph python-chess Move lists.
        Returns:
            Dict with flat scores, candidate_ptr, candidate_graph_index, and
            candidate_uci.
        Side effects:
            None.
        """

        node_embeddings, graph_context = self.encode_board(batch)
        batch_size = graph_context.shape[0]
        if len(candidate_moves) != batch_size:
            raise ValueError(
                f"Expected {batch_size} candidate lists; got {len(candidate_moves)}."
            )

        ptr = getattr(batch, "ptr", None)
        if ptr is None:
            counts = torch.bincount(batch.batch, minlength=batch_size)
            ptr = torch.cat(
                [
                    torch.zeros(1, device=batch.batch.device, dtype=torch.long),
                    counts.cumsum(0),
                ]
            )

        source_indices = []
        target_indices = []
        graph_indices = []
        promotion_ids = []
        candidate_uci = []
        candidate_ptr = [0]

        for graph_index, moves in enumerate(candidate_moves):
            if not moves:
                raise ValueError(f"Graph {graph_index} has no legal candidates.")
            base = int(ptr[graph_index].item())
            for move in moves:
                source_indices.append(base + int(move.from_square))
                target_indices.append(base + int(move.to_square))
                graph_indices.append(graph_index)
                promotion_ids.append(promotion_id(move))
                candidate_uci.append(move.uci())
            candidate_ptr.append(len(source_indices))

        device = node_embeddings.device
        source = torch.tensor(source_indices, dtype=torch.long, device=device)
        target = torch.tensor(target_indices, dtype=torch.long, device=device)
        graph_index_tensor = torch.tensor(graph_indices, dtype=torch.long, device=device)
        promotion_tensor = torch.tensor(promotion_ids, dtype=torch.long, device=device)
        candidate_ptr_tensor = torch.tensor(candidate_ptr, dtype=torch.long, device=device)

        features = torch.cat(
            [
                node_embeddings[source],
                node_embeddings[target],
                graph_context[graph_index_tensor],
                self.promotion_embedding(promotion_tensor),
            ],
            dim=1,
        )
        scores = self.scorer(features).squeeze(-1)
        return {
            "scores": scores,
            "candidate_ptr": candidate_ptr_tensor,
            "candidate_graph_index": graph_index_tensor,
            "candidate_uci": candidate_uci,
        }

    def forward(self, batch, candidate_moves):
        """Run legal-candidate scoring for a PyG batch.

        Parameters:
            batch: Batched PyG object.
            candidate_moves: List of per-graph python-chess Move lists.
        Returns:
            Flat scalar scores grouped by candidate_ptr.
        Side effects:
            None.
        """

        return self.score_candidates(batch, candidate_moves)

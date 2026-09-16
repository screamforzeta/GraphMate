"""Post-move GNN reranker for MODEL A4.

Purpose:
    Score the frozen A3 Top-K candidates after applying each move and encoding
    the resulting chess position with a dedicated GAT.
Input:
    A3 candidate representations plus post-move PyG graphs.
Output:
    One reranking logit per A3 Top-K candidate.
Role:
    Implements the no-timing MODEL_A4_POSTMOVE_GNN_RERANKER architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


OFFICIAL_A4_K = 5


class PostMoveGATEncoder(nn.Module):
    """Encode resulting-position graphs produced after candidate moves.

    Parameters:
        input_dim: Node feature count.
        edge_dim: Edge feature count.
        hidden_per_head: Hidden channels per attention head.
        heads: Number of attention heads.
        global_feature_dim: Number of graph-level features.
        dropout: Dropout probability.
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
        global_feature_dim=4,
        dropout=0.30,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.hidden_per_head = hidden_per_head
        self.heads = heads
        self.global_feature_dim = global_feature_dim
        self.dropout_probability = dropout
        self.node_hidden_dim = hidden_per_head * heads
        self.context_dim = self.node_hidden_dim + global_feature_dim

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

    def _validate_batch(self, batch):
        """Validate a post-move PyG batch before encoding."""

        for attribute in ("x", "edge_index", "edge_attr", "global_features", "batch"):
            if not hasattr(batch, attribute) or getattr(batch, attribute) is None:
                raise ValueError(f"PostMoveGATEncoder requires batch.{attribute}.")
        if batch.x.ndim != 2 or batch.x.shape[1] != self.input_dim:
            raise ValueError(f"batch.x must have shape [num_nodes,{self.input_dim}].")
        if batch.edge_attr.ndim != 2 or batch.edge_attr.shape[1] != self.edge_dim:
            raise ValueError(f"batch.edge_attr must have shape [num_edges,{self.edge_dim}].")
        if batch.global_features.ndim != 2:
            raise ValueError("batch.global_features must have shape [batch_size,global_dim].")
        if batch.global_features.shape[1] != self.global_feature_dim:
            raise ValueError(
                f"batch.global_features must have {self.global_feature_dim} columns."
            )

    def forward(self, batch):
        """Return post-move graph contexts for candidate-result graphs.

        Parameters:
            batch: Batched PyG resulting-position graphs.
        Returns:
            Tensor [num_candidate_graphs, 132] with default dimensions.
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
        return torch.cat([graph_embedding, batch.global_features], dim=1)


class ChessA4PostMoveReranker(nn.Module):
    """Rerank frozen A3 Top-K candidates with post-move graph context.

    Parameters:
        node_hidden_dim: A3 source/destination node embedding dimension.
        graph_context_dim: A3 original graph context dimension.
        promotion_embedding_dim: A3 promotion embedding dimension.
        score_feature_dim: Number of A3 scalar score features.
        scorer_hidden_dim: Hidden dimension of the A4 scorer MLP.
        dropout: Dropout probability.
        postmove_encoder: Optional custom resulting-position encoder.
    Returns:
        None.
    Side effects:
        Initializes trainable A4 modules only.
    """

    def __init__(
        self,
        node_hidden_dim=128,
        graph_context_dim=132,
        promotion_embedding_dim=8,
        score_feature_dim=2,
        scorer_hidden_dim=128,
        dropout=0.30,
        postmove_encoder=None,
    ):
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.graph_context_dim = graph_context_dim
        self.promotion_embedding_dim = promotion_embedding_dim
        self.score_feature_dim = score_feature_dim
        self.scorer_hidden_dim = scorer_hidden_dim
        self.dropout_probability = dropout
        self.postmove_encoder = postmove_encoder or PostMoveGATEncoder(dropout=dropout)
        self.postmove_context_dim = self.postmove_encoder.context_dim
        self.candidate_dim = (
            node_hidden_dim
            + node_hidden_dim
            + graph_context_dim
            + promotion_embedding_dim
            + self.postmove_context_dim
            + score_feature_dim
        )
        self.scorer = nn.Sequential(
            nn.Linear(self.candidate_dim, scorer_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def forward(self, a3_candidate_features, postmove_batch, score_features):
        """Score one flat list of A4 candidates.

        Parameters:
            a3_candidate_features: Tensor [C,396] from frozen A3 embeddings.
            postmove_batch: Batched resulting-position PyG graphs.
            score_features: Tensor [C,2] containing raw and centered A3 scores.
        Returns:
            Flat tensor [C] with A4 reranking logits.
        Side effects:
            None.
        """

        if a3_candidate_features.ndim != 2:
            raise ValueError("a3_candidate_features must be rank 2.")
        if score_features.ndim != 2 or score_features.shape[1] != self.score_feature_dim:
            raise ValueError(
                f"score_features must have shape [num_candidates,{self.score_feature_dim}]."
            )
        postmove_context = self.postmove_encoder(postmove_batch)
        if postmove_context.shape[0] != a3_candidate_features.shape[0]:
            raise ValueError("Post-move graph count must match candidate feature count.")
        if score_features.shape[0] != a3_candidate_features.shape[0]:
            raise ValueError("Score feature count must match candidate feature count.")
        features = torch.cat(
            [a3_candidate_features, postmove_context, score_features],
            dim=1,
        )
        return self.scorer(features).squeeze(-1)

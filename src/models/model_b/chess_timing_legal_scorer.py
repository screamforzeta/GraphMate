"""Timing-aware extension of the A3 legal-candidate chess GAT scorer.

Model B keeps A3 as its architectural baseline: it scores only legal
candidate moves and uses the same grouped candidate API. Its experimental
difference is a compact timing context built from PyG graph attributes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.model_a.chess_legal_scorer import ChessGATLegalMoveScorer
from src.models.model_a.chess_legal_scorer import promotion_id


class ChessGATTimingLegalMoveScorer(ChessGATLegalMoveScorer):
    """A3 legal-candidate scorer extended with synthetic timing features.

    Required batch attributes:
        previous_move_time, original_move_time, time_is_synthetic.

    The timing tensors are graph-level attributes with shape [B,1] after PyG
    batching. No timing fallback is used, because silent no-timing behavior
    would make B experimentally indistinguishable from A3.
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
        timing_hidden_dim=16,
        dropout=0.30,
    ):
        super().__init__(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_per_head=hidden_per_head,
            heads=heads,
            num_layers=num_layers,
            global_feature_dim=global_feature_dim,
            scorer_hidden_dim=scorer_hidden_dim,
            promotion_embedding_dim=promotion_embedding_dim,
            dropout=dropout,
        )
        self.timing_input_dim = 3
        self.timing_hidden_dim = timing_hidden_dim
        self.timing_encoder = nn.Sequential(
            nn.Linear(self.timing_input_dim, timing_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.candidate_dim = self.candidate_dim + timing_hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(self.candidate_dim, scorer_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(scorer_hidden_dim, 1),
        )

    def _extract_timing_features(self, batch):
        """Return normalized graph-level timing features with shape [B,3]."""

        required = (
            "previous_move_time",
            "original_move_time",
            "time_is_synthetic",
        )
        missing = [
            attribute
            for attribute in required
            if not hasattr(batch, attribute) or getattr(batch, attribute) is None
        ]
        if missing:
            raise ValueError(
                "ChessGATTimingLegalMoveScorer requires timing attributes: "
                + ", ".join(missing)
            )

        batch_size = int(batch.global_features.shape[0])
        features = []
        for attribute in required:
            value = getattr(batch, attribute)
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=batch.global_features.device)
            value = value.to(device=batch.global_features.device, dtype=torch.float)
            value = value.view(batch_size, -1)
            if value.shape[1] != 1:
                raise ValueError(f"batch.{attribute} must have one value per graph.")
            features.append(value)
        return torch.cat(features, dim=1)

    def encode_board(self, batch):
        """Return A3 board encoding with timing context appended."""

        node_embeddings, graph_context = super().encode_board(batch)
        timing_context = self.timing_encoder(self._extract_timing_features(batch))
        return node_embeddings, torch.cat([graph_context, timing_context], dim=1)

    def score_candidates(self, batch, candidate_moves):
        """Score legal candidates using A3 candidate construction plus timing."""

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


ChessGATWithTiming = ChessGATTimingLegalMoveScorer

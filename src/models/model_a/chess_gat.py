"""Graph-level GAT classifier for chess puzzle move prediction.

Purpose:
    Implement the no-timing chess GAT baseline defined in the architecture
    plan. The model consumes PyTorch Geometric chess graphs and predicts one
    move class per graph.
Input:
    Batched PyG data with x, edge_index, edge_attr, global_features, and batch.
Output:
    Raw graph-level logits with shape [batch_size, num_classes].
Role:
    First trainable model for the project, before synthetic timing is added.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class ChessGATNoTiming(nn.Module):
    """Classify chess puzzle target moves with a no-timing GAT.

    Parameters:
        input_dim: Number of node features per square.
        edge_dim: Number of tactical edge features per edge.
        hidden_per_head: Hidden channels produced by each attention head.
        heads: Number of GAT attention heads.
        num_layers: Number of GAT layers. Only 2 is supported in this baseline.
        global_feature_dim: Number of graph-level features concatenated after
            pooling.
        classifier_hidden_dim: Hidden size of the graph-level classifier.
        num_classes: Number of move classes from the move encoder vocabulary.
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
        classifier_hidden_dim=128,
        num_classes=None,
        dropout=0.10,
    ):
        super().__init__()

        if num_layers != 2:
            raise ValueError(
                "ChessGATNoTiming currently supports only num_layers=2."
            )
        if num_classes is None or num_classes <= 0:
            raise ValueError(
                "num_classes must be a positive integer from the move encoder."
            )

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.hidden_per_head = hidden_per_head
        self.heads = heads
        self.num_layers = num_layers
        self.global_feature_dim = global_feature_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.num_classes = num_classes
        self.dropout_probability = dropout
        self.node_hidden_dim = hidden_per_head * heads

        # GAT layer 1 maps square features [N,15] to node states [N,128].
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )

        # GAT layer 2 keeps the same final hidden size for graph pooling.
        self.gat2 = GATConv(
            in_channels=self.node_hidden_dim,
            out_channels=hidden_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
        )

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

        classifier_input_dim = (
            self.node_hidden_dim
            + global_feature_dim
        )

        # The classifier maps pooled graph features to raw move logits.
        self.classifier = nn.Sequential(
            nn.Linear(
                classifier_input_dim,
                classifier_hidden_dim,
            ),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(
                classifier_hidden_dim,
                num_classes,
            ),
        )

    def _validate_batch(self, batch):
        """Validate required batch attributes and inexpensive tensor shapes.

        Parameters:
            batch: Batched PyTorch Geometric data object.
        Returns:
            None.
        Side effects:
            Raises ValueError with a readable message when an invariant fails.
        """

        required_attributes = [
            "x",
            "edge_index",
            "edge_attr",
            "global_features",
            "batch",
        ]

        for attribute in required_attributes:
            if not hasattr(batch, attribute) or getattr(batch, attribute) is None:
                raise ValueError(
                    f"ChessGATNoTiming requires batch.{attribute}."
                )

        if batch.x.ndim != 2:
            raise ValueError(
                "batch.x must have rank 2 with shape [num_nodes, input_dim]."
            )
        if batch.x.shape[1] != self.input_dim:
            raise ValueError(
                f"batch.x feature dimension must be {self.input_dim}; "
                f"got {batch.x.shape[1]}."
            )

        if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
            raise ValueError(
                "batch.edge_index must have shape [2, num_edges]."
            )

        if batch.edge_attr.ndim != 2:
            raise ValueError(
                "batch.edge_attr must have rank 2 with shape "
                "[num_edges, edge_dim]."
            )
        if batch.edge_attr.shape[0] != batch.edge_index.shape[1]:
            raise ValueError(
                "batch.edge_attr row count must match the number of edges."
            )
        if batch.edge_attr.shape[1] != self.edge_dim:
            raise ValueError(
                f"batch.edge_attr feature dimension must be {self.edge_dim}; "
                f"got {batch.edge_attr.shape[1]}."
            )

        if batch.global_features.ndim != 2:
            raise ValueError(
                "batch.global_features must have rank 2 with shape "
                "[batch_size, global_feature_dim]."
            )
        if batch.global_features.shape[1] != self.global_feature_dim:
            raise ValueError(
                "batch.global_features feature dimension must be "
                f"{self.global_feature_dim}; got "
                f"{batch.global_features.shape[1]}."
            )

        batch_size = int(
            batch.batch.max().item()
        ) + 1 if batch.batch.numel() else 0

        if batch.global_features.shape[0] != batch_size:
            raise ValueError(
                "batch.global_features row count must match the number of "
                f"graphs in batch; got {batch.global_features.shape[0]} and "
                f"{batch_size}."
            )

    def forward(self, batch):
        """Run graph-level move classification.

        Parameters:
            batch: Batched PyG object containing x, edge_index, edge_attr,
                global_features, and batch.
        Returns:
            Raw logits with shape [batch_size, num_classes].
        Side effects:
            None.
        """

        self._validate_batch(batch)

        # Node GAT block: [N,15] -> [N,128] -> [N,128].
        x = self.gat1(
            batch.x,
            batch.edge_index,
            edge_attr=batch.edge_attr,
        )
        x = self.activation(x)
        x = self.dropout(x)

        x = self.gat2(
            x,
            batch.edge_index,
            edge_attr=batch.edge_attr,
        )
        node_embeddings = self.activation(x)

        # Pool 64-square node embeddings into one representation per graph.
        graph_embedding = global_mean_pool(
            node_embeddings,
            batch.batch,
        )

        # Append graph-level chess features before move classification.
        classifier_input = torch.cat(
            [
                graph_embedding,
                batch.global_features,
            ],
            dim=1,
        )

        return self.classifier(classifier_input)


def count_trainable_parameters(model):
    """Count trainable parameters in a PyTorch model.

    Parameters:
        model: PyTorch module.
    Returns:
        Number of parameters with requires_grad=True.
    Side effects:
        None.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

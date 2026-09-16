# Model A4 — Post-Move GNN Reranker No Timing

## Identity

Model A4 is the frozen post-move reranker built on frozen A3 retrieval.

## Input

For each original position:

1. frozen A3 scores all legal moves;
2. A3 Top-5 are selected without target injection;
3. each candidate is applied;
4. the resulting position is encoded as a graph.

## Architecture

Source: `src/models/model_a/chess_postmove_reranker.py`.

Post-move GAT:

- `GATConv(15, 32, heads=4, edge_dim=5)`
- ELU
- dropout `0.30`
- `GATConv(128, 32, heads=4, edge_dim=5)`
- ELU
- global mean pool
- concatenate resulting-position global features

Candidate vector:

| Component | Dim |
|---|---:|
| A3 source embedding | 128 |
| A3 destination embedding | 128 |
| original graph context | 132 |
| promotion embedding | 8 |
| post-move graph context | 132 |
| A3 score features | 2 |
| Total | 530 |

Parameters:

- A3 frozen: `71,593`
- A4 postmove encoder: `20,608`
- A4 scorer: `68,097`
- A4 trainable: `88,705`
- total inference: `160,298`

## Training

A4 trains only on rerankable examples where the target is already in A3 Top-5. It uses grouped CE over the Top-5 candidate group. Validation checkpoint selection uses validation end-to-end Top1.

## Results

Frozen validation:

- best epoch: `32`
- validation end-to-end Top1: `0.8543892243381328`
- validation loss: `0.19448498769802403`

Terminal test:

- shared N: `8610`
- A4 end-to-end Top1: `0.854123112659698`
- delta vs A3: `+17.851335656213696` pp
- A3 Recall@5: `0.9185830429732869`
- conditional Top1: `0.9298267796181565`
- errors: `1256`

## Limitations

A4 cannot recover a target outside A3 Top-5. It predicts one next move, not a full autonomous Mate-in-N line.


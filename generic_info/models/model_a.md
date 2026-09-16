# Model A — Raw No-Timing Baseline

## Identity

Model A is the historical no-timing `ChessGATNoTiming` baseline. It encodes a chess graph and predicts over the fixed train-only move vocabulary.

## Input

- node features: 15
- edge features: 5
- global features: 4
- timing: none
- target: fixed vocabulary class `move_to_idx[TargetMove]`

## Architecture

Source: `src/models/model_a/chess_gat.py`.

- GAT board encoder
- global mean pooling
- concatenation with 4 global features
- classifier over the move vocabulary

## Training

The final convergence run uses Adam, lr `5e-4`, weight decay `1e-4`, dropout `0.30`, batch size `128`, seed `42`, ReduceLROnPlateau, AMP on CUDA, and early stopping.

## Results

Frozen official baseline:

- best epoch: `162`
- stop epoch: `174`
- Top1: approximately `39.62%`
- Top3: approximately `55.81%`
- Top5: approximately `62.75%`

## Limitations

Model A can emit illegal moves because it predicts over a global move vocabulary.


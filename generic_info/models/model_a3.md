# Model A3 — Legal-Candidate Scorer No Timing

## Identity

Model A3 is the official frozen no-timing baseline.

Run ID: `MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING`.

## Input

- graph node dimension: 15
- edge dimension: 5
- global dimension: 4
- no timing
- candidate set: all legal moves from `python-chess`

## Architecture

Source: `src/models/model_a/chess_legal_scorer.py`.

- `GATConv(15, 32, heads=4, edge_dim=5)`
- ELU
- dropout `0.30`
- `GATConv(128, 32, heads=4, edge_dim=5)`
- ELU
- global mean pool
- concatenate 4 global features
- candidate vector:
  - source node embedding 128
  - destination node embedding 128
  - graph context 132
  - promotion embedding 8
- scorer MLP: `Linear(candidate_dim, 128) -> ELU -> Dropout(0.30) -> Linear(128, 1)`

## Training

Uses grouped/listwise CE over legal candidates. Default optimizer is Adam with lr `5e-4`, weight decay `1e-4`, batch size `128`, seed `42`, ReduceLROnPlateau, early stopping patience `12`, AMP on CUDA.

## Results

Official frozen test:

- N: `8610`
- CE/NLL: `1.0832485489175157`
- Top1: `0.675609756097561`
- Top3: `0.8565621370499419`
- Top5: `0.9185830429732869`
- Top10: `0.9706155632984901`
- mean rank: `2.224274099883856`
- median rank: `1.0`
- illegal Top1: `0.0`

## Limitations

A3 sees only the original position. It does not encode the post-move consequences of each candidate.


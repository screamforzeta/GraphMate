# Model A2 — Legal-Masked No-Timing Model

## Identity

Model A2 is the legal-masked no-timing ablation. It keeps the fixed-vocabulary classifier family but masks illegal moves.

## Input

Same graph representation as Model A:

- `x`: `[64, 15]`
- `edge_attr`: `[E, 5]`
- `global_features`: `[1, 4]`
- no timing

## Architecture

Same broad architecture as Model A. Legal masking is applied to restrict predictions to legal moves.

## Training and Evaluation

Entrypoint: `src/cli/training/train_model_a2_legal_mask.py`.

Evaluator: `src/evaluation/model_a/model_a_vs_a2.py`.

## Results

Frozen comparison values are stored in A/A2 evaluation artifacts. A2 improves legality but was superseded by A3.

## Limitations

A2 is still a fixed-vocabulary classifier and does not natively score the variable legal candidate set.


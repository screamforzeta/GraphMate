# Pre-LLM Project Status

## Data

- Puzzle pipeline: complete for Lichess mate-in-1 through mate-in-5.
- Game pipeline: implemented for streaming PGN sampling and cleaning.
- Final PyG split sizes: train `68,958`, validation `8,612`, test `8,610`.
- Timing data: synthetic puzzle timing dataset and Model B ablation implemented. Synthetic timing is labelled as synthetic and is not measured human puzzle thinking time.

## Representation

- Graph version: 64 square nodes, node dimension 15, edge dimension 5, global dimension 4.
- Sparse PyG `edge_index`, multilabel `edge_attr`, graph-level `global_features`.
- Lichess semantics: `OriginalFEN -> Moves[0] -> solver FEN`, `TargetMove = Moves[1]`.

## Frozen Models

| Model | Frozen status | Checkpoint status |
|---|---|---|
| A | Frozen historical baseline | convergence checkpoint path documented |
| A1 | Historical best-legal mode | uses Model A checkpoint |
| A2 | Frozen legal-mask ablation | checkpoint path documented |
| A3 | Frozen official no-timing baseline | checkpoint path documented |
| A4 | Frozen post-move reranker | checkpoint path documented |
| B | Frozen timing-aware ablation | checkpoint path documented |

## Results

Major frozen outcomes:

- A3 test Top1: `67.5609756097561%`.
- A4 test end-to-end Top1: `85.4123112659698%`.
- A4 improves A3 by `+17.8513` pp on the shared Lichess test population.
- Model B synthetic timing Top1: `65.9814%`, below A3.

## Research Questions

RQ2 timing: complete for the implemented synthetic timing representation. The implemented timing input did not improve move prediction relative to A3.

RQ1 GNN vs LLM: not yet complete. It requires a held-out/classic puzzle benchmark and a fair LLM inference protocol.

## Streamlit

Current Streamlit capabilities:

- puzzle browsing across train/validation/test CSV splits;
- MateDepth filtering from Mate-in-1 through Mate-in-5;
- chessboard and graph visualization with edge filters;
- move-by-move Mate-in-N input via legal move selection;
- ground-truth reveal and solution playback;
- A/A1/A2/A3 model registry and next-move inference paths where checkpoints exist;
- A4 Top-5 reranker explanation;
- Model B timing feature explanation;
- frozen result dashboard loaded from artifacts/snapshot values.

## Remaining Work

- Create held-out classic mate-in-N puzzle set.
- Implement LLM baseline prompts, parsing, and scoring.
- Define fair GNN-vs-LLM protocol for next-move and/or solution-line guidance.
- Run final held-out statistical analysis.
- Produce final thesis/report visualizations.

READY_FOR_HELD_OUT_PHASE = YES

READY_FOR_LLM_BENCHMARK_PHASE = NO until held-out dataset and LLM protocol are implemented.

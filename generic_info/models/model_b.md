# Model B — Timing-Aware Legal-Candidate Scorer

## Identity

Model B is the timing-aware controlled variant of A3.

Run ID: `MODEL_B_TIMING_LEGAL_MOVE_SCORER`.

## Input

Same as A3 plus graph-level timing attributes:

- `previous_move_time`
- `original_move_time`
- `time_is_synthetic`

The timing input dimension is 3.

## Architecture

Source: `src/models/model_b/chess_timing_legal_scorer.py`.

Model B inherits A3 and appends a timing encoder:

```text
Linear(3, 16) -> ELU -> Dropout(0.30)
```

The timing context is concatenated to the graph context before candidate scoring.

## Training

Uses the same grouped legal-candidate training pipeline as A3 through `src/training/model_b/model_b_timing_legal_scorer.py`.

## Results

Frozen timing ablation conclusion:

- A3 Top1: `67.5609756097561%`
- B synthetic Top1: `65.98141695702672%`
- B neutral Top1: approximately `66.0975609756%`
- B - A3: approximately `-1.5796` pp
- synthetic - neutral: approximately `-0.1161` pp
- A3 vs B McNemar: approximately `p = 1.22e-6`
- B synthetic vs neutral: approximately `p = 0.253`

## Limitations

The implemented synthetic timing representation did not improve move prediction relative to A3. This result does not prove that timing is useless for chess; it only evaluates this synthetic timing design and dataset.

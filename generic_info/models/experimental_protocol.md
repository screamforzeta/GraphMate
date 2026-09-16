# Experimental Protocol

## Frozen Status

Models A, A1, A2, A3, A4, and B are frozen for the pre-LLM handoff. This means no further architecture tuning, checkpoint selection, hyperparameter changes, or terminal test reuse for model development.

## Data Population

Final PyG graph counts:

- train: `68,958`
- validation: `8,612`
- test: `8,610`

The raw test CSV contains `8,620` rows, but `10` targets are OOV under the train-only move vocabulary and are skipped by the PyG graph generation.

## Train/Validation/Test Semantics

- Training and checkpoint selection use train/validation only.
- Test evaluations are post-freeze terminal evaluations.
- A4 test-set post-move representations are legitimate only after A4 is frozen.
- LLM and held-out classic puzzle evaluation are not complete.

## Output Semantics

A/A1/A2/A3/A4/B predict the next solver move from the solver-position FEN. They do not autonomously solve the entire Mate-in-N line.

For UI or analysis, using ground-truth opponent replies must be labelled as evaluation along the reference puzzle line, not autonomous full-line generation.


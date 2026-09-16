# Model A1 — Best-Legal Diagnostic Mode

## Identity

Model A1 is a historical naming layer for best-legal inference over Model A logits. Repository evidence does not show a separate A1 architecture or checkpoint.

## Input and Architecture

Same as Model A. The difference is inference-time post-filtering: rank Model A logits, discard illegal moves for the current board, and choose the best legal move.

## Training

No separate training. Uses the Model A checkpoint.

## Results

Reported in A/A2/A3 comparison artifacts as `best_legal` metrics where available.

## Limitations

Best-legal filtering improves legality but does not retrain the representation or scorer. It remains tied to the fixed vocabulary.


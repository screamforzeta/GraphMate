# Held-Out Classic Puzzle Schema

The held-out classic dataset is not created in this implementation pass.

Required fields:

| Field | Meaning |
|---|---|
| `puzzle_id` | Stable puzzle identifier |
| `source` | Dataset/source label |
| `initial_solver_fen` | FEN at the solving side's first move |
| `reference_moves_uci` | Canonical UCI line from solver position |
| `mate_depth` | Number of solving-side moves |

Optional fields:

- title
- composer
- year
- source_reference
- notes

Classic problems may exist online and may therefore be present in LLM pretraining corpora. Do not claim this benchmark is contamination-free without evidence.


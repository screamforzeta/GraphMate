# Chess Graph Representation

This document freezes the graph representation implemented in `src/graph/`.

## Core Object

Each puzzle position is represented as a PyTorch Geometric `Data` object:

```text
Data(x, edge_index, edge_attr, y, global_features)
```

- `x`: `[64, 15]`, one node per board square.
- `edge_index`: `[2, E]`, sparse directed PyG edge list.
- `edge_attr`: `[E, 5]`, multilabel relation vector.
- `y`: scalar class index for `TargetMove` when fixed-vocabulary models need it.
- `global_features`: `[1, 4]`, graph-level state.
- Metadata includes `fen`, `target_move`, `puzzle_id`, `rating`, `mate_depth`, and `source_row_index` when generated from final CSV splits.

## Node Features

Source: `src/graph/node_features.py`.

| Index | Feature | Semantics |
|---:|---|---|
| 0 | pawn | Piece one-hot pawn |
| 1 | knight | Piece one-hot knight |
| 2 | bishop | Piece one-hot bishop |
| 3 | rook | Piece one-hot rook |
| 4 | queen | Piece one-hot queen |
| 5 | king | Piece one-hot king |
| 6 | color | `1.0` white, `0.0` black or empty |
| 7 | occupied | `1.0` if a piece is on the square |
| 8 | normalized row | `chess.square_rank(square) / 7.0` |
| 9 | normalized column | `chess.square_file(square) / 7.0` |
| 10 | attacked_by_white | number of white attackers |
| 11 | attacked_by_black | number of black attackers |
| 12 | legal_mobility | count of legal moves from the square, summed over both side-to-move settings |
| 13 | is_pinned | pin flag for occupied pieces |
| 14 | piece_value | pawn 1, knight/bishop 3, rook 5, queen 9, king 0 |

`is_check` is not a node feature.

## Edge Features

Source: `src/graph/edge_features.py`.

| Index | Feature | Semantics |
|---:|---|---|
| 0 | legal_move | Directed legal move from source square to destination |
| 1 | attack | Source piece attacks an enemy piece |
| 2 | defend | Source piece defends a friendly piece |
| 3 | pin | Pin relation |
| 4 | check_line | Relation along a checking line |

Relations are multilabel. If the same square pair has multiple relations, the edge feature vector stores their elementwise maximum.

## Global Features

Source: `src/graph/graph_builder.py::extract_global_features`.

| Index | Feature | Formula |
|---:|---|---|
| 0 | side_to_move | `1.0` if white to move else `0.0` |
| 1 | is_check | `1.0` if current side is in check else `0.0` |
| 2 | fullmove_number_normalized | `min(board.fullmove_number, 200) / 200.0` |
| 3 | halfmove_clock_normalized | `min(board.halfmove_clock, 100) / 100.0` |

Global features describe only the position. They do not include rating, MateDepth, themes, target move, solution moves, or timing.

## Lichess Puzzle Semantics

Source: `src/data/preprocess/prepare_puzzles_dataset.py::transform_lichess_puzzle_fields`.

The raw Lichess puzzle FEN is the position before the setup move. The project transforms it as:

```text
OriginalFEN
-> apply Moves[0]
-> FEN shown to solver and stored in Data.fen
-> TargetMove = Moves[1]
```

`Moves[1:]` is the puzzle solution sequence from the solver position. Models A/A2/A3/A4/B are trained to predict the next target move, not the full Mate-in-N line.


# Frozen Checkpoints

[English](README.md) | [Italiano](README.it.md)

This directory contains only the final frozen checkpoints needed for reproducible local inference and evaluation. Training outputs, trial checkpoints, `last.pt` files, and raw experiment artifacts remain under ignored `artifacts/` paths.

| Model | Role | Repository path | SHA256 | Size | Timing | Benchmark role |
| --- | --- | --- | --- | --- | --- | --- |
| A | Global-vocabulary no-timing GAT baseline | `checkpoints/model_a/best.pt` | `1a72d0b6f675b50c829d76c63343e7016a569c00f993f063b0d0b75061de9b70` | ~3.2 MB | no | Lichess baseline |
| A2 | Legal-masked global-vocabulary GAT | `checkpoints/model_a2/best.pt` | `20a76ae5648ded7f8fe3eb835eddd5ec0a87c9043989c37a65b4e8b9dc6901c1` | ~3.1 MB | no | Lichess baseline |
| A3 | Legal-move candidate scorer | `checkpoints/model_a3/best.pt` | `4efec653a451da7585f3663847c8dc5caaa8ebffadea677617e2f496e4253b80` | ~943 KB | no | Lichess baseline and YACPDB external generalization |
| B | Timing-aware A3-controlled variant | `checkpoints/model_b/best.pt` | `b8bbad2420ce301582ad08377b1c80f56609513a8dae201600557e255b87d26f` | ~952 KB | synthetic timing | Lichess timing comparison; not applicable to YACPDB classic |
| A4 | A3 Top-5 post-move GNN reranker | `checkpoints/model_a4/best.pt` | `0cb73acf70487efa5c93f715a5a60c0aa09d64792d894301efaaaf47b2801e99` | ~1.1 MB | no | Lichess reranking and YACPDB external generalization |

The hashes above were verified after copying from the official local experiment artifacts. Do not redirect training output into this directory.

Model A1 is an inference mode over Model A, so it has no separate checkpoint.

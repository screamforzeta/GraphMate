# Resources

[English](README.md) | [Italiano](README.it.md)

This directory contains small, version-controlled resources required by the frozen GraphMate models.

## Move Encoder

`move_encoder/` contains the canonical move vocabulary used by the global-vocabulary models and compatibility tooling:

- `move_to_idx.json`
- `idx_to_move.json`
- `move_encoder_stats.json`

These files were copied out of ignored experiment artifacts so public evaluation code can resolve move indices without depending on local training-output directories.

Do not regenerate or edit these files when inspecting the final results.

# Held-Out Classic Dataset Report

Status: `AVAILABILITY_READY`

The external classic Mate-in-N dataset is not frozen yet. This pass implements generic import, YACPDB-specific availability scanning, key validation, duplicate detection, Lichess contamination audit, manifest, and documentation infrastructure. No real external source file has been imported in this development VM.

## Purpose

The original project specification requires an external held-out set of classic Mate-in-N problems, independent from the Lichess puzzle data used for training, validation, and testing.

The set is intended for final external evaluation only. It must not be used for training, fine-tuning, model selection, prompt tuning, parser tuning, or hyperparameter tuning.

## Source And Provenance

The importer accepts `.json`, `.jsonl`, or `.csv` files from a documented external public source. Source metadata is preserved where available:

- source name;
- source URL;
- original problem ID;
- author/composer;
- publication/reference;
- license/provenance;
- original source metadata.

No source is bundled and no synthetic final dataset is created.

### YACPDB Support

YACPDB support is implemented in:

```text
src/data/heldout_sources/yacpdb.py
```

The public website exposes a web search interface and documented Py2Web problem embedding format, but no stable documented REST download API was identified in this pass. The importer therefore consumes local `.json`, `.jsonl`, or `.csv` export/cache records and does not scrape live pages.

Accepted YACPDB records are limited to orthodox directmates `#1` through `#10` with a legal, structurally identified key move. Helpmates, selfmates, series problems, fairy conditions, twins, malformed positions, and unsupported key notation are rejected.

Real `gateway/ql` records use `algebraic.white` and `algebraic.black` piece lists rather than FEN. The YACPDB adapter converts standard orthodox pieces to FEN with this policy:

- White to move for directmates;
- no fabricated castling rights;
- no fabricated en-passant square;
- halfmove/fullmove set to canonical neutral values `0 1`;
- records requiring unknown historical state are rejected.

The key extractor ignores set-play/defense lines such as `1...`, skips tries marked with `?`, and requires exactly one actual solver-side `1.` key before validating legality.

## Canonical Schema

Final canonical rows contain:

- `heldout_id`
- `source`
- `source_problem_id`
- `fen`
- `side_to_move`
- `mate_depth`
- `stipulation`
- `problem_type`
- `key_move_uci`
- `solution_uci`
- `solution_plies`
- `source_solution_raw`
- `solution_tree`
- `principal_line_uci`
- `source_url`
- `source_reference`
- `author`
- `publication`
- `publication_date`
- `license`
- `validation_status`
- `dataset_version`
- `key_validation_status`
- `line_validation_status`
- `forced_mate_verification_status`
- `source_metadata_json`

`solution_uci` is a JSON array of UCI moves containing the complete source-provided principal line.

For YACPDB v1, `key_move_uci` stores the validated first solution move, `source_solution_raw` preserves the raw source solution, and `solution_tree` / `principal_line_uci` remain empty until a reviewed tree parser exists.

## Mate-in-N Definition

Mate-in-N means N moves by the solving side. A complete principal line normally has up to `2*N - 1` plies.

The validator checks that the supplied line is legal, ends in checkmate, and contains exactly `N` solving-side moves.

## Validation Procedure

Implemented in:

```text
src/data/heldout_classic.py
```

Checks:

- valid FEN;
- side-to-move agreement;
- legal sequential UCI solution line;
- final checkmate;
- final mating move by the original solving side;
- mate depth agreement;
- malformed/truncated line rejection;
- exact FEN duplicate rejection;
- normalized board-state duplicate rejection;
- Lichess train/val/test overlap rejection.

YACPDB-specific checks:

- stipulation is an orthodox directmate `#1`..`#10`;
- FEN is supplied or deterministically built from YACPDB `algebraic` piece lists;
- source solution structure identifies an actual key move;
- key move is legal in the source FEN;
- capture markers must agree with the board;
- problem ID is present for provenance.

## Duplicate Policy

Duplicates are rejected by:

- exact FEN;
- normalized board-state key.

The normalized key uses the first four FEN fields: board, side to move, castling rights, and en-passant square.

## Lichess Contamination Audit

Held-out FENs are compared against local Lichess splits:

- `data/final/puzzles/train.csv`
- `data/final/puzzles/val.csv`
- `data/final/puzzles/test.csv`

The audit checks exact solver-facing FEN, normalized solver-facing FEN, and exact `OriginalFEN`.

## Limitation

A legal principal line ending in mate is not proof of forced Mate-in-N against all defenses.

This infrastructure distinguishes:

- `LINE_VALIDATED`
- `FORCED_MATE_VERIFIED`

No engine verification is performed in this pass.

## Reproducibility

Build command template:

```bash
./venv/bin/python -m src.data.heldout_classic \
  --source data/heldout_classic/raw/<external_source>.jsonl \
  --dataset-version v1 \
  --source-name <public-source-name> \
  --source-url <source-url> \
  --source-license <license-or-provenance> \
  --per-depth 20 \
  --seed 42
```

YACPDB availability scan:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb \
  --mode availability \
  --source data/heldout_classic/raw/yacpdb/<export>.jsonl \
  --output-root data/heldout_classic
```

YACPDB candidate build:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb \
  --mode build \
  --source data/heldout_classic/raw/yacpdb/<export>.jsonl \
  --output-root data/heldout_classic \
  --dataset-version yacpdb_v1 \
  --per-depth 20 \
  --seed 42
```

Outputs:

```text
data/heldout_classic/final/heldout_classic.csv
data/heldout_classic/final/manifest.json
data/heldout_classic/processed/rejected_records.jsonl
data/heldout_classic/processed/yacpdb_availability.json
data/heldout_classic/processed/yacpdb_manual_review.json
```

## Current Counts

No real external source has been imported in this VM.

- raw count: not available
- accepted count: not available
- rejected count: not available
- MateDepth distribution: not available
- dataset fingerprint: not available
- lifecycle status: `BUILDING`
- YACPDB infrastructure status: `AVAILABILITY_READY`

## Held-Out Status

This dataset must remain unseen by all models until the dataset content, version, validation report, contamination audit, and fingerprint are frozen.

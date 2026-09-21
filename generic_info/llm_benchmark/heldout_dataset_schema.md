# Held-Out Classic Mate-in-N Dataset

Status: `VALIDATED_NOT_FROZEN`

The held-out classic dataset is a true external evaluation set. It must remain independent from the Lichess train/validation/test splits and must not be used for model training, fine-tuning, hyperparameter choices, prompt tuning, or parser changes.

The YACPDB candidate dataset v1 has been built from the cached bounded
availability scan, but the final dataset is not frozen. It still requires an
independent forced-mate verification/review pass.

## Location

```text
data/heldout_classic/
  raw/
    yacpdb/
  processed/
  final/
    yacpdb_classic_v1/
```

This area is separate from `data/final/puzzles/`.

## Canonical Schema

The current authoritative candidate rows are written to:

```text
data/heldout_classic/final/yacpdb_classic_v1/dataset.jsonl
```

A CSV derivative is also generated at:

```text
data/heldout_classic/final/yacpdb_classic_v1/dataset.csv
```

Required fields:

| Field | Meaning |
|---|---|
| `heldout_id` | Stable internal ID derived from source metadata when absent |
| `source` | Public external source label |
| `source_problem_id` | Original source problem ID |
| `fen` | Solver-facing starting FEN |
| `side_to_move` | `white` or `black`, matching the FEN |
| `mate_depth` | Number of solving-side moves |
| `stipulation` | Source stipulation, for example `#3` |
| `problem_type` | Normalized problem type, for example `directmate` |
| `key_move_uci` | First solution move in UCI when only key validation is available |
| `solution_uci` | JSON array of complete principal line in UCI |
| `solution_plies` | Number of plies in `solution_uci` |
| `source_solution_raw` | Original source solution text |
| `solution_tree` | JSON solution tree when a reviewed tree parser exists |
| `principal_line_uci` | JSON principal line when normalized separately |
| `source_url` | Source URL when available |
| `source_reference` | Publication or collection reference |
| `author` | Composer/author when available |
| `publication` | Publication field when distinct from source reference |
| `publication_date` | Publication year/date when available |
| `license` | License/provenance field |
| `validation_status` | Current semantic validation state |
| `dataset_version` | Explicit held-out dataset version |
| `key_validation_status` | Key-move validation state |
| `line_validation_status` | Principal-line validation status |
| `forced_mate_verification_status` | Engine-proof status |
| `source_metadata_json` | Preserved non-canonical source fields |

YACPDB v1 records are key-validated, not line-normalized. For those rows:

- `key_move_uci` is required;
- `source_solution_raw` is preserved;
- `solution_uci` is empty;
- `solution_tree` is empty;
- `principal_line_uci` is empty;
- `line_validation_status = NOT_NORMALIZED`;
- `forced_mate_verification_status = NOT_VERIFIED_ENGINE_NOT_USED`.

For real YACPDB `gateway/ql` records, the raw source may not contain `fen`. The project normalizes supported orthodox `algebraic.white` / `algebraic.black` lists into FEN before writing canonical rows. Castling and en-passant rights are not invented from piece placement; they are set absent unless explicitly supplied by a FEN source.

## Mate-in-N Semantics

Mate-in-N means **N moves by the solving side**. A complete principal line normally contains up to:

```text
2*N - 1 plies
```

The validator preserves the source-provided MateDepth and checks that the legal supplied line ends with checkmate after exactly `N` solving-side moves.

## Validation

Implemented validator:

```bash
./venv/bin/python -m src.data.heldout_classic \
  --source data/heldout_classic/raw/<external_source>.jsonl \
  --dataset-version v1 \
  --source-name <public-source-name> \
  --source-url <source-url> \
  --source-license <license-or-provenance>
```

Validation checks:

- FEN parses with `python-chess`;
- `side_to_move` agrees with FEN;
- every UCI solution move is legal sequentially;
- full supplied line reaches checkmate;
- final mating move is made by the original solving side;
- solver-move count agrees with `mate_depth`;
- truncated/malformed lines are rejected;
- exact FEN duplicates are rejected;
- normalized board-state duplicates are rejected;
- overlap with Lichess train/val/test solver-facing FENs is rejected.

Invalid rows are written to:

```text
data/heldout_classic/processed/rejected_records.jsonl
```

YACPDB-specific rejected records and manual review files are written to:

```text
data/heldout_classic/processed/yacpdb_rejected_records.json
data/heldout_classic/processed/yacpdb_manual_review.json
```

## YACPDB Import

YACPDB support lives in:

```text
src/data/heldout_sources/yacpdb.py
```

The importer reads local JSON/JSONL/CSV export/cache files. It does not assume a live API, does not scrape HTML pages, and does not run engines.

Accepted rows must be orthodox directmates `#1`..`#10` with a legal structurally identified key move. Full YACPDB solution trees are preserved as raw text only until a dedicated tree parser is reviewed and tested.

The structural key parser skips set-play/defense lines, skips tries/refutations, and rejects ambiguous keys. This is intentionally stricter than selecting the first legal move-like token.

Availability command:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb \
  --mode availability \
  --source data/heldout_classic/raw/yacpdb/<export>.jsonl \
  --output-root data/heldout_classic
```

Build command:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb_candidate_build \
  --output-root data/heldout_classic \
  --dataset-version yacpdb_classic_v1 \
  --per-depth 20 \
  --seed 42
```

Candidate build metadata:

- availability fingerprint: `a7694af29724e10907d3b51d759d1ef4f38492e35cfb410b40150400696a2880`
- selection seed: `42`
- canonical ordering: numeric YACPDB `source_problem_id`
- selected rows: 200
- selected rows per MateDepth: exactly 20 for each MateIn1..MateIn10
- dataset fingerprint: `bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a`
- lifecycle: `VALIDATED_NOT_FROZEN`
- forced-mate verification: `NOT_VERIFIED_ENGINE_NOT_USED`

## Line Validation vs Forced Mate

`LINE_VALIDATED` means the supplied principal line is legal and ends in checkmate.

It does **not** prove the position is a forced Mate-in-N against all legal defenses. No Stockfish or engine verification is run in this pass. A stronger forced-mate protocol must be frozen before any official held-out model benchmark.

## Lichess Contamination Audit

The builder compares held-out FENs against all local Lichess splits:

- `data/final/puzzles/train.csv`
- `data/final/puzzles/val.csv`
- `data/final/puzzles/test.csv`

It checks:

- exact solver-facing FEN overlap;
- normalized solver-facing board-state overlap;
- exact `OriginalFEN` overlap.

The normalized board-state key is the first four FEN fields:

```text
piece placement + side to move + castling rights + en-passant square
```

Halfmove and fullmove counters are ignored.

## Lifecycle

Lifecycle states:

- `BUILDING`: infrastructure/source import exists, but no final validated source is frozen.
- `VALIDATED`: real external source imported, validation/contamination checks complete, manifest generated.
- `FROZEN`: final content, version, and fingerprint are frozen before any model sees the data.

The builder writes `VALIDATED`, not `FROZEN`. Freezing is a separate decision after reviewing provenance, validation, contamination, distribution, and fingerprint.

## Manifest

The builder writes:

```text
data/heldout_classic/final/manifest.json
```

It includes source metadata, schema version, validator version, counts, rejection reasons, duplicate/overlap policies, MateDepth distribution, deterministic fingerprint, and explicit statements that model outputs were not used and the data has not been used for tuning.

# YACPDB Methodology

## Purpose

YACPDB is the external source used in this project to construct a classic
Mate-in-N held-out benchmark. In this repository, YACPDB is treated as a source
of classical chess compositions rather than as another game-derived tactical
puzzle sample.

A Lichess puzzle is derived from a played game position. A classical chess
composition is intentionally constructed around a stipulation, such as
directmate `#N`, and may differ substantially from naturally occurring game
positions. This distinction is useful because the held-out benchmark is meant
to test external generalization beyond the Lichess training/test distribution.

External background facts about YACPDB, its history, maintainers, licensing,
and database scope require bibliography support in the final report:
`[EXTERNAL REFERENCE REQUIRED]`.

## Why YACPDB Was Selected

YACPDB was selected for methodological reasons specific to this project:

- it provides chess compositions rather than another random sample of
  game-derived tactical positions;
- it supports directmate stipulations across the desired MateDepth range;
- it makes it possible to build an external benchmark distinct from Lichess;
- records expose source/problem identifiers and solution text useful for
  provenance;
- the available data supported balanced coverage from `#1` through `#10`.

This does not imply that YACPDB is universally superior to other chess-problem
sources. It was appropriate for the project goal: a fixed, external,
composition-based Mate-in-N benchmark.

## Acquisition

The repository used a public but undocumented YACPDB query endpoint:

```text
GET https://yacpdb.org/gateway/ql?q=<QUERY>&p=<PAGE>
```

Endpoint classification:

```text
PUBLIC_BUT_UNDOCUMENTED
```

Queries used for the availability scan:

```text
Stip("^#1$")
...
Stip("^#10$")
```

Pagination used `p=<page>`. The observed result page size was 100 records.
No authentication was required during project acquisition. Because this is not
documented as a stable official API, reproducibility depends on the cached raw
responses and recorded fingerprints in the repository artifacts.

## Filtering

The importer accepts only conservative orthodox directmate records:

- directmate stipulations `#1` through `#10`;
- supported orthodox pieces and positions;
- structurally identifiable legal root key;
- no Lichess overlap under the repository contamination policy.

Rejected categories include, when encountered:

- helpmates;
- selfmates;
- series problems;
- fairy conditions or unsupported fairy pieces;
- unsupported twins;
- malformed positions;
- unsupported key notation;
- illegal extracted source keys;
- capture-marker inconsistencies;
- ambiguous or missing actual key.

The implementation source of truth is:

```text
src/data/heldout_sources/yacpdb.py
src/data/heldout_sources/yacpdb_position.py
src/data/heldout_sources/yacpdb_solution.py
src/data/heldout_sources/yacpdb_availability.py
```

## Position Normalization

YACPDB records may represent positions as piece lists rather than canonical
FEN. The project normalizes supported orthodox `algebraic.white` and
`algebraic.black` piece lists into standard FEN.

Normalization policy:

- directmates are normalized with White to move as implemented;
- no castling rights are fabricated;
- no en-passant square is fabricated;
- halfmove/fullmove counters use neutral defaults;
- Popeye/YACPDB knight notation is mapped carefully where relevant.

The canonical FEN in `dataset.jsonl` is the benchmark position. Later
verification and model evaluation use that normalized position, not an
alternative reconstruction.

## Source Key Extraction

The benchmark target is the root key move: the first solving-side move of the
composition. This is distinct from parsing the full composition solution tree.

The structural extractor avoids treating the following as the key:

- set play;
- defensive continuations;
- tries and refutations;
- later solution plies.

The root-key target is directly comparable to the next-move prediction task
used elsewhere in the project: given a chess position, predict the correct
solver move.

## Availability Scan

The bounded YACPDB availability scan inspected up to 200 source records per
MateDepth.

```text
records inspected: 200 per MateDepth, 2000 total
source-reported total across queries: 430401
availability fingerprint: a7694af29724e10907d3b51d759d1ef4f38492e35cfb410b40150400696a2880
```

Clean eligible counts:

| MateDepth | Clean eligible |
|---:|---:|
| 1 | 53 |
| 2 | 186 |
| 3 | 191 |
| 4 | 189 |
| 5 | 193 |
| 6 | 181 |
| 7 | 132 |
| 8 | 120 |
| 9 | 126 |
| 10 | 169 |
| Total | 1540 |

These counts describe the bounded inspected sample, not the full YACPDB
population. Query order is server-defined and was treated as `UNKNOWN`; the
scan was not a random sample.

## Final Candidate Selection

The final candidate dataset selected:

```text
20 problems per MateDepth
MateDepth #1 through #10
200 total problems
seed = 42
dataset fingerprint = bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a
```

The only balancing dimension was MateDepth. The selection was fixed before
external held-out model evaluation. This prevents performance-based selection
bias: no A3/A4/Model B/LLM result influenced which YACPDB problems entered the
benchmark.


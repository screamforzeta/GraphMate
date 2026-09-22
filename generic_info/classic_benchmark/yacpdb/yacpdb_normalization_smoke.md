# YACPDB Normalization Smoke

Status: `READY_FOR_AVAILABILITY_SCAN`

This diagnostic pass tested real YACPDB `gateway/ql` records without running a broad availability scan, building a final dataset, or using model/engine outputs.

## Scope

Unique real YACPDB problems contacted in this pass: `8`.

IDs:

```text
26026, 4, 36411, 297, 49265, 44165, 341021, 412003
```

Raw responses are preserved under:

```text
data/heldout_classic/raw/yacpdb/normalization_smoke/
```

## Real Format Observed

Positions are encoded as:

```yaml
algebraic:
  white: [Kc6, Qf3, Sb5, Pa3]
  black: [Kc4, Se4, Pd4]
```

Observed orthodox grammar:

| Element | Syntax |
|---|---|
| King | `K<square>` |
| Queen | `Q<square>` |
| Rook | `R<square>` |
| Bishop | `B<square>` |
| Knight | `S<square>` in YACPDB/Popeye; `N<square>` accepted for compatibility |
| Pawn | `P<square>` |
| Square | `[a-h][1-8]` |
| Multiple pieces | YAML/JSON list per color |

Unsupported constructs fail closed:

- fairy pieces/specifications such as `Royal Zf3`;
- neutral pieces;
- malformed squares;
- duplicate occupancy;
- missing kings;
- non-directmate stipulations.

## FEN Policy

YACPDB algebraic piece lists do not provide historical state.

The normalizer therefore uses this composition-position policy:

| FEN field | Policy |
|---|---|
| Piece placement | built from `algebraic.white` and `algebraic.black` |
| Side to move | White for orthodox directmates `#1`..`#10` |
| Castling rights | none, because not present in source |
| En-passant | none, because not present in source |
| Halfmove clock | `0` |
| Fullmove number | `1` |

Records requiring castling or en-passant historical state are rejected rather than repaired.

## Solution Syntax Observed

Observed solution constructs:

- set play before the actual solution, starting with `1...`;
- tries marked with `?` and refuted by `but ... !`;
- actual keys marked with `!`;
- threats after the key;
- defenses/branches after the key;
- comments in `{...}`;
- check/mate markers `+` and `#`;
- Popeye coordinate moves such as `Qf3-d1`, `Qa1-a8`, `Sf2-g4`;
- captures with `*` in branches and mates.

The parser does not reconstruct the full solution tree in this pass. It extracts only the structurally identified actual key.

## Key Extraction Algorithm

The parser:

1. Splits the raw solution into lines.
2. Ignores empty lines and `1...` defense/set-play lines.
3. Considers only solver-side `1.` lines.
4. Rejects lines marked as tries/refutations with `?` or `but`.
5. Requires exactly one remaining structural key token.
6. Resolves that token against the starting board.
7. Validates legal move membership.

It never chooses a key because it is merely the first legal move encountered.

## Diagnostic Records

| ID | Stipulation | Status | Reason | Generated FEN | Key token | Key UCI | Method |
|---:|---|---|---|---|---|---|---|
| 26026 | `#2` | ACCEPT |  | `8/8/2K5/1N6/2kpn3/P4Q2/8/8 w - - 0 1` | `Qf3-d1` | `f3d1` | `popeye_coordinate` |
| 297 | `#2` | ACCEPT |  | `1n6/2pR3b/4p3/4k2N/1R1nP1pp/8/1K1N4/Q7 w - - 0 1` | `Qa1-a8` | `a1a8` | `popeye_coordinate` |
| 341021 | `h#8` | REJECT | `UNSUPPORTED_NON_DIRECTMATE` |  |  |  |  |
| 36411 | `#2` | ACCEPT |  | `3Q1R1n/7P/2R3Np/5prk/R5N1/2K1Q1P1/8/5b1r w - - 0 1` | `Qd8-d1` | `d8d1` | `popeye_coordinate` |
| 4 | `#2` | ACCEPT |  | `7n/3NR3/1P3p2/1p1kbN1B/1p6/1K6/6b1/1Q6 w - - 0 1` | `Qb1-f1` | `b1f1` | `popeye_coordinate` |
| 412003 | `h#6.5` | REJECT | `UNSUPPORTED_NON_DIRECTMATE` |  |  |  |  |
| 44165 | `h#2` | REJECT | `UNSUPPORTED_NON_DIRECTMATE` |  |  |  |  |
| 49265 | `#3` | ACCEPT |  | `6R1/2K5/8/8/5p2/6p1/1Q3Nk1/8 w - - 0 1` | `Sf2-g4` | `f2g4` | `popeye_coordinate` |

MateDepths represented in accepted real records: `2`, `3`.

Moremover directmate was not found by cheap ID-based probing in this pass. Promotion and castling were not observed in real smoke records; they remain covered by synthetic unit tests only.

## Readiness Gate

PASS:

- actual `gateway/ql` records can be consumed directly;
- algebraic position conversion is deterministic;
- unsupported/fairy constructs fail closed;
- no castling/en-passant rights are fabricated;
- set play and tries are skipped structurally;
- the key is not selected by legality search;
- extracted keys are validated for legality;
- ID `26026` is covered as regression behavior;
- no model or engine was used.

Remaining limitations:

- no full solution tree normalization;
- no forced-mate proof;
- no broad availability statistics yet;
- castling/en-passant real examples remain unsupported unless source historical state is explicit.


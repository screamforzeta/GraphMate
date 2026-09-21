# YACPDB Availability Scan

Status: `COMPLETE`

Availability is measured within a bounded discovery sample of up to 200 unique source records per MateDepth. Query order is `UNKNOWN`; this is not a random sample.

## Funnel

| MateDepth | Source count | Inspected | Filter pass | Position OK | Key extracted | Key legal | Unique eligible | Clean unique eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 543 | 200 | 194 | 191 | 53 | 53 | 53 | 53 |
| 2 | 217456 | 200 | 192 | 192 | 186 | 186 | 186 | 186 |
| 3 | 141902 | 200 | 194 | 194 | 191 | 191 | 191 | 191 |
| 4 | 42331 | 200 | 197 | 197 | 189 | 189 | 189 | 189 |
| 5 | 13852 | 200 | 198 | 198 | 193 | 193 | 193 | 193 |
| 6 | 6439 | 200 | 199 | 198 | 181 | 181 | 181 | 181 |
| 7 | 3554 | 200 | 197 | 197 | 132 | 132 | 132 | 132 |
| 8 | 2149 | 200 | 196 | 196 | 120 | 120 | 120 | 120 |
| 9 | 1238 | 200 | 198 | 198 | 126 | 126 | 126 | 126 |
| 10 | 937 | 200 | 196 | 196 | 169 | 169 | 169 | 169 |
| TOTAL | 430401 | 2000 | 1961 | 1957 | 1540 | 1540 | 1540 | 1540 |

## Rejections

| Rejection reason | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `AMBIGUOUS_KEY` | 21 | 4 | 1 | 4 | 5 | 10 | 7 | 3 | 4 | 5 | 64 |
| `CAPTURE_MARKER_MISMATCH` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `KEY_NOT_LEGAL` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `KEY_PIECE_PREFIX_MISMATCH` | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 7 |
| `MISSING_KING` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `NO_ACTUAL_KEY` | 91 | 0 | 2 | 0 | 0 | 4 | 48 | 69 | 64 | 14 | 292 |
| `TRY_ONLY` | 12 | 1 | 0 | 1 | 0 | 1 | 3 | 3 | 3 | 1 | 25 |
| `UNKNOWN_CASTLING_RIGHTS` | 1 | 1 | 0 | 2 | 0 | 1 | 0 | 0 | 1 | 4 | 10 |
| `UNSUPPORTED_ALGEBRAIC_POSITION` | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 3 |
| `UNSUPPORTED_FAIRY_CONDITION` | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 |
| `UNSUPPORTED_MOVE_NOTATION` | 4 | 0 | 0 | 1 | 0 | 1 | 6 | 1 | 0 | 3 | 16 |
| `UNSUPPORTED_TWIN` | 5 | 8 | 6 | 3 | 2 | 0 | 3 | 4 | 2 | 4 | 37 |

## Depth Feasibility

- MateIn1: 53 - >= 20 clean candidates
- MateIn2: 186 - >= 20 clean candidates
- MateIn3: 191 - >= 20 clean candidates
- MateIn4: 189 - >= 20 clean candidates
- MateIn5: 193 - >= 20 clean candidates
- MateIn6: 181 - >= 20 clean candidates
- MateIn7: 132 - >= 20 clean candidates
- MateIn8: 120 - >= 20 clean candidates
- MateIn9: 126 - >= 20 clean candidates
- MateIn10: 169 - >= 20 clean candidates

## Manual Review Sample

- MateIn1 ID 29464: `Ke1-e2` -> `e1e2`, pieces=21, parser=popeye_coordinate
- MateIn1 ID 140062: `Qe6-c4` -> `e6c4`, pieces=6, parser=popeye_coordinate
- MateIn2 ID 4: `Qb1-f1` -> `b1f1`, pieces=14, parser=popeye_coordinate
- MateIn2 ID 5: `Bd3-e4` -> `d3e4`, pieces=8, parser=popeye_coordinate
- MateIn3 ID 29476: `Bd1-a4` -> `d1a4`, pieces=6, parser=popeye_coordinate
- MateIn3 ID 33245: `Bd4-a7` -> `d4a7`, pieces=16, parser=popeye_coordinate
- MateIn4 ID 29188: `Qf6-g7` -> `f6g7`, pieces=23, parser=popeye_coordinate
- MateIn4 ID 41291: `Ba4-d1` -> `a4d1`, pieces=7, parser=popeye_coordinate
- MateIn5 ID 41477: `Ke5-d6` -> `e5d6`, pieces=6, parser=popeye_coordinate
- MateIn5 ID 41480: `Ke4-d5` -> `e4d5`, pieces=7, parser=popeye_coordinate
- MateIn6 ID 50857: `b2-b3` -> `b2b3`, pieces=18, parser=popeye_coordinate
- MateIn6 ID 65050: `Sg1-e2` -> `g1e2`, pieces=22, parser=popeye_coordinate
- MateIn7 ID 29697: `Bg6-h7` -> `g6h7`, pieces=22, parser=popeye_coordinate
- MateIn7 ID 65536: `Se4-g5` -> `e4g5`, pieces=18, parser=popeye_coordinate
- MateIn8 ID 67291: `Bc2-h7` -> `c2h7`, pieces=8, parser=popeye_coordinate
- MateIn8 ID 67293: `e2-e4` -> `e2e4`, pieces=19, parser=popeye_coordinate
- MateIn9 ID 67413: `Ka1-b1` -> `a1b1`, pieces=16, parser=popeye_coordinate
- MateIn9 ID 67414: `Sc7-d5` -> `c7d5`, pieces=14, parser=popeye_coordinate
- MateIn10 ID 67483: `b2-b3` -> `b2b3`, pieces=24, parser=popeye_coordinate
- MateIn10 ID 67484: `Qe2-a2` -> `e2a2`, pieces=10, parser=popeye_coordinate

## Integrity

- No parser/normalizer semantic changes were made during the scan.
- No final dataset was built or frozen.
- No model, LLM, Stockfish, or Popeye inference was run.

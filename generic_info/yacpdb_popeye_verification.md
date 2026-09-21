# YACPDB Popeye Verification

Popeye executable: `NOT_FOUND`
Installation/provenance: Popeye executable not available locally; see documentation for server installation/run commands.
Dataset fingerprint: `bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a`
Verification config fingerprint: `641526504333b4eb491e7558cd42f6f88f8d941e2eaf9a234a212e08920f1302`
Timeout seconds: `0.0`
Canonical dataset mutated: `false`

Popeye is used only as an independent chess-composition verifier before dataset freeze. It is not a model, baseline, target generator, or selection mechanism.

## Input Format

The adapter writes `BeginProblem`, `Option NoBoard`, `Stipulation #N`, and `Forsyth <board>` from the canonical FEN board field. It does not fabricate castling or en-passant state.

## Summary

- total: 200
- verified forced mate: 0
- failed: 0
- unverifiable: 0
- timeouts: 0
- Popeye errors: 0
- unique key matches: 0
- multiple keys including source: 0
- key mismatches: 0

## By MateDepth

| MateDepth | Total | Verified | Failed | Unverifiable | Timeout | Popeye error | Unique key | Multiple key | Key mismatch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 9 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Anomalies

No anomalies.

## Immutability

- No YACPDB resampling.
- No selected-ID changes.
- No target changes.
- No parser/normalizer semantic changes.
- No GNN/LLM/Ollama/Stockfish involvement.
- Dataset remains not frozen.

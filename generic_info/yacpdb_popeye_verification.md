# YACPDB Popeye Verification

Popeye executable: `NOT_FOUND`
Installation/provenance: Popeye executable not available locally; see documentation for server installation/run commands.
Dataset fingerprint: `bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a`
Verification config fingerprint: `d88e7001c3b01451701c957d05de2c7b41a0a4f6da75d9d522f0763684a1cead`
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

## Two-Stage Timeout Strategy

The official verification protocol is now two-stage:

1. Pass 1 verifies all 200 YACPDB candidate problems with `timeout_seconds = 300`.
2. Pass 2 verifies only Pass-1 rows whose structured result has `verification_reason = TIMEOUT`, using `timeout_seconds = 1200`.

The retry pass is explicit and provenance-preserving. It reads the first-pass
`results.jsonl`, selects only timeout rows, and writes to a separate directory
such as:

```text
data/heldout_classic/verification/yacpdb_classic_v1/popeye_retry_1200s/
```

Timeouts are retried instead of replaced because replacing computationally
difficult compositions with easier ones would introduce solver-runtime
selection bias into the held-out benchmark.

Important semantics:

- `TIMEOUT` is not evidence of an invalid composition.
- `TIMEOUT` is not evidence of a wrong YACPDB key.
- A second timeout after 1200 seconds remains computationally inconclusive.
- No timeout problem may be replaced or resampled automatically.

Manual retry command:

```bash
./venv/bin/python -m src.verification.popeye \
  --dataset-dir data/heldout_classic/final/yacpdb_classic_v1 \
  --verification-root data/heldout_classic/verification/yacpdb_classic_v1/popeye_retry_1200s \
  --popeye-executable ./tools/popeye/py \
  --timeout-seconds 1200 \
  --retry-timeouts-from data/heldout_classic/verification/yacpdb_classic_v1/popeye/results.jsonl
```

The retry summary records:

- `verification_pass = timeout_retry`
- `parent_verification_config_fingerprint`
- `parent_timeout_seconds`
- `parent_results_path`
- `retry_timeout_seconds`
- `retry_selection_reason = TIMEOUT`
- `retry_selected_count`

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

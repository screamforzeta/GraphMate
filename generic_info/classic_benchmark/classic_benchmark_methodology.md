# Classic Held-Out Benchmark Methodology

## Pipeline Overview

The external classic benchmark is built as a fixed, composition-based held-out
set for Mate-in-N next-move evaluation.

```text
YACPDB
  -> acquisition
  -> orthodox directmate filtering
  -> position normalization
  -> source root-key extraction
  -> bounded availability scan
  -> balanced fixed selection
  -> candidate dataset
  -> Popeye independent verification
  -> timeout retry
  -> verification consolidation
  -> accepted-key policy
  -> formal freeze
  -> future model evaluation
```

The benchmark is independent from Lichess train/validation/test data and was
fixed before graph-model or LLM evaluation on this held-out set.

## Provenance Table

| Stage | Input | Output count | Main criterion | Artifact / fingerprint |
|---|---|---:|---|---|
| YACPDB availability scan | YACPDB `Stip("^#N$")`, N=1..10 | 2000 inspected | bounded source scan | `a7694af29724e10907d3b51d759d1ef4f38492e35cfb410b40150400696a2880` |
| Clean eligible pool | inspected records | 1540 clean eligible | orthodox directmate, legal source key, no contamination | `data/heldout_classic/processed/yacpdb_availability.json` |
| Fixed candidate selection | clean eligible pool | 200 | 20 per MateDepth, seed 42 | `bb1b2d7c3858e3b2ffad58fd561534acf1e29bb2321b4af52a57256398ec9a5a` |
| Popeye Pass 1 | 200 candidates | 183 verified, 17 timeout | 300 s/problem | `7500621cd95b841a0a105ec37f02b821ea3ca0fd116a78f7d94c244c7d9f7bb1` |
| Popeye Pass 2 | 17 Pass-1 timeouts | 10 verified, 7 timeout | 1200 s/problem | `92628f0a0fb1bd69101ca13fedd691d2aede0c00f24183fc89f494a9cbd1661f` |
| Consolidated verification | both Popeye passes | 193 verified, 7 timeout | Pass 1 authoritative except timeout retry | `yacpdb_popeye_consolidation_v1` |
| Formal frozen benchmark | canonical + consolidated artifacts | 200 | immutable benchmark identity | `freeze_manifest.json` |

## Accepted-Key Policy

The canonical source dataset stores one `key_move_uci`, the YACPDB source key.
The frozen benchmark adds a derived scoring view:

```text
accepted_key_moves_uci
```

Policy:

- verified unique: the single Popeye root key is accepted;
- verified multiple: all Popeye-verified root keys are accepted;
- residual timeout: the YACPDB source key is accepted with explicit
  `YACPDB_SOURCE_UNVERIFIED_TIMEOUT` provenance.

Future model evaluation on this benchmark must score:

```text
prediction in accepted_key_moves_uci
```

This applies only to the external classic benchmark. Lichess evaluation
semantics remain unchanged.

## Freeze Identity

The formal freeze is defined by:

- canonical dataset fingerprint and file hashes;
- selected IDs;
- consolidated verification artifacts;
- accepted-key policy version;
- Popeye verification provenance;
- consolidation version;
- post-freeze immutability rules.

The freeze fingerprint excludes volatile fields such as timestamps.

## Post-Freeze Rules

After `yacpdb_classic_v1` is frozen, the following must not change:

- selected problem IDs;
- canonical FENs;
- MateDepth;
- source keys;
- accepted key sets;
- verification classifications;
- accepted-key policy version;
- benchmark-defining artifact bytes.

Any future correction must create a new benchmark version, for example
`yacpdb_classic_v2`. Model evaluation outputs are not part of the benchmark
identity and may be added later without changing the freeze fingerprint.

## Threats To Validity And Limitations

- The YACPDB endpoint used by the project is public but undocumented.
- Only orthodox directmates are included.
- The target is root-key prediction, not full solution-tree generation.
- Seven cases lack independent Popeye completion within the available compute
  budget.
- Multi-key compositions require set-valued ground truth.
- Balanced 20-per-depth sampling is intentional and not representative of
  natural occurrence frequency.
- Classical compositions differ distributionally from Lichess game-derived
  puzzles.
- The external benchmark size is 200; subgroup uncertainty grows at fine
  granularity.
- External background details about YACPDB and Popeye require bibliography
  references in the final thesis: `[EXTERNAL REFERENCE REQUIRED]`.


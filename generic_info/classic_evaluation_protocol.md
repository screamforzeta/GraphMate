# Classic Held-Out Evaluation Protocol

## Purpose

The external classic benchmark evaluates whether the frozen chess systems
generalize outside the Lichess puzzle distribution used for development. The
benchmark is based on YACPDB orthodox directmate compositions and is treated as
a fixed held-out test set.

The goal is not to train, tune, or select new models. The benchmark is used only
after the dataset and verification artifacts have been frozen.

YACPDB is distributionally different from Lichess: problems are composed chess
studies/directmates rather than game-derived puzzle rows. This makes the
benchmark an external generalization test, not a replacement for the original
Lichess test split. [EXTERNAL REFERENCE REQUIRED]

## Frozen Benchmark Identity

Benchmark:

```text
yacpdb_classic_v1
```

Canonical dataset:

```text
data/heldout_classic/final/yacpdb_classic_v1/
```

Freeze manifest:

```text
data/heldout_classic/final/yacpdb_classic_v1/freeze_manifest.json
```

Freeze fingerprint:

```text
57ff2b7c725a4ef802817581fac9430f8c6416a6de54dafcb1deb7b471716b60
```

Lifecycle:

```text
FROZEN
```

The benchmark contains 200 positions: 20 positions for each MateDepth #1 through
#10.

Popeye verification state:

| Category | Count |
|---|---:|
| Verified total | 193 |
| Verified unique key | 188 |
| Verified multiple keys including source key | 5 |
| Residual timeout/source-only | 7 |

Accepted-key policy:

```text
accepted_classic_keys_v1
```

The seven residual timeout cases retain their YACPDB source key as the accepted
target, but they remain distinguishable from independently Popeye-verified
positions.

## Why Evaluation Happens Only After Freeze

The benchmark was selected, normalized, independently checked, consolidated,
and frozen before any held-out model evaluation. This prevents model-output
driven dataset changes.

Post-freeze rules:

- selected problem IDs are immutable;
- canonical FENs are immutable;
- MateDepth labels are immutable;
- source keys are immutable;
- accepted-key sets are immutable;
- verification categories are immutable;
- future corrections require a new benchmark version, not silent replacement.

## Evaluated Systems

The planned external comparison includes:

| System | Role | Frozen status |
|---|---|---|
| Model A3 | No-timing legal-move scorer | Frozen Lichess baseline |
| Model B | Timing-aware legal-move scorer | Frozen timing ablation |
| Model A4 | Frozen A3 Top-5 retrieval plus post-move GAT reranker | Frozen strongest GNN |
| Qwen 3.5 4B | LLM baseline | Frozen prompt/parser protocol |
| Qwen 3.5 9B | LLM baseline | Frozen prompt/parser protocol |

No model is retrained for this benchmark.

## Common Sample Schema

Each classic sample exposes:

| Field | Meaning |
|---|---|
| `heldout_id` | Frozen benchmark ID |
| `source_problem_id` | YACPDB source problem ID |
| `fen` | Canonical solver-position FEN |
| `mate_depth` | Directmate depth #1..#10 |
| `source_key_move_uci` | Source/YACPDB key move |
| `accepted_key_moves_uci` | Set of accepted UCI key moves |
| `accepted_key_basis` | Popeye unique, Popeye multiple, or source-only timeout |
| `verification_status` | Final consolidated verification status |
| `forced_mate_verified` | Whether Popeye independently verified the mate |

Canonical rows and consolidated accepted-key rows are joined by `heldout_id`.
The loader fails closed if IDs or core fields disagree.

## Common Prediction Schema

Every evaluated system writes a common core:

| Field | Meaning |
|---|---|
| `benchmark_version` | `yacpdb_classic_v1` |
| `freeze_fingerprint` | Frozen benchmark fingerprint |
| `heldout_id` | Sample ID |
| `source_problem_id` | Source problem ID |
| `mate_depth` | MateDepth |
| `model_id` | System identity |
| `model_family` | GNN/LLM family |
| `model_version` | Checkpoint/protocol version |
| `predicted_move_uci` | Top-1 predicted move |
| `accepted_key_moves_uci` | Frozen accepted key set |
| `source_key_move_uci` | Source key |
| `is_correct` | Accepted-key membership result |
| `is_legal` | Legality status when applicable |
| `parse_success` | Parser status for LLMs |
| `verification_status` | Frozen verification status |
| `accepted_key_basis` | Frozen accepted-key provenance |
| `runtime_seconds` | Per-sample runtime if measured |
| `ranked_moves_uci` | Optional ranked list for GNNs |
| `diagnostics` | Model-specific metadata |

GNN and LLM diagnostics remain separate when their semantics differ.

## Scoring

Primary correctness is set-valued:

```text
prediction in accepted_key_moves_uci
```

The evaluator must not use:

```text
prediction == source_key_move_uci
```

This matters for multi-key compositions where Popeye verified more than one
valid key including the source key.

Primary metric:

```text
Top-1 accepted-key accuracy
```

For ranked GNN outputs, the framework also supports:

- Top-3 accepted-key accuracy;
- Top-5 accepted-key accuracy;
- rank of first accepted key;
- mean and median accepted-key rank.

For LLMs that return one move, Top-K metrics are not fabricated. LLM reports
retain parse failure, illegal move, wrong legal move, and runtime failure
diagnostics according to the frozen LLM benchmark semantics.

## Stratification

Every summary must support:

- ALL-200;
- POPEYE-VERIFIED-193 only;
- MateDepth #1 through #10;
- verification category:
  - `POPEYE_VERIFIED_UNIQUE`;
  - `POPEYE_VERIFIED_MULTIPLE`;
  - `YACPDB_SOURCE_UNVERIFIED_TIMEOUT`.

The unresolved timeout/source-only group is reported separately because these
seven positions are not independently Popeye-verified.

## Model A3 Protocol

A3 consumes the canonical YACPDB FEN directly. Classic positions are already
solver positions, so no Lichess `Moves[0]` transformation is applied.

A3 candidate generation:

```text
chess.Board(fen).legal_moves
```

Candidate scoring uses the frozen A3 legal-move scorer. The model does not
require the global 1786-class move vocabulary at inference. Promotion handling
uses the same compact promotion embedding as the Lichess A3 pipeline.

Compatibility result:

```text
A3_CLASSIC_COMPATIBILITY = READY
```

The current implementation provides the adapter contract and fixture tests. The
official 200-position inference run has not been executed in this task.

## Model A4 Protocol

A4 is evaluated end-to-end:

```text
canonical FEN
-> frozen A3 scores all legal moves
-> real A3 Top-5 retrieval
-> apply each retrieved candidate move
-> build post-move graph
-> A4 reranks retrieved candidates
```

Accepted keys are used only after prediction for scoring. They are never passed
to retrieval or reranking. A4 cannot recover positions where the accepted key is
not in the real A3 Top-5 candidate set.

Compatibility result:

```text
A4_CLASSIC_COMPATIBILITY = READY
```

The adapter records the frozen A3 and A4 checkpoint identities and states that
retrieval does not use target or accepted-key information.

## Model B Timing Compatibility

Model B is a timing-aware extension of A3. The implementation requires three
graph-level timing fields:

```text
previous_move_time
original_move_time
time_is_synthetic
```

The timing encoder is:

```text
Linear(3, 16) -> ELU -> Dropout
```

The timing embedding is concatenated to the candidate context before scoring.
The model intentionally has no silent no-timing fallback.

YACPDB classic compositions do not provide the Lichess-derived timing context
used to create Model B inputs. The protocol therefore does not invent timing
values from MateDepth, IDs, random sampling, Lichess test rows, or model
outputs.

Evaluated options:

| Option | Decision |
|---|---|
| Do not evaluate B when timing is unavailable | Scientifically safest default |
| Neutral/zero timing ablation | Not selected unless a separate frozen protocol is explicitly versioned |
| New synthetic timing distribution | Rejected for this protocol |

Selected protocol:

```text
MODEL_B_CLASSIC_COMPATIBILITY = NOT_APPLICABLE_TIMING_UNAVAILABLE
CLASSIC_TIMING_PROTOCOL = not_applicable_timing_unavailable
```

If a later study wants to evaluate the timing-aware architecture without
position-specific timing, it must define a separate versioned neutral-ablation
protocol and clearly report that it is not timed YACPDB performance.

## Qwen Protocol

The external classic LLM benchmark reuses the frozen LLM protocol.

Prompt version:

```text
llm_chess_uci_v1
```

System prompt:

```text
You are solving a chess position.
Return exactly one legal chess move for the side to move.
Do not provide explanation or analysis.
```

User prompt:

```text
FEN:
{fen}

Return your move in UCI notation only.
```

The prompt includes only the canonical FEN. It does not include MateDepth,
source keys, accepted keys, YACPDB metadata, Popeye status, legal moves, or
hints.

Strict parser:

```text
strict_uci_v1
```

The relaxed parser remains a diagnostic parser only if the frozen LLM reporting
protocol includes it. Prompt and parser semantics are unchanged for YACPDB.

Compatibility result:

```text
QWEN_CLASSIC_COMPATIBILITY = READY
```

The framework builds prompts and parses responses without contacting Ollama in
fixture tests.

## Official-Run Protection

The official evaluator must require an explicit official mode. Without
`--official`, it refuses to evaluate the frozen 200 positions.

An official run must:

1. verify the freeze manifest and fingerprint;
2. record the freeze fingerprint;
3. record code/git provenance when available;
4. record model/checkpoint identity;
5. create a model/run-specific output directory;
6. refuse to overwrite a completed official run;
7. support safe resume without duplicate predictions;
8. persist per-problem records;
9. regenerate summaries from persisted prediction records.

## Run Identity

The same official run is defined by:

- benchmark version;
- freeze fingerprint;
- model ID;
- checkpoint/model version;
- evaluation protocol version;
- accepted-key policy version;
- LLM prompt/parser version where applicable;
- timing protocol for Model B where applicable.

Performance metrics are not part of run identity.

## Executable Implementation

Entrypoint:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout
```

The CLI has two mutually exclusive modes:

- `--official`: evaluates the frozen 200-position benchmark after freeze
  verification;
- `--smoke-test`: evaluates a dedicated synthetic non-YACPDB fixture position.

Running without either flag is refused. Smoke mode is intended only for
pre-run tensor/device/protocol checks and must not use any heldout ID beginning
with `yacpdb_classic_v1_`.

### A3 Inference Path

```text
canonical YACPDB FEN
-> build_graph(fen, dummy vocabulary target used only for Data.y)
-> python-chess legal move enumeration
-> frozen A3 candidate scorer
-> ranked legal UCI moves
-> accepted-key scoring after prediction
```

The dummy vocabulary target is not the accepted key and is not used for
candidate scoring. It only satisfies the existing PyG graph schema inherited
from the Lichess training pipeline.

### A4 Inference Path

```text
canonical YACPDB FEN
-> original-position graph
-> frozen A3 scores all legal moves
-> genuine A3 Top-5
-> apply each Top-5 candidate
-> build post-move graph
-> frozen A4 reranker
-> ranked A4 result
-> accepted-key scoring after prediction
```

A4 diagnostics include:

- A3 retrieved Top-5;
- A3 retrieval scores;
- whether any accepted key was retrieved;
- A4 reranked order;
- whether the final error is a retrieval failure or a reranking failure.

Retrieval success is:

```text
intersection(A3_top5, accepted_key_moves_uci) is non-empty
```

If retrieval succeeds but A4 Top-1 is not accepted, the example is counted as a
reranking failure. If retrieval fails, it is not counted as a reranking failure.

### Qwen Inference Path

Qwen uses the frozen prompt and strict parser:

```text
canonical FEN
-> llm_chess_uci_v1 prompt
-> Ollama qwen3.5:4b or qwen3.5:9b
-> strict_uci_v1 parse
-> legal move check
-> set-valued accepted-key scoring
```

The primary result is strict-parser based. Relaxed parsing remains a separate
diagnostic protocol and is not substituted for strict scoring here.

### Model B Exclusion

Requesting Model B under this protocol fails before inference and creates no
prediction artifact. This is not a model failure. It records the methodological
decision that the frozen classic benchmark has no position-specific timing
fields compatible with Model B.

## Output Structure

Planned output tree:

```text
artifacts/classic_benchmark/
  yacpdb_classic_v1/
    <model_id>/
      <run_id>/
        config.json
        predictions.jsonl
        summary.json
        by_mate_depth.json
        by_verification_status.json
        COMPLETED
```

GNN configs record checkpoint SHA256 when the checkpoint file is available.
LLM configs record exact Ollama model ID and prompt/parser versions; immutable
Ollama model digests should be recorded when available without changing the
frozen protocol.

## Commands

Freeze verification:

```bash
./venv/bin/python -m src.verification.freeze_classic_benchmark \
  --dataset-dir data/heldout_classic/final/yacpdb_classic_v1 \
  --consolidated-dir data/heldout_classic/verification/yacpdb_classic_v1/consolidated \
  --verify
```

Safe A3 smoke test:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --smoke-test \
  --model MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING \
  --device cuda:0 \
  --amp
```

Safe A4 smoke test:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --smoke-test \
  --model MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING \
  --device cuda:0 \
  --amp
```

Official A3 run:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --official \
  --model MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING \
  --device cuda:0 \
  --amp
```

Official A4 run:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --official \
  --model MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING \
  --device cuda:0 \
  --amp
```

Official Qwen runs:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --official \
  --model qwen3.5:4b

./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --official \
  --model qwen3.5:9b
```

Resume:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout \
  --official \
  --resume \
  --model <same-model-id-and-config-as-original-run>
```

## External Classic Benchmark Evaluation Protocol

The final report should describe the external classic benchmark as follows:

The YACPDB classic benchmark was frozen before held-out evaluation. A3 and A4
are applied without retraining. A4 remains an end-to-end retrieval and reranking
test because the frozen A3 Top-5 set is generated from the position only. Qwen
is evaluated with the same frozen prompt/parser protocol used for the LLM
benchmark. Model B is not evaluated under this protocol because YACPDB does not
provide the timing fields required by the timing-aware architecture.

Correctness is set-valued for independently verified multi-key compositions.
Primary results are reported over all 200 benchmark positions and separately
over the 193 independently Popeye-verified positions. Results are stratified by
MateDepth #1 through #10 and by verification category.

## Limitations

- YACPDB classic compositions differ from Lichess game-derived puzzles.
- The benchmark contains 200 positions, balanced by MateDepth, not by natural
  frequency.
- Seven positions remain source-key accepted after Popeye timeout.
- Multi-key problems require set-valued scoring.
- Popeye is validation infrastructure, not a competitor.
- Model B timing cannot be interpreted as real timing on YACPDB because YACPDB
  has no empirical timing fields.
- Academic background on YACPDB, Popeye, and chess composition verification
  needs external citations. [EXTERNAL REFERENCE REQUIRED]

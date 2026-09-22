# Popeye Verification Methodology

## What Popeye Is

Popeye is used in this project as an independent chess-composition solver and
verifier. Its role is to verify YACPDB directmate claims and identify valid
root key moves for the fixed held-out benchmark.

Popeye is not:

- one of the machine-learning models being benchmarked;
- a training target generator;
- a component used to improve A3, A4, or Model B;
- a benchmark competitor.

External historical and project facts about Popeye require citations in the
final report: `[EXTERNAL REFERENCE REQUIRED]`.

## Why Popeye Instead Of Stockfish

Stockfish is a high-performance chess engine designed primarily for ordinary
game play, position evaluation, and engine search. It can find mates in many
positions, but this benchmark validation task is narrower and more specific:

```text
Given an orthodox directmate composition with stipulation #N,
verify whether the stipulated forced mate exists and identify valid root keys.
```

Popeye is designed for chess-composition solving and accepts problem
stipulations such as directmate `#N` directly. Its output is naturally aligned
with composition verification: it reports solution structures and root keys in
the problem-solving setting.

The methodological reason for using Popeye is therefore task suitability:

- Stockfish: general/game-oriented engine and search.
- Popeye: composition/stipulation-oriented solver.

This is not a claim that Stockfish is incapable of finding mates, nor that
Popeye is a stronger chess engine in general. A Stockfish-based verifier would
require a separately designed forced-mate validation procedure and careful
interpretation of engine output against exact directmate semantics. Popeye
already operates in that domain.

Using Popeye also avoids validating YACPDB merely by trusting YACPDB's own
solution text.

## Why Not Use A GNN Or LLM As Verifier

The benchmarked models must not validate their own ground truth. GNNs and LLMs
are the systems being evaluated downstream. Popeye is external to those learned
systems and is used only to audit the dataset.

## Adapter

The repository adapter is implemented in:

```text
src/verification/popeye.py
```

The conversion pipeline is:

```text
canonical FEN
  -> Popeye Forsyth board field
  -> BeginProblem
     Option NoBoard
     Stipulation #N
     Forsyth ...
     EndProblem
  -> Popeye stdout/stderr
  -> structural root-key parser
  -> verified_keys_uci
  -> comparison with YACPDB source key
```

Implementation details:

- Popeye version: `v4.103`;
- server executable path used during verification: `tools/popeye/py`;
- observed banner: `Popeye Linux-7.0.0-31-generic-unknown-64Bit v4.103 (1024 MB)`;
- knights are mapped to Popeye `S/s` notation;
- castling and en-passant state are not fabricated;
- version detection parses the real Popeye banner from supported execution;
- root-key parsing uses structural root `1.` solution lines, not arbitrary
  move-looking tokens.

The Popeye binary is external tooling. It is not part of the benchmark
identity unless explicitly installed and recorded by path/version; repository
policy should avoid committing opaque binaries unless deliberately approved.

## Multiple Keys

Popeye may find more than one root key for a composition under the normalized
canonical position. Such cases are not automatically invalid.

For this benchmark:

- if the YACPDB source key is among the Popeye-verified root keys;
- all independently verified root keys become accepted benchmark answers.

This prevents false-negative scoring: a model should not be marked wrong for
predicting another independently verified root key.

The consolidated benchmark contains five multi-key cases, listed in:

```text
data/heldout_classic/verification/yacpdb_classic_v1/consolidated/multi_key_cases.jsonl
```

## Timeout Policy

Timeout is different from verification failure.

Pass 1:

```text
200 problems
300 seconds/problem
183 verified
17 timeout
0 key mismatches
0 Popeye errors
```

Pass 2:

```text
17 Pass-1 timeout cases only
1200 seconds/problem
10 additional verified
7 timeout
0 key mismatches
0 Popeye errors
```

Combined:

```text
193 / 200 independently verified by Popeye = 96.5%
7 / 200 unresolved due to compute timeout = 3.5%
0 key mismatches
0 Popeye errors
```

A timeout does not mean:

- invalid composition;
- incorrect YACPDB solution;
- no forced mate.

It means Popeye did not finish independent verification within the allocated
compute budget.

## Why Timeout Problems Were Not Replaced

Replacing slow positions after observing Popeye runtime would bias the held-out
benchmark toward solver-easier compositions. The fixed candidate set was chosen
before held-out model evaluation and is preserved even when some cases remain
computationally inconclusive.

## Why The Retry Stopped At 1200 Seconds

The second pass increased the per-problem budget from 300 seconds to 1200
seconds and resolved 10 of 17 timeouts. Seven remained, concentrated in
MateDepth #9 and #10.

Stopping at this point is a compute-budget decision, not proof that the seven
remaining cases are correct. The project can report both:

- performance on all 200 benchmark problems;
- performance on the independently Popeye-verified 193-problem subset.

Those epistemic categories must not be silently merged.


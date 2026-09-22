# Frozen Model Family

## Repository Mapping

| Label | Repository meaning | Main files/artifacts |
|---|---|---|
| A | Raw no-timing fixed-vocabulary `ChessGATNoTiming` baseline | `src/models/model_a/chess_gat.py`, `src/training/model_a/convergence.py`, `checkpoints/model_a/best.pt` |
| A1 | Historical best-legal inference mode over Model A logits, not a separate architecture/checkpoint | `src/inference/model_a/no_timing_multimodel.py::best_legal_summary` |
| A2 | Legal-masked no-timing fixed-vocabulary model | `src/training/model_a/model_a2_legal_mask.py`, `src/cli/training/train_model_a2_legal_mask.py` |
| A3 | Official no-timing legal-candidate scorer | `src/models/model_a/chess_legal_scorer.py`, `src/training/model_a/model_a3_legal_scorer.py` |
| A4 | Frozen A3 Top-5 retrieval plus post-move GAT reranker | `src/models/model_a/chess_postmove_reranker.py`, `src/training/model_a/model_a4_postmove_reranker.py` |
| B | Timing-aware controlled variant of A3 | `src/models/model_b/chess_timing_legal_scorer.py`, `src/training/model_b/model_b_timing_legal_scorer.py` |

The A1 label is a naming caveat: repository evidence shows it as an inference/evaluation mode for Model A, not as a distinct trainable model.

## Lineage

```text
A raw global move classifier
  -> A1 best-legal diagnostic post-filter
  -> A2 legal-masked fixed-vocabulary model
  -> A3 legal-candidate scorer
  -> A4 post-move Top-5 reranker

A3
  -> B timing-aware legal-candidate ablation
```

## Definitive Results

### A3 Official Test

| Metric | Value |
|---|---:|
| N | 8610 |
| CE/NLL | 1.0832485489175157 |
| Top1 | 0.675609756097561 |
| Top3 | 0.8565621370499419 |
| Top5 | 0.9185830429732869 |
| Top10 | 0.9706155632984901 |
| Mean legal target rank | 2.224274099883856 |
| Median legal target rank | 1.0 |
| Illegal Top1 | 0.0 |

### A4 Frozen Validation and Terminal Test

Checkpoint: `checkpoints/model_a4/best.pt`.

Frozen validation:

- best epoch: `32`
- validation end-to-end Top1: `0.8543892243381328`
- validation loss: `0.19448498769802403`

Terminal test:

| Metric | Value |
|---|---:|
| Shared test N | 8610 |
| A3 parity | PASS |
| A4 end-to-end Top1 | 0.854123112659698 |
| Delta vs A3 | +17.851335656213696 pp |
| Rerankable | 7909 |
| Unrerankable | 701 |
| A3 Recall@5 ceiling | 0.9185830429732869 |
| A4 conditional Top1 | 0.9298267796181565 |
| A4 errors | 1256 |
| A4 conditional CE/NLL | 0.19621933876661724 |

Paired transitions:

| Transition | Count |
|---|---:|
| both correct | 5702 |
| A3 correct / A4 wrong | 115 |
| A3 wrong / A4 correct | 1652 |
| both wrong | 1141 |

McNemar: `n01=1652`, `n10=115`, continuity-corrected chi-square approximately `1335.1986`, exact p-value should be reported as `p < 0.001` if underflow occurs.

### A4 MateDepth Terminal Breakdown

| MateDepth | N | A3 Top1 | A4 Top1 | Delta pp |
|---|---:|---:|---:|---:|
| MateIn1 | 1995 | 0.7954887218045112 | 0.9393483709273183 | +14.385964912280702 |
| MateIn2 | 1999 | 0.6978489244622311 | 0.8819409704852427 | +18.409204602301152 |
| MateIn3 | 1999 | 0.6673336668334167 | 0.8459229614807404 | +17.858929464732366 |
| MateIn4 | 1999 | 0.5872936468234117 | 0.7838919459729865 | +19.65982991495748 |
| MateIn5 | 618 | 0.529126213592233 | 0.7427184466019418 | +21.35922330097087 |

### A4 Rating Terminal Breakdown

| Rating | N | A3 Top1 | A4 Top1 | Delta pp |
|---|---:|---:|---:|---:|
| <1200 | 3879 | 0.8205723124516628 | 0.9554008765145656 | +13.482856406290281 |
| 1200-1599 | 2176 | 0.6585477941176471 | 0.8690257352941176 | +21.047794117647058 |
| 1600-1999 | 1564 | 0.5134271099744245 | 0.7583120204603581 | +24.48849104859335 |
| 2000-2399 | 811 | 0.4315659679408138 | 0.6041923551171393 | +17.26263871763255 |
| 2400+ | 180 | 0.26666666666666666 | 0.45 | +18.333333333333332 |

The `2400+` bucket is small and should not be overinterpreted.

### A4 Error Decomposition

- total errors: `1256`
- retrieval failures: `701` (`55.8121%` of A4 errors)
- reranking failures: `555` (`44.1879%` of A4 errors)
- gap to retrieval ceiling: `6.445993031358888` pp

A4 cannot recover targets outside frozen A3 Top-5. More than half of remaining A4 errors are imposed by upstream retrieval.

### Model B Timing Ablation

Model B extends A3 with graph-level timing inputs:

- `previous_move_time`
- `original_move_time`
- `time_is_synthetic`

Frozen timing conclusion:

- A3 Top1: `67.5609756097561%`
- B synthetic Top1: `65.98141695702672%`
- B neutral Top1: approximately `66.0975609756%`
- B - A3: approximately `-1.5796` pp
- synthetic - neutral: approximately `-0.1161` pp
- A3 vs B McNemar: approximately `p = 1.22e-6`
- B synthetic vs neutral: approximately `p = 0.253`

Valid conclusion: the implemented synthetic timing representation did not improve move prediction relative to A3. This does not imply that timing cannot help chess models in general.


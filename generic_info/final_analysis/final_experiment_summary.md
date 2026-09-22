# Final Experiment Summary

This report is generated from existing official artifacts only. No training, official inference, LLM inference, Popeye verification, dataset rebuild, or checkpoint mutation is performed.

## Experiment Inventory
Included experiments: 12. Excluded/diagnostic artifacts: 4.

Excluded artifacts include the invalid first Qwen 4B classic HTTP-404 attempt, smoke tests, and GPT-OSS protocol-failure evidence outside the primary comparison.

## Dataset and Evaluation Protocols
Lichess is the only benchmark used for the literal timed-GNN-vs-LLM comparison because Model B timing inputs are available there. YACPDB classic is an external generalization benchmark for A3/A4 and LLMs; Model B is N/A - TIMING_UNAVAILABLE.

## Lichess Primary Results
| model | N | Top1_percent | Top3_percent | Top5_percent | mean_rank | protocol_notes |
| --- | --- | --- | --- | --- | --- | --- |
| A | 8610 | 39.6167 | 55.8072 | 62.7526 | 4.0115 | Frozen no-timing baseline |
| A1 | 8610 | 49.8142 | 70.4994 | 79.5587 | N/A | Inference-only; no separate checkpoint |
| A2 | 8610 | 49.2567 | 71.4983 | 81.1731 | 3.7220 | Legal class masking |
| A3 | 8610 | 67.5610 | 85.6562 | 91.8583 | 2.2243 | Matched no-timing baseline for RQ2 |
| B | 8610 | 65.9814 | 84.8432 | 91.3705 | 2.3029 | Synthetic timing; not applicable to classic benchmark |
| A4 | 8610 | 85.4123 | N/A | N/A | N/A | Top1 only in terminal summary |
| Qwen 3.5 4B strict | 8610 | 0.0697 | N/A | N/A | N/A | Strict UCI, one generated move |
| Qwen 3.5 9B strict | 8610 | 0.1161 | N/A | N/A | N/A | Strict UCI, one generated move |

## Timing Ablation / RQ2
| model | timing_mode | N | Top1_percent | Top3_percent | Top5_percent | delta_Top1_vs_A3_pp |
| --- | --- | --- | --- | --- | --- | --- |
| A3 | no timing | 8610 | 67.5610 | 85.6562 | 91.8583 | 0.0000 |
| B_SYNTHETIC_TIMING | synthetic timing | 8610 | 65.9814 | 84.8432 | 91.3705 | -1.5796 |
| B_NEUTRAL_TIMING | neutralized timing | 8610 | 66.0976 | 84.8432 | 91.3705 | -1.4634 |
| B_ZERO_TIMING_AND_FLAG | zero timing | 8610 | 66.0976 | 84.8548 | 91.3705 | -1.4634 |

RQ2 bounded conclusion: The implemented synthetic timing signal did not improve predictive performance relative to the matched no-timing A3 baseline.

## LLM Comparison / RQ1
| mate_depth | N | B_Top1_percent | Qwen4_Top1_percent | Qwen9_Top1_percent | B_minus_Qwen4_pp | B_minus_Qwen9_pp |
| --- | --- | --- | --- | --- | --- | --- |
| MateIn1 | 1995 | 78.1454 | 0.1003 | 0.1003 | 78.0451 | 78.0451 |
| MateIn2 | 1999 | 68.3342 | 0.0500 | 0.0500 | 68.2841 | 68.2841 |
| MateIn3 | 1999 | 65.4827 | 0.1501 | 0.1001 | 65.3327 | 65.3827 |
| MateIn4 | 1999 | 56.7284 | 0.0000 | 0.2001 | 56.7284 | 56.5283 |
| MateIn5 | 618 | 50.6472 | 0.0000 | 0.1618 | 50.6472 | 50.4854 |

RQ1 is computed on Lichess only. A3 and A4 are contextual no-timing graph baselines in the associated depth table and figures.

## External Classic Benchmark
| model | ALL200_Top1_percent | Top3_percent | Top5_percent | correct_count | Popeye193_Top1_percent | notes |
| --- | --- | --- | --- | --- | --- | --- |
| A3 | 11.0000 | 23.5000 | 34.5000 | 22 | 11.3990 | N/A |
| A4 | 15.5000 | 29.5000 | 34.5000 | 31 | 15.0259 | N/A |
| Qwen 3.5 4B strict | 0.0000 | N/A | N/A | 0 | 0.0000 | N/A |
| Qwen 3.5 9B strict | 0.5000 | N/A | N/A | 1 | 0.5181 | N/A |
| B | N/A | N/A | N/A | N/A | N/A | N/A - TIMING_UNAVAILABLE |

Classic mate-depth buckets contain only N=20 problems each, so interval estimates are wide and individual buckets should not be overinterpreted.

## LLM Failure Analysis
| benchmark | model | outcome | count | percent |
| --- | --- | --- | --- | --- |
| Lichess | Qwen 3.5 4B strict | correct | 6 | 0.0697 |
| Lichess | Qwen 3.5 4B strict | wrong_legal | 130 | 1.5099 |
| Lichess | Qwen 3.5 4B strict | illegal | 5894 | 68.4553 |
| Lichess | Qwen 3.5 4B strict | parse_failure | 2580 | 29.9652 |
| Lichess | Qwen 3.5 4B strict | runtime_failure | 0 | 0.0000 |
| Lichess | Qwen 3.5 9B strict | correct | 10 | 0.1161 |
| Lichess | Qwen 3.5 9B strict | wrong_legal | 413 | 4.7967 |
| Lichess | Qwen 3.5 9B strict | illegal | 2899 | 33.6702 |
| Lichess | Qwen 3.5 9B strict | parse_failure | 5288 | 61.4170 |
| Lichess | Qwen 3.5 9B strict | runtime_failure | 0 | 0.0000 |
| YACPDB classic | Qwen 3.5 4B strict | correct | 0 | 0.0000 |
| YACPDB classic | Qwen 3.5 4B strict | wrong_legal | 3 | 1.5000 |
| YACPDB classic | Qwen 3.5 4B strict | illegal | 101 | 50.5000 |
| YACPDB classic | Qwen 3.5 4B strict | parse_failure | 96 | 48.0000 |
| YACPDB classic | Qwen 3.5 4B strict | runtime_failure | 0 | 0.0000 |
| YACPDB classic | Qwen 3.5 9B strict | correct | 1 | 0.5000 |
| YACPDB classic | Qwen 3.5 9B strict | wrong_legal | 6 | 3.0000 |
| YACPDB classic | Qwen 3.5 9B strict | illegal | 64 | 32.0000 |
| YACPDB classic | Qwen 3.5 9B strict | parse_failure | 129 | 64.5000 |
| YACPDB classic | Qwen 3.5 9B strict | runtime_failure | 0 | 0.0000 |

The LLM analysis separates chess correctness, legality, strict parse compliance, and runtime failure.

## External Generalization
| model | lichess_Top1_percent | classic_Top1_percent | classic_minus_lichess_pp |
| --- | --- | --- | --- |
| A3 | 67.5610 | 11.0000 | -56.5610 |
| A4 | 85.4123 | 15.5000 | -69.9123 |
| Qwen 3.5 4B strict | 0.0697 | 0.0000 | -0.0697 |
| Qwen 3.5 9B strict | 0.1161 | 0.5000 | 0.3839 |

Performance under the external composition distribution was substantially lower for the graph models; this is an out-of-distribution/external-generalization result, not an IID claim that the benchmark is simply harder.

## Statistical Analysis
Lichess A3 vs B paired test: {'status': 'COMPUTED', 'method': 'McNemar exact two-sided binomial', 'n01_a_wrong_b_correct': 321, 'n10_a_correct_b_wrong': 457, 'discordant': 778, 'p_value': 1.2238376595768834e-06}

Classic A3 vs A4 paired test: {'status': 'COMPUTED', 'method': 'McNemar exact two-sided binomial', 'n01_a_wrong_b_correct': 16, 'n10_a_correct_b_wrong': 7, 'discordant': 23, 'p_value': 0.0931396484375}

A3 vs B Top1 effect size: -1.579559 percentage points.

Wilson 95% confidence intervals are stored in `data/statistical_tests.json` and included in primary/depth CSV outputs.

## Limitations
Timing is synthetic and rating-conditioned. The classic benchmark has no timing fields and cannot answer the timed-model comparison. GPT-OSS is retained as protocol-failure evidence, not as a primary benchmark row. Relaxed LLM parser results remain diagnostic and are not substituted for strict results.

## Direct Answers
RQ1: On Lichess MateIn1 through MateIn5, Model B Top1 is far above both strict Qwen baselines for every depth with available artifacts.

RQ2: The implemented synthetic timing signal did not improve predictive performance relative to A3; Top1 changed by a negative amount in the controlled comparison.

## Missing Analyses / Unavailable Data
Model B on YACPDB classic is unavailable because timing inputs are unavailable. No unsupported YACPDB timing values are generated.

## Figures
- `figures/lichess_topk_comparison.svg`
- `figures/lichess_topk_comparison.png`
- `figures/lichess_mate_depth_top1.svg`
- `figures/lichess_mate_depth_top1.png`
- `figures/lichess_rating_top1.svg`
- `figures/lichess_rating_top1.png`
- `figures/timing_ablation.svg`
- `figures/timing_ablation.png`
- `figures/classic_top1_comparison.svg`
- `figures/classic_top1_comparison.png`
- `figures/classic_mate_depth_top1.svg`
- `figures/classic_mate_depth_top1.png`
- `figures/llm_failure_composition_lichess.svg`
- `figures/llm_failure_composition_lichess.png`
- `figures/llm_failure_composition_classic.svg`
- `figures/llm_failure_composition_classic.png`
- `figures/generalization_lichess_vs_classic.svg`
- `figures/generalization_lichess_vs_classic.png`
- `figures/a4_classic_retrieval_reranking.svg`
- `figures/a4_classic_retrieval_reranking.png`

## Artifact Provenance
See `source_manifest.json` and `data/experiment_inventory.json`.

## Validation
- Lichess A3/A4 N = 8610
- Classic frozen dataset and verification totals validated
- Classic A3/A4 TopK counts validated
- Classic Qwen valid-attempt category sums validated
- A4 classic retrieval ceiling validated

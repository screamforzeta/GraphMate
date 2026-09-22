# Model A4 Terminal Evaluation

This report compares frozen A3 and frozen A4 on the shared terminal test population.

## Test Lock

- TEST_USED_FOR_TRAINING = NO
- TEST_USED_FOR_CHECKPOINT_SELECTION = NO
- TARGET_INJECTED_IN_TOP5 = NO

## A3 Parity

- A3_TEST_PARITY = `PASS`
- A3 Top1 = `0.675609756097561`
- A3 Top5 = `0.9185830429732869`

## A4 Terminal Test

- A4_END_TO_END_N = `8610`
- A3_RECALL_AT_5 = `0.9185830429732869`
- A4_CONDITIONAL_TOP1 = `0.9298267796181565`
- A4_END_TO_END_TOP1 = `0.854123112659698`
- DELTA_TOP1_PP = `17.851335656213696`

## Paired Comparison

```json
{
  "both_wrong": 1141,
  "both_correct": 5702,
  "a3_wrong_a4_correct": 1652,
  "a3_correct_a4_wrong": 115,
  "a3_correct_preserved_rate": 0.9802303592917311,
  "a3_errors_recovered_rate_rerankable_definition": 0.7896749521988528,
  "a3_errors_recovered_rate_all_a3_wrong": 0.5914786967418546,
  "net_correct_gain": 1537,
  "delta_top1_pp": 17.851335656213696
}
```

## McNemar

```json
{
  "n01_a3_wrong_a4_correct": 1652,
  "n10_a3_correct_a4_wrong": 115,
  "continuity_corrected_chi_square": 1335.1986417657047,
  "exact_two_sided_binomial_p": 0.0
}
```

## Retrieval vs Reranking Errors

```json
{
  "retrieval_failures": 701,
  "reranking_failures": 555,
  "retrieval_failure_fraction_of_a4_errors": 0.5581210191082803,
  "reranking_failure_fraction_of_a4_errors": 0.44187898089171973
}
```

# Final Experiment Analysis Layer

This analysis layer consolidates already completed official Progetto-Damiani experiments into reproducible tables, statistics, figures, and a generated summary report.

It is analysis-only. It reads existing artifacts and does not train models, run official inference, mutate checkpoints, rebuild datasets, edit frozen accepted-key files, or modify `TimeGNN-main/`.

## Source Selection

The CLI uses explicit artifact paths and official run metadata rather than selecting the latest directory. It includes official valid Lichess graph, LLM, and YACPDB classic runs, and records excluded artifacts in `source_manifest.json`.

Invalid infrastructure attempts, smoke tests, incomplete/unofficial runs, and GPT-OSS primary-comparison exclusions are not included in primary accuracy tables.

## Primary vs Diagnostic Results

Strict UCI LLM results are primary. Relaxed parser results remain diagnostic and are not substituted into primary comparisons.

Model A1 is documented as an inference-only best-legal filtering mode over Model A, not as an independent architecture or checkpoint.

## Statistical Methods

Top1 accuracy intervals use Wilson 95% confidence intervals. Paired binary comparisons use an exact two-sided McNemar/binomial test when aligned per-sample prediction records are available.

The implemented tests are intentionally focused on the main hypotheses: A3 vs Model B for timing, and A3 vs A4 where aligned prediction records support comparison.

## RQ1 Mapping

RQ1 asks for which mate depths the timed GNN provides better move guidance than the selected LLM baselines. The literal timed-GNN-vs-LLM comparison is computed on Lichess only, because Model B timing inputs exist there.

YACPDB classic is not used for this timed comparison.

## RQ2 Mapping

RQ2 uses the controlled A3 vs Model B comparison. A3 is the matched no-timing legal-move candidate scorer; Model B is the timing-aware variant using synthetic timing features.

The generated conclusion is bounded to this experiment: the implemented synthetic timing signal did not improve predictive performance relative to the matched no-timing A3 baseline.

## Classic Benchmark Role

YACPDB classic is an external generalization benchmark for A3/A4 and LLMs. It has 200 frozen compositions, 20 per mate depth from #1 through #10, with accepted-key scoring.

Model B classic results are `N/A - TIMING_UNAVAILABLE`; the pipeline does not fabricate timing values.

## Invalid-Run Handling

The first Qwen 3.5 4B classic attempt is preserved as provenance but excluded from accuracy because it failed with `HTTP_404_RUNTIME_FAILURE`. The second attempt with the same scientific run identity is the valid official result.

## Limitations

Classic mate-depth buckets are small (`N=20`), so per-depth intervals are wide. YACPDB differs from Lichess in distribution and construction; the analysis describes lower external-distribution performance without calling the benchmark objectively harder.

LLM results separate strict output-format compliance, legality, runtime failures, and correctness.

## Reproduction

Run:

```bash
./venv/bin/python -m src.cli.analysis.build_final_experiment_report
```

Generated outputs are written under `generic_info/final_analysis/`, including machine-readable data in `generic_info/final_analysis/data/`, figures in `generic_info/final_analysis/figures/`, `source_manifest.json`, and `final_experiment_summary.md`.


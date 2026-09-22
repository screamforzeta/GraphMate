# Final Analysis

This directory is the version-controlled home for the consolidated final scientific analysis.

The analysis pipeline reads official raw experiment artifacts from ignored `artifacts/` and writes canonical report material here:

- `final_experiment_analysis.md`: methodology and reproduction notes.
- `final_experiment_summary.md`: generated scientific summary.
- `source_manifest.json`: included and excluded source-artifact inventory.
- `data/`: generated CSV/JSON tables and statistical outputs.
- `figures/`: generated SVG and PNG figures by default.

Regenerate with:

```bash
./venv/bin/python -m src.cli.analysis.build_final_experiment_report
```

PDF figures are omitted by default. Use `--include-pdf` only when PDF exports are explicitly needed.


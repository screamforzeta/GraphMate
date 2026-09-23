# Final Analysis

[English](README.md) | [Italiano](README.it.md)

This directory is the version-controlled home for GraphMate's consolidated final scientific analysis.

The analysis pipeline reads official raw experiment artifacts from ignored `artifacts/` paths and writes canonical report material here:

- [final_experiment_analysis.md](final_experiment_analysis.md): methodology, source selection, statistics, exclusions, and reproduction notes.
- [final_experiment_summary.md](final_experiment_summary.md): generated scientific summary.
- [source_manifest.json](source_manifest.json): included and excluded source-artifact inventory.
- `data/`: generated CSV/JSON tables and statistical outputs.
- `figures/`: generated SVG and PNG figures by default.

Regeneration entry point:

```bash
./venv/bin/python -m src.cli.analysis.build_final_experiment_report --help
```

Run the command without `--help` only when the expected local official artifacts are available. PDF figures are omitted by default; use `--include-pdf` only when PDF exports are explicitly needed.

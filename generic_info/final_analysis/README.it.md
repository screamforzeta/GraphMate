# Analisi Finale

[English](README.md) | [Italiano](README.it.md)

Questa directory è la sede versionata dell'analisi scientifica finale consolidata di GraphMate.

La pipeline di analisi legge gli artifact sperimentali ufficiali da percorsi `artifacts/` ignorati da Git e scrive qui il materiale canonico:

- [final_experiment_analysis.md](final_experiment_analysis.md): metodologia, selezione sorgenti, statistiche, esclusioni e note di riproduzione.
- [final_experiment_summary.md](final_experiment_summary.md): summary scientifico generato.
- [source_manifest.json](source_manifest.json): inventario degli artifact inclusi ed esclusi.
- `data/`: tabelle CSV/JSON e output statistici generati.
- `figures/`: figure SVG e PNG generate di default.

Entry point di rigenerazione:

```bash
./venv/bin/python -m src.cli.analysis.build_final_experiment_report --help
```

Esegui il comando senza `--help` solo quando gli artifact ufficiali locali attesi sono disponibili. Le figure PDF sono omesse di default; usa `--include-pdf` solo quando servono esplicitamente export PDF.

# Progetto Damiani - Chess GNN

Pipeline Python per trasformare puzzle e partite Lichess in grafi PyTorch Geometric e addestrare modelli Graph Neural Network / Graph Attention Network su puzzle mate-in-n.

Il baseline scientifico no-timing è `ChessGATNoTiming`: un GAT graph-level che predice la prima mossa corretta della soluzione di un puzzle. Model A è ora completato, convergente secondo il protocollo definito e congelato come riferimento no-timing per confronti futuri.

## Stato Attuale

Implementato:

- download puzzle Lichess `.csv.zst`;
- download streaming PGN Lichess con campionamento senza scaricare tutto l'archivio mensile;
- preprocessing puzzle mate-in-1 fino a mate-in-5;
- parsing e cleaning partite;
- cleaning puzzle;
- split train/validation/test;
- vocabulary mosse train-only;
- feature nodi e archi;
- graph builder PyG;
- dataset PyG sharded completo;
- validator rappresentazione;
- modello `ChessGATNoTiming`;
- trainer standard, adaptive storico, benchmark runtime e progressive training;
- diagnosi/fix shard-aware per subset progressivi Pilot/Confirmation, validato sul server;
- `MODEL_A_CONVERGENCE_RUN_V1` completata e congelata come baseline no-timing;
- variante `MODEL_A2_LEGAL_MASK_NO_TIMING` implementata e pronta per training come ablation no-timing legal-masked;
- debugger Streamlit standalone per grafo/scacchiera.

Non implementato: Model B timing-aware, legal move masking, nuove feature temporali, valutazione MateDepth dedicata e confronto LLM.

Baseline full completato: `MODEL_A_FULL_BASELINE_V1`.

- dataset: train `68.958`, validation `8.612`, test `8.610`;
- modello: `ChessGATNoTiming`;
- config selezionata: lr `5e-4`, weight decay `1e-4`, dropout `0.30`, batch `128`;
- final test: Top1 circa `31,87%`, Top3 circa `47,72%`, Top5 circa `54,29%`.

Il run a 60 epoch ha esaurito il budget con il miglior risultato validation all'ultima epoch. La successiva `MODEL_A_CONVERGENCE_RUN_V1`, inclusa continuation controllata, ha raggiunto `CONVERGED_BY_EARLY_STOPPING`.

Baseline ufficiale congelato: `MODEL_A_NO_TIMING_FROZEN_BASELINE`.

- best epoch: `162`;
- stop epoch: `174`;
- stop reason: `EARLY_STOPPING`;
- final test: Top1 circa `39,62%`, Top3 circa `55,81%`, Top5 circa `62,75%`.

## Documentazione

- [Architettura del progetto](generic_info/project_architecture.md): come sono organizzati moduli, pipeline, dati, training e artifact.
- [Scelte architetturali](generic_info/architectural_choices.md): perché sono state prese le principali decisioni progettuali.
- [Model A no-timing](generic_info/model_a_no_timing.md): report completo del baseline congelato `ChessGATNoTiming`.
- [Streamlit Model A verification](generic_info/streamlit_model_a_verification.md): sezione UI per training umano sui puzzle, inference read-only e diagnostica Mate-in-1.
- [Model A2 legal mask no-timing](generic_info/model_a2_legal_mask_no_timing.md): piano/protocollo della variante legal-masked senza timing.
- [Model A vs A2 evaluator](generic_info/model_a_vs_a2_evaluation.md): valutazione post-hoc raw/best-legal/masked tra i due baseline.
- [Analisi TimeGNN](generic_info/timegnn_info.md): audit della libreria esterna `TimeGNN-main/`.
- [Guida training TimeGNN/GNN](generic_info/timegnn_gnn_training_guide.md): note di integrazione future.
- [Piano architettura Chess GAT](generic_info/chess_gat_architecture_plan.md): piano tecnico del modello chess-specific.

## Comandi Principali

Pipeline completa:

```bash
python3 main.py
```

Generazione dataset PyG sharded:

```bash
./venv/bin/python -m src.graph.pyg_dataset --graphs-per-shard 1000 --overwrite
```

Validazione rappresentazioni:

```bash
./venv/bin/python -m src.validate_representations --csv-sample 1000 --graph-sample 500 --seed 42
```

Training standard:

```bash
./venv/bin/python -m src.train_chess_gat
```

Training progressivo full consigliato:

```bash
./venv/bin/python -m src.train_chess_gat_progressive \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp \
  --max-runtime-hours 10 \
  --seed 42
```

Resume progressivo:

```bash
./venv/bin/python -m src.train_chess_gat_progressive --resume --device cuda
```

Historical convergence run Model A:

```bash
./venv/bin/python -m src.train_chess_gat_convergence \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp \
  --max-epochs 150 \
  --early-stopping-patience 12 \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 3 \
  --min-learning-rate 1e-6 \
  --seed 42
```

Historical resume convergence run:

```bash
./venv/bin/python -m src.train_chess_gat_convergence --resume --device cuda
```

Historical continuation convergence run:

```bash
./venv/bin/python -m src.train_chess_gat_convergence \
  --resume \
  --extend-max-epochs 300 \
  --device cuda
```

Training Model A2 legal-masked no-timing:

```bash
./venv/bin/python -m src.train_chess_gat_legal_mask \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp
```

Resume Model A2:

```bash
./venv/bin/python -m src.train_chess_gat_legal_mask --resume --device cuda
```

Post-hoc Model A vs A2 evaluation:

```bash
./venv/bin/python -m src.evaluate_model_a_vs_a2 \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp
```

Benchmark runtime:

```bash
./venv/bin/python -m src.benchmark_chess_gat_training \
  --batch-sizes 32,64,128,256 \
  --num-workers 0,2,4 \
  --warmup-batches 10 \
  --benchmark-batches 100 \
  --device cuda
```

Diagnostica access pattern subset progressivi, da eseguire sul server prima della prossima run lunga:

```bash
./venv/bin/python -m src.benchmark_progressive_subset_loading \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory
```

Debugger Streamlit:

```bash
streamlit run src/graph/debug/streamlit_graph_debugger.py
```

Il debugger Streamlit è standalone e non deve essere importato in `main.py`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Se `torch` o `torch-geometric` richiedono wheel specifiche CUDA, seguire le istruzioni ufficiali per la piattaforma usata.

## Dati e Artifact

I dati generati sono sotto `data/`. I dataset PyG sono sharded:

```text
data/pyg/
  manifest.json
  train/shard_00000.pt ...
  val/shard_00000.pt ...
  test/shard_00000.pt ...
```

Conteggi attuali:

- train: 68.958 grafi;
- validation: 8.612 grafi;
- test: 8.610 grafi.

`artifacts/move_to_idx.json` e `artifacts/idx_to_move.json`, se versionati, fissano la vocabulary delle mosse. La vocabulary è costruita solo sul train set per evitare leakage; target OOV di validation/test vengono esclusi dalla rappresentazione PyG e conteggiati.

## Test

```bash
./venv/bin/python -m pytest -q
./venv/bin/python -m compileall -q main.py src tests
```

Stato dell'audit corrente: `89 passed, 2 skipped`; `compileall` passa.

## Note Repository

`TimeGNN-main/` è codice esterno/vendor/reference. Non viene modificato dalla pipeline del progetto.

`data/`, `artifacts/`, cache Python, `.pytest_cache`, build temporanee e file `.pt.tmp` sono ignorati in `.gitignore` perché generati o potenzialmente pesanti.

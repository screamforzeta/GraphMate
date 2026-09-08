# Progetto Damiani - Chess GNN

Pipeline Python per trasformare puzzle e partite Lichess in grafi PyTorch Geometric e addestrare modelli Graph Neural Network / Graph Attention Network su puzzle mate-in-n.

L'obiettivo scientifico attuale è `ChessGATNoTiming`: un GAT graph-level che predice la prima mossa corretta della soluzione di un puzzle. In seguito verrà progettata una variante timing-aware, con confronto timing/no timing, valutazione per MateDepth e confronto con modelli LLM.

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
- pipeline dedicata `MODEL_A_CONVERGENCE_RUN_V1` pronta;
- debugger Streamlit standalone per grafo/scacchiera.

Non implementato: run di convergenza finale, Model B timing-aware, legal move masking, nuove feature temporali, valutazione MateDepth dedicata e confronto LLM.

Baseline full completato: `MODEL_A_FULL_BASELINE_V1`.

- dataset: train `68.958`, validation `8.612`, test `8.610`;
- modello: `ChessGATNoTiming`;
- config selezionata: lr `5e-4`, weight decay `1e-4`, dropout `0.30`, batch `128`;
- final test: Top1 circa `31,87%`, Top3 circa `47,72%`, Top5 circa `54,29%`.

Il run ha esaurito il budget configurato di `60` epoch con il miglior risultato di validation all'ultima epoch. La pipeline `MODEL_A_CONVERGENCE_RUN_V1` e pronta per una run piu lunga da zero, con stessa architettura/config, scheduler ed early stopping. Se una convergence run terminale finisce ancora per `MAX_EPOCHS_REACHED` con best all'ultima epoch, puo essere estesa esplicitamente con `--resume --extend-max-epochs`.

## Documentazione

- [Architettura del progetto](generic_info/project_architecture.md): come sono organizzati moduli, pipeline, dati, training e artifact.
- [Scelte architetturali](generic_info/architectural_choices.md): perché sono state prese le principali decisioni progettuali.
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

Convergence run Model A:

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

Resume convergence run:

```bash
./venv/bin/python -m src.train_chess_gat_convergence --resume --device cuda
```

Continuation convergence run:

```bash
./venv/bin/python -m src.train_chess_gat_convergence \
  --resume \
  --extend-max-epochs 300 \
  --device cuda
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

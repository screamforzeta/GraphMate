# Project Architecture

Questo documento descrive come è organizzato il progetto e come scorre la pipeline end-to-end. Le motivazioni progettuali sono raccolte in [architectural_choices.md](architectural_choices.md).

## Obiettivo

`Progetto-Damiani` rappresenta posizioni di puzzle Lichess come grafi PyTorch Geometric e addestra un modello GAT graph-level per predire la prima mossa corretta della soluzione.

Stato implementato:

- pipeline dati Lichess;
- correzione semantica puzzle;
- move vocabulary train-only;
- graph representation PyG;
- dataset PyG sharded;
- `ChessGATNoTiming`;
- training standard, adaptive storico, benchmark runtime e progressive training;
- validator e test suite.

Stato pianificato:

- Model B timing-aware;
- semantica timing robusta;
- legal move masking;
- valutazione per MateDepth;
- confronto con LLM.

## Directory Tree

```text
Progetto-Damiani/
├── main.py
├── README.md
├── requirements.txt
├── src/
│   ├── download/
│   │   ├── download_puzzles.py
│   │   └── download_games.py
│   ├── preprocess/
│   │   ├── preprocess_puzzles.py
│   │   ├── parse_games.py
│   │   ├── clean_games.py
│   │   ├── clean_puzzles.py
│   │   ├── prepare_puzzles_dataset.py
│   │   └── prepare_games_dataset.py
│   ├── graph/
│   │   ├── move_encoder.py
│   │   ├── node_features.py
│   │   ├── edge_features.py
│   │   ├── graph_builder.py
│   │   ├── pyg_dataset.py
│   │   └── debug/streamlit_graph_debugger.py
│   ├── models/
│   │   └── chess_gat.py
│   ├── training/
│   │   ├── chess_gat_trainer.py
│   │   ├── adaptive_controller.py
│   │   ├── progressive_controller.py
│   │   ├── convergence.py
│   │   └── metrics.py
│   ├── train_chess_gat.py
│   ├── train_chess_gat_auto.py
│   ├── train_chess_gat_progressive.py
│   ├── benchmark_chess_gat_training.py
│   ├── validate_representations.py
│   ├── smoke_test_chess_gat.py
│   └── reset_data.py
├── tests/
├── generic_info/
│   ├── project_architecture.md
│   ├── architectural_choices.md
│   ├── timegnn_info.md
│   ├── timegnn_gnn_training_guide.md
│   └── chess_gat_architecture_plan.md
├── data/
├── artifacts/
└── TimeGNN-main/
```

`TimeGNN-main/` è vendor/reference esterno. Non viene modificato dal codice del progetto.

## Pipeline End-to-End

```text
Lichess raw data
↓
download
↓
parsing / cleaning
↓
semantic correction puzzle
↓
train / validation / test
↓
move vocabulary train-only
↓
graph construction
↓
PyG sharding
↓
representation validator
↓
ChessGATNoTiming
↓
training / progressive training
↓
best checkpoint
↓
final test
```

Entry point principale:

```bash
python3 main.py
```

## Data Layer

`src/download/download_puzzles.py` scarica il database puzzle Lichess `.csv.zst`.

`src/download/download_games.py` scarica PGN Lichess in streaming e campiona senza dover salvare l'intero archivio mensile.

`src/preprocess/preprocess_puzzles.py` filtra e prepara puzzle mate-in-n.

`src/preprocess/parse_games.py`, `clean_games.py` e `prepare_games_dataset.py` gestiscono le partite PGN.

`src/preprocess/clean_puzzles.py` e `prepare_puzzles_dataset.py` producono i CSV finali puzzle.

## Lichess Puzzle Semantics

Lichess fornisce una FEN prima della mossa setup dell'avversario.

Nel progetto:

- `OriginalFEN` conserva la FEN originale Lichess;
- `Moves[0]` è la setup move;
- applicando `Moves[0]` si ottiene la posizione mostrata al solver;
- `FEN` è la posizione dopo `Moves[0]`;
- `Moves[1]` è la prima mossa corretta del solver;
- `TargetMove = Moves[1]`.

La pipeline valida che la setup move sia legale e che `TargetMove` sia legale nella posizione trasformata.

## Move Vocabulary

`src/graph/move_encoder.py` costruisce:

- `artifacts/move_to_idx.json`;
- `artifacts/idx_to_move.json`;
- `artifacts/move_encoder_stats.json`.

La vocabulary è costruita solo dal train set per evitare leakage validation/test.

Stato corrente:

- `num_classes = 1786`;
- validation OOV: `8 / 8620`, circa `0.09%`;
- test OOV: `10 / 8620`, circa `0.12%`.

I target OOV di validation/test non vengono aggiunti alla vocabulary e vengono esclusi dalla rappresentazione PyG.

## Graph Representation

Ogni posizione usa 64 nodi, uno per casella.

Singolo grafo:

```text
x               [64, 15]
edge_index      [2, E]
edge_attr       [E, 5]
y               scalar graph-level class
global_features [1, 4]
```

Batch PyG:

```text
x               [64B, 15]
edge_index      [2, total_E]
edge_attr       [total_E, 5]
y               [B]
global_features [B, 4]
```

### Node Features

Feature per nodo:

1. pawn
2. knight
3. bishop
4. rook
5. queen
6. king
7. color
8. occupied
9. row normalized
10. col normalized
11. attacked_by_white
12. attacked_by_black
13. legal_mobility
14. is_pinned
15. piece_value

Non esiste una feature nodo `is_check`.

### Edge Features

`edge_index` è sparse COO PyG, non una matrice densa 64x64.

Ogni coppia `(src, dst)` compare al massimo una volta. Le relazioni sono aggregate in `edge_attr` multilabel binario:

1. legal_move
2. attack
3. defend
4. pin
5. check_line

Esempio: `[1, 1, 0, 0, 0]` indica legal move e attack sulla stessa edge.

`check_line` rappresenta la relazione checker -> king. Non enumera tutte le caselle intermedie lungo il raggio.

### Global Features

`src/graph/graph_builder.py` produce `global_features` con shape `[1,4]`:

1. side_to_move;
2. is_check;
3. fullmove_number normalized;
4. halfmove_clock normalized.

La shape `[1,4]` è intenzionale: con batching PyG diventa `[B,4]`.

## PyG Dataset Sharding

`src/graph/pyg_dataset.py` genera il dataset sharded:

```text
data/pyg/
  manifest.json
  train/shard_00000.pt ...
  val/shard_00000.pt ...
  test/shard_00000.pt ...
```

Stato corrente:

- train: `68,958` grafi, `69` shard;
- validation: `8,612` grafi, `9` shard;
- test: `8,610` grafi, `9` shard;
- default shard size: circa `1000` grafi.

`ShardedPyGDataset` carica lazy gli shard, mantiene una LRU cache per processo e supporta uno shard-aware sampler. Ogni grafo conserva `source_row_index` come metadata di tracciabilità, non come feature del modello.

La generazione usa una directory temporanea `data/pyg_building/` e sostituisce `data/pyg/` solo a build completa. Ogni shard è scritto con file `.tmp` e rename finale.

### Data Access Samplers

FULL TRAINING DATA ACCESS:

- il dataset completo viene percorso con sampler shard-aware in train;
- gli indici vengono raggruppati per shard;
- a ogni epoch cambiano in modo deterministico sia l'ordine degli shard sia l'ordine degli esempi dentro lo shard;
- la membership e il contenuto del dataset non cambiano.

PILOT/CONFIRMATION TRAIN DATA ACCESS:

- i subset restano definiti da una membership random deterministica seed-controlled;
- Pilot usa il prefisso del subset Confirmation, quindi `Pilot subset of Confirmation`;
- il sampler converte gli indici visibili del `Subset` negli shard sorgente e legge gli stessi esempi in ordine shard-local;
- `set_epoch(epoch)` cambia l'ordine di lettura tra epoch senza usare randomness globale non controllata.

PILOT/CONFIRMATION VALIDATION DATA ACCESS:

- la membership validation resta deterministica e separata dal train;
- non c'e shuffle per la validation;
- l'ordine e shard-local e stabile tra epoch, cosi la valutazione resta riproducibile e cache-friendly.

Il benchmark diagnostico dedicato e:

```bash
./venv/bin/python -m src.benchmark_progressive_subset_loading \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory
```

Su VM si puo usare `--pattern-only` per verificare solo transizioni/cache simulate; la validazione prestazionale finale resta server-side.

## Model Layer

`src/models/chess_gat.py` contiene `ChessGATNoTiming`, il modello attuale.

Forma logica:

```text
x [64B,15]
edge_index [2,E]
edge_attr [E,5]
↓
GATConv(15 -> 32, heads=4, edge_dim=5)
↓
ELU + Dropout
↓
GATConv(128 -> 32, heads=4, edge_dim=5)
↓
ELU
↓
global_mean_pool -> [B,128]
↓
concat global_features -> [B,132]
↓
Linear(132,128) + ELU + Dropout
↓
Linear(128,1786)
↓
logits [B,1786]
```

## Training Layer

`src/training/chess_gat_trainer.py` implementa il trainer standard:

- `CrossEntropyLoss`;
- `Adam`;
- validation loss per early stopping;
- Top1/Top3/Top5;
- best checkpoint;
- test solo dopo reload del best checkpoint;
- seed e device handling;
- AMP opzionale;
- nessun scheduler nel trainer base.

Default standard:

- `batch_size=32`;
- `learning_rate=1e-3`;
- `max_epochs=30`;
- `patience=5`;
- `weight_decay=0`;
- `seed=42`.

`src/training/adaptive_controller.py` è il controller adaptive storico trial-based.

`src/training/progressive_controller.py` è il controller multi-fidelity corrente:

```text
Pilot -> Confirmation -> Full -> Final Test
```

## Benchmark Layer

`src/benchmark_chess_gat_training.py` misura throughput senza fare convergence training. Usa solo train e salva report in `artifacts/benchmarks/`.

Benchmark RTX A2000 noto:

- batch 32, workers 0, AMP false: circa `3531 graphs/s`;
- batch 128, workers 0, AMP true: circa `4758 graphs/s`;
- batch 256, workers 0, AMP true: circa `4904 graphs/s`.

Runtime progressivo scelto:

- `batch_size=128`;
- `num_workers=0`;
- `pin_memory=True`;
- `persistent_workers=False`;
- `prefetch_factor=None`;
- `non_blocking=True`;
- `amp=True`;
- `sampler=shard_aware`.

Baseline full reale su RTX A2000 documentato come `MODEL_A_FULL_BASELINE_V1`:

- dataset: train `68,958`, validation `8,612`, test `8,610`;
- config: lr `5e-4`, weight decay `1e-4`, dropout `0.30`, batch `128`;
- Full: `60` epoch, best epoch `60`, stop reason `MAX_EPOCHS_REACHED`;
- best validation: loss `3.621614`, Top1 `31.86%`, Top3 `46.96%`, Top5 `53.69%`;
- final test: loss `3.590105`, Top1 `31.87%`, Top3 `47.72%`, Top5 `54.29%`.

Il run ha raggiunto il budget epoch configurato mentre il miglior risultato validation era all'ultima epoch. Questo motiva una futura `MODEL_A_CONVERGENCE_RUN` con stessa architettura e stessa config selezionata, budget maggiore, scheduler ed early stopping fino a plateau empirico.

## Progressive Training

Default:

- Pilot: `train=12000`, `val=2000`, `trials=4`, `epochs=10`, `patience=3`;
- Confirmation: `train=30000`, `val=4000`, `top_k=2`, `epochs=15`, `patience=4`;
- Full: all train/val, `epochs=60`, `patience=8`;
- Final test: all test, una sola volta.

Il Full stage usa `ReduceLROnPlateau` su validation loss con:

- `factor=0.5`;
- `patience=3`;
- `min_lr=1e-6`.

## Validation And Tests

`src/validate_representations.py` verifica:

- semantica Lichess;
- target e legalità;
- vocabulary e OOV;
- shape PyG;
- edge multilabel;
- batching;
- coerenza CSV -> PyG tramite `source_row_index`.

Verdict corrente: `REPRESENTATION_VALID`.

Test suite:

```bash
./venv/bin/python -m pytest -q
```

Stato audit: `89 passed, 2 skipped`.

## Utilities

`src/reset_data.py` elimina solo dati/artifact rigenerabili allowlisted:

- `data/raw`;
- `data/processed`;
- `data/final`;
- `data/pyg`;
- move vocabulary artifacts.

Non deve toccare source, documentazione o `TimeGNN-main/`.

`src/smoke_test_chess_gat.py` verifica compatibilità graph -> model.

`src/graph/debug/streamlit_graph_debugger.py` è una app standalone. Non va importata in `main.py`.

## Artifact Policy

`data/` e `artifacts/` possono diventare grandi e sono normalmente ignorati. La vocabulary può essere versionata deliberatamente se serve fissare le classi tra generazione, training e valutazione.

Cache, `.pyc`, `.pytest_cache`, `__MACOSX`, build temporanee e `.pt.tmp` non fanno parte del source tree concettuale.

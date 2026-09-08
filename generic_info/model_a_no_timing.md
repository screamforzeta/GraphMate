# Model A — ChessGATNoTiming

## 1. Executive Summary

`ChessGATNoTiming` è il baseline no-timing congelato del progetto. Il modello prende una posizione di puzzle Lichess già corretta secondo la semantica del solver, la rappresenta come grafo PyTorch Geometric a 64 nodi e predice la prima mossa UCI della soluzione.

Risultato ufficiale congelato:

```text
MODEL_A_NO_TIMING_FROZEN_BASELINE
status = FROZEN
convergence_status = CONVERGED_BY_EARLY_STOPPING
best_epoch = 162
stop_epoch = 174
test_loss = 3.078363185864846
test_top1 = 39.6167%
test_top3 = 55.8072%
test_top5 = 62.7526%
```

La convergenza qui significa convergenza empirica sotto il protocollo definito: la validation loss non migliora per tutta la patience di early stopping dopo più riduzioni del learning rate. Non implica massimo globale dell'architettura o del task.

## 2. Scientific Objective

Model A risolve un task supervised graph-level: data una posizione puzzle, predire la mossa soluzione target.

Il suo ruolo scientifico è essere il riferimento senza timing contro cui confrontare esperimenti futuri. Non usa timing, event IDs, legal-move masking o Model B.

## 3. Dataset

Split CSV finali:

| Split | CSV rows |
|---|---:|
| Train | 68,958 |
| Validation | 8,620 |
| Test | 8,620 |
| Total | 86,198 |

Split PyG utilizzabili:

| Split | PyG graphs | Note |
|---|---:|---|
| Train | 68,958 | nessun OOV rispetto alla vocabulary train |
| Validation | 8,612 | 8 target OOV esclusi |
| Test | 8,610 | 10 target OOV esclusi |

La vocabulary è costruita solo sul train set. Questo produce `1,786` classi. Gli OOV validation/test sono gestiti esplicitamente, non persi silenziosamente.

Fonti locali verificate: `data/pyg/manifest.json`, `artifacts/move_encoder_stats.json`.

## 4. Lichess Puzzle Semantics

La FEN raw di Lichess rappresenta la posizione prima della setup move dell'avversario.

Semantica corretta:

1. leggere `OriginalFEN`;
2. applicare `Moves[0]`;
3. la posizione risultante è quella vista dal solver;
4. la soluzione inizia da `Moves[1]`;
5. `TargetMove = Moves[1]`.

Una versione precedente della pipeline usava erroneamente raw FEN + `Moves[0]` come target. Questo errore fondazionale è stato corretto prima degli esperimenti finali di Model A. Senza questa correzione il modello avrebbe imparato la setup move dell'avversario, non la risposta del solver.

## 5. Graph Representation

Ogni posizione è un grafo con 64 nodi, uno per casella.

Node features: `x [64,15]`.

| Index | Feature |
|---:|---|
| 0-5 | piece one-hot: pawn, knight, bishop, rook, queen, king |
| 6 | color |
| 7 | occupied |
| 8 | normalized row |
| 9 | normalized col |
| 10 | attacked_by_white |
| 11 | attacked_by_black |
| 12 | legal_mobility |
| 13 | is_pinned |
| 14 | piece_value |

Non esiste una node feature `is_check`.

Edge representation:

```text
edge_index [2,E] sparse COO
edge_attr  [E,5]
```

Ogni `(src,dst)` compare al massimo una volta. Le relazioni tattiche sono aggregate in un vettore multilabel:

1. `legal_move`;
2. `attack`;
3. `defend`;
4. `pin`;
5. `check_line`.

Una versione precedente produceva parallel edges separati per relazione. La rappresentazione finale aggrega invece le relazioni nello stesso edge multilabel.

Dettagli importanti:

- pin detection geometrica corretta;
- legal mobility calcolata per entrambi i colori;
- `check_line = checker -> king`;
- `global_features [1,4]`;
- target graph-level `y`.

Global features:

| Index | Feature |
|---:|---|
| 0 | side_to_move |
| 1 | is_check |
| 2 | fullmove_number normalized |
| 3 | halfmove_clock normalized |

## 6. Move Target and Vocabulary

Il target è una classificazione graph-level:

```text
y = move_to_idx[TargetMove]
TargetMove = prima mossa UCI della soluzione puzzle
```

Artifact:

- `artifacts/move_to_idx.json`;
- `artifacts/idx_to_move.json`;
- `artifacts/move_encoder_stats.json`.

La vocabulary train-only evita leakage da validation/test. Limite metodologico: le `1,786` classi coprono le mosse osservate nel train, non tutte le mosse UCI teoricamente possibili.

Alternative future possibili, non parte di Model A congelato:

- theoretical UCI vocabulary;
- from-square / to-square heads;
- promotion head;
- legal move masking.

## 7. Model Architecture

Fonte: `src/models/chess_gat.py`.

Architettura:

```text
x [64B,15]
edge_index [2,E]
edge_attr [E,5]

GATConv(in=15, out=32, heads=4, edge_dim=5)
-> [64B,128]
-> ELU
-> Dropout

GATConv(in=128, out=32, heads=4, edge_dim=5)
-> [64B,128]
-> ELU

global_mean_pool
-> graph embedding [B,128]

concat global_features [B,4]
-> [B,132]

Linear(132,128)
-> ELU
-> Dropout
-> Linear(128,1786)
```

Parameter count verificato localmente con `num_classes=1786`: `268,026` trainable parameters.

Dropout finale della run congelata: `0.30`.

Architettura e hyperparameter sono concetti separati: la struttura GAT resta fissa, mentre lr/weight decay/batch/scheduler appartengono al protocollo di training.

## 8. Training Protocol

Protocollo finale:

| Component | Value |
|---|---|
| Optimizer | Adam |
| Loss | CrossEntropyLoss |
| Initial LR | 5e-4 |
| Weight decay | 1e-4 |
| Dropout | 0.30 |
| Batch size | 128 |
| Seed | 42 |
| Metrics | Top1, Top3, Top5 |
| Checkpoint selection | minimum validation loss |

Runtime server:

- CUDA;
- AMP true;
- `num_workers=0`;
- `pin_memory=true`;
- `non_blocking=true`;
- `ShardAwareSampler`.

Scheduler:

```text
ReduceLROnPlateau(mode=min, factor=0.5, patience=3, min_lr=1e-6)
```

Early stopping:

```text
patience = 12
min_delta = 0.0
monitor = validation loss
```

## 9. Experimental History

La storia Model A ha seguito questa sequenza:

1. correzione semantica Lichess;
2. costruzione rappresentazione PyG;
3. esperimento iniziale circa 5k;
4. passaggio al dataset full sharded;
5. progressive training;
6. fix performance subset;
7. `MODEL_A_FULL_BASELINE_V1` a 60 epoch;
8. `MODEL_A_CONVERGENCE_RUN_V1` da zero a 150 epoch;
9. continuation controllata 151-174;
10. freeze del baseline no-timing.

## 10. Initial 5k Experiment

Esperimento ridotto:

| Field | Value |
|---|---:|
| Train | circa 5,000 |
| Validation | 4,997 |
| Test | 4,994 |
| Best trial epoch | 11 |

Best adaptive configuration:

- lr `5e-4`;
- dropout `0.30`;
- weight decay `1e-4`;
- batch size `32`.

Validation:

- loss `6.245944`;
- Top1 `4.42%`;
- Top5 `13.09%`.

Test:

- loss `6.184599`;
- Top1 `5.2663%`;
- Top3 `10.1922%`;
- Top5 `13.7765%`.

Questo esperimento non rappresenta il risultato finale. Serviva a verificare pipeline, ottenere una configurazione plausibile e mostrare i limiti del training su subset ridotto.

Fonte locale parziale: `artifacts/adaptive_training/chess_gat_no_timing/final_report.json`.

## 11. Full Dataset Transition

Il passaggio full ha introdotto dataset PyG sharded:

| Split | Graphs | Shards | OOV skipped |
|---|---:|---:|---:|
| Train | 68,958 | 69 | 0 |
| Validation | 8,612 | 9 | 8 |
| Test | 8,610 | 9 | 10 |

Circa `1000` grafi per shard.

Componenti:

- `ShardedPyGDataset`;
- manifest;
- lazy loading;
- LRU cache;
- shard-aware sampling;
- `source_row_index`;
- atomic build tramite directory temporanea e rename finale.

## 12. Progressive Training Experiment

Pilot:

| Trial | lr | wd | dropout | batch | best epoch | val loss | Top1 | Top3 | Top5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | .0005 | .0001 | .3 | 128 | 10 | 5.96671484 | 5.25% | 10.55% | 14.75% |
| 2 | .00025 | .0001 | .3 | 128 | N/A | 6.3032656 | 4.40% | N/A | 11.85% |
| 3 | .0005 | .001 | .3 | 128 | N/A | 5.9803047 | 4.70% | N/A | 13.75% |
| 4 | .0005 | .0001 | .4 | 128 | N/A | 5.9820249 | 5.15% | N/A | 14.15% |

Promossi:

- lr `.0005`, wd `.0001`, dropout `.3`;
- lr `.0005`, wd `.001`, dropout `.3`.

Confirmation:

| Config | Train | Val | Best epoch | Val loss | Top1 | Top3 | Top5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr .0005 wd .0001 drop .3 | 30,000 | 4,000 | 15 | 5.3856451 | 9.125% | 18.425% | 23.15% |
| lr .0005 wd .001 drop .3 | 30,000 | 4,000 | 15 | 5.7197434 | 6.375% | 12.80% | 17.925% |

La configurazione finale congelata deriva da questo stage.

## 13. Subset Performance Problem and Fix

Problema osservato: Pilot e Confirmation erano molto più lenti del Full.

Runtime indicativi pre-fix:

- Pilot: circa `1997 s` per trial;
- Confirmation: circa `5686-5726 s` per run;
- Full: circa `17.4 s/epoch`.

Root cause: la membership casuale dei subset era corretta, ma il traversal naive saltava continuamente fra shard. Con LRU cache piccola questo causava reload continui. Non era un problema del modello o della GPU.

Pattern diagnostic:

| Case | Old shard transitions | Old estimated shard loads |
|---|---:|---:|
| Pilot 12k | 11,829 | 11,642 |
| Confirmation 30k | 29,573 | 29,141 |
| Full 68,958 | 68 | 69 |

Fix: `ShardAwareSampler`.

Proprietà:

- membership invariata;
- deterministic shuffle;
- traversal shard-local;
- `set_epoch(epoch)`;
- Pilot ancora sottoinsieme di Confirmation;
- nessun cambio scientifico ai sample.

Server validation data-loading only:

| Case | Elapsed | Graphs/s | Transitions | Shard loads | Cache hit |
|---|---:|---:|---:|---:|---:|
| Pilot 12k | 8.7386 s | 1373.21 | 68 | 69 | 99.425% |
| Confirmation 30k | 9.2540 s | 3241.83 | 68 | 69 | 99.770% |
| Full 68,958 | 10.6042 s | 6502.88 | 68 | 69 | 99.900% |

Il benchmark finale distingue `BEFORE = ANALYTICAL_PATTERN_SIMULATION` da `AFTER = REAL_SERVER_TRAVERSAL`.

## 14. MODEL_A_FULL_BASELINE_V1

Primo vero baseline full dataset:

| Field | Value |
|---|---|
| Config | lr .0005, wd .0001, dropout .3, batch 128 |
| Epochs | 60 |
| Best epoch | 60 |
| Stop reason | MAX_EPOCHS_REACHED |

Validation:

- loss `3.6216142139`;
- Top1 `31.8625%`;
- Top3 `46.9577%`;
- Top5 `53.6925%`.

Final test:

- loss `3.59010546`;
- Top1 `31.8699%`;
- Top3 `47.7236%`;
- Top5 `54.2857%`.

Rispetto al 5k:

- Test Top1: `5.2663% -> 31.8699%`;
- Test Top5: `13.7765% -> 54.2857%`.

Questo non è una ablation isolata del solo dataset size: sono cambiati anche batch size, runtime config e training orchestration.

## 15. Why Baseline V1 Was Not Considered Converged

Il baseline 60 epoch non fu considerato convergente perché:

- `max_epochs = 60`;
- `epochs_completed = 60`;
- `best_epoch = 60`;
- validation loss ancora in discesa nelle ultime epoch;
- stop per `MAX_EPOCHS_REACHED`;
- nessun early stopping.

`best_epoch == max_epochs` con validation ancora in miglioramento indica che il budget stava troncando il training. La decisione di fare una convergence run non fu basata sul test.

## 16. MODEL_A_CONVERGENCE_RUN_V1

La convergence run è una nuova run da zero, non continuation del baseline 60.

Motivo: ottenere una traiettoria riproducibile completa con protocollo di convergenza definito fin dall'inizio.

Differenze principali:

- max epochs `150`;
- `ReduceLROnPlateau`;
- early stopping patience `12`;
- best checkpoint by validation loss;
- test isolato fino al terminal state.

## 17. Epoch 150 Terminal State

Risultato terminale a epoch 150:

| Field | Value |
|---|---|
| status | COMPLETED |
| stop_reason | MAX_EPOCHS_REACHED |
| convergence_status | MAX_EPOCH_BUDGET_EXHAUSTED |
| epochs | 150 |
| best_epoch | 150 |
| best_val_loss | 3.123774577671021 |
| final LR | 1.25e-4 |
| LR reductions | 2 |
| early_stopping_counter | 0 |

Test #1 della convergence trajectory:

- loss `3.098223075523332`;
- Top1 `39.291521%`;
- Top3 `55.389082%`;
- Top5 `62.195122%`;
- examples `8610`.

## 18. Why the Run Was Continued

Epoch 150 non dimostrava ancora convergenza:

- `best_epoch = 150/150`;
- best validation loss all'ultima epoch;
- `early_stopping_counter = 0`;
- scheduler sopra `min_lr`;
- stop ancora `MAX_EPOCHS_REACHED`.

La continuation fu decisa solo su stato training/validation. Le metriche test epoch 150 non furono usate per scheduler, checkpoint, config o stopping.

## 19. Continuation 151–174

Continuation controllata della stessa trajectory:

```text
last.pt epoch 150 -> epoch 151
max_epochs 150 -> 300
```

Preservati:

- model;
- optimizer;
- scheduler;
- AMP scaler;
- LR corrente;
- early stopping;
- history;
- best checkpoint;
- config.

Il test precedente è stato archiviato come osservazione intermedia.

Eventi principali:

| Epoch | Event |
|---:|---|
| 160 | LR `1.25e-4 -> 6.25e-5` |
| 161 | new best, val loss `3.111613` |
| 162 | new final best, val loss `3.1078798007920794` |
| 166 | LR `6.25e-5 -> 3.125e-5` |
| 170 | LR `3.125e-5 -> 1.5625e-5` |
| 174 | LR `1.5625e-5 -> 7.8125e-6`; early stopping counter `12` |

Finale:

- stop epoch `174`;
- stop reason `EARLY_STOPPING`;
- best resta epoch `162`.

Questi valori vengono dal risultato server fornito nel prompt; gli artifact finali non sono presenti nella VM.

## 20. Evidence of Convergence

Model A ha raggiunto convergenza empirica secondo il protocollo definito.

Evidenze:

1. il budget massimo non è più la causa dello stop;
2. best validation loss a epoch `162`;
3. training continua per altre `12` epoch;
4. nessuna epoch `163-174` migliora il best;
5. early stopping patience `12` esaurita;
6. `ReduceLROnPlateau` interviene ripetutamente;
7. LR progressivamente ridotto;
8. ulteriori riduzioni non producono un nuovo best;
9. stop reason finale `EARLY_STOPPING`;
10. best checkpoint separato dal terminal epoch.

Questa è evidenza di plateau empirico per questa architettura, questi dati, questi hyperparameter, questo seed e questo stopping protocol.

## 21. Final Model A Results

`MODEL_A_NO_TIMING_FROZEN_BASELINE`:

| Field | Value |
|---|---|
| Best checkpoint | epoch 162 |
| Best validation loss | 3.1078798007920794 |
| Stop epoch | 174 |
| Stop reason | EARLY_STOPPING |
| Final LR | 7.8125e-6 |
| LR reductions | 6 |
| Convergence status | CONVERGED_BY_EARLY_STOPPING |

Final test after best reload:

| Metric | Value |
|---|---:|
| loss | 3.078363185864846 |
| Top1 | 39.6167% |
| Top3 | 55.8072% |
| Top5 | 62.7526% |
| examples | 8,610 |

## 22. Comparison Across Experiments

| Experiment | Train size | Best/stop epoch | Val loss | Val Top1 | Val Top5 | Test loss | Test Top1 | Test Top3 | Test Top5 | Status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Initial ~5k experiment | ~5,000 | best 11 | 6.245944 | 4.42% | 13.09% | 6.184599 | 5.2663% | 10.1922% | 13.7765% | preliminary |
| MODEL_A_FULL_BASELINE_V1 | 68,958 | best/stop 60 | 3.621614 | 31.8625% | 53.6925% | 3.590105 | 31.8699% | 47.7236% | 54.2857% | budget-limited |
| Convergence @ epoch 150 | 68,958 | best/stop 150 | 3.123775 | N/A | N/A | 3.098223 | 39.2915% | 55.3891% | 62.1951% | budget-limited |
| Final convergence | 68,958 | best 162 / stop 174 | 3.107880 | N/A | N/A | 3.078363 | 39.6167% | 55.8072% | 62.7526% | frozen |

N/A indica metriche non disponibili negli artifact locali o non fornite nel prompt.

Delta `MODEL_A_FULL_BASELINE_V1 -> final`:

- loss: `-0.5117422741`;
- Top1: `+7.7468 pp`;
- Top3: `+8.0836 pp`;
- Top5: `+8.4669 pp`.

Delta `epoch 150 -> final`:

- loss: `-0.0198598897`;
- Top1: `+0.3252 pp`;
- Top3: `+0.4181 pp`;
- Top5: `+0.5575 pp`.

La continuation ha dato miglioramenti marginali ma reali; il punto scientifico principale era verificare plateau/early stopping validation-driven.

## 23. Generalization

Baseline V1 mostrava validation e test molto vicini. La convergence finale mantiene test performance coerente con il trend validation osservato.

Non si può affermare assenza assoluta di overfitting, ma dai risultati disponibili non emerge un evidente generalization collapse.

## 24. Test-Set Methodology

Il test set è stato osservato più volte nella storia:

1. baseline V1;
2. convergence terminal state epoch 150;
3. final convergence dopo continuation.

Nella convergence trajectory `test_evaluation_count = 2`: epoch 150 e finale sono due evaluation della stessa run continuata.

La continuation 150->300 fu decisa usando solo stato validation/training: best epoch al massimo budget, best validation all'ultima epoch, early stopping counter zero, scheduler sopra min LR.

Il test non fu usato per scheduler, checkpoint, stopping o config. Tuttavia il test non può essere descritto come completamente untouched. Questa limitazione deve restare esplicita in ogni report scientifico futuro.

## 25. Performance / Runtime

Training runtime server durante continuation: train epoch circa `15.3-16.2 s`, validation circa `1.6-1.9 s`.

Questi tempi sono full training epoch runtime, diversi dai benchmark data-loading only.

Benchmark training iniziale su RTX A2000 6GB:

| Config | Graph/s | Estimated full epoch |
|---|---:|---:|
| batch 32, no AMP | ~3530.69 | ~19.53 s |
| batch 64, no AMP | ~4131 | ~16.69 s |
| batch 128, AMP | ~4757.85 | ~14.49 s |
| batch 256, AMP | ~4904.12 | ~14.06 s |

Batch 128 è stato mantenuto come compromesso operativo. Il sistema non è documentato come VRAM-bound; il modello usa poca VRAM rispetto ai 6GB disponibili.

## 26. Known Limitations

- vocabulary train-derived fissa;
- OOV validation/test;
- classificazione su 1786 mosse;
- nessun legal-move masking;
- nessuna vocabulary UCI teorica completa;
- singolo seed/run principale per la convergence finale;
- test osservato più volte storicamente;
- convergenza empirica non equivale a massimo globale;
- distribuzione puzzle diversa dal gioco generale;
- target = mossa soluzione puzzle, non valutazione generale della qualità delle mosse;
- nessun timing in Model A per design.

## 27. Frozen Baseline Definition

```text
MODEL_A_NO_TIMING_FROZEN_BASELINE
Status: FROZEN
```

Definizione:

| Component | Frozen value |
|---|---|
| Model | ChessGATNoTiming |
| Vocabulary | 1,786 train-derived classes |
| Architecture | current `src/models/chess_gat.py` |
| Initial LR | 5e-4 |
| Weight decay | 1e-4 |
| Dropout | 0.30 |
| Batch size | 128 |
| Seed | 42 |
| Scheduler | ReduceLROnPlateau |
| Early stopping | patience 12 |
| Best checkpoint | epoch 162 |
| Final test loss | 3.078363185864846 |
| Final Top1 | 39.6167% |
| Final Top3 | 55.8072% |
| Final Top5 | 62.7526% |
| Convergence | CONVERGED_BY_EARLY_STOPPING |

## 28. Reproducibility

Representation validation:

```bash
./venv/bin/python -m src.validate_representations \
  --csv-sample 1000 \
  --graph-sample 500 \
  --seed 42
```

Historical convergence command:

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

Historical continuation command:

```bash
./venv/bin/python -m src.train_chess_gat_convergence \
  --resume \
  --extend-max-epochs 300 \
  --device cuda
```

Do not relaunch these commands to create the frozen baseline again unless intentionally reproducing the experiment in a separate controlled environment.

Test suite:

```bash
./venv/bin/python -m pytest -q
./venv/bin/python -m compileall -q main.py src tests
```

Benchmark commands:

```bash
./venv/bin/python -m src.benchmark_chess_gat_training --device cuda
./venv/bin/python -m src.benchmark_progressive_subset_loading --device cuda --batch-size 128 --num-workers 0 --pin-memory --non-blocking --amp
```

## 29. Final Verdict

Model A is complete, converged under the defined protocol, and frozen as the official no-timing baseline.

Future changes to architecture, features, target, vocabulary, masking, optimizer protocol or hyperparameters are not the same Model A baseline. They must be treated as a new variant, ablation or experiment.

## Source of Truth

MODEL DEFINITION:

- `src/models/chess_gat.py`

TRAINING / CONVERGENCE LOGIC:

- `src/train_chess_gat_convergence.py`
- `src/training/convergence_run.py`
- `src/training/chess_gat_trainer.py`

DATA REPRESENTATION:

- `src/graph/node_features.py`
- `src/graph/edge_features.py`
- `src/graph/graph_builder.py`
- `src/graph/pyg_dataset.py`
- `data/pyg/manifest.json`
- `artifacts/move_encoder_stats.json`

FINAL TRAINING ARTIFACTS:

- expected server path: `artifacts/convergence_training/chess_gat_no_timing/best.pt`
- expected server path: `artifacts/convergence_training/chess_gat_no_timing/last.pt`
- expected server path: `artifacts/convergence_training/chess_gat_no_timing/history.json`
- expected server path: `artifacts/convergence_training/chess_gat_no_timing/controller_state.json`
- expected server path: `artifacts/convergence_training/chess_gat_no_timing/final_report.json`
- expected server path: `artifacts/convergence_training/chess_gat_no_timing/final_report.md`
- expected server snapshot: `artifacts/convergence_training/chess_gat_no_timing/snapshots/epoch_150_terminal/`

LOCAL AUDIT NOTE:

- The VM used for this documentation does not contain the final convergence artifact directory. Final convergence values in this document come from the server results supplied for this audit.
- Locally verified artifacts include `data/pyg/manifest.json`, `artifacts/move_encoder_stats.json`, `artifacts/adaptive_training/chess_gat_no_timing/final_report.json`, and benchmark report files under `artifacts/benchmarks/`.

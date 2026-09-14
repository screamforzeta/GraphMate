# Model B Timing Dataset

Questo documento definisce la pipeline timing usata per preparare i grafi puzzle per `MODEL_B_TIMING_LEGAL_MOVE_SCORER`.

## Stato

- `TIMING_GENERATOR_FOUND = NO`: nella repository non esisteva un generatore ufficiale dei tre campi richiesti da Model B.
- `REAL_GAME_TIMING_AVAILABLE = PARTIAL`: i game sample contengono clock `%clk` e CSV con `MoveTimes`, ma questi tempi appartengono a partite generiche e non sono allineati uno-a-uno ai puzzle.
- Il dataset prodotto deriva dai puzzle PyG ufficiali e viene scritto in `data/pyg_puzzles_timing/`.

## Campi

| Field | Semantic Meaning | Unit | Source | Normalization | Available At Inference? |
| --- | --- | --- | --- | --- | --- |
| `previous_move_time` | Contesto sintetico del tempo della mossa precedente alla posizione del solver. | z-score unitless di `log1p(seconds)` | generatore sintetico deterministico | train-only mean/std | sì |
| `original_move_time` | Tempo sintetico associato alla decisione corrente del puzzle. | z-score unitless di `log1p(seconds)` | generatore sintetico deterministico | train-only mean/std | sì |
| `time_is_synthetic` | Indicatore synthetic/real. Per ora è costante `1.0`. | binary flag | policy dataset | nessuna | sì |

I secondi raw sono mantenuti come metadata aggiuntivi: `previous_move_time_seconds` e `original_move_time_seconds`.

## Distribuzione

Il tempo corrente usa una LogNormal condizionata in modo smooth dal rating:

```text
difficulty = clamp((rating - 600) / (2400 - 600), 0, 1)
median_time = 12.0 * (1 + 0.55 * difficulty)
T_current ~ LogNormal(log(median_time), 0.45)
T_current = clamp(T_current, 0.5, 120.0)
```

Il tempo precedente non è una copia del tempo corrente. È un campione correlato nello spazio log:

```text
log_prev_center = 0.70 * log(T_current) + 0.30 * log(median_time)
T_previous ~ LogNormal(log_prev_center, 0.35)
T_previous = clamp(T_previous, 0.5, 120.0)
```

La normalizzazione è:

```text
normalized_time = (log1p(raw_seconds) - train_mean) / train_std
```

`train_mean` e `train_std` sono fit esclusivamente sullo split train e poi congelati per val/test.

## Riproducibilità

Ogni campione usa una seed stabile derivata da:

```text
global_seed + stable hash(PuzzleId, stream)
```

Quindi lo stesso puzzle mantiene gli stessi timing indipendentemente da DataLoader order, batch size, shuffle o numero di worker.

## Leakage Policy

La generazione usa solo:

- `PuzzleId`, per seed deterministica;
- `Rating`, come metadata disponibile nel setup sperimentale.

Non usa:

- `TargetMove`;
- continuation della soluzione;
- posizioni future;
- `MateDepth`;
- `Themes`;
- label test;
- correttezza del modello.

Poiché il rating influenza il timing, il manifest marca `potential_shortcut = true`. Questo è un confound controllato: la relazione è smooth e limitata, ma va considerata nell'interpretazione dell'ablation timing/no-timing.

## Comandi

Generazione:

```bash
./venv/bin/python -m src.cli.data.generate_puzzle_timing_dataset --overwrite
```

Training Model B, solo dopo audit:

```bash
./venv/bin/python -m src.cli.training.train_model_b_timing_legal_scorer \
  --dataset-root data/pyg_puzzles_timing \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp
```

Non avviare full training prima di verificare manifest, statistiche e compatibilità batch/forward/backward.

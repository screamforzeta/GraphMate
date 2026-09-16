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

## Ablation Ufficiale A3 Vs Model B

Il training ufficiale di `MODEL_B_TIMING_LEGAL_MOVE_SCORER` va confrontato con A3 senza modificare pesi, checkpoint, dataset o generatore timing.

### Official Model Performance

| Metric | A3 no-timing | Model B synthetic timing | Delta B-A3 |
| --- | ---: | ---: | ---: |
| N | 8610 | 8610 | 0 |
| Candidate CE / NLL | 1.0832485489175157 | 1.1394022388048426 | +0.0561536898873269 |
| Top1 | 0.675609756097561 | 0.6598141695702672 | -1.57955965272938 pp |
| Top3 | 0.8565621370499419 | 0.8484320557491289 | -0.81300813008130 pp |
| Top5 | 0.9185830429732869 | 0.913704994192799 | -0.48780487804879 pp |
| Mean legal target rank | 2.224274099883856 | 2.302903600464576 | +0.07862950058072 |
| Median legal target rank | 1.0 | 1.0 | 0 |
| Illegal Top1 | 0.0 | 0.0 | 0 |

Questi valori sono authoritative. A3 non deve essere sovrascritto con metriche di Model B.

### Post-Hoc Diagnostic Ablations

L'evaluator dedicato produce anche:

- `B_SYNTHETIC_TIMING`: checkpoint B con `previous_move_time`, `original_move_time`, `time_is_synthetic` originali.
- `B_NEUTRAL_TIMING`: stesso checkpoint B, ma `previous_move_time=0`, `original_move_time=0`, `time_is_synthetic=1`.
- `B_ZERO_TIMING_AND_FLAG`: diagnostica secondaria con timing a zero e `time_is_synthetic=0`.

`B_NEUTRAL_TIMING` non è A3: conserva timing encoder, candidate dim `412`, parametri aggiuntivi e pesi appresi durante training con timing. Il confronto `B_SYNTHETIC_TIMING` vs `B_NEUTRAL_TIMING` misura solo una sensitivity post-hoc dello stesso checkpoint.

### Report Command

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_b_timing_ablation \
  --device cuda \
  --batch-size 128 \
  --non-blocking \
  --amp
```

Output atteso:

- `artifacts/model_b_timing_ablation/summary.json`
- `artifacts/model_b_timing_ablation/report.md`
- `artifacts/model_b_timing_ablation/paired_test_rows.json`

Il report separa `OFFICIAL MODEL PERFORMANCE` da `POST-HOC DIAGNOSTIC ABLATIONS`, include transition counts, breakdown per MateDepth/rating, bucket timing, McNemar, distribuzione timing sintetica vs reale e esempi di errore.

## Real Game Timing Reference

La source-of-truth locale per i tempi reali è la pipeline games già persistita:

- `data/raw/games/sample_1000_games.pgn`: PGN campionato con commenti Lichess `[%clk ...]`.
- `data/processed/games/games_metadata.csv`: output del parser PGN con `ClockValues` e `MoveTimes`.
- `data/processed/games/games_clean.csv`: games validati.
- `data/final/games/games_train.csv`, `games_val.csv`, `games_test.csv`: split finali usati come sorgente preferita per il confronto descrittivo.

Semantica:

- `ClockValues`: clock residuo dopo ogni ply, in secondi, estratto dai commenti PGN `[%clk H:MM:SS]`.
- `MoveTimes`: stima del tempo di pensiero in secondi calcolata come differenza tra clock residui consecutivi: `clock[i-1] - clock[i]`.

Limitazioni note:

- Il primo ply non ha clock precedente.
- Con incremento, premove, lag o clock che aumenta, la differenza può essere negativa; la pipeline la salva come `None`.
- I valori `0.0` esistono ma non sono usati nel confronto distributional, perché la policy dell'evaluator tiene solo durate positive.
- I tempi reali provengono da partite Lichess campionate, non dagli stessi puzzle del test set.

Filtering policy dell'evaluator:

- parse con `ast.literal_eval`, perché i CSV contengono liste Python serializzate con `None`;
- flatten delle liste `MoveTimes`;
- conversione a `float` seconds;
- scarto di missing, non numerici, non finiti e `<= 0`;
- nessun clipping arbitrario.

Audit locale sui final games split:

| Counter | Value |
| --- | ---: |
| games scanned | 757 |
| games with timing | 757 |
| raw timing values | 50,469 |
| valid positive timing values | 25,469 |
| discarded missing | 21,788 |
| discarded non-positive | 3,212 |

Statistiche in secondi:

| Metric | Real games | Synthetic puzzle test |
| --- | ---: | ---: |
| N | 25,469 | 8,610 |
| mean | 29.4984 | 16.4443 |
| median | 11.0000 | 14.8018 |
| std | 49.2107 | 8.0505 |
| p05 | 1.0000 | 6.8593 |
| p25 | 4.0000 | 10.7548 |
| p50 | 11.0000 | 14.8018 |
| p75 | 32.0000 | 20.2762 |
| p95 | 116.0000 | 31.5825 |
| p99 | 286.3200 | 43.7166 |
| max | 432.0000 | 92.6349 |
| skewness | 3.6122 | 1.5671 |

Statistiche `log1p(seconds)`:

| Metric | Real games | Synthetic puzzle test |
| --- | ---: | ---: |
| N | 25,469 | 8,610 |
| mean | 2.6006 | 2.7640 |
| median | 2.4849 | 2.7601 |
| std | 1.2582 | 0.4339 |
| p05 | 0.6931 | 2.0617 |
| p25 | 1.6094 | 2.4643 |
| p50 | 2.4849 | 2.7601 |
| p75 | 3.4965 | 3.0576 |
| p95 | 4.7622 | 3.4838 |
| p99 | 5.6606 | 3.8003 |
| max | 6.0707 | 4.5394 |
| skewness | 0.3222 | 0.0678 |

Ratio descrittivi:

- synthetic mean / real mean: circa `0.5575`;
- synthetic median / real median: circa `1.3456`;
- synthetic p95 / real p95: circa `0.2723`.

KS two-sample, se `scipy` è disponibile:

- seconds statistic: `0.3399`, p-value `0.0`;
- log1p statistic: `0.3399`, p-value `0.0`.

Il KS è solo descrittivo. Con sample grandi e popolazioni non equivalenti, il p-value non è un verdetto di validità o invalidità del timing sintetico.

Interpretazione prudente: synthetic v1 ha centro in log-space vicino alla mediana reale, ma è molto più stretta e con coda alta molto più corta rispetto ai tempi reali osservati nelle partite. Questo confronto è una calibrazione marginale/distributional reference, non una misura di errore per-sample.

Aggiornamento solo distribuzione, senza rifare inference:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_b_timing_ablation \
  --timing-distribution-only
```

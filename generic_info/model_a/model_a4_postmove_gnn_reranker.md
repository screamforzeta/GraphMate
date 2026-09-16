# Model A4 — Post-Move GNN Reranker No Timing

## Stato

`MODEL_A4_POSTMOVE_GNN_RERANKER_NO_TIMING` implementa una variante no-timing successiva al baseline frozen A3.

Il modello non sostituisce A3: usa A3 come retrieval model frozen e impara solo a rerankare le candidate Top-5 prodotte da A3.

## Motivazione

Gli audit post-hoc di A3 hanno mostrato che:

- A3 Top1 test ufficiale: `0.675609756097561`
- A3 Top5 ceiling test ufficiale: `0.9185830429732869`
- errori A3 Top1: `2793`
- errori recuperabili entro Top5: `2092`
- errori oltre Top5: `701`

L'audit ha evidenziato segnali forti dopo l'applicazione della mossa candidata, per esempio riduzione delle legal moves avversarie e frequenza di check. A4 verifica se una GNN sulla posizione risultante può imparare questi segnali senza usare feature tattiche handcrafted.

Il test-set audit ha motivato la famiglia architetturale, ma non deve essere usato per training, hyperparameter selection, checkpoint selection o debugging scientifico di A4.

## Architettura

Pipeline:

1. posizione originale;
2. frozen A3 legal move scorer;
3. selezione Top-5 con ordering ufficiale device-side;
4. applicazione di ogni candidate Top-5;
5. costruzione del grafo PyG della posizione risultante;
6. post-move GAT dedicato;
7. scorer MLP A4;
8. reranking finale delle Top-5.

A4 non ricerca tra tutte le legal moves. Il suo candidate set è esattamente la Top-5 di A3, o meno di 5 mosse se la posizione ha meno legal moves.

## Frozen A3

A3 viene caricato da:

```bash
artifacts/model_a3_legal_move_scorer_no_timing/best.pt
```

I parametri A3 sono forzati a `requires_grad = False` e A3 resta in `eval()`.

Il checkpoint A4 dichiara il checkpoint A3 richiesto e la cache include fingerprint SHA256 del checkpoint A3.

## Candidate Representation

Per ogni candidate A3 Top-5, A4 concatena:

| Component | Dim |
|---|---:|
| A3 source node embedding | 128 |
| A3 destination node embedding | 128 |
| A3 original graph context | 132 |
| A3 promotion embedding | 8 |
| post-move graph context | 132 |
| A3 score features | 2 |
| Total | 530 |

Score features:

- `a3_raw_score`: logit raw prodotto da A3 per la candidate;
- `a3_score_centered_within_top5`: `raw_score - mean(raw_scores_top5)`.

A3 rank è mantenibile come metadata/debug, ma non è usato come input learned in A4 v1.

## Post-Move Graph

Per ogni candidate:

1. si parte dalla solver-position FEN;
2. si applica la candidate con `python-chess`;
3. si costruisce il grafo risultante con la rappresentazione ufficiale del progetto.

Il grafo risultante mantiene:

- node features: 15;
- edge features: 5;
- global features: 4;
- `edge_index` sparse PyG;
- `edge_attr` multilabel.

Le global features sono ricalcolate sulla resulting position, quindi cambiano correttamente `side_to_move`, `is_check`, `fullmove_number_normalized` e `halfmove_clock_normalized`.

## Training Semantics

A4 impara solo su esempi rerankable:

```text
target in A3 Top5 -> rerankable
target not in A3 Top5 -> unrerrankable
```

Gli esempi unrerrankable non entrano nella loss. Non viene mai inserito artificialmente il target nella Top-5.

Loss:

```text
CE(scores_for_A3_top5_candidates, target_candidate_index)
```

La loss è grouped/listwise come A3, ma limitata al gruppo Top-5.

## Metriche Validation

Checkpoint selection:

```text
MAX validation end-to-end Top1
```

Tie-breaker:

```text
min validation loss
```

Metriche riportate:

- `A3_VAL_TOP5_RECALL`
- `A4_VAL_CONDITIONAL_TOP1`
- `A4_VAL_END_TO_END_TOP1`
- paired transitions:
  - A3 correct / A4 correct
  - A3 correct / A4 wrong
  - A3 wrong / A4 correct
  - A3 wrong / A4 wrong
- `A3_CORRECT_PRESERVED_RATE`
- `A3_ERRORS_RECOVERED_RATE`

End-to-end Top1 considera sbagliati gli esempi in cui il target non è nella Top-5 A3.

## Cache

La cache A4 è separata dai dati PyG originali:

```text
data/model_a4_postmove/
  train/examples.pt
  train/summary.json
  val/examples.pt
  val/summary.json
  manifest.json
```

La cache contiene solo train e validation durante lo sviluppo A4.

Provenance:

- checkpoint A3 path;
- SHA256 checkpoint A3;
- `K = 5`;
- graph representation version;
- split;
- numero esempi;
- timestamp.

La cache viene rifiutata se checkpoint A3, `K` o graph representation version non corrispondono.

## Test-Set Lock

Durante implementazione, training, smoke test, benchmark e checkpoint selection:

```text
TEST_SET_USED_FOR_A4_TRAINING = NO
TEST_SET_USED_FOR_CHECKPOINT_SELECTION = NO
TEST_SET_EVALUATED = NO
```

Il test set dovrà essere valutato una sola volta in una task separata dopo il freeze finale di A4.

## Limitazioni

A4 v1 non usa:

- timing;
- Stockfish;
- Level-2 opponent-response graphs;
- minimax;
- handcrafted tactical audit features;
- target rank come feature.

A4 può migliorare solo esempi in cui A3 recupera già il target nella Top-5.

## Comandi Server

Costruzione cache train + validation:

```bash
./venv/bin/python -m src.cli.data.build_model_a4_postmove_cache \
  --device cuda \
  --batch-size 128
```

Smoke test:

```bash
./venv/bin/python -m src.cli.training.train_model_a4_postmove \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp \
  --smoke \
  --max-epochs 3
```

Benchmark cached validation:

```bash
./venv/bin/python -m src.cli.benchmarks.benchmark_model_a4_postmove \
  --device cuda \
  --batch-size 128 \
  --max-batches 10 \
  --amp
```

Full training:

```bash
./venv/bin/python -m src.cli.training.train_model_a4_postmove \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp \
  --max-epochs 300
```

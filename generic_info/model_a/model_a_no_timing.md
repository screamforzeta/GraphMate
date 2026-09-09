# Famiglia No-Timing — Model A, Model A2, Model A3

Questo documento chiude formalmente la fase no-timing del progetto. La famiglia no-timing nasce con `Model A`, passa per l'ablation `Model A2`, e arriva al baseline finale `Model A3`.

```text
MODEL_A3_NO_TIMING_FROZEN_BASELINE = YES
```

## 1. Obiettivo Della Baseline No-Timing

L'obiettivo è risolvere puzzle Lichess mate-in-n usando grafi PyTorch Geometric senza informazioni temporali o timing sintetici. Il task supervisionato è predire la prima mossa UCI corretta della soluzione dalla posizione vista dal solver.

La fase no-timing serve come riferimento scientifico per le future varianti timing-aware. Nessun modello no-timing usa event IDs, tempi di pensiero o feature temporali.

## 2. Rappresentazione Del Grafo

Ogni posizione è un grafo con 64 nodi, uno per casella.

```text
x [64,15]
edge_index [2,E]
edge_attr [E,5]
global_features [1,4]
y = target graph-level
```

Node features:

| Index | Feature |
|---:|---|
| 0-5 | piece one-hot: pawn, knight, bishop, rook, queen, king |
| 6 | color |
| 7 | occupied |
| 8 | normalized row |
| 9 | normalized column |
| 10 | attacked_by_white |
| 11 | attacked_by_black |
| 12 | legal_mobility |
| 13 | is_pinned |
| 14 | piece_value |

Edge features multilabel:

| Index | Feature |
|---:|---|
| 0 | legal_move |
| 1 | attack |
| 2 | defend |
| 3 | pin |
| 4 | check_line |

Global features:

| Index | Feature |
|---:|---|
| 0 | side_to_move |
| 1 | is_check |
| 2 | fullmove_number normalized |
| 3 | halfmove_clock normalized |

Queste global features descrivono solo lo stato della posizione. Non contengono rating, MateDepth, themes, target move, solution/future moves o informazioni derivate dalla risposta corretta.

## 3. Semantica Lichess

La FEN raw Lichess rappresenta la posizione prima della setup move dell'avversario.

Pipeline corretta:

1. leggere `OriginalFEN`;
2. applicare `Moves[0]`;
3. salvare la posizione risultante come `FEN`;
4. usare `Moves[1]` come `TargetMove`.

Quindi `Data.fen` è la posizione vista dal solver e `Data.target_move` è la prima risposta UCI corretta. `Data.y`, quando disponibile, è `move_to_idx[TargetMove]`.

## 4. Dataset E Vocabulary

| Split | CSV rows | PyG graphs | Note |
|---|---:|---:|---|
| Train | 68,958 | 68,958 | nessun OOV |
| Validation | 8,620 | 8,612 | 8 OOV esclusi |
| Test | 8,620 | 8,610 | 10 OOV esclusi |

La vocabulary `move_to_idx.json` è costruita solo sul train set e contiene `1,786` classi. Questo evita leakage da validation/test.

## 5. Model A — Global Move Classification

Model A è il primo baseline no-timing. Formula il problema come classificazione globale della mossa corretta sull'intera vocabulary train-only da `1,786` mosse.

```text
x [N,15]
edge_index [2,E]
edge_attr [E,5]

GATConv(15,32,heads=4,edge_dim=5) -> 128 -> ELU -> Dropout(0.30)
GATConv(128,32,heads=4,edge_dim=5) -> 128 -> ELU
global_mean_pool -> 128
concat global_features [4] -> 132
Linear(132,128) -> ELU -> Dropout -> Linear(128,1786)
```

No timing. No event IDs. No legal mask durante il training. La scelta era intenzionalmente semplice: misurare quanto una GAT chess-specific potesse apprendere predicendo una classe globale di mossa.

## 6. Training E Risultati Model A

Checkpoint ufficiale:

```text
artifacts/convergence_training/chess_gat_no_timing/best.pt
```

| Metric | Value |
|---|---:|
| best epoch | 162 |
| stop epoch | 174 |
| stop reason | EARLY_STOPPING |
| test N | 8,610 |
| loss | 3.0783631859 |
| Top1 | 39.6167247421% |
| Top3 | 55.8072009402% |
| Top5 | 62.7526132501% |

Model A ha raggiunto convergenza empirica: la validation loss non migliora per tutta la patience dopo più riduzioni del learning rate. Questo non implica massimo globale del task.

## 7. Limiti Osservati Di Model A

Il limite più importante emerso è la legalità delle mosse:

```text
raw illegal Top1 rate ≈ 30.2323%
```

Il modello poteva assegnare massima probabilità a mosse impossibili. Questo mostra che la formulazione globale obbligava il modello anche ad apprendere un vincolo già noto: la legalità della mossa.

Model A best-legal filtra le predizioni illegali dopo il forward, senza cambiare il modello:

| Metric | Model A Best-Legal |
|---|---:|
| Top1 | 49.8142% |
| Top3 | 70.4994% |
| Top5 | 79.5587% |

Questo forte salto ha motivato una ablation controllata.

## 8. Model A2 — Legal Mask Training

Model A2 non è un'architettura completamente diversa. È una ablation per rispondere alla domanda:

```text
Quanto del limite di Model A deriva dal fatto che il classificatore può scegliere mosse illegali?
```

A2 mantiene stessa rappresentazione, stessa GAT architecture, stessa vocabulary, stesso dataset/split, stesso optimizer, lr iniziale `5e-4`, weight decay `1e-4`, dropout `0.30`, batch size `128`, seed `42` e model selection su validation loss.

Cambia solo lo spazio delle azioni durante loss/metriche:

1. ricostruisce la posizione da `Data.fen`;
2. genera mosse legali con `python-chess`;
3. mappa le mosse legali nella vocabulary;
4. maschera logits illegali con valore finito `-1e4`;
5. applica CrossEntropy sullo spazio legalmente consentito.

## 9. Perché È Stato Creato A2

A2 isola l'effetto della legalità. Se A2 avesse risolto il problema, allora il limite principale di Model A sarebbe stato solo la presenza di mosse illegali nello spazio di output.

Il risultato è stato più interessante: legal filtering è fondamentale, ma legal-mask training non basta.

## 10. Risultati E Limiti Di A2

| Metric | A2 Masked |
|---|---:|
| loss | 1.7831716187 |
| masked Top1 | 49.2566782811% |
| masked Top3 | 71.4982578480% |
| masked Top5 | 81.1730545946% |
| masked illegal Top1 | 0% |

Confronto chiave:

| Metric | Model A Best-Legal | A2 Masked |
|---|---:|---:|
| Top1 | 49.8142% | 49.2567% |
| Top3 | 70.4994% | 71.4983% |
| Top5 | 79.5587% | 81.1731% |

Interpretazione:

- legal filtering produce un grande miglioramento;
- A2 non migliora il Top1 globale rispetto al best-legal post-hoc di Model A;
- A2 migliora leggermente Top3/Top5;
- il limite non è semplicemente "allenare con legal mask";
- il problema sembra essere nella formulazione globale della decisione.

MateIn1:

| Metric | Model A Best-Legal | A2 |
|---|---:|---:|
| Top1 | 56.14% | 52.53% |

A2 peggiora MateIn1 rispetto al best-legal post-hoc di Model A. Questo è stato il segnale principale che ha motivato A3.

## 11. Motivazione Per Model A3

Il passaggio concettuale è:

```text
A / A2: score di tutte le 1786 classi globali
A3: score solo delle mosse legali realmente disponibili nella posizione
```

La domanda scientifica diventa: è migliore una formulazione candidate-ranking rispetto a una classificazione globale delle mosse?

A3 elimina il classifier finale da `1,786` classi e usa un insieme variabile di candidate legali per ogni grafo.

## 12. Model A3 — Legal Move Candidate Scorer

A3 mantiene lo stile dell'encoder GAT di Model A ma cambia la formulazione dell'output.

Proprietà:

- no timing;
- no event IDs;
- no global move-classification head;
- legal candidates generate da `Data.fen`;
- target = `Data.target_move`;
- grouped/listwise CrossEntropy per grafo;
- output variabile = numero di mosse legali.

```text
NO_1786_CLASSIFIER_IN_A3_FORWARD = YES
```

## 13. Scelte Architetturali A3

Encoder:

```text
GATConv(15,32,heads=4,edge_dim=5) -> 128
ELU
Dropout(0.30)
GATConv(128,32,heads=4,edge_dim=5) -> 128
ELU
global_mean_pool -> 128
concat global_features [4] -> graph context 132
```

Candidate representation:

| Component | Dim |
|---|---:|
| source node embedding | 128 |
| destination node embedding | 128 |
| graph context | 132 |
| promotion embedding | 8 |
| total | 396 |

Scorer:

```text
Linear(396,128) -> ELU -> Dropout(0.30) -> Linear(128,1)
```

Parametri:

| Component | Parameters |
|---|---:|
| encoder | 20,608 |
| scorer | 50,985 |
| total | 71,593 |

A3 ha molti meno parametri di Model A, circa `71.6k` contro circa `268k`. Il miglioramento non può quindi essere attribuito semplicemente a maggiore capacità parametrica.

## 14. Training Protocol A3

| Component | Value |
|---|---|
| optimizer | Adam |
| lr | 5e-4 |
| weight decay | 1e-4 |
| dropout | 0.30 |
| batch size | 128 |
| seed | 42 |
| scheduler | ReduceLROnPlateau |
| factor / patience / min LR | 0.5 / 3 / 1e-6 |
| early stopping patience | 12 |
| max epochs | 300 |
| AMP | enabled on server |
| num_workers | 0 |
| pin_memory | true |
| non_blocking | true |
| model selection | validation loss only |

Il test viene eseguito solo dopo training terminale e reload del best checkpoint.

## 15. Validation / Parity

Pretraining candidate validation:

| Split | Valid | Total | Candidate errors |
|---|---:|---:|---:|
| train | 68,958 | 68,958 | 0 |
| val | 8,612 | 8,612 | 0 |
| test | 8,610 | 8,610 | 0 |

Candidate generation:

```text
batch.fen -> python-chess legal_moves -> candidate list
```

Grouped CE:

```text
-log exp(score_target) / sum(exp(score_legal_moves_same_graph))
```

Ogni grafo ha il proprio denominatore. Candidate di grafi diversi non condividono il softmax.

Parity finale ufficiale:

```text
MODEL_A_PARITY = PASS
MODEL_A2_PARITY = PASS
MODEL_A3_PARITY = PASS
```

A3 canonical shared-test:

| Metric | Value |
|---|---:|
| N | 8,610 |
| candidate CE/NLL | 1.0832485489175157 circa |
| Top1 | 0.675609756097561 |
| Top3 | 0.8565621370499419 |
| Top5 | 0.9185830429732869 |
| mean legal rank | 2.224274099883856 |
| median legal rank | 1.0 |
| illegal Top1 | 0.0 |

TopK/rank hanno parity near-exact. Solo CE/NLL usa tolerance assoluta `1e-6` per differenze floating-point CUDA/AMP. Questa tolerance non influenza ranking metrics, model selection o interpretazione TopK.

## 16. Risultati Globali

| Metric | A Raw | A Best-Legal | A2 Masked | A3 |
|---|---:|---:|---:|---:|
| Top1 | 39.6167% | 49.8142% | 49.2567% | 67.5610% |
| Top3 | 55.8072% | 70.4994% | 71.4983% | 85.6562% |
| Top5 | 62.7526% | 79.5587% | 81.1731% | 91.8583% |
| Mean legal rank | 4.0115 | 4.0115 | 3.7220 | 2.2243 |
| Median legal rank | 2 | 2 | 2 | 1 |
| Illegal Top1 | 30.2323% | 0% | 0% | 0% |

Deltas A3:

| Comparison | Top1 | Top3 | Top5 |
|---|---:|---:|---:|
| A3 vs A raw | +27.9443 pp | +29.8490 pp | +29.1057 pp |
| A3 vs A best-legal | +17.7468 pp | +15.1568 pp | +12.2997 pp |
| A3 vs A2 | +18.3043 pp | +14.1580 pp | +10.6852 pp |

## 17. MateDepth Analysis

Top1 per MateDepth:

| MateDepth | A Raw | A Best-Legal | A2 | A3 |
|---|---:|---:|---:|---:|
| MateIn1 | 47.22% | 56.14% | 52.53% | 79.55% |
| MateIn2 | 40.62% | 52.83% | 52.08% | 69.78% |
| MateIn3 | 42.42% | 52.33% | 52.38% | 66.73% |
| MateIn4 | 32.47% | 42.12% | 43.12% | 58.73% |
| MateIn5 | 25.89% | 36.41% | 39.32% | 52.91% |

A3 domina A/A2 in ogni MateDepth e risolve chiaramente la debolezza MateIn1 mostrata da A2. L'accuracy diminuisce comunque con la profondità: la difficoltà residua non è più principalmente legalità della mossa, ma ranking/rappresentazione/comprensione della posizione.

## 18. Rating Analysis

A3 Top1 per rating:

| Rating bucket | N | A3 Top1 |
|---|---:|---:|
| <1200 | 3,879 | 82.0572% |
| 1200-1599 | 2,176 | 65.8548% |
| 1600-1999 | 1,564 | 51.3427% |
| 2000-2399 | 811 | 43.1566% |
| 2400+ | 180 | 26.6667% |

La performance cala all'aumentare della difficoltà/rating. A3 resta superiore ai predecessori in tutti i bucket. Il bucket `2400+` va interpretato con cautela per `N=180`.

Questa subgroup analysis è descrittiva e post-hoc. Non è stata usata per model selection.

## 19. Confronto A Vs A2 Vs A3

La sequenza sperimentale mostra:

1. Model A dimostra che una GAT chess-specific può apprendere il task, ma soffre lo spazio globale da `1,786` classi.
2. Model A best-legal mostra che legal filtering recupera molti errori.
3. Model A2 mostra che allenare con legal mask migliora Top3/Top5 ma non risolve Top1 e peggiora MateIn1 rispetto al best-legal post-hoc.
4. Model A3 riformula il problema come ranking tra candidate legali e migliora nettamente tutte le metriche principali.

## 20. Interpretazione Scientifica

Conclusione principale:

- legal filtering è fondamentale;
- legal-mask training da solo non risolve il problema;
- candidate-based legal move scoring è molto più adatto della classificazione globale su vocabulary;
- A3 migliora molto pur usando meno parametri;
- il task residuo è ranking tra poche mosse plausibili, non eliminazione di mosse illegali.

Non si deve affermare causalità su componenti interni specifici: A3 cambia sia output parametrization sia inductive bias.

## 21. Baseline No-Timing Ufficiale

La baseline no-timing ufficiale del progetto è:

```text
MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING
```

Model A resta baseline iniziale storica. Model A2 resta ablation controllata sulla legal mask. A3 è il riferimento no-timing da usare per confronti futuri.

## 22. Implicazioni Per Il Modello Timing-Aware

La futura variante timing-aware dovrà partire dalla formulazione candidate-based di A3, mantenendo invariati per quanto possibile:

- dataset/split;
- rappresentazione base del grafo;
- candidate generation;
- target semantics;
- training protocol;
- metriche shared-test.

Il timing dovrà essere aggiunto come nuova componente sperimentale controllata. Non deve riportare il progetto alla classificazione globale su vocabulary come output principale.

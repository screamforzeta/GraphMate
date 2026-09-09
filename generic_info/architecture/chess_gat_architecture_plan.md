# Chess GAT Architecture Plan

> Nota di stato: questo documento è un piano storico. `ChessGATNoTiming` è stato implementato, addestrato, portato a convergenza empirica e congelato come `MODEL_A_NO_TIMING_FROZEN_BASELINE`. Il riferimento definitivo aggiornato è [`model_a_no_timing.md`](model_a_no_timing.md).

## Obiettivo

Questa relazione progetta due architetture GAT future per il progetto chess, senza implementarle:

- `ChessGATNoTiming`
- `ChessGATWithTiming`

Il confronto deve essere una ablation quanto piu possibile fair: stessa rappresentazione chess, stesso backbone logico, stesso pooling, stessa classification head, stessa procedura di training. La differenza principale deve essere solo l'informazione temporale e il modo in cui modula l'attention.

Dataset attuale validato:

```python
Data(
    x=[64, 15],
    edge_index=[2, E],
    edge_attr=[E, 5],
    y=graph_level_move_target,
    global_features=[1, 4],
)
```

Target: classificazione graph-level della mossa `TargetMove`, con `num_classes` derivato dal move encoder, non hardcoded.

## Verifica Delle Conclusioni Precedenti

Ho verificato direttamente i file TimeGNN rilevanti:

| Conclusione | Verifica nel codice | Esito |
| --- | --- | --- |
| `DualGATModel` e node-level | `TimeGNN-main/src/timegnn/models/gat_basic.py`: `out = self.fc(x)` dopo GAT su nodi, senza pooling | Confermata |
| `DualGATTimeAwareModel` e node-level | `TimeGNN-main/src/timegnn/models/gat_time_decay.py`: `out = self.fc(x)`, nessun pooling | Confermata |
| `TimeAwareGATConv` usa edge-level timing | Usa `edge_attr` come `time_diff` in `exp(-lambda_decay * time_diff)` | Confermata |
| `DualGATModel` puo usare `edge_attr` numerico | Costruttore passa `edge_dim` a `GATConv`; default `edge_dim=1`, configurabile | Confermata |
| `DualGATTimeAwareModel` non usa il nostro tactical `edge_attr=[E,5]` | Nel forward fa `edge_attr = data_event.time` | Confermata |
| `TimeAwareETGATConv` separa edge features e time | Forward: `edge_attr` e `time` sono argomenti separati | Confermata |
| `DualGATTimeAwareETModel` usa `edge_type` discreto | Forward: `edge_attr = self.edge_type_emb(data_event.edge_type)` | Confermata |
| `PrefixGCNClassifier` mostra pooling graph-level | Usa `global_mean_pool(x, data.batch)` e concatena `sequence_features` | Confermata |
| `gat_outcome.py` mostra graph-level GAT pattern | Fa `node_logits = model(event_data)` e `global_mean_pool(node_logits, batch)` | Confermata |

Differenza importante rispetto a un design ideale: `gat_outcome.py` fa pooling sui logits node-level, non sugli embedding prima della classification head. Per chess conviene fare pooling sugli embedding nodali e poi classificare il grafo.

## Decisione Architetturale Principale

### Strategia 1: riuso quasi diretto di `DualGATModel` e `DualGATTimeAwareModel`

| Criterio | Valutazione |
| --- | --- |
| Fair ablation | Debole: `DualGATModel` puo usare `edge_attr=[E,5]`, mentre `DualGATTimeAwareModel` usa `data.time` come edge attr temporale e non usa tactical edge features nello stesso modo. |
| Compatibilita edge_attr `[E,5]` | Buona per `DualGATModel(edge_dim=5)`, scarsa per `DualGATTimeAwareModel`. |
| Compatibilita timing | Buona per `DualGATTimeAwareModel`, ma il timing sostituisce il canale edge_attr. |
| Necessita `event_ids` | Alta: entrambi richiedono `data.event_ids`. |
| Codice custom | Medio: servono `event_ids`, wrapper graph-level e adattatori. |
| Rischio bias architetturali | Alto: il ramo embedding eventi nasce per event log, non per caselle chess. |
| Chiarezza scientifica | Media: difficile sostenere che l'unica differenza sia il timing. |
| Manutenibilita | Media. |
| Compatibilita PyG | Buona. |
| Facilita test | Media. |

### Strategia 2: wrapper/backbone chess-specific usando layer TimeGNN rilevanti

| Criterio | Valutazione |
| --- | --- |
| Fair ablation | Forte: i due modelli possono condividere struttura, dimensioni, pooling, head e tactical edge features. |
| Compatibilita edge_attr `[E,5]` | Forte: progettata esplicitamente. |
| Compatibilita timing | Forte: `edge_time` rimane separato e viene usato solo nel modello B. |
| Necessita `event_ids` | Nessuna: si evita il vincolo event-log. |
| Codice custom | Medio: servono classi wrapper/backbone, ma piccole e testabili. |
| Rischio bias architetturali | Basso: niente embedding event-log non necessari. |
| Chiarezza scientifica | Alta: stessa architettura, differenza controllata sul timing. |
| Manutenibilita | Alta. |
| Compatibilita PyG | Alta. |
| Facilita test | Alta: `logits = model(batch)` per entrambi. |

### Raccomandazione

Raccomando **Strategia 2**.

Motivo: usare direttamente i modelli dual-path TimeGNN obbliga a introdurre `event_ids` e porta una differenza non fair tra no-timing e timing, perche il modello no-timing puo consumare `edge_attr=[E,5]` con `GATConv(edge_dim=5)`, mentre `DualGATTimeAwareModel` tratta `data.time` come edge feature temporale e non mantiene lo stesso canale tattico. Un wrapper chess-specific permette invece di usare le stesse tactical edge features in entrambi e cambiare solo il meccanismo temporale.

## Event IDs

Opzioni valutate:

| Opzione | Pro | Contro | Decisione |
| --- | --- | --- | --- |
| Square IDs `0..63` | Stabili, facili, compatibili con embedding TimeGNN | Duplicano in parte `row`/`col`; aggiungono bias posizionale hard-coded; servono solo per soddisfare modelli event-log | Non raccomandati per baseline principale |
| Piece IDs | Informazione semantica scacchistica | Ambigui per caselle vuote; instabili; gia coperti da piece one-hot | Non raccomandati |
| Nessun `event_ids` | Modello piu pulito; usa solo feature progettate; fairer | Richiede non usare direttamente i dual-path originali | Raccomandato |
| Altro mapping | Possibile in esperimenti futuri | Rischio arbitrarieta | Non necessario ora |

Decisione:

```text
USES_EVENT_IDS = NO
```

Le coordinate normalizzate `row` e `col` sono gia presenti in `x`. Aggiungere square embeddings sarebbe una seconda codifica posizionale appresa. Non e necessariamente sbagliata, ma per una prima baseline scientificamente pulita conviene evitare informazione duplicata e non necessaria.

## Tactical Edge Features

Il nostro `edge_attr=[E,5]` e multilabel:

| Indice | Feature |
| ---: | --- |
| 0 | `legal_move` |
| 1 | `attack` |
| 2 | `defend` |
| 3 | `pin` |
| 4 | `check_line` |

Requisito: `ChessGATNoTiming` e `ChessGATWithTiming` devono usare esattamente le stesse tactical edge features.

### Comportamento nativo TimeGNN

| Componente | Tactical `edge_attr=[E,5]` |
| --- | --- |
| `GATConv(edge_dim=5)` usato in `DualGATModel` | Compatibile se `edge_dim=5`. |
| `TimeAwareGATConv` | Non compatibile direttamente: interpreta `edge_attr` come tempo. |
| `TimeAwareETGATConv` | Accetta `edge_attr` e `time` separati, ma nel wrapper TimeGNN `edge_attr` nasce da `edge_type_emb`. |
| `DualGATTimeAwareETModel` | Non compatibile direttamente: richiede `edge_type` discreto e crea embedding interno. |

### Adattamento proposto

Creare due layer/backbone chess-specific:

- No timing: usare `torch_geometric.nn.GATConv(..., edge_dim=5)`.
- With timing: usare una variante ispirata a `TimeAwareETGATConv`, ma con `edge_attr=[E,5]` multilabel come vettore numerico diretto e `edge_time=[E,1]` separato.

Non convertire `edge_attr` in un singolo `edge_type`: un arco puo essere contemporaneamente legal move e attack, oppure altra combinazione. La conversione single-label perderebbe informazione e introdurrebbe arbitrarieta.

Decisione:

```text
USES_TACTICAL_EDGE_ATTR = YES, in both models
EDGE_FEATURE_FORMAT = multilabel float tensor [E,5]
EDGE_TYPE_DISCRETE = NO for baseline architectures
```

## MODEL A: ChessGATNoTiming

### Input

| Attributo | Shape dopo batching | Note |
| --- | --- | --- |
| `batch.x` | `[64 * B, 15]` | Node features. |
| `batch.edge_index` | `[2, E_total]` | Sparse PyG edges. |
| `batch.edge_attr` | `[E_total, 5]` | Tactical multilabel features. |
| `batch.global_features` | `[B, 4]` | Graph-level chess features. |
| `batch.batch` | `[64 * B]` | PyG graph assignment vector. |
| `batch.y` | `[B]` | Target move class. |

### Forward

```text
batch.x [64B,15]
batch.edge_index [2,E]
batch.edge_attr [E,5]
    -> GAT layer 1: GATConv(15, 32, heads=4, concat=True, edge_dim=5)
    -> node h1 [64B,128]
    -> activation ELU
    -> dropout 0.10
    -> optional LayerNorm/BatchNorm over [64B,128]
    -> GAT layer 2: GATConv(128, 32, heads=4, concat=True, edge_dim=5)
    -> node embeddings [64B,128]
    -> activation ELU
    -> global_mean_pool(node_embeddings, batch.batch)
    -> graph_embedding [B,128]
    -> concat global_features [B,4]
    -> classifier input [B,132]
    -> Linear(132, 128)
    -> ELU
    -> dropout 0.10
    -> Linear(128, num_classes)
    -> logits [B,num_classes]
```

### Config iniziale

| Parametro | Valore |
| --- | --- |
| `input_dim` | 15 |
| `edge_dim` | 5 |
| `hidden_per_head` | 32 |
| `heads` | 4 |
| `node_hidden_dim` finale | 128 |
| `num_layers` | 2 |
| `concat` | True |
| `activation` | ELU |
| `dropout` | 0.10 |
| `normalization` | preferenza: none per primissimo smoke test; LayerNorm o BatchNorm1d dopo se instabile |
| `pooling` | `global_mean_pool` |
| `global_feature_dim` | 4 |
| `classifier_hidden_dim` | 128 |
| `output_dim` | `num_classes`, letto dal move encoder |

Questa configurazione e piccola/media e adatta ai primi test su RTX A2000 6GB. Il costo principale sara la classification head verso circa 1,786 classi, non i 64 nodi.

## MODEL B: ChessGATWithTiming

### Obiettivo

Il modello B deve essere identico al modello A in tutto tranne che nel meccanismo di attention, che riceve anche timing.

### Input

| Attributo | Shape dopo batching | Note |
| --- | --- | --- |
| `batch.x` | `[64 * B, 15]` | Identico al modello A. |
| `batch.edge_index` | `[2, E_total]` | Identico. |
| `batch.edge_attr` | `[E_total, 5]` | Identico: tactical features. |
| `batch.edge_time` | `[E_total, 1]` | Timing edge-level, separato. |
| `batch.global_features` | `[B, 4]` | Identico. |
| `batch.batch` | `[64 * B]` | Identico. |
| `batch.y` | `[B]` | Identico. |

Shape raccomandata:

```text
edge_time = float tensor [E,1]
```

Motivo: TimeGNN usa valori edge-level compatibili con `E`; `[E,1]` evita ambiguita di broadcasting e permette eventuale proiezione lineare del tempo.

### Forward

```text
batch.x [64B,15]
batch.edge_index [2,E]
batch.edge_attr [E,5]
batch.edge_time [E,1]
    -> Time/tactical-aware GAT layer 1
    -> node h1 [64B,128]
    -> activation ELU
    -> dropout 0.10
    -> same normalization choice as A
    -> Time/tactical-aware GAT layer 2
    -> node embeddings [64B,128]
    -> activation ELU
    -> global_mean_pool(node_embeddings, batch.batch)
    -> graph_embedding [B,128]
    -> concat global_features [B,4]
    -> same classifier as A
    -> logits [B,num_classes]
```

### Differenza rispetto al modello A

Solo attention:

```text
No timing:     attention = f(x_i, x_j, edge_attr)
With timing:   attention = f(x_i, x_j, edge_attr, edge_time)
```

Pooling, classifier, hidden sizes, dropout, activation, split, optimizer e loss devono restare identici.

## Integrazione Timing Nell'Attention

### Formula TimeGNN reale

In `TimeAwareGATConv`:

```python
cat_ij = torch.cat([x_i, x_j], dim=-1)
alpha = torch.einsum("ehc,hc->eh", cat_ij, self.att)
alpha = F.leaky_relu(alpha, self.negative_slope)

decay = torch.exp(-self.lambda_decay * time_diff).unsqueeze(-1)
alpha = alpha * decay
```

In `TimeAwareETGATConv`:

```python
edge_type_score = torch.einsum("ed,hd->eh", edge_attr, self.edge_type_att)
alpha = alpha + edge_type_score
decay = torch.exp(-self.lambda_decay * time).unsqueeze(-1)
alpha = alpha * decay
```

### Criticita

Dal codice TimeGNN il decay sembra applicato ai valori di attention/logit dentro `message`, e non e evidente una softmax standard dopo il decay. Questo e diverso dal pattern classico GAT in cui:

```text
raw score -> LeakyReLU -> softmax per neighborhood -> dropout -> weighted message
```

Questa scelta potrebbe funzionare, ma va testata con attenzione per stabilita numerica e interpretabilita.

### Variante chess proposta

Per il progetto chess propongo un layer custom piccolo ispirato a `TimeAwareETGATConv`, ma con edge multilabel diretto:

```text
base score from source/target node embeddings
+ tactical edge contribution from edge_attr=[E,5]
+ optional temporal term from edge_time=[E,1]
-> normalized attention over incoming edges
-> message aggregation
```

Oppure, se si vuole essere piu fedeli a TimeGNN:

```text
base score + tactical edge contribution
-> temporal decay exp(-lambda * edge_time)
-> attention modulation
```

La scelta finale va implementata e testata esplicitamente. Se l'obiettivo principale e ablation timing/no timing, la soluzione migliore e:

- stesso tactical edge projection in entrambi;
- nel modello B aggiungere solo un termine/modulazione temporale;
- evitare che il timing sostituisca `edge_attr`.

## Fair Ablation

| Parameter | No Timing | With Timing |
| --- | --- | --- |
| Node input dim | 15 | 15 |
| Tactical edge dim | 5 | 5 |
| `edge_index` | stesso | stesso |
| Timing input | assente | `edge_time=[E,1]` |
| Hidden per head | 32 | 32 |
| Heads | 4 | 4 |
| Layers | 2 | 2 |
| Concat heads | True | True |
| Node embedding dim | 128 | 128 |
| Activation | ELU | ELU |
| Dropout | 0.10 | 0.10 |
| Normalization | stessa scelta | stessa scelta |
| Pooling | `global_mean_pool` | `global_mean_pool` |
| Global feature dim | 4 | 4 |
| Classifier | `Linear(132,128)->ELU->Dropout->Linear(128,C)` | stesso |
| Output dim | `num_classes` | `num_classes` |
| Loss futura | `CrossEntropyLoss` | `CrossEntropyLoss` |
| Optimizer futuro | Adam, stesso LR | Adam, stesso LR |
| Batch size | stesso | stesso |
| Split/seed | stesso | stesso |

### Parameter count

Il modello timing aggiungera pochi parametri se:

- `lambda_decay` e scalare non appreso, come in TimeGNN;
- `edge_time` viene usato solo come decay;
- tactical edge projection resta identica.

Se invece si aggiunge una proiezione temporale `Linear(1, heads)`, il costo e comunque trascurabile rispetto alla classification head.

## Graph-Level Wrapper

### Pooling

| Pooling | Pro | Contro | Decisione |
| --- | --- | --- | --- |
| `global_mean_pool` | Stabile rispetto al numero di nodi; usato da TimeGNN; adatto per 64 nodi fissi | Può diluire segnali tattici locali | Baseline raccomandata |
| `global_add_pool` | Mantiene somma dei segnali | Dipende dalla dimensione del grafo; qui 64 fisso, ma puo amplificare scale | Non baseline |
| `global_max_pool` | Cattura pattern forti/locali | Puo ignorare distribuzione globale | Possibile ablation futura |

Decisione:

```text
POOLING = global_mean_pool
```

Wrapper:

```text
node_embeddings [64B,128]
batch.batch [64B]
-> global_mean_pool
graph_embedding [B,128]
global_features [B,4]
-> concat
classifier_input [B,132]
-> MLP
logits [B,num_classes]
```

## Classification Head

Il target e classificazione mossa. Il modello non deve hardcodare `1786`.

In futuro:

```python
with open("artifacts/move_to_idx.json") as f:
    move_to_idx = json.load(f)

num_classes = len(move_to_idx)
```

Output:

```text
logits = [B, num_classes]
```

Loss:

```python
criterion = torch.nn.CrossEntropyLoss()
loss = criterion(logits, batch.y)
```

Nessuna softmax dentro il modello durante training. `CrossEntropyLoss` si aspetta logits raw.

## OOV

I grafi con target OOV sono gia esclusi da `src/graph/pyg_dataset.py` quando `SKIP_OOV_TARGETS=True`.

Scelta consigliata:

```text
UNK_CLASS = NO
```

Il training loop non deve aggiungere una classe `<UNK>` se i grafi caricati sono gia compatibili con `move_to_idx`. Deve solo validare che:

- `batch.y` sia `long`;
- `0 <= y < num_classes`;
- train/val/test `.pt` siano quelli generati con lo stesso encoder.

## Normalization, Activation, Dropout

TimeGNN supporta:

- activation: `relu`, `elu`, `gelu`, `leaky_relu`;
- dropout tra hidden layers;
- `BatchNorm1d` opzionale nei modelli multi-layer.

Decisione baseline:

| Componente | Scelta |
| --- | --- |
| Activation | ELU |
| Dropout | 0.10 |
| Normalization | None nel primissimo smoke test; se serve stabilita, `LayerNorm(128)` o `BatchNorm1d(128)` in entrambi |

Motivo: ELU e default nei GAT TimeGNN; dropout 0.10 e leggero; evitare normalizzazione inizialmente riduce complessita. Se si introduce norm, deve essere identica in entrambi i modelli.

## Attention Heads

Con `GATConv(out_channels=32, heads=4, concat=True)`:

```text
output_dim_per_layer = out_channels * heads = 32 * 4 = 128
```

Esempio:

```text
Layer 1:
input [N,15]
out_channels=32
heads=4
concat=True
output [N,128]

Layer 2:
input [N,128]
out_channels=32
heads=4
concat=True
output [N,128]
```

Per mantenere fairness, il modello timing deve produrre la stessa shape `[N,128]` a ogni layer.

## Stima Parameter Count

Stima approssimativa per configurazione:

- `input_dim=15`;
- `edge_dim=5`;
- `hidden_per_head=32`;
- `heads=4`;
- `node_hidden_dim=128`;
- `classifier_hidden=128`;
- `num_classes circa 1786`.

| Componente | Parametri indicativi |
| --- | ---: |
| GAT layer 1 node projection | `15 * 128 = 1,920` |
| GAT layer 1 edge projection/attention | qualche centinaio / migliaio |
| GAT layer 2 node projection | `128 * 128 = 16,384` |
| GAT layer 2 edge projection/attention | qualche centinaio / migliaio |
| Classifier `132 -> 128` | `~17k` |
| Classifier `128 -> 1786` | `~230k` |
| Totale no-timing | ordine `270k-350k` |
| Extra timing | da `0` a poche centinaia/migliaia, in base alla modulazione |

Componenti costosi:

- classifier finale verso molte classi;
- heads e hidden dim nei layer GAT;
- eventuali proiezioni edge/timing ripetute per layer.

La dimensione proposta e compatibile con primi test su RTX A2000 6GB. La memoria sara dominata da batch size, edge count totale e attivazioni, non dal numero assoluto di parametri.

## File Structure Futura

Non creare ora, ma nella fase successiva suggerisco:

```text
src/models/
├── __init__.py
├── chess_gat.py
└── time_aware_gat.py
```

Possibile contenuto:

| File | Contenuto futuro |
| --- | --- |
| `src/models/chess_gat.py` | `ChessGATNoTiming`, wrapper graph-level, classifier head condivisibile. |
| `src/models/time_aware_gat.py` | Layer timing-aware chess-specific ispirato a `TimeAwareETGATConv`, `ChessGATWithTiming`. |
| `src/models/__init__.py` | Export classi principali. |
| `tests/test_chess_gat_models.py` | Test forward/backward/fairness. |

Una alternativa ancora piu ordinata:

```text
src/models/chess_gat.py
src/models/layers.py
src/models/config.py
```

Ma per iniziare conviene evitare troppi file.

## Forward API

API desiderata:

```python
logits = model(batch)
```

Per `ChessGATNoTiming`, attributi richiesti:

```text
batch.x
batch.edge_index
batch.edge_attr
batch.global_features
batch.batch
```

Per `ChessGATWithTiming`, attributi richiesti:

```text
batch.x
batch.edge_index
batch.edge_attr
batch.edge_time
batch.global_features
batch.batch
```

`ChessGATNoTiming` non deve leggere o richiedere `edge_time`.

`ChessGATWithTiming` deve fallire in modo chiaro se `edge_time` manca, per evitare training accidentalmente identico al modello A.

## Test Da Implementare Dopo

### MODEL A

- single graph forward;
- batch forward;
- output shape `[B,C]`;
- logits finiti, nessun NaN/Inf;
- backward works con `CrossEntropyLoss`;
- accetta `edge_attr=[E,5]` multilabel.

### MODEL B

- stessi test del modello A;
- `edge_time` richiesto;
- `edge_time` shape `[E,1]`;
- timing influenza realmente l'output a parita di pesi/input;
- logits finiti;
- backward works.

### Fairness

- stesso output shape;
- stessa hidden dim;
- parameter count comparabile;
- stessi input eccetto `edge_time`;
- stesso classifier;
- stessa pooling function.

### Edge/global

- nessuna conversione single-label di `edge_attr`;
- `global_features` concatenate correttamente;
- batching PyG produce `global_features=[B,4]`;
- `batch.y` compatibile con `[B]`.

## Timing Data Requirements

Non generare timing ora. Il futuro preprocessing timing dovra produrre:

```text
edge_time: torch.FloatTensor
shape: [E,1]
same edge order as edge_index
finite values only
no NaN/Inf
normalized range recommended: [0,1]
dtype: float32
```

Requisiti:

- `edge_time[k]` deve riferirsi esattamente all'arco `edge_index[:, k]`;
- se gli edge vengono riordinati o aggregati, `edge_time` deve essere riordinato/aggregato nello stesso modo;
- val/test devono usare la stessa normalizzazione imparata o definita sul train, se il timing deriva da distribuzioni statistiche.

## Rischio Principale: Semantica Del Timing

TimeGNN nasce da event logs: un edge rappresenta una transizione temporale fra evento `i` ed evento `i+1`. Nei nostri grafi chess, invece, un edge rappresenta relazioni tattiche tra caselle:

- legal move;
- attack;
- defend;
- pin;
- check line.

Non esiste automaticamente un tempo naturale per ogni edge tattico.

### Opzioni per rappresentare il timing

| Opzione | Pro | Contro | Compatibilita TimeGNN | Valore scientifico |
| --- | --- | --- | --- | --- |
| A. stesso move time replicato su tutti gli edge | Facile; edge-level shape compatibile | Se il valore e uguale per tutti gli edge dello stesso grafo, il decay e costante: non cambia l'attention relativa tra archi, modifica solo scala globale del grafo/layer | Alta formalmente | Basso per temporal attention |
| B. timing node-level | Puo assegnare tempo/importanza a caselle o pezzi | Serve definire semantica: tempo per pezzo? per destinazione? non nativo nei layer TimeGNN | Media con wrapper | Medio se ben motivato |
| C. timing graph-level concatenato dopo pooling | Semanticamente corretto se il timing e "time to move" del puzzle o della posizione | Non testa temporal attention sugli edge; il modello B differisce nella head/input globale | Non usa TimeGNN temporal layer | Alto se il timing disponibile e davvero graph-level |
| D. timing edge-level per modulare attention | Allineato a TimeGNN e al confronto GAT timing/no timing | Serve costruire un tempo per edge non arbitrario; rischio semantico alto | Alta con custom TimeAware layer | Alto solo se edge_time ha significato tattico |
| E. hybrid: graph-level move time + edge-level allocation tattica | Permette usare timing globale senza renderlo costante su tutti gli edge | Richiede formula motivata, per esempio distribuire tempo su edge in base a ruolo tattico o salienza | Media/alta con adapter | Potenzialmente alto, ma non pronto senza progetto specifico |

### Punto cruciale

Se un singolo move time viene replicato su tutti gli edge:

```text
edge_time[k] = same_value_for_graph
```

allora `exp(-lambda * edge_time)` e uguale per tutti gli edge del grafo. Questo non cambia le preferenze relative fra archi nello stesso neighborhood. Di fatto il timing diventa una scala globale, non un segnale temporale locale. Per una ablation "temporal attention" sarebbe debole.

## Decisione Finale Sul Timing

Raccomandazione:

```text
TIMING_REPRESENTATION = HYBRID, not ready as pure edge-level yet
```

Spiegazione:

- Se il timing sintetico rappresenta solo "tempo della mossa" o "tempo del puzzle", la rappresentazione piu semanticamente corretta e graph-level, concatenata dopo pooling.
- Se vogliamo usare TimeGNN-style temporal attention, dobbiamo progettare un `edge_time` con variazione per edge e significato tattico. Esempi concettuali:
  - tempo allocato agli edge legal_move in base alla probabilita/salienza della mossa;
  - edge_time diverso per attacchi, difese, pin e check_line secondo una regola derivata e dichiarata;
  - timing relativo a source/destination square o piece mobility, ma solo se motivato.
- Una versione hybrid potrebbe usare:
  - `graph_time=[B,1]` come feature globale reale/sintetica;
  - `edge_time=[E,1]` derivato da una decomposizione tattica per testare temporal attention.

Quindi:

```text
MODEL_A_IMPLEMENTATION_STATUS = HISTORICAL_READY_MARKER_SUPERSEDED_BY_FROZEN_BASELINE
READY_TO_IMPLEMENT_MODEL_B = NO, finche non fissiamo la semantica di edge_time
TIMING_SEMANTICS_BLOCKER = YES
```

## Training Compatibility

Il design e compatibile con `torch_geometric.loader.DataLoader`.

Batch attuale atteso:

```python
batch.x               # [64B,15]
batch.edge_index      # [2,E_total]
batch.edge_attr       # [E_total,5]
batch.global_features # [B,4]
batch.y               # [B]
batch.batch           # [64B]
```

Per il modello B futuro:

```python
batch.edge_time       # [E_total,1]
```

Accorgimenti PyG:

- `global_features` e gia `[1,4]` per grafo, quindi il batching produce `[B,4]`;
- `edge_time` deve essere un attributo edge-level con prima dimensione pari a `edge_index.shape[1]`;
- se si salva `edge_time` in `Data`, PyG concatenera lungo la dimensione 0 come per `edge_attr`;
- il training loop deve spostare tutto su device con `batch = batch.to(device)`;
- `batch.y` deve essere `long` e shape `[B]`.

## Raccomandazione Finale

### MODEL A

```text
ARCHITECTURE = Chess-specific edge-aware GAT graph classifier
BACKBONE = 2-layer GATConv(edge_dim=5), hidden_per_head=32, heads=4
USES_TACTICAL_EDGE_ATTR = YES
USES_EVENT_IDS = NO
POOLING = global_mean_pool
GLOBAL_FEATURES = concatenate after pooling
CLASSIFIER = Linear(128+4,128) -> ELU -> Dropout(0.10) -> Linear(128,num_classes)
OUTPUT = logits [B,num_classes]
```

### MODEL B

```text
ARCHITECTURE = Chess-specific tactical + timing-aware GAT graph classifier
BACKBONE = same 2-layer structure as MODEL A, but attention also receives edge_time
USES_TACTICAL_EDGE_ATTR = YES
TIMING_REPRESENTATION = HYBRID recommended; pure EDGE_LEVEL only after semantic design
USES_EVENT_IDS = NO
POOLING = global_mean_pool
GLOBAL_FEATURES = same as MODEL A
CLASSIFIER = same as MODEL A
OUTPUT = logits [B,num_classes]
```

### FAIRNESS

```text
SAME_BACKBONE_STRUCTURE = YES
SAME_HIDDEN_DIMS = YES
SAME_CLASSIFIER = YES
SAME_TACTICAL_EDGE_ATTR = YES
ONLY_MAIN_DIFFERENCE_IS_TIMING = YES, after edge_time semantics are fixed
```

### IMPLEMENTATION DECISION

```text
MODEL_A_IMPLEMENTATION_STATUS = HISTORICAL_READY_MARKER_SUPERSEDED_BY_FROZEN_BASELINE
READY_TO_IMPLEMENT_MODEL_B = NO
TIMING_SEMANTICS_BLOCKER = YES
```

Motivazione finale:

- `ChessGATNoTiming` e pronto da implementare: il dataset attuale contiene gia tutto cio che serve.
- `ChessGATWithTiming` e architetturalmente chiaro, ma non scientificamente pronto finche non definiamo cosa significa un tempo per edge tattico chess.
- Non bisogna usare `event_ids` solo per aderire ai modelli TimeGNN originari.
- Non bisogna comprimere `edge_attr=[E,5]` in `edge_type` single-label.
- Il modo piu fair e costruire due wrapper chess-specific piccoli, usando TimeGNN come riferimento per GAT/time-aware attention, non come API high-level diretta.

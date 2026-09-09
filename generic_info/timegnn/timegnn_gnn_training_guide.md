# TimeGNN GNN/GAT Training Guide per il Progetto Chess

> Nota di stato: questa guida rimane utile per l'integrazione futura di TimeGNN e del modello timing-aware. Il baseline no-timing definitivo è ora documentato in [`model_a_no_timing.md`](model_a_no_timing.md).

## Scopo

Questa relazione tecnica analizza `TimeGNN-main/` con focus esclusivo su modelli GNN/GAT, layer, forward pass e training. Non sostituisce `generic_info/timegnn/timegnn_info.md`: qui l'obiettivo e operativo, cioe capire cosa possiamo riusare per due futuri modelli chess:

- **MODELLO A**: GAT/GNN senza timing.
- **MODELLO B**: GAT/GNN con timing sintetici.

Il nostro dataset attuale e gia in formato PyTorch Geometric:

```python
Data(
    x=[64, 15],
    edge_index=[2, E],
    edge_attr=[E, 5],
    y=graph_level_move_target,
    global_features=[1, 4],
)
```

Il task chess e **graph-level classification** su circa 1,786 classi mossa. TimeGNN nasce invece per event log/process mining, quindi molte API sono pensate per next-event prediction token-level.

## Fonti Analizzate

| Area | File |
| --- | --- |
| Guida generale gia presente | `generic_info/timegnn/timegnn_info.md` |
| Progetto chess | `README.md`, `src/graph/graph_builder.py`, `src/graph/pyg_dataset.py`, `src/validate_representations.py` |
| Modelli TimeGNN | `TimeGNN-main/src/timegnn/models/*.py` |
| Ricette training | `TimeGNN-main/src/timegnn/recipes/*.py` |
| Dati/collate PyG | `TimeGNN-main/src/timegnn/data/pyg.py`, `TimeGNN-main/src/timegnn/data/transformer.py`, `TimeGNN-main/src/timegnn/data/encoding.py` |
| API pubblica | `TimeGNN-main/src/timegnn/api.py`, `TimeGNN-main/src/timegnn/sklearn.py`, `TimeGNN-main/src/timegnn/__init__.py` |
| Training utilities | `TimeGNN-main/src/timegnn/models/training.py`, `TimeGNN-main/src/timegnn/train/early_stopping.py` |
| Test TimeGNN | `TimeGNN-main/tests/test_models_recipes.py`, `TimeGNN-main/tests/test_api_naming.py` |
| Packaging | `TimeGNN-main/pyproject.toml`, `TimeGNN-main/README.md` |

## Inventario Completo GNN/GAT

| Classe/funzione | File | Tipo | Input richiesti | Output | Livello | Timing | Edge features | `event_ids` | Compat. chess |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `DualGATModel` | `src/timegnn/models/gat_basic.py` | GAT dual-path | `data.x`, `data.edge_index`, `data.edge_attr`, `data.event_ids` | logits per nodo `[N, output_dim]` | Node-level | NO | SI, passa `edge_attr` a `GATConv`, default `edge_dim=1` | Obbligatorio | MEDIUM |
| `TimeAwareGATConv` | `src/timegnn/models/gat_time_decay.py` | Layer temporal GAT | `x`, `edge_index`, `edge_attr` usato come tempo | embedding nodo `[N, heads*out]` | Node-level layer | SI | Solo tempo scalare/vettore semplice | NO | MEDIUM |
| `DualGATTimeAwareModel` | `src/timegnn/models/gat_time_decay.py` | GAT dual-path temporal | `data.x`, `data.edge_index`, `data.time`, `data.event_ids` | logits per nodo `[N, output_dim]` | Node-level | SI | NO tactical; `data.time` entra come decay | Obbligatorio | MEDIUM |
| `DualGAT2EdgesModel` | `src/timegnn/models/gat_status_emb.py` | Edge-aware GAT | `data.x`, `data.edge_index`, `data.edge_type`, `data.edge_time_diff`, `data.event_ids` | logits per nodo `[N, output_dim]` | Node-level | PARZIALE: usa `edge_time_diff` come edge feature, non custom decay | SI, ma edge type discreto + time diff | Obbligatorio | LOW/MEDIUM |
| `TimeAwareETGATConv` | `src/timegnn/models/gat_time_decay_status_emb.py` | Temporal edge-type GAT layer | `x`, `edge_index`, `edge_attr`, `time` | embedding nodo `[N, heads*out]` | Node-level layer | SI | SI, edge embedding vettoriale | NO | MEDIUM |
| `DualGATTimeAwareETModel` | `src/timegnn/models/gat_time_decay_status_emb.py` | Temporal edge-aware GAT | `data.x`, `data.edge_index`, `data.edge_type`, `data.time`, `data.event_ids` | logits per nodo `[N, output_dim]` | Node-level | SI | SI, ma `edge_type` discreto embedded | Obbligatorio | MEDIUM con adapter |
| `PrefixGCNClassifier` | `src/timegnn/models/prefix_gcn.py` | GCN graph-level | `data.x`, `data.edge_index`, `data.edge_attr`, `data.event_ids`, `sequence_features` | logits grafo `[B, output_dim]` | Graph-level | NO diretto | `edge_attr` come `edge_weight` per `GCNConv` | Obbligatorio | MEDIUM |
| `train_gat_outcome` | `src/timegnn/recipes/gat_outcome.py` | Training graph-level wrapper | Event log dataframe | modello + history | Graph-level recipe | Dipende da `mode` | Dipende da `mode` | Generati dal transformer | MEDIUM come pattern |

### Dipendenze Reali

Da `TimeGNN-main/pyproject.toml`:

| Tipo | Dipendenze |
| --- | --- |
| Python | `>=3.9` |
| Base | `numpy>=1.24`, `pandas>=2.0`, `torch>=2.1`, `scikit-learn>=1.3` |
| Optional PyG | `torch-geometric>=2.4`, `torch-scatter>=2.1.2`, `torch-sparse>=0.6.18` |
| Analysis | `matplotlib`, `seaborn`, `scipy`, `nltk`, `pyxdameraulevenshtein` |

**Certo dal codice TimeGNN**: i modelli moderni usano PyTorch e PyTorch Geometric. Non ho trovato uso moderno di DGL o TensorFlow; TensorFlow compare solo nel codice legacy.

## Analisi Dei Layer Principali

### `DualGATModel`

File: `TimeGNN-main/src/timegnn/models/gat_basic.py`

Costruttore reale:

```python
DualGATModel(
    num_event_features: int,
    num_embedding_features: int,
    embedding_dims: int,
    gat_hidden_dim_event: int,
    gat_hidden_dim_embed: int,
    gat_hidden_dim_concat: int,
    output_dim: int,
    num_heads: int,
    edge_dim: int = 1,
    num_layers: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "elu",
)
```

Forward reale:

```python
def forward(self, data_event):
    edge_attr = data_event.edge_attr
    x_embed = self.embedding(data_event.event_ids.view(-1))
    x_embed = self._run_path(..., x_embed, data_event.edge_index, edge_attr)
    x_event = self._run_path(..., data_event.x, data_event.edge_index, edge_attr)
    x = torch.cat([x_embed, x_event], dim=1)
    x = self._run_path(..., x, data_event.edge_index, edge_attr)
    out = self.fc(x)
    return out
```

Shape attese:

| Attributo | Shape attesa |
| --- | --- |
| `data_event.x` | `[N, num_event_features]` |
| `data_event.event_ids` | `[N]` o `[N, 1]`, valori `0 <= id < num_embedding_features` |
| `data_event.edge_index` | `[2, E]` |
| `data_event.edge_attr` | `[E, edge_dim]` |
| output | `[N, output_dim]` |

**Certo dal codice TimeGNN**: non contiene pooling e non contiene classification head graph-level. La `fc` e applicata a ogni nodo.

**Per chess**: possiamo passare `x=[64,15]`, `edge_index` e `edge_attr=[E,5]` impostando `edge_dim=5`, ma dobbiamo aggiungere `event_ids`. Possibili mapping:

- square IDs `0..63`: semplice e stabile, ma introduce embedding assoluto della casella;
- piece IDs: non naturale per caselle vuote e cambia con la posizione;
- nessun event embedding: richiede wrapper/custom model che usa solo il path `x`.

Raccomandazione: se si vuole riusare il dual-path quasi invariato, usare **square IDs 0..63** come `event_ids`. Se si vuole un modello piu pulito per chess, usare direttamente `torch_geometric.nn.GATConv` o estrarre solo il pattern dei layer, evitando il ramo `event_ids`.

### `TimeAwareGATConv`

File: `TimeGNN-main/src/timegnn/models/gat_time_decay.py`

Costruttore reale:

```python
TimeAwareGATConv(
    in_channels: int,
    out_channels: int,
    heads: int = 1,
    concat: bool = True,
    lambda_decay: float = 0.1,
    **kwargs,
)
```

Forward reale:

```python
def forward(self, x, edge_index, edge_attr=None, return_attention=False):
    x = self.lin(x)
    x = x.view(-1, self.heads, self.out_channels)
    out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
    ...
```

Uso del timing:

```python
time_diff = edge_attr
decay = torch.exp(-self.lambda_decay * time_diff).unsqueeze(-1)
alpha = alpha * decay
```

**Certo dal codice TimeGNN**:

- `edge_attr` e interpretato come tempo/differenza temporale.
- Il timing entra direttamente nell'attenzione tramite decay esponenziale.
- Non supporta tactical `edge_attr=[E,5]` come feature di arco generiche in modo separato.
- Non fa pooling.
- Non e graph-level.

**Nota tecnica**: il layer custom ridefinisce attention/message rispetto a `GATConv`; dal codice non risulta una normale softmax esplicita sull'attenzione dopo il decay. Conviene testarlo bene quando verra usato.

### `DualGATTimeAwareModel`

File: `TimeGNN-main/src/timegnn/models/gat_time_decay.py`

Costruttore reale:

```python
DualGATTimeAwareModel(
    num_event_features: int,
    num_embedding_features: int,
    embedding_dims: int,
    gat_hidden_dim_event: int,
    gat_hidden_dim_embed: int,
    gat_hidden_dim_concat: int,
    output_dim: int,
    num_heads: int,
    lambda_decay: float,
    num_layers: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "elu",
)
```

Forward:

```python
def forward(self, data_event, return_attention: bool = False):
    edge_attr = data_event.time
    edge_index = data_event.edge_index
    ...
    out = self.fc(x)
    return out
```

**Certo dal codice TimeGNN**: `data_event.time` deve essere compatibile con gli archi, perche viene passato ai layer come `edge_attr`. Il nome `time` puo trarre in inganno: qui funziona come edge-level time/delta tensor.

**Per chess**: servirebbe creare un attributo separato, per esempio `graph.edge_time` o `graph.time`, con shape compatibile con `E`. Non bisogna mescolare il timing dentro le 5 feature tattiche.

### `TimeAwareETGATConv`

File: `TimeGNN-main/src/timegnn/models/gat_time_decay_status_emb.py`

Costruttore reale:

```python
TimeAwareETGATConv(
    in_channels: int,
    out_channels: int,
    heads: int = 1,
    concat: bool = True,
    lambda_decay: float = 0.1,
    edge_type_dim: Optional[int] = None,
    **kwargs,
)
```

Forward reale:

```python
def forward(self, x, edge_index, edge_attr=None, time=None, return_attention=False):
    x = self.lin(x)
    x = x.view(-1, self.heads, self.out_channels)
    out = self.propagate(..., edge_attr=edge_attr, time=time)
```

Uso di timing e edge type:

```python
edge_type_score = torch.einsum("ed,hd->eh", edge_attr, self.edge_type_att)
alpha = alpha + edge_type_score
decay = torch.exp(-self.lambda_decay * time).unsqueeze(-1)
alpha = alpha * decay
```

**Certo dal codice TimeGNN**:

- `edge_attr` non e un vettore tattico arbitrario: nel modello wrapper viene creato da `nn.Embedding(edge_type)`.
- `time` e separato da `edge_attr`.
- Il layer richiede `edge_type_dim`.
- Non supporta direttamente edge multilabel discreti multipli; supporta un vettore edge_attr gia embedded.

### `DualGATTimeAwareETModel`

File: `TimeGNN-main/src/timegnn/models/gat_time_decay_status_emb.py`

Costruttore reale:

```python
DualGATTimeAwareETModel(
    num_event_features: int,
    num_embedding_features: int,
    embedding_dims: int,
    gat_hidden_dim_event: int,
    gat_hidden_dim_embed: int,
    gat_hidden_dim_concat: int,
    output_dim: int,
    num_heads: int,
    num_edge_types: int,
    edge_type_dim: int,
    lambda_decay: float,
    num_layers: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "elu",
)
```

Forward:

```python
def forward(self, data_event, return_attention: bool = False):
    edge_type = data_event.edge_type
    edge_attr = self.edge_type_emb(edge_type)
    time = data_event.time
    edge_index = data_event.edge_index
    ...
    out = self.fc(x)
    return out
```

**Certo dal codice TimeGNN**:

- Richiede `data_event.edge_type` con dtype long e valori `0 <= edge_type < num_edge_types`.
- Richiede `data_event.time`, edge-level.
- Usa `event_ids`.
- Produce logits node-level.
- Non contiene pooling graph-level.

**Per chess**: il nostro `edge_attr=[E,5]` e multilabel: un arco puo essere contemporaneamente `legal_move` e `attack`, ecc. Convertirlo a un singolo `edge_type` perderebbe informazione o richiederebbe combinazioni di label come categorie. Questa e una scelta progettuale, non un passaggio automatico.

### `DualGAT2EdgesModel`

File: `TimeGNN-main/src/timegnn/models/gat_status_emb.py`

Forward:

```python
edge_type = data_event.edge_type
edge_time = data_event.edge_time_diff
type_vec = self.edge_type_emb(edge_type)
edge_attr = torch.cat([edge_time, type_vec], dim=-1)
...
out = self.fc(x)
```

**Certo dal codice TimeGNN**: usa `GATConv` standard con `edge_attr = [edge_time_diff, edge_type_embedding]`. Timing qui e solo una feature concatenata, non un decay custom nell'attention come `TimeAwareGATConv`.

### `PrefixGCNClassifier`

File: `TimeGNN-main/src/timegnn/models/prefix_gcn.py`

Forward:

```python
def forward(self, data, sequence_features):
    d = self.embedding(data.event_ids.squeeze(-1))
    d = self._run_path(..., d, data.edge_index, data.edge_attr)
    f = data.x
    f[f == -1] = 0
    f = self._run_path(..., f, data.edge_index, data.edge_attr)
    x = torch.cat([d, f], dim=1)
    x = self._run_path(..., x, data.edge_index, data.edge_attr)
    graph_emb = global_mean_pool(x, data.batch)
    seq_out = self.seq_proj(sequence_features)
    out = self.classifier(...)
    return out
```

**Certo dal codice TimeGNN**:

- E il modello piu chiaramente graph-level.
- Usa `global_mean_pool` sugli embedding nodali.
- Accetta feature graph-level esterne come `sequence_features`.
- E GCN, non GAT.
- Passa `data.edge_attr` a `GCNConv` come `edge_weight`, quindi si aspetta valori compatibili con pesi scalari o formato supportato da PyG; il nostro `edge_attr=[E,5]` non e direttamente compatibile come edge weight scalare.

**Per chess**: il pattern `node embeddings -> global_mean_pool -> concat global_features -> classifier` e molto utile, ma la classe specifica non e ideale se vogliamo GAT e edge_attr multilabel.

## Training Reale Nella Libreria

### Next-event GAT recipes

File:

- `TimeGNN-main/src/timegnn/recipes/gat_basic.py`
- `TimeGNN-main/src/timegnn/recipes/gat_time_decay.py`
- `TimeGNN-main/src/timegnn/recipes/gat_status_emb.py`
- `TimeGNN-main/src/timegnn/recipes/gat_time_decay_status_emb.py`

Flusso reale:

```text
pandas event log
-> encoding event/sequence/time
-> PyG Data list
-> CustomDataset
-> torch.utils.data.DataLoader + custom_collate_fn
-> model(event_data)
-> logits per nodo
-> flatten logits/labels
-> mask labels != -1
-> CrossEntropyLoss(ignore_index=-1)
-> backward
-> Adam
-> evaluate_epoch
-> EarlyStopping
```

Valori reali dalle config:

| Config | batch | lr | epochs | patience | dropout | heads | layers | loss | optimizer | scheduler | weight decay | checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `GATBasicConfig` | 16 | `1e-3` | 10 | 3 | 0.0 | 4 | 1 | `CrossEntropyLoss(ignore_index=-1)` | Adam | Nessuno | Nessuno | Nessuno |
| `GATTimeDecayConfig` | 16 | `1e-3` | 10 | 3 | 0.0 | 4 | 1 | `CrossEntropyLoss(ignore_index=-1)` | Adam | Nessuno | Nessuno | Nessuno |
| `GATStatusEmbConfig` | 16 | `1e-3` | 10 | 3 | 0.0 | 4 | 1 | `CrossEntropyLoss(ignore_index=-1)` | Adam | Nessuno | Nessuno | Nessuno |
| `GATTimeDecayStatusConfig` | 16 | `1e-3` | 10 | 3 | 0.0 | 4 | 1 | `CrossEntropyLoss(ignore_index=-1)` | Adam | Nessuno | Nessuno | Nessuno |
| `PrefixGCNConfig` | 32 | `1e-3` | 10 | 3 | 0.0 | n/a | 1 | `CrossEntropyLoss()` | Adam | Nessuno | Nessuno | Nessuno |
| `GATOutcomeConfig` | 16 | `1e-3` | 10 | 3 | 0.0 | 4 | 1 | `CrossEntropyLoss()` | Adam | Nessuno | Nessuno | Nessuno |

Device handling:

```python
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
event_data = event_data.to(device)
labels = labels.to(device)
```

Early stopping:

- `EarlyStopping(patience=cfg.patience, delta=cfg.delta)`.
- Monitora `test_loss` nelle ricette, non una validation separata.
- Non salva automaticamente il best checkpoint.

Gradient clipping, scheduler, class weighting, checkpoint/save/load: **non presenti nel training GAT moderno**.

### Graph-level outcome recipe

File: `TimeGNN-main/src/timegnn/recipes/gat_outcome.py`

Pattern reale:

```python
node_logits = model(event_data)
graph_logits = global_mean_pool(node_logits, event_data.batch)
loss = criterion(graph_logits, labels)
```

**Certo dal codice TimeGNN**: questa e graph-level classification, ma fa pooling sui **logits per nodo**, non sugli embedding prima della classifier. Per chess, pooling sugli embedding e poi classifier e probabilmente piu corretto.

## TimeGNN API / Commands We Need

Non ho trovato CLI shell ufficiale per training. Gli entrypoint sono Python API.

### No timing: `DualGATModel`

```python
from timegnn.models.gat_basic import DualGATModel

model = DualGATModel(
    num_event_features=15,
    num_embedding_features=64,
    embedding_dims=32,
    gat_hidden_dim_event=64,
    gat_hidden_dim_embed=64,
    gat_hidden_dim_concat=128,
    output_dim=num_move_classes,
    num_heads=4,
    edge_dim=5,
    num_layers=2,
    dropout=0.1,
)

node_logits = model(batch)
```

Input aggiuntivo da creare per chess:

```python
batch.event_ids  # long tensor [N], probabilmente square IDs ripetuti per grafo
```

Serve wrapper graph-level:

```python
node_embeddings_or_logits = model(batch)
graph_repr = global_mean_pool(node_embeddings_or_logits, batch.batch)
logits = classifier(torch.cat([graph_repr, batch.global_features], dim=1))
```

Questo snippet e una **proposta di adattamento chess**, non una funzione nativa TimeGNN.

### Timing: `DualGATTimeAwareModel`

```python
from timegnn.models.gat_time_decay import DualGATTimeAwareModel

model = DualGATTimeAwareModel(
    num_event_features=15,
    num_embedding_features=64,
    embedding_dims=32,
    gat_hidden_dim_event=64,
    gat_hidden_dim_embed=64,
    gat_hidden_dim_concat=128,
    output_dim=num_move_classes,
    num_heads=4,
    lambda_decay=0.01,
    num_layers=2,
    dropout=0.1,
)

batch.time = edge_time  # [E] or [E, 1], edge-level timing
node_logits = model(batch)
```

Input aggiuntivi:

- `batch.event_ids`;
- `batch.time` edge-level, separato da `batch.edge_attr`.

### Timing + edge type: `DualGATTimeAwareETModel`

```python
from timegnn.models.gat_time_decay_status_emb import DualGATTimeAwareETModel

model = DualGATTimeAwareETModel(
    num_event_features=15,
    num_embedding_features=64,
    embedding_dims=32,
    gat_hidden_dim_event=64,
    gat_hidden_dim_embed=64,
    gat_hidden_dim_concat=128,
    output_dim=num_move_classes,
    num_heads=4,
    num_edge_types=num_edge_types,
    edge_type_dim=32,
    lambda_decay=0.01,
    num_layers=2,
)

batch.edge_type = edge_type.long()  # [E]
batch.time = edge_time             # [E] or [E, 1]
node_logits = model(batch)
```

**Certo dal codice TimeGNN**: `edge_type` e un singolo indice per arco. Non accetta direttamente `edge_attr=[E,5]` multilabel.

## Compatibilita Dei Candidati per Chess

| Candidato | Compatibility | Adaptation effort | Suitability for chess | Motivazione |
| --- | --- | --- | --- | --- |
| `DualGATModel` | MEDIUM | MEDIUM | HIGH | Usa GAT e puo accettare `edge_attr=[E,5]` con `edge_dim=5`, ma richiede `event_ids` e wrapper graph-level. |
| `DualGATTimeAwareModel` | MEDIUM | MEDIUM/HIGH | HIGH per ablation timing | Backbone simile a `DualGATModel`, ma usa `data.time` al posto di tactical edge_attr; serve separare timing e relazioni tattiche. |
| `DualGATTimeAwareETModel` | MEDIUM | HIGH | MEDIUM/HIGH | Integra timing + edge type, ma richiede edge type discreti single-label; mismatch col nostro multilabel. |
| `DualGAT2EdgesModel` | LOW/MEDIUM | HIGH | MEDIUM | Edge type + edge time diff, ma niente decay custom; richiede conversione edge multilabel a tipo discreto. |
| `PrefixGCNClassifier` | MEDIUM | MEDIUM | MEDIUM | Gia graph-level e concatena feature globali, ma e GCN e tratta `edge_attr` come edge_weight, non come multilabel. |
| `train_gat_outcome` | MEDIUM come pattern | HIGH se usato diretto | MEDIUM | Graph-level reale, ma input event-log e pooling sui logits. Utile da imitare, non plug-and-play. |

## Timing / Temporal Component

### Cosa e certo dal codice TimeGNN

| Componente | Dove entra il tempo | Shape implicita | Effetto |
| --- | --- | --- | --- |
| `DualGATTimeAwareModel` | `data_event.time` passato come `edge_attr` a `TimeAwareGATConv` | compatibile con edge count `E` | `exp(-lambda_decay * time)` moltiplica attention logits |
| `DualGAT2EdgesModel` | `data_event.edge_time_diff` concatenato a edge type embedding | `[E, 1]` | feature di arco per `GATConv` standard |
| `DualGATTimeAwareETModel` | `data_event.time` separato da `edge_type` | compatibile con `E` | edge type score sommato ad attention, poi time decay |
| Transformer TimeGNN | da timestamp event-log | sequenze temporali event-log | crea edge time diffs tra eventi consecutivi |

### Per il progetto chess

**Proposta di adattamento**:

```python
edge_attr = tactical_features      # [E, 5], non temporale
edge_time = synthetic_timing       # [E, 1], temporale
global_features = graph_features   # [B, 4] dopo batching
```

Non bisogna fare:

```python
edge_attr = torch.cat([tactical_features, edge_time], dim=1)
```

se l'obiettivo e confrontare in modo pulito no timing vs timing con differenza controllata. Meglio tenere timing in un attributo separato (`time`, `edge_time` o nome wrapper interno).

## Edge Attr Multilabel vs Edge Type TimeGNN

Il nostro `edge_attr=[E,5]` e multilabel:

- `legal_move`;
- `attack`;
- `defend`;
- `pin`;
- `check_line`.

TimeGNN edge-type models invece usano:

```python
edge_type = data_event.edge_type       # [E], long
edge_attr = self.edge_type_emb(edge_type)
```

Quindi `TimeAwareETGATConv` puo usare edge features embedded, ma `DualGATTimeAwareETModel` costruisce quelle feature da un **singolo edge_type discreto**.

Strategie possibili, tutte da decidere prima dell'implementazione:

| Strategia | Pro | Contro |
| --- | --- | --- |
| Ignorare `edge_attr` nei primi modelli | Ablation timing piu pulita e semplice | Perde relazioni tattiche gia estratte |
| Usare `DualGATModel(edge_dim=5)` per no timing | Usa tactical multilabel direttamente | Il timing model comparabile non usa lo stesso canale edge_attr |
| Creare wrapper custom edge-aware per `[E,5]` + timing separato | Mantiene tutta l'informazione | Richiede codice nuovo |
| Convertire multilabel a edge type combinatorio | Compatibile con `edge_type` discreto | Aumenta tipi, perde struttura multilabel, categorie rare |
| Prioritizzare una sola label per edge | Semplice | Perde informazione e introduce arbitrarieta |

Raccomandazione: per una ablation seria, non comprimere subito `edge_attr=[E,5]` a una sola label. Meglio iniziare con backbone GAT comparabile e aggiungere un adapter chiaro per timing.

## Event IDs

### Cosa rappresentano in TimeGNN

Nel codice TimeGNN, `event_ids` sono ID interi degli eventi del log:

- creati da `encode_label_event`;
- usati da `nn.Embedding`;
- shape `[N]` o `[N,1]`;
- range `0 <= id < num_embedding_features`;
- obbligatori nei modelli dual-path (`DualGATModel`, `DualGATTimeAwareModel`, `DualGAT2EdgesModel`, `DualGATTimeAwareETModel`, `PrefixGCNClassifier`).

### Per chess

Opzioni:

| Opzione | Valutazione |
| --- | --- |
| Evitare `event_ids` | Pulito, ma richiede wrapper/custom model. |
| Square IDs `0..63` | Stabile e naturale per 64 nodi/caselle; raccomandato se si usa dual-path TimeGNN quasi invariato. |
| Piece IDs | Non stabile per caselle vuote e non identifica la posizione sulla board. |
| Move IDs | Non node-level; non adatto a `event_ids`. |

Raccomandazione: per prototipo rapido con modelli TimeGNN, usare square IDs. Per modello finale piu rigoroso, costruire wrapper GAT che non dipenda da `event_ids`, oppure rendere il ramo embedding opzionale.

## Global Graph Features

**Certo dal codice TimeGNN**:

- I GAT dual-path non supportano nativamente `global_features`.
- `PrefixGCNClassifier` supporta un tensore graph-level `sequence_features`.
- `gat_outcome` usa graph-level pooling, ma non concatena `global_features` chess.

**Pattern proposto per chess**:

```text
node embeddings [N_total, H]
-> global_mean_pool(node_embeddings, batch) = [B, H]
-> concat global_features [B, 4]
-> MLP/classification head
-> logits [B, num_move_classes]
```

Questo wrapper e necessario per usare `global_features=[1,4]` del nostro dataset.

## Recommended Architecture

### MODELLO A: ChessGATNoTiming

**Raccomandazione**:

```text
RECOMMENDED_NO_TIMING_MODEL = DualGATModel backbone + graph-level wrapper
```

Struttura proposta:

```text
x=[64,15]
edge_index=[2,E]
edge_attr=[E,5]
event_ids=[64] opzionale/square IDs se si riusa DualGATModel
-> GAT backbone
-> node embeddings/logits intermediate
-> global_mean_pool
-> concat global_features=[B,4]
-> MLP
-> logits=[B,1786 circa]
```

Parametri indicativi per confronto:

| Parametro | Valore iniziale ragionevole |
| --- | --- |
| layers | 2 |
| hidden event path | 64 |
| hidden embed path | 64 |
| hidden concat path | 128 |
| heads | 4 |
| dropout | 0.1 |
| loss | `CrossEntropyLoss()` |
| optimizer | `Adam(lr=1e-3)` |
| pooling | `global_mean_pool` |

**Certo dal codice TimeGNN**: `DualGATModel` produce logits node-level. Il wrapper graph-level e una nostra aggiunta futura.

### MODELLO B: ChessGATWithTiming

**Raccomandazione**:

```text
RECOMMENDED_TIMING_MODEL = TimeAwareGATConv-based backbone + graph-level wrapper
```

Per confronto piu pulito:

```text
stesso x
stesso edge_index
stesso numero layer
stesso hidden dim
stessi heads
stesso pooling
stesse global_features
stesso classifier
stesso optimizer/loss/split/seed
unica differenza: edge_time entra nella attention
```

Input proposto:

```python
batch.x              # [N_total, 15]
batch.edge_index     # [2, E_total]
batch.edge_attr      # [E_total, 5], tactical features separate
batch.edge_time      # [E_total, 1], synthetic timing
batch.global_features # [B, 4]
batch.y              # [B]
```

**Certo dal codice TimeGNN**: `DualGATTimeAwareModel` usa `data_event.time` come edge-level decay, ma non usa tactical `edge_attr=[E,5]`. Per una versione equa e completa servira un wrapper o layer custom che combina:

- tactical edge features;
- timing separato;
- pooling graph-level.

### Alternativa timing piu edge-aware

```text
DualGATTimeAwareETModel
```

Pro:

- ha timing decay;
- ha edge type score.

Contro:

- richiede `edge_type` single-label;
- non supporta direttamente multilabel `[E,5]`;
- richiede `event_ids`;
- produce node-level logits.

Questa alternativa e utile solo se decidiamo una strategia esplicita per trasformare il multilabel tattico in tipi discreti o in embedding separati.

## Training Pattern Consigliato per Chess

**Da non riusare direttamente**: le ricette high-level `timegnn.gat_basic(...)`, `timegnn.gat_time_decay(...)`, ecc., perche partono da DataFrame event-log e rifanno encoding/split non compatibili con i nostri `.pt`.

**Da riusare come riferimento**:

- `EarlyStopping`;
- config dataclass style;
- loop `model.train()`, `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`;
- metriche loss/accuracy;
- device handling;
- `global_mean_pool` pattern da `gat_outcome`/`PrefixGCNClassifier`.

Training loop chess proposto:

```python
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

train_graphs = torch.load("data/pyg/train_graphs.pt", weights_only=False)
val_graphs = torch.load("data/pyg/val_graphs.pt", weights_only=False)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

for batch in train_loader:
    batch = batch.to(device)
    optimizer.zero_grad()
    logits = model(batch)       # [B, num_move_classes]
    loss = criterion(logits, batch.y)
    loss.backward()
    optimizer.step()
```

Questo e un pattern proposto per chess, non una funzione gia presente in TimeGNN.

## Validation/Test

TimeGNN implementa:

- next-event `evaluate_epoch`: loss/accuracy token-level con mask `labels != -1`;
- prefix GCN `evaluate_epoch`: loss/accuracy graph-level;
- outcome `_eval_epoch_outcome`: graph-level loss/accuracy;
- metriche sequence-level in `timegnn/metrics/sequence.py`.

Per chess servono metriche dedicate:

- top-1 accuracy;
- top-k accuracy;
- loss media;
- evaluation per `MateDepth`;
- confusion/error analysis per mossa;
- gestione OOV gia fatta a dataset generation.

## Componenti Riutilizzabili

### Quasi senza modifiche

| Componente | Uso |
| --- | --- |
| `TimeAwareGATConv` | Layer base per timing-aware attention, con wrapper. |
| `EarlyStopping` | Stop su validation loss. |
| Config dataclass pattern | Organizzazione esperimenti. |
| `global_mean_pool` pattern | Pooling graph-level. |

### Adattabili

| Componente | Adattamento richiesto |
| --- | --- |
| `DualGATModel` | Aggiungere `event_ids`, estrarre embedding o wrappare output per graph-level. |
| `DualGATTimeAwareModel` | Fornire `time` edge-level e wrapper graph-level; separare tactical edge_attr. |
| `DualGATTimeAwareETModel` | Creare `edge_type` discreti o riscrivere adapter per multilabel. |
| `train_gat_outcome` | Usare solo come riferimento; input event-log non compatibile con `.pt` chess. |

### Troppo specifici per process mining

| Parte | Perche |
| --- | --- |
| `EventLogTransformer` | Parte da colonne case/event/time/status, non da grafi chess gia costruiti. |
| `encode_label_event`, `prepare_data_y` | Costruiscono next-event labels sequenziali, non target graph-level chess. |
| `length_stratified_split` | Splitta per lunghezza sequenza event-log, non per MateDepth o dataset gia pronto. |
| `TimeGNNClassifier` | Wrapper sklearn su DataFrame event-log, non su PyG chess dataset. |

## Problemi Da Risolvere Prima Dell'Implementazione

| Priorita | Problema | Dettaglio |
| --- | --- | --- |
| BLOCKER | Output node-level vs graph-level | I GAT principali producono `[N, C]`; chess richiede `[B, C]`. Serve wrapper con pooling. |
| BLOCKER | `event_ids` obbligatori | I modelli dual-path richiedono embedding lookup. Decidere square IDs o rimuovere ramo. |
| BLOCKER | Timing format | Definire `edge_time=[E,1]` sintetico e conservarlo separato da `edge_attr`. |
| BLOCKER | Edge multilabel incompatibile con edge type | `DualGATTimeAwareETModel` usa singolo `edge_type`, non `[E,5]` multilabel. |
| IMPORTANT | Global features | Nessun GAT TimeGNN concatena `global_features`; serve classifier wrapper. |
| IMPORTANT | Fair ablation | No timing e timing devono avere stesso pooling/head/training. |
| IMPORTANT | Output dimensione circa 1,786 | Classification head deve usare vocab size corrente. |
| IMPORTANT | OOV | Val/test OOV sono gia saltati nei grafi; metriche devono usare grafi effettivi. |
| IMPORTANT | Batch handling | Usare `torch_geometric.loader.DataLoader`; gestire `batch.global_features` shape `[B,4]`. |
| MINOR | Checkpointing | TimeGNN non salva checkpoint; aggiungerlo nel training chess. |
| MINOR | Scheduler/class weights | Non presenti in TimeGNN; opzionali dopo baseline. |
| MINOR | Memory usage | 64 nodi per grafo e molte classi: monitorare batch size. |

## Cheat Sheet

| Classe | Import | Forward | Ruolo | Timing | Edge-aware | Graph-level | Uso probabile |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `DualGATModel` | `from timegnn.models.gat_basic import DualGATModel` | `model(data)` | GAT no timing dual-path | NO | SI via `edge_attr` | NO | Base no-timing con wrapper |
| `TimeAwareGATConv` | `from timegnn.models.gat_time_decay import TimeAwareGATConv` | `layer(x, edge_index, edge_attr=time)` | Layer attention con decay | SI | NO tactical | NO | Base per timing custom |
| `DualGATTimeAwareModel` | `from timegnn.models.gat_time_decay import DualGATTimeAwareModel` | `model(data, return_attention=False)` | GAT timing dual-path | SI | NO tactical | NO | Candidate timing con wrapper |
| `DualGAT2EdgesModel` | `from timegnn.models.gat_status_emb import DualGAT2EdgesModel` | `model(data)` | GAT edge type + time diff | PARZIALE | SI single type | NO | Solo con adapter edge_type |
| `TimeAwareETGATConv` | `from timegnn.models.gat_time_decay_status_emb import TimeAwareETGATConv` | `layer(x, edge_index, edge_attr, time)` | Layer timing + edge vector | SI | SI embedded | NO | Candidate custom edge/timing |
| `DualGATTimeAwareETModel` | `from timegnn.models.gat_time_decay_status_emb import DualGATTimeAwareETModel` | `model(data, return_attention=False)` | GAT timing + edge type | SI | SI single type | NO | Timing avanzato con adapter |
| `PrefixGCNClassifier` | `from timegnn.models.prefix_gcn import PrefixGCNClassifier` | `model(data, sequence_features)` | GCN graph classifier | NO | Edge weight | SI | Pattern pooling/global features |
| `EarlyStopping` | `from timegnn.train.early_stopping import EarlyStopping` | `early_stopping(val_loss)` | Training utility | n/a | n/a | n/a | Riutilizzabile |

## Conclusione Pratica

```text
RECOMMENDED_NO_TIMING_MODEL = DualGATModel backbone wrapped for graph-level classification
RECOMMENDED_TIMING_MODEL = TimeAwareGATConv/DualGATTimeAwareModel-style backbone wrapped for graph-level classification
GRAPH_LEVEL_WRAPPER_REQUIRED = YES
CUSTOM_TRAINING_LOOP_REQUIRED = YES
EDGE_FEATURE_ADAPTER_REQUIRED = YES
TIMING_ADAPTER_REQUIRED = YES
```

Motivazione:

- TimeGNN offre layer e modelli GAT utili, ma i GAT principali sono node-level e process-mining oriented.
- Il nostro task richiede pooling graph-level, concatenazione di `global_features` e classifier sulle classi mossa.
- Per il modello no-timing `DualGATModel` e il candidato piu vicino perche usa GAT e puo ricevere `edge_attr` numerico.
- Per il modello timing, la scelta piu pulita e basarsi su `TimeAwareGATConv`/`DualGATTimeAwareModel`, mantenendo stesso backbone/head del no-timing e aggiungendo solo `edge_time`.
- `DualGATTimeAwareETModel` e interessante ma non direttamente compatibile con `edge_attr=[E,5]` multilabel, perche richiede un singolo `edge_type` discreto per arco.
- Le ricette TimeGNN high-level non vanno usate direttamente sui nostri `.pt`: servira un training loop chess dedicato che carica `data/pyg/train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`.

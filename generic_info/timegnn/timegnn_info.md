# Analisi libreria esterna `TimeGNN-main/`

## Scopo dell'analisi

Questa nota documenta la libreria esterna `TimeGNN-main/` senza modificarne i file. L'obiettivo e capire come riusarla nel progetto chess GNN per costruire due linee future:

1. GNN/GAT senza timing.
2. GNN/GAT con timing sintetici.

Conclusione breve: la libreria usa PyTorch e PyTorch Geometric, quindi e tecnologicamente vicina ai nostri grafi PyG. Non e pero direttamente plug-and-play con `Data(x, edge_index, edge_attr, y, global_features)`, perche i modelli TimeGNN sono pensati per event log sequenziali e si aspettano attributi come `event_ids`, `time`, `edge_type`, `edge_time_diff` e label token-level. Per il progetto chess servono adattatori o wrapper model-specific.

## 1. Struttura generale della libreria

Path principale analizzato: `TimeGNN-main/`.

| Path | Ruolo |
| --- | --- |
| `TimeGNN-main/README.md` | Panoramica della libreria, quick start, ricette disponibili e API pubblica. |
| `TimeGNN-main/pyproject.toml` | Metadati package, dipendenze base e optional extras. |
| `TimeGNN-main/src/timegnn/` | Package moderno principale. |
| `TimeGNN-main/src/timegnn/api.py` | Funzioni high-level: `gat_basic`, `gat_status`, `gat_time_decay`, `gat_time_decay_status`, `prefix_gcn`, `gat_outcome`. |
| `TimeGNN-main/src/timegnn/models/` | Modelli GAT/GCN, baseline e loop training shared. |
| `TimeGNN-main/src/timegnn/recipes/` | Ricette end-to-end con config dataclass, preprocessing event-log, training e history. |
| `TimeGNN-main/src/timegnn/data/` | Encoding event log, trasformatori sklearn-style, dataset/collate PyG. |
| `TimeGNN-main/src/timegnn/train/` | `EarlyStopping` e trainer minimale per baseline. |
| `TimeGNN-main/src/timegnn/metrics/` | Accuracy, top-k, BLEU, DLS, analisi sequenze. |
| `TimeGNN-main/src/timegnn/visuals/` | Visualizzazioni attention/topology. |
| `TimeGNN-main/src/timegnn/sklearn.py` | Wrapper `TimeGNNClassifier` in stile sklearn. |
| `TimeGNN-main/src/legacy/` | Codice legacy/notebook-style, anche TensorFlow in `DataEncoder.py`; non e il layer consigliato. |
| `TimeGNN-main/examples/` | Notebook dimostrativi. |
| `TimeGNN-main/tests/` | Test unitari e smoke test ricette/modelli. |

### Entrypoint/script eseguibili

Non ho trovato CLI script dedicati. Gli entrypoint pratici sono funzioni Python:

- high-level API da `timegnn`: `gat_basic(...)`, `gat_time_decay(...)`, ecc.;
- ricette da `timegnn.recipes`: `train_gat_basic(...)`, `train_gat_time_decay(...)`, ecc.;
- wrapper sklearn-style: `TimeGNNClassifier(...).fit(...)`;
- pipeline generica: `Pipeline(Config(...)).fit()`, ma principalmente per baseline registry.

## 2. Dipendenze

Da `TimeGNN-main/pyproject.toml`:

| Tipo | Dipendenze |
| --- | --- |
| Python | `requires-python >= 3.9` |
| Base | `numpy>=1.24`, `pandas>=2.0`, `torch>=2.1`, `scikit-learn>=1.3` |
| Optional `pyg` | `torch-geometric>=2.4`, `torch-scatter>=2.1.2`, `torch-sparse>=0.6.18` |
| Optional `analysis` | `matplotlib>=3.7`, `seaborn>=0.13`, `scipy>=1.10`, `nltk>=3.8`, `pyxdameraulevenshtein>=1.7.1` |

Framework effettivamente usati:

- PyTorch: modelli, training loop, loss, optimizer.
- PyTorch Geometric: `GATConv`, `GCNConv`, `Data`, `Batch`, `global_mean_pool`.
- scikit-learn: encoder/scaler, train/test utility, metriche/report.
- TensorFlow compare solo in `src/legacy/DataEncoder.py`, quindi non e parte della API moderna consigliata.
- DGL non risulta usato.

Nota sui test: `tests/test_models_recipes.py` verifica PyG e controlla se `GATConv.propagate` supporta `edge_attr`; alcune funzioni time-aware possono dipendere dalla versione PyG.

## 3. Modelli disponibili

| File | Classe/Funzione | Tipo | Timing | Edge type | Note |
| --- | --- | --- | --- | --- | --- |
| `src/timegnn/models/gat_basic.py` | `DualGATModel` | GAT token-level | No | Usa `edge_attr` numerico | Doppio ramo: embedding eventi + feature nodo. |
| `src/timegnn/models/gat_status_emb.py` | `DualGAT2EdgesModel` | GAT token-level | Usa `edge_time_diff` come edge feature | Si | Embedding per tipo arco/status. |
| `src/timegnn/models/gat_time_decay.py` | `TimeAwareGATConv` | Layer GAT custom | Si | No | Applica decadimento esponenziale all'attenzione. |
| `src/timegnn/models/gat_time_decay.py` | `DualGATTimeAwareModel` | GAT token-level | Si | No | Usa `data_event.time` come edge/time attribute. |
| `src/timegnn/models/gat_time_decay_status_emb.py` | `TimeAwareETGATConv` | Layer GAT custom | Si | Si | Combina decay temporale e score edge-type. |
| `src/timegnn/models/gat_time_decay_status_emb.py` | `DualGATTimeAwareETModel` | GAT token-level | Si | Si | Variante piu completa. |
| `src/timegnn/models/prefix_gcn.py` | `PrefixGCNClassifier` | GCN graph-level | No diretto | `edge_attr` come edge weight | Usa `global_mean_pool` e feature sequenza. |
| `src/timegnn/models/baseline.py` | `BaselineMostFrequentModel` | Baseline tabellare | No | No | Solo evento piu frequente, non utile per chess graph. |

### Architettura comune dei modelli GAT

I modelli GAT principali usano una logica "dual-path":

- path embedding: `event_ids -> nn.Embedding -> GAT`;
- path feature nodo: `x -> GAT`;
- concat path: concat dei due output -> GAT -> `Linear(output_dim)`.

Questo separa semanticamente identificativo evento e feature numeriche/categoriche del nodo. Per i nostri grafi chess non abbiamo `event_ids` naturali; possiamo crearli artificialmente o semplificare il modello.

## 4. API utile per noi

### `DualGATModel`

- Path: `TimeGNN-main/src/timegnn/models/gat_basic.py`
- Firma:

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

- Input `forward(data_event)`: PyG `Data`/`Batch` con `x`, `edge_index`, `edge_attr`, `event_ids`.
- Output: logits per nodo, shape indicativa `[num_nodes_in_batch, output_dim]`.
- Uso minimo:

```python
from torch_geometric.loader import DataLoader
from timegnn.models.gat_basic import DualGATModel

model = DualGATModel(
    num_event_features=15,
    num_embedding_features=64,
    embedding_dims=32,
    gat_hidden_dim_event=32,
    gat_hidden_dim_embed=32,
    gat_hidden_dim_concat=64,
    output_dim=num_moves,
    num_heads=4,
    edge_dim=5,
)

batch = next(iter(DataLoader(graphs, batch_size=8)))
batch.event_ids = batch.event_ids.long()
logits_per_node = model(batch)
```

Per classificazione puzzle serve pooling graph-level o scelta di un nodo, perche il modello produce logits per nodo.

### `DualGATTimeAwareModel`

- Path: `TimeGNN-main/src/timegnn/models/gat_time_decay.py`
- Timing: si, via `TimeAwareGATConv`.
- Parametro temporale: `lambda_decay`.
- Input `forward(data_event)`: `x`, `edge_index`, `event_ids`, `time`; opzionale `return_attention=True`.
- Output: logits per nodo, oppure `(logits, attention_dict)`.

```python
from timegnn.models.gat_time_decay import DualGATTimeAwareModel

model = DualGATTimeAwareModel(
    num_event_features=15,
    num_embedding_features=64,
    embedding_dims=32,
    gat_hidden_dim_event=32,
    gat_hidden_dim_embed=32,
    gat_hidden_dim_concat=64,
    output_dim=num_moves,
    num_heads=4,
    lambda_decay=0.01,
)

batch.time = batch.synthetic_edge_time.view(-1, 1)
logits_per_node = model(batch)
```

Attenzione: nel codice `forward()` usa `edge_attr = data_event.time`; quindi `time` deve avere lunghezza compatibile con gli archi, non necessariamente un timestamp per nodo.

### `DualGAT2EdgesModel`

- Path: `TimeGNN-main/src/timegnn/models/gat_status_emb.py`
- Input: `x`, `edge_index`, `event_ids`, `edge_type`, `edge_time_diff`.
- Output: logits per nodo.
- Utile se vogliamo mappare i nostri edge label (`legal_move`, `attack`, `defend`, `pin`, `check_line`) in un singolo `edge_type` categoriale.
- Limite: il nostro `edge_attr` e multilabel a 5 dimensioni; il modello si aspetta un solo indice `edge_type` per arco.

### `DualGATTimeAwareETModel`

- Path: `TimeGNN-main/src/timegnn/models/gat_time_decay_status_emb.py`
- Input: `x`, `edge_index`, `event_ids`, `edge_type`, `time`.
- Output: logits per nodo, opzionalmente attention/decay/edge-type scores.
- Variante piu vicina a "GAT con timing + tipo arco", ma richiede adattamento edge multilabel -> tipo discreto.

### `PrefixGCNClassifier`

- Path: `TimeGNN-main/src/timegnn/models/prefix_gcn.py`
- Firma principale:

```python
PrefixGCNClassifier(
    num_event_features: int,
    gcn_hidden_dims: int,
    num_embedding_features: int,
    embedding_dims: int,
    gcn_hidden_dims_embedding: int,
    gcn_hidden_dims_concat: int,
    num_sequence_features: int,
    fc_hidden_dims: int,
    fc_hidden_dims_concat: int,
    output_dim: int,
    num_layers: int = 1,
    dropout: float = 0.0,
    use_batch_norm: bool = False,
    activation: str = "relu",
)
```

- Input `forward(data, sequence_features)`: PyG batch con `x`, `edge_index`, `edge_attr`, `event_ids`, `batch` e un tensore graph-level `sequence_features`.
- Output: logits graph-level `[batch_size, output_dim]`.
- E piu vicino al nostro problema, perche `TargetMove` e una label per grafo.

```python
from timegnn.models.prefix_gcn import PrefixGCNClassifier

model = PrefixGCNClassifier(
    num_event_features=15,
    gcn_hidden_dims=64,
    num_embedding_features=64,
    embedding_dims=16,
    gcn_hidden_dims_embedding=32,
    gcn_hidden_dims_concat=64,
    num_sequence_features=4,
    fc_hidden_dims=32,
    fc_hidden_dims_concat=64,
    output_dim=num_moves,
)

logits = model(batch, batch.global_features)
```

Nota: `GCNConv` usa `edge_attr` come `edge_weight`, quindi serve un vettore 1D per arco, non una matrice multilabel `[num_edges, 5]`.

### Training loop condivisi

Path: `TimeGNN-main/src/timegnn/models/training.py`.

| Funzione | Input | Output | Note |
| --- | --- | --- | --- |
| `train_epoch(model, loader, optimizer, criterion, device)` | loader con `(event_data, labels)` | `(avg_loss, accuracy)` | Token-level; ignora label `-1`. |
| `evaluate_epoch(model, loader, criterion, device)` | loader con `(event_data, labels)` | `(avg_loss, accuracy)` | Token-level; no gradient. |

Per i nostri puzzle graph-level conviene scrivere un loop simile, ma con logits `[batch_size, num_moves]` e label `batch.y`.

### Collate e dataset

Path: `TimeGNN-main/src/timegnn/data/pyg.py`.

| Oggetto | Scopo |
| --- | --- |
| `CustomDataset(event_features, y)` | Wrapper semplice per lista grafi + label. |
| `PrefixDataset(event_features, sequence_features, y)` | Wrapper per grafi + feature globali + label. |
| `custom_collate_fn(batch)` | Batch PyG + padding label sequenziali. |
| `custom_collate_prefix(batch)` | Batch PyG + stack feature sequenza + label. |
| `custom_collate_graph(batch)` | Batch PyG + label graph-level. |

Per i nostri dataset `.pt`, PyG `DataLoader` standard puo bastare. `custom_collate_graph` e utile se creiamo `CustomDataset(graphs, labels)`.

## 5. Training

### Trainer disponibili

- `BasicTrainer` in `src/timegnn/train/trainer.py`: trainer minimale per `BaselineMostFrequentModel`, non utile per GAT chess.
- Loop GAT token-level in `src/timegnn/models/training.py`.
- Loop GCN graph-level in `src/timegnn/models/prefix_gcn.py`.
- Loop outcome graph-level in `src/timegnn/recipes/gat_outcome.py`, funzioni private `_train_epoch_outcome` e `_eval_epoch_outcome`.

### Training loop delle ricette

Le ricette:

- costruiscono input PyG da DataFrame event-log;
- fanno split interno train/test;
- creano `DataLoader`;
- istanziano modello;
- usano `torch.optim.Adam`;
- usano `nn.CrossEntropyLoss`, spesso con `ignore_index=-1`;
- applicano `EarlyStopping`;
- salvano `history` come lista di dizionari.

Non ho trovato scheduler learning rate nelle ricette moderne.

### GPU/CPU

Le ricette usano:

```python
device = device or ("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch.to(device)
```

### Checkpoint/save/load

- `BaselineMostFrequentModel` implementa `save()`/`load()` JSON.
- I modelli PyTorch GAT/GCN non hanno checkpoint automatico nelle ricette.
- Per il nostro progetto va aggiunto esplicitamente:

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config_dict,
        "move_to_idx": move_to_idx,
    },
    "artifacts/checkpoints/gat_no_timing.pt",
)
```

## 6. Validation/Test

### Funzioni disponibili

| Path | Funzione | Tipo |
| --- | --- | --- |
| `src/timegnn/models/training.py` | `evaluate_epoch` | Loss/accuracy token-level. |
| `src/timegnn/models/gat_time_decay.py` | `evaluate_epoch(..., return_attention=True)` | Loss/accuracy + attention per grafi. |
| `src/timegnn/models/gat_time_decay_status_emb.py` | `evaluate_epoch(..., return_attention=True)` | Loss/accuracy + attention/decay/edge-type scores. |
| `src/timegnn/models/prefix_gcn.py` | `evaluate_epoch` | Loss/accuracy graph-level per prefix GCN. |
| `src/timegnn/metrics/sequence.py` | `predict`, `top_k_accuracy`, `predict_per_sequence`, ecc. | Metriche sequence/token. |
| `src/timegnn/metrics/basic.py` | `accuracy_score` | Accuracy semplice pandas/numpy. |

### Per validation chess

Il nostro target e graph-level: una mossa per puzzle. La validation consigliata:

```python
model.eval()
correct = 0
total = 0
loss_sum = 0.0

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        logits = model(batch)          # [batch_size, num_moves]
        loss = criterion(logits, batch.y)
        pred = logits.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.numel()
        loss_sum += loss.item() * batch.y.numel()

val_loss = loss_sum / total
val_acc = correct / total
```

Se si riusa `DualGATModel`, serve pooling graph-level:

```python
from torch_geometric.nn import global_mean_pool

node_logits = dual_gat(batch)
graph_logits = global_mean_pool(node_logits, batch.batch)
```

Questa e la stessa idea usata in `recipes/gat_outcome.py`.

## 7. Hyperparameters

Le config principali sono dataclass in `TimeGNN-main/src/timegnn/recipes/`.

| Config | File | Parametri importanti |
| --- | --- | --- |
| `GATBasicConfig` | `recipes/gat_basic.py` | `embedding_dims`, `gat_hidden_dim_event`, `gat_hidden_dim_embed`, `gat_hidden_dim_concat`, `num_heads`, `num_layers`, `dropout`, `use_batch_norm`, `activation`, `batch_size`, `lr`, `num_epochs`, `patience`, `delta`, `test_size`. |
| `GATStatusEmbConfig` | `recipes/gat_status_emb.py` | Come sopra + `edge_type_dim`. |
| `GATTimeDecayConfig` | `recipes/gat_time_decay.py` | Come base + `lambda_decay`. |
| `GATTimeDecayStatusConfig` | `recipes/gat_time_decay_status_emb.py` | Come base + `edge_type_dim`, `lambda_decay`, `n_bins`. |
| `PrefixGCNConfig` | `recipes/prefix_gcn.py` | `prefix_size`, GCN hidden dims, FC dims, dropout, batch, lr, epochs. |
| `GATOutcomeConfig` | `recipes/gat_outcome.py` | GAT dims + `edge_type_dim`, `lambda_decay`, batch/lr/epochs. |

Le API high-level accettano sia `config=...` sia override flat:

```python
from timegnn import GATBasicConfig, gat_basic

cfg = GATBasicConfig(num_layers=2, dropout=0.1, num_heads=4)
result = gat_basic(df, case_col="sequence", event_col="event", time_col="time", config=cfg)
```

Per il nostro progetto conviene definire config locali in `src/train/` o `src/models/`, non usare direttamente le recipe event-log.

## 8. Formato dati richiesto

### Formato TimeGNN moderno

La libreria accetta principalmente:

1. DataFrame event-log tramite API/recipes.
2. PyG `Data` generati internamente dai transformer.

Attributi PyG usati dai modelli:

| Attributo | Usato da | Significato TimeGNN |
| --- | --- | --- |
| `x` | Tutti | Feature nodo/evento. |
| `edge_index` | Tutti | Archi sequenziali evento i -> i+1. |
| `edge_attr` | `DualGATModel`, `PrefixGCNClassifier`, alcuni preparatori | Time diff o edge weight. |
| `event_ids` | GAT/GCN principali | ID categoriale evento per embedding. |
| `time` | Time-aware models | Valori temporali/edge time per decay. |
| `edge_type` | Status/time+status models | Tipo di transizione categoriale. |
| `edge_time_diff` | Status model | Time diff scalare per edge. |
| `batch` | PyG Batch | Indice grafo per pooling/attention split. |

### Compatibilita con i nostri grafi chess

Nostro formato attuale:

```python
Data(
    x=x,                         # [64, node_feature_dim]
    edge_index=edge_index,       # [2, num_edges]
    edge_attr=edge_attr,         # [num_edges, 5]
    y=y,                         # target move class
)
graph.global_features = ...     # [4]
```

Compatibilita:

| Campo chess | TimeGNN | Stato |
| --- | --- | --- |
| `x` | `x` | Compatibile. |
| `edge_index` | `edge_index` | Compatibile. |
| `edge_attr` multilabel `[E,5]` | `edge_attr` o `edge_type` | Parzialmente compatibile: GAT base supporta `edge_dim=5`; GCN no, perche `GCNConv` vuole edge weight 1D. |
| `y` graph-level | Ricette GAT next-event token-level | Non compatibile direttamente; serve pooling/graph classifier. |
| `global_features` | `sequence_features` o custom concat | Serve adattatore. |
| Timing sintetico | `time` / `edge_time_diff` | Serve generazione e normalizzazione. |
| Edge type multilabel | `edge_type` singolo | Serve mapping multilabel -> single type oppure modello custom che usa edge_attr multilabel. |

## 9. Timing / temporal component

### Come TimeGNN rappresenta il tempo

La libreria usa due concetti:

- `scaled_time_diffs`: differenze temporali tra eventi consecutivi, normalizzate;
- `node_times`: timestamp per nodo/evento, usato in alcune preparazioni ma poi nei modelli time-aware viene passato come `data_event.time`.

Nei layer:

- `TimeAwareGATConv` calcola `decay = exp(-lambda_decay * time_diff)` e moltiplica l'attenzione.
- `TimeAwareETGATConv` combina:
  - attenzione GAT base,
  - score da edge-type embedding,
  - decay temporale esponenziale.

### Dove inserire timing sintetici nei puzzle

Per i puzzle chess possiamo aggiungere:

- `graph.time`: tensore `[num_edges, 1]` con timing sintetico associato agli archi;
- oppure `graph.edge_time_diff`: tensore `[num_edges, 1]` per modelli status;
- eventualmente `graph.global_features` esteso con timing aggregati.

Esempi di timing sintetici plausibili:

- tempo previsto per puzzle in base a `Rating` e `MateDepth`;
- tempo per edge legale/attacco/difesa proporzionale al tipo tattico;
- rumore controllato per simulare distribuzioni bullet/blitz/rapid;
- scalar edge time normalizzato in `[0,1]`.

### Differenza pratica no-timing vs timing

No-timing:

- usa `x`, `edge_index`, `edge_attr`;
- ignora feature temporali;
- baseline pulita per capire quanto bastano struttura e tattica.

Timing:

- aggiunge `time` o `edge_time_diff`;
- usa decay temporale nell'attenzione;
- permette ablation: stesso modello/dati, con o senza timing.

## 10. Algebraic Decoupling

Non ho trovato un modulo o una classe chiamata esplicitamente "Algebraic Decoupling".

Pero esiste una separazione architetturale simile nei modelli dual-path:

- `event_ids` -> embedding path;
- `x` -> feature path;
- concat -> fusion path;
- nei modelli time/status, `time` e `edge_type` modificano l'attenzione sugli archi.

File rilevanti:

- `src/timegnn/models/gat_basic.py`: separa embedding evento e feature evento.
- `src/timegnn/models/gat_status_emb.py`: aggiunge edge-type embedding.
- `src/timegnn/models/gat_time_decay.py`: separa feature grafo e decay temporale.
- `src/timegnn/models/gat_time_decay_status_emb.py`: combina feature nodo, edge type e tempo.

Per il nostro progetto possiamo interpretare questa idea cosi:

- node features chess: `x`;
- edge features tattiche: `edge_attr` o `edge_type`;
- temporal features sintetiche: `time` / `edge_time_diff`;
- global features: concatenazione dopo pooling, non gestita nativamente dai GAT TimeGNN.

## 11. Integrazione proposta con il nostro progetto

### File del nostro progetto coinvolti

| Area | File/cartella futura consigliata | Ruolo |
| --- | --- | --- |
| Caricamento dataset | `src/train/loaders.py` | Caricare `data/pyg/train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`. |
| Modelli | `src/models/` o `src/gnn_models/` | Wrapper chess-specific attorno ai layer/modelli TimeGNN. |
| Training | `src/train/train_gat.py` | Training loop graph-level per `TargetMove`. |
| Validation/test | `src/train/evaluate.py` | Accuracy, top-k, loss per MateDepth. |
| Timing sintetici | `src/timing/` o `src/graph/timing_features.py` | Generare `time`/`edge_time_diff` per ogni grafo. |
| Config | `src/train/config.py` | Iperparametri no-timing/timing. |
| Checkpoint | `artifacts/checkpoints/` | `state_dict`, config, vocabulary. |

### Caricamento dei nostri `.pt`

```python
import torch
from torch_geometric.loader import DataLoader

train_graphs = torch.load("data/pyg/train_graphs.pt")
val_graphs = torch.load("data/pyg/val_graphs.pt")
test_graphs = torch.load("data/pyg/test_graphs.pt")

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
```

### Adattatore minimo per GAT no-timing

Se si vuole riusare `DualGATModel`, aggiungere attributi richiesti:

```python
def adapt_chess_graph_no_timing(data):
    # Square id 0..63 as embedding id.
    data.event_ids = torch.arange(data.num_nodes, dtype=torch.long).view(-1, 1)

    # DualGATModel can use edge_dim=5 if edge_attr is [E, 5].
    data.edge_attr = data.edge_attr.float()
    return data
```

Serve poi pooling graph-level:

```python
from torch_geometric.nn import global_mean_pool

node_logits = model(batch)
graph_logits = global_mean_pool(node_logits, batch.batch)
loss = criterion(graph_logits, batch.y)
```

### Adattatore minimo per GAT con timing

```python
def adapt_chess_graph_timing(data):
    data.event_ids = torch.arange(data.num_nodes, dtype=torch.long).view(-1, 1)

    # Required by DualGATTimeAwareModel.forward().
    # Shape should align with edges: [num_edges, 1].
    data.time = data.synthetic_edge_time.float().view(-1, 1)
    return data
```

Per `DualGATTimeAwareETModel` serve anche:

```python
data.edge_type = chess_edge_attr_to_type(data.edge_attr).long()
```

### Cosa manca per compatibilita

- Un modello graph-level GAT chess-specific oppure un wrapper che faccia pooling dopo i logits nodo.
- Gestione di `global_features`: concatenarle dopo pooling:

```python
graph_emb = global_mean_pool(node_hidden_or_logits, batch.batch)
z = torch.cat([graph_emb, batch.global_features], dim=1)
logits = classifier(z)
```

- Conversione `edge_attr` multilabel in:
  - vettore continuo per `GATConv(edge_dim=5)`, oppure
  - singolo `edge_type` per modelli status.
- Timing sintetico per edge in shape `[E,1]`.
- Checkpointing esplicito.
- Evaluation per `MateDepth`.

## 12. Piano operativo consigliato

### Step 1 - Baseline GNN/GAT senza timing

- Creare modello chess-specific usando PyG.
- Opzione A: usare direttamente `torch_geometric.nn.GATConv` e prendere ispirazione da `DualGATModel`.
- Opzione B: wrapper attorno a `DualGATModel` con `event_ids` = square id e pooling graph-level.
- Input: `x`, `edge_index`, `edge_attr`.
- Output: logits `[batch_size, num_moves]`.

### Step 2 - Training loop

- Usare `CrossEntropyLoss`.
- Optimizer iniziale: `Adam(lr=1e-3)`.
- Batch con `torch_geometric.loader.DataLoader`.
- Salvare history train/val loss e accuracy.

### Step 3 - Validation accuracy

- Calcolare:
  - top-1 accuracy;
  - top-k accuracy, ad esempio k=3/5;
  - loss media;
  - accuracy per `MateDepth`.

### Step 4 - Checkpoint

- Salvare best model su validation loss:

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "config": config,
    },
    "artifacts/checkpoints/gat_no_timing_best.pt",
)
```

### Step 5 - Timing sintetici

- Aggiungere generatore di feature temporali:
  - per grafo,
  - per edge,
  - normalizzate.
- Salvare nei grafi come `time` o `edge_time_diff`.

### Step 6 - GNN/GAT con timing

- Usare `TimeAwareGATConv` o `DualGATTimeAwareModel` come base.
- Verificare shape `time == [num_edges, 1]`.
- Valutare `lambda_decay`.

### Step 7 - Ablation timing vs no timing

- Stessi split.
- Stessa dimensione modello il piu possibile.
- Confrontare:
  - no timing;
  - timing sintetico;
  - timing random/control;
  - timing rimosso a inference se utile.

## 13. Possibili problemi

| Problema | Impatto | Mitigazione |
| --- | --- | --- |
| TimeGNN nasce per event log sequenziali | I nostri puzzle sono grafi tattici non sequenze | Usare solo modelli/layer, non le ricette DataFrame. |
| Label token-level vs graph-level | `TargetMove` e per puzzle, non per nodo | Pooling graph-level + classifier. |
| `event_ids` obbligatorio | I nostri nodi non sono eventi categoriali | Usare square id 0..63 o piece id come embedding; oppure rimuovere path embedding in wrapper custom. |
| `edge_attr` multilabel | Alcuni modelli vogliono edge scalar o edge type discreto | Per GAT base usare `edge_dim=5`; per status creare mapping multilabel -> type. |
| `GCNConv` con `edge_attr` 2D | `PrefixGCNClassifier` non accetta direttamente `[E,5]` come edge weight | Usare edge weight scalare o modificare modello/wrapper. |
| Timing mancante | Modelli time-aware richiedono `time`/`edge_time_diff` | Generare timing sintetici normalizzati. |
| Output classi mosse numeroso | `output_dim = len(move_to_idx)` puo essere grande | Monitorare RAM/GPU, top-k, class imbalance. |
| Checkpoint non automatico | Ricette non salvano best model | Implementare checkpoint nel nostro training. |
| PyG version mismatch | Test gia segnalano possibili problemi con `edge_attr` in propagate | Verificare versione PyG nel venv prima di training serio. |
| `global_features` non gestito nativamente dai GAT | Informazioni globali perse | Concatenare dopo pooling graph-level. |

## Esempio di wrapper consigliato per chess GAT no-timing

Questo e uno sketch operativo, non codice gia presente nella libreria:

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class ChessGATNoTiming(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim, heads, num_moves):
        super().__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=heads, edge_dim=edge_dim)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=edge_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_moves),
        )

    def forward(self, data):
        x = self.gat1(data.x, data.edge_index, edge_attr=data.edge_attr)
        x = torch.relu(x)
        x = self.gat2(x, data.edge_index, edge_attr=data.edge_attr)
        graph_emb = global_mean_pool(x, data.batch)
        z = torch.cat([graph_emb, data.global_features], dim=1)
        return self.classifier(z)
```

## Esempio di wrapper consigliato per chess GAT con timing

```python
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from timegnn.models.gat_time_decay import TimeAwareGATConv


class ChessGATTiming(nn.Module):
    def __init__(self, node_dim, global_dim, hidden_dim, heads, num_moves, lambda_decay):
        super().__init__()
        self.gat1 = TimeAwareGATConv(
            node_dim,
            hidden_dim,
            heads=heads,
            edge_dim=1,
            lambda_decay=lambda_decay,
        )
        self.gat2 = TimeAwareGATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            edge_dim=1,
            lambda_decay=lambda_decay,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_moves),
        )

    def forward(self, data):
        time = data.time.float().view(-1, 1)
        x = self.gat1(data.x, data.edge_index, edge_attr=time)
        x = torch.relu(x)
        x = self.gat2(x, data.edge_index, edge_attr=time)
        graph_emb = global_mean_pool(x, data.batch)
        z = torch.cat([graph_emb, data.global_features], dim=1)
        return self.classifier(z)
```

## Conclusione pratica per il nostro progetto

`TimeGNN-main/` e utile come base tecnica per modelli PyTorch Geometric, soprattutto per:

- pattern GAT con `edge_attr`;
- layer temporale `TimeAwareGATConv`;
- variante con edge-type attention `TimeAwareETGATConv`;
- training loop semplici con loss/accuracy;
- early stopping;
- metriche top-k e funzioni di analisi;
- esempi di pooling graph-level in `gat_outcome.py`.

Non e direttamente compatibile con i nostri grafi chess PyG nel senso "carico `train_graphs.pt` e chiamo una recipe". Le recipe si aspettano DataFrame event-log e costruiscono grafi sequenziali internamente. La compatibilita migliore e a livello di modelli/layer, non a livello di pipeline.

Per procedere in modo pulito:

1. Non usare le high-level recipes per i puzzle chess.
2. Creare wrapper chess-specific che accettano direttamente `Data(x, edge_index, edge_attr, y, global_features)`.
3. Riutilizzare `GATConv`/`TimeAwareGATConv`/idee dual-path dove servono.
4. Implementare training/validation/checkpoint nel nostro progetto.
5. Aggiungere timing sintetici come `data.time` o `data.edge_time_diff`.
6. Confrontare no-timing vs timing sugli stessi split e con la stessa vocabulary `move_to_idx`.

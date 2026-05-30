# TimeGNN

A general-purpose temporal Graph Neural Network library for **process mining**
and **event-log prediction**.  It provides several GAT / GCN model variants
that turn sequential event logs into temporal graphs and learn to predict the
next activity or the case-level outcome.

## Features

| Recipe                  | Task             | Model class              | Time-aware | Edge-type embed |
|-------------------------|------------------|--------------------------|:----------:|:---------------:|
| `gat_basic`             | next-event       | `DualGATModel`           |     –      |        –        |
| `gat_status`            | next-event       | `DualGAT2EdgesModel`     |     –      |       ✓         |
| `gat_time_decay`        | next-event       | `DualGATTimeAwareModel`  |     ✓      |        –        |
| `gat_time_decay_status` | next-event       | `DualGATTimeAwareETModel`|     ✓      |       ✓         |
| `prefix_gcn`            | next-event       | `PrefixGCNClassifier`    |     –      |        –        |
| `gat_outcome`           | case outcome     | any GAT variant          |     ✓/–    |       ✓/–       |

## Installation

```bash
pip install -e ".[pyg,analysis]"
```

> **Note:** PyTorch Geometric wheels depend on your CUDA / CPU version.
> See <https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html>.

## Quick start

### High-level API (one function call)

```python
import pandas as pd
import timegnn

df = pd.read_csv("examples/data/Helpdesk.csv")

result = timegnn.gat_time_decay(
    df,
    case_col="sequence",
    event_col="event",
    time_col="time",
    cat_event=["event"],
    num_event=["sn1"],
    seq_cols=["sn1"],
    num_seq=["sn1"],
    num_epochs=10,
)

model   = result["model"]
history = result["history"]
```

### API config: topology + early stopping

All high-level API functions accept either:

1. `config=...` with a typed dataclass config.
2. Flat keyword overrides (`**overrides`) for quick experiments.

New minimal controls now available across recipes:

- `num_layers`: stack multiple GAT/GCN layers per path.
- `dropout`: dropout between hidden layers.
- `use_batch_norm`: enable BatchNorm between hidden layers.
- `activation`: choose hidden activation (`relu`, `elu`, `gelu`, `leaky_relu`).
- `patience`: early stopping patience on validation/test loss.
- `delta`: minimum loss improvement to reset patience.

Model-specific controls:

- `lambda_decay`: exponential time-decay strength for time-aware models
    (`gat_time_decay`, `gat_time_decay_status`, and `gat_outcome` when
    `mode` uses time decay).
- `edge_type_dim`: transition/status embedding size for edge-type models
    (`gat_status`, `gat_time_decay_status`, and `gat_outcome` when
    `mode` uses status edges).
- `prefix_size`: sliding window length for `prefix_gcn`.
- `mode`: choose the backbone in `gat_outcome`
    (`gat_basic`, `gat_time_decay`, `gat_status`, `gat_time_decay_status`).

Example with overrides:

```python
result = timegnn.gat_basic(
    df,
    case_col="sequence",
    event_col="event",
    time_col="time",
    cat_event=["event"],
    num_event=[],
    seq_cols=["sn1"],
    cat_seq=[],
    num_seq=["sn1"],
    num_layers=2,
    dropout=0.1,
    use_batch_norm=True,
    activation="gelu",
    num_epochs=30,
    patience=4,
    delta=1e-4,
)
```

Example with explicit config object:

```python
from timegnn import GATBasicConfig, gat_basic

cfg = GATBasicConfig(
    embedding_dims=64,
    gat_hidden_dim_event=32,
    gat_hidden_dim_embed=128,
    gat_hidden_dim_concat=256,
    num_heads=4,
    num_layers=3,
    dropout=0.2,
    use_batch_norm=True,
    activation="gelu",
    num_epochs=50,
    patience=5,
    delta=1e-4,
)

result = gat_basic(
    df,
    case_col="sequence",
    event_col="event",
    time_col="time",
    cat_event=["event"],
    num_event=[],
    seq_cols=["sn1"],
    cat_seq=[],
    num_seq=["sn1"],
    config=cfg,
)
```

Available config classes:

- `GATBasicConfig`
- `GATStatusEmbConfig`
- `GATTimeDecayConfig`
- `GATTimeDecayStatusConfig`
- `PrefixGCNConfig`
- `GATOutcomeConfig`

### Plot model topology

You can visualize the architecture tree of a trained model:

```python
from timegnn import plot_model_topology

fig, ax = plot_model_topology(
    model,
    max_depth=4,
    title="Dual GAT Topology",
    save_path="topology.png",  # optional
)
```

This shows module hierarchy and trainable parameter counts per block.

### Outcome prediction

```python
result = timegnn.gat_outcome(
    df,
    case_col="Case ID",
    event_col="Activity",
    time_col="Complete Timestamp",
    outcome_col="outcome",
    mode="gat_time_decay_status",
    status_col="Activity",
    num_epochs=10,
)
```

### Sklearn-style wrapper

```python
from timegnn import TimeGNNClassifier

clf = TimeGNNClassifier(
    model_type="gat_basic",
    case_col="sequence",
    event_col="event",
    time_col="time",
    cat_event=["event"],
    num_event=["sn1"],
    seq_cols=["sn1"],
    num_seq=["sn1"],
    num_epochs=5,
)
clf.fit(train_df)
preds  = clf.predict(test_df)
acc    = clf.score(test_df)
```

### Pipeline (Config-driven)

```python
from timegnn import Config, Pipeline

cfg = Config(
    model="baseline_most_frequent",
    task="next_event",
    data_source="examples/data/Helpdesk.csv",
    time_col="time",
    case_col="sequence",
    event_col="event",
)

pipe = Pipeline(cfg)
pipe.fit()
metrics = pipe.evaluate()
```

## Project layout

```
src/timegnn/
├── api.py                  # User-facing one-call functions
├── sklearn.py              # Sklearn-compatible wrapper
├── __init__.py             # Public re-exports
├── core/                   # Config, Pipeline, Registry, base classes
├── data/                   # Ingestion, encoding, PyG datasets, transformers
├── encoders/               # Label encoder
├── metrics/                # Accuracy, BLEU, DLS, top-k, sequence analysis
├── models/                 # DualGAT*, PrefixGCN, baseline, shared training loops
├── recipes/                # High-level training recipes with config dataclasses
├── train/                  # BasicTrainer, EarlyStopping
├── utils/                  # Logging, pad_sequences
└── visuals/                # Attention heatmaps, rank dominance, timeline plots
examples/
├── data/                   # Sample CSV datasets
└── *.ipynb                 # Jupyter notebooks demonstrating each recipe
```

## Running tests

```bash
pip install pytest
pytest
```

## License

MIT

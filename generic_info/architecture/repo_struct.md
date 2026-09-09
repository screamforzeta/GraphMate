# Struttura Repository

Questo documento descrive la struttura logica della repository dopo il refactor preparatorio alla fase timing-aware.

## Root

```text
.
├── main.py
├── README.md
├── requirements.txt
├── src/
├── tests/
├── generic_info/
├── artifacts/
├── data/
└── TimeGNN-main/
```

- `main.py`: pipeline dati principale fino alla generazione dei grafi PyG.
- `src/`: codice sviluppato nel progetto.
- `tests/`: test automatici pytest.
- `generic_info/`: documentazione tecnica e progettuale.
- `artifacts/`: checkpoint, vocabolari e report generati/versionabili quando necessario.
- `data/`: dati generati o scaricati, non versionati.
- `TimeGNN-main/`: libreria esterna/vendor, non modificata dal progetto.

## Codice Sorgente

```text
src/
├── audit/
│   └── model_a/
├── benchmarks/
├── cli/
│   ├── audit/
│   ├── benchmarks/
│   ├── data/
│   ├── evaluation/
│   └── training/
├── common/
├── data/
│   ├── download/
│   └── preprocess/
├── evaluation/
│   ├── model_a/
│   └── model_b/
├── graph/
│   └── debug/
├── inference/
│   ├── model_a/
│   └── model_b/
├── models/
│   ├── model_a/
│   └── model_b/
├── training/
│   ├── common/
│   ├── model_a/
│   └── model_b/
└── validation/
```

## Responsabilità

| Path | Ruolo |
| --- | --- |
| `src/data/download/` | Download puzzle Lichess e PGN campionati. |
| `src/data/preprocess/` | Preprocessing, cleaning, parsing e split dataset. |
| `src/graph/` | Feature nodi/archi, graph builder e dataset PyG. |
| `src/graph/debug/` | Debugger Streamlit standalone. |
| `src/models/model_a/` | Architetture no-timing Model A/A3. |
| `src/models/model_b/` | Namespace futuro per modelli timing-aware. |
| `src/training/common/` | Metriche e utilità condivise. |
| `src/training/model_a/` | Training Model A, A2 e A3 no-timing. |
| `src/training/model_b/` | Namespace futuro per training timing-aware. |
| `src/evaluation/model_a/` | Evaluator post-hoc Model A/A2/A3. |
| `src/evaluation/model_b/` | Namespace futuro per evaluation timing-aware. |
| `src/inference/model_a/` | Inference read-only no-timing e multi-model. |
| `src/inference/model_b/` | Namespace futuro per inference timing-aware. |
| `src/audit/model_a/` | Audit pre-training e controlli specifici Model A. |
| `src/benchmarks/` | Benchmark runtime e data loading. |
| `src/validation/` | Validator delle rappresentazioni generate. |
| `src/cli/` | Entrypoint eseguibili con `python -m`. |

## Entrypoint Principali

```bash
python3 main.py
./venv/bin/python -m src.graph.pyg_dataset --graphs-per-shard 1000 --overwrite
./venv/bin/python -m src.validation.representations --csv-sample 1000 --graph-sample 500 --seed 42
./venv/bin/python -m src.cli.training.train_model_a3_legal_scorer --device cuda --batch-size 128 --num-workers 0 --pin-memory --non-blocking --amp
./venv/bin/python -m src.cli.evaluation.evaluate_model_a_vs_a2_vs_a3 --device cuda --batch-size 128 --non-blocking --amp
streamlit run src/graph/debug/streamlit_graph_debugger.py
```

Il debugger Streamlit resta un'applicazione standalone e non deve essere importato in `main.py`.

## Documentazione

```text
generic_info/
├── architecture/
├── model_a/
├── project/
├── reference/
└── timegnn/
```

- `generic_info/architecture/`: architettura repo, decisioni progettuali e piani tecnici.
- `generic_info/model_a/`: documentazione definitiva della fase no-timing Model A/A2/A3.
- `generic_info/timegnn/`: analisi della libreria `TimeGNN-main/` e note per Model B.
- `generic_info/project/`: note operative generali.
- `generic_info/reference/`: materiali esterni o consegne di riferimento.

## Regole di Integrità

- Non modificare `TimeGNN-main/`.
- Non spostare `data/` o `artifacts/`.
- Non cambiare dataset, split, vocabulary, checkpoint o semantica dei modelli durante refactor strutturali.
- Model A/A2/A3 restano sotto namespace `model_a`.
- Model B ha solo namespace preparatori finché non vengono introdotti timing sintetici e modelli timing-aware.

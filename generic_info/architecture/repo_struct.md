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
- `data/heldout_classic/`: area separata per il dataset esterno classic Mate-in-N (`raw/`, `processed/`, `final/`).
- `data/heldout_classic/raw/yacpdb/`: area per export/cache locali YACPDB, non per scraping live.
- `data/heldout_classic/final/yacpdb_classic_v1/`: candidate dataset YACPDB `VALIDATED_NOT_FROZEN`, non ancora frozen.
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
│   ├── heldout_sources/
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
├── verification/
└── validation/
```

## Responsabilità

| Path | Ruolo |
| --- | --- |
| `src/data/download/` | Download puzzle Lichess e PGN campionati. |
| `src/data/preprocess/` | Preprocessing, cleaning, parsing e split dataset. |
| `src/data/timing/` | Generazione timing sintetici per dataset puzzle Model B. |
| `src/data/heldout_classic.py` | Import, validazione e manifest del dataset esterno classic Mate-in-N. |
| `src/data/heldout_sources/yacpdb.py` | Importer YACPDB da export/cache locale, availability scan e build key-validated. |
| `src/data/heldout_sources/yacpdb_client.py` | Client YACPDB minimo per QL query/fetch/raw-cache con guardrail `max-records`. |
| `src/data/heldout_sources/yacpdb_availability.py` | Scan bounded #1..#10 da cache/API YACPDB con fingerprint e report. |
| `src/data/heldout_sources/yacpdb_candidate_build.py` | Build offline deterministico del candidate dataset YACPDB 20 x 10 da scan cache. |
| `src/data/heldout_sources/yacpdb_position.py` | Normalizzazione YACPDB `algebraic` -> FEN per posizioni ortodosse. |
| `src/data/heldout_sources/yacpdb_solution.py` | Estrazione strutturale della key evitando set play, tries e refutazioni. |
| `src/graph/` | Feature nodi/archi, graph builder e dataset PyG. |
| `src/graph/debug/` | Debugger Streamlit standalone. |
| `src/models/model_a/` | Architetture no-timing Model A/A3. |
| `src/models/model_b/` | Architettura timing-aware Model B basata su A3. |
| `src/training/common/` | Metriche e utilità condivise. |
| `src/training/model_a/` | Training Model A, A2 e A3 no-timing. |
| `src/training/model_b/` | Wrapper training Model B sul protocollo A3. |
| `src/evaluation/model_a/` | Evaluator post-hoc Model A/A2/A3. |
| `src/evaluation/model_b/` | Namespace futuro per evaluation timing-aware. |
| `src/inference/model_a/` | Inference read-only no-timing e multi-model. |
| `src/inference/model_b/` | Namespace futuro per inference timing-aware. |
| `src/verification/popeye.py` | Adapter Popeye per verifica forced-mate YACPDB separata dal dataset e dai modelli. |
| `src/audit/model_a/` | Audit pre-training e controlli specifici Model A. |
| `src/benchmarks/` | Benchmark runtime e data loading. |
| `src/validation/` | Validator delle rappresentazioni generate. |
| `src/cli/` | Entrypoint eseguibili con `python -m`. |

## Entrypoint Principali

```bash
python3 main.py
./venv/bin/python -m src.graph.pyg_dataset --graphs-per-shard 1000 --overwrite
./venv/bin/python -m src.validation.representations --csv-sample 1000 --graph-sample 500 --seed 42
./venv/bin/python -m src.cli.data.generate_puzzle_timing_dataset --overwrite
./venv/bin/python -m src.data.heldout_classic --source data/heldout_classic/raw/<external_source>.jsonl --dataset-version v1
./venv/bin/python -m src.data.heldout_sources.yacpdb_client --mode fetch-id --problem-id 26026 --max-records 3
./venv/bin/python -m src.data.heldout_sources.yacpdb_availability --output-root data/heldout_classic --per-depth-cap 200 --timeout 20
./venv/bin/python -m src.data.heldout_sources.yacpdb_candidate_build --output-root data/heldout_classic --dataset-version yacpdb_classic_v1 --per-depth 20 --seed 42
./venv/bin/python -m src.verification.popeye --dataset-dir data/heldout_classic/final/yacpdb_classic_v1 --verification-root data/heldout_classic/verification/yacpdb_classic_v1/popeye --popeye-executable <path> --timeout-seconds 300
./venv/bin/python -m src.cli.training.train_model_a3_legal_scorer --device cuda --batch-size 128 --num-workers 0 --pin-memory --non-blocking --amp
./venv/bin/python -m src.cli.training.train_model_b_timing_legal_scorer --dataset-root data/pyg_puzzles_timing --device cuda --batch-size 128 --num-workers 0 --pin-memory --non-blocking --amp
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
- Model B è presente come estensione timing-aware di A3, ma richiede dataset timing ufficiale e audit prima di training/evaluation.

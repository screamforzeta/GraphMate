STRUTTURA DELLA REPO

chess-gnn-project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── puzzles/
│   └── heldout/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── graphs/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── results/
│
├── reports/
│
├── requirements.txt
├── README.md
└── main.py

--------------------------
TimeGNN-main (analisi)
--------------------------
- **Scopo:** Libreria Python per Graph Neural Networks temporali orientata a process mining e predizione su log di eventi (next-event, case outcome).
- **Licenza:** MIT (vedi [TimeGNN-main/LICENSE](TimeGNN-main/LICENSE)).
- **Requisiti minimi:** Python >= 3.9, `numpy>=1.24`, `pandas>=2.0`, `torch>=2.1`, `scikit-learn>=1.3` (definiti in [TimeGNN-main/pyproject.toml](TimeGNN-main/pyproject.toml)).
- **Extra opzionali:**
	- `pyg` extra: `torch-geometric>=2.4`, `torch-scatter>=2.1.2`, `torch-sparse>=0.6.18` (richiesto per le ricette GAT/GCN e i notebook). Note: le ruote PyG dipendono da CUDA/CPU — seguire la guida ufficiale PyG.
	- `analysis` extra: `matplotlib`, `seaborn`, `scipy`, `nltk`, `pyxdameraulevenshtein` per visualizzazioni e metriche avanzate.
- **Installazione rapida:**
	- Sviluppo con PyG e strumenti di analisi: `pip install -e "TimeGNN-main.[pyg,analysis]"` (attenzione alla compatibilità di `torch`/CUDA).
- **Entry-point & API alto livello:** vedere [TimeGNN-main/src/timegnn/api.py](TimeGNN-main/src/timegnn/api.py)
	- Funzioni one-call: `gat_basic`, `gat_status`, `gat_time_decay`, `gat_time_decay_status`, `prefix_gcn`, `gat_outcome`.
	- Ogni funzione accetta un `pandas.DataFrame` e parametri come `case_col`, `event_col`, `time_col`, colonne categorical/numerical, `config` (dataclass) oppure override flat (`num_layers`, `dropout`, `patience`, ...).
- **Esportazioni pubbliche principali:** definite in [TimeGNN-main/src/timegnn/__init__.py](TimeGNN-main/src/timegnn/__init__.py)
	- `Config`, `Pipeline`, `ModelRegistry`, `model_registry`, `BaselineMostFrequentModel`, `TimeGNNClassifier`
	- Ricette/config: `GATBasicConfig`, `GATOutcomeConfig`, `GATStatusEmbConfig`, `GATTimeDecayConfig`, `GATTimeDecayStatusConfig`, `PrefixGCNConfig`
	- Metriche helper: `accuracy_score`, `average_bleu_score`, `compute_dls_and_exact_match`, `top_k_accuracy`, `predict`, `sequence_level_top_k_accuracy`, ecc.
	- Visuals: plot di attention, timeline, rank dominance, topology (`plot_model_topology`), t-test tabelle, ecc.
- **Struttura principale del codice:** (cartelle sotto `TimeGNN-main/src/timegnn/`)
	- `core/`: `Config`, `Pipeline`, `ModelRegistry` (usare `Pipeline(Config(...))` per flussi end-to-end).
	- `data/`: ingestion (`read_events_csv`), `EventSchema`, `EventLogTransformer`, `PrefixGCNTransformer`, funzioni PyG di preparazione dei dati.
	- `encoders/`: etichettatura (ad es. `BasicLabelEncoder`).
	- `models/`: implementazioni principali: `BaselineMostFrequentModel`, `DualGATModel`, `DualGAT2EdgesModel`, `DualGATTimeAwareModel`, `DualGATTimeAwareETModel`, `PrefixGCNClassifier` e helper di training (`train_epoch`, `evaluate_epoch`). (vedi [TimeGNN-main/src/timegnn/models](TimeGNN-main/src/timegnn/models)).
	- `recipes/`: funzioni di training di alto livello (`train_gat_basic`, `train_gat_status_emb`, `train_gat_time_decay`, `train_gat_time_decay_status_emb`, `train_prefix_gcn`, `train_gat_outcome`) che orchestrano `EventLogTransformer`, dataclass di config e loop di training.
	- `train/`: `BasicTrainer`, `EarlyStopping` (comportamento di early stopping e loop di training comune).
	- `metrics/`: funzioni per valutazione (accuracy, BLEU, DLS, analisi di sequenza, top-k).
	- `visuals/`: funzioni per plot e analisi delle attention maps e topologie.
- **Transformer / preprocessing:**
	- `EventLogTransformer`: costruisce gli input PyG partendo da DataFrame — gestisce encodings categoriali/numerici, scaling dei tempi, transizioni di stato, padding e modalità multiple (`gat_basic`, `gat_time_decay`, `gat_status`, `gat_time_decay_status`).
	- `PrefixGCNTransformer`: per compiti prefix-based (sliding window).
- **Modelli disponibili (nominativi):**
	- `baseline_most_frequent` → `BaselineMostFrequentModel` (registro: `model_registry.register("baseline_most_frequent")`).
	- `DualGATModel` (gat_basic), `DualGAT2EdgesModel` (gat_status), `DualGATTimeAwareModel` (gat_time_decay), `DualGATTimeAwareETModel` (gat_time_decay_status), `PrefixGCNClassifier`.
- **Sklearn wrapper:** `TimeGNNClassifier` in [TimeGNN-main/src/timegnn/sklearn.py](TimeGNN-main/src/timegnn/sklearn.py) per `fit/predict/score` stile scikit-learn.
- **Esempi / notebook:** diverse notebook dimostrative in `TimeGNN-main/examples/` (per ogni recipe è presente un notebook). I CSV di esempio citati nel README potrebbero non essere inclusi nella copia locale.
- **Tests:** ci sono test unitari in `TimeGNN-main/tests/` che coprono import, API naming, flusso core (schema, encoder, baseline, trainer, pipeline) e l'esecuzione delle ricette (salvo PyG). Per eseguire i test: `pip install pytest` quindi `pytest -q` nella root.
- **Come usare rapidamente (esempi sintetici):**
	- High-level quick-call: importare `timegnn` e chiamare `timegnn.gat_basic(...)` passandogli il `DataFrame` e le colonne.
	- Pipeline config-driven: creare `Config(...)` e `Pipeline(cfg)` poi `fit()` / `evaluate()`.
- **Limitazioni / note operative importanti:**
	- Le ricette GAT/GCN richiedono `torch_geometric` e relative estensioni; l'installazione può richiedere compatibilità con la versione di `torch` e il supporto CUDA. In ambienti senza PyG i test/notebook verranno saltati (vedi `pytest.importorskip("torch_geometric")`).
	- Alcune implementazioni controllano se `GATConv.propagate` supporta `edge_attr` — questo condiziona l'esecuzione dei modelli con attributi di bordo.
	- Non ci sono documentazione estesa oltre al `README.md` e i notebook; per approfondire l'implementazione internamente guardare i file sotto `src/timegnn/`.
- **Dove guardare prima (punti di ingresso per sviluppo):**
	- API one-call: [TimeGNN-main/src/timegnn/api.py](TimeGNN-main/src/timegnn/api.py)
	- Config / Pipeline: [TimeGNN-main/src/timegnn/core/config.py](TimeGNN-main/src/timegnn/core/config.py), [TimeGNN-main/src/timegnn/core/pipeline.py](TimeGNN-main/src/timegnn/core/pipeline.py)
	- Transformer: [TimeGNN-main/src/timegnn/data/transformer.py](TimeGNN-main/src/timegnn/data/transformer.py)
	- Models: [TimeGNN-main/src/timegnn/models](TimeGNN-main/src/timegnn/models)
	- Tests: [TimeGNN-main/tests](TimeGNN-main/tests)

------
Aggiornamento completato: questa sezione è stata aggiunta automaticamente con le informazioni rilevabili dal sorgente e dai test presenti.
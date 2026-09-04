# Progetto Damiani - Chess GNN

Questo repository costruisce una pipeline Python per trasformare puzzle e partite Lichess in grafi utilizzabili da modelli Graph Neural Network.

L'obiettivo finale è addestrare e valutare modelli GNN/GAT su puzzle Lichess mate-in-n, usando come target la prima mossa corretta della soluzione. In una fase successiva il progetto integrerà timing sintetici, farà ablation timing vs no timing e confronterà le prestazioni con modelli LLM.

## Stato Attuale

La pipeline implementata copre:

- download del database puzzle Lichess in formato `.csv.zst`;
- download streaming di PGN Lichess con campionamento diretto, senza scaricare l'intero archivio mensile da circa 30GB;
- preprocessing dei puzzle mate-in-1 fino a mate-in-5;
- parsing delle partite PGN;
- cleaning delle partite;
- cleaning dei puzzle;
- split train/validation/test;
- encoding delle mosse;
- estrazione delle feature dei nodi;
- estrazione delle feature degli archi;
- costruzione dei grafi PyTorch Geometric;
- generazione dei dataset PyG serializzati;
- debugger Streamlit con visualizzazione del grafo e della scacchiera.

## Struttura Cartelle

```text
.
├── main.py
├── README.md
├── artifacts/
│   ├── move_to_idx.json
│   ├── idx_to_move.json
│   └── move_encoder_stats.json
├── src/
│   ├── download/
│   │   ├── download_games.py
│   │   └── download_puzzles.py
│   ├── preprocess/
│   │   ├── preprocess_puzzles.py
│   │   ├── parse_games.py
│   │   ├── clean_games.py
│   │   ├── clean_puzzles.py
│   │   ├── prepare_games_dataset.py
│   │   └── prepare_puzzles_dataset.py
│   └── graph/
│       ├── move_encoder.py
│       ├── node_features.py
│       ├── edge_features.py
│       ├── graph_builder.py
│       ├── pyg_dataset.py
│       └── debug/
│           └── streamlit_graph_debugger.py
└── TimeGNN-main/
```

`TimeGNN-main/` è trattata come libreria esterna/vendor e non fa parte del codice sviluppato direttamente in questa pipeline.

## Pipeline Completa

Il comando principale esegue gli step in sequenza:

```bash
python3 main.py
```

La pipeline scarica i dati raw, crea i CSV processati/finali, costruisce la vocabulary delle mosse, genera le feature di nodi e archi, costruisce oggetti `torch_geometric.data.Data` e salva i dataset PyG in `data/pyg/`.

Per i puzzle Lichess, la colonna `Moves` viene interpretata così: `Moves[0]` è la mossa che porta dalla FEN originale alla posizione effettivamente mostrata al solver/modello; `Moves[1]` è la prima mossa della soluzione e diventa `TargetMove`. Il dataset finale conserva `OriginalFEN` per debug e usa `FEN` come posizione trasformata dopo `Moves[0]`.

## Debugger Streamlit

Il debugger è un'applicazione standalone:

```bash
streamlit run src/graph/debug/streamlit_graph_debugger.py
```

Non deve essere importato in `main.py`, perché Streamlit esegue codice UI già in fase di import.

## Rappresentazione del Grafo

Ogni posizione è rappresentata come grafo PyTorch Geometric:

- 64 nodi, uno per ogni casella della scacchiera;
- `edge_index` sparso PyG, non una matrice di adiacenza;
- `edge_attr` multilabel per descrivere le relazioni tra caselle, con un solo edge aggregato per coppia `(src, dst)`;
- oggetto finale: `Data(x, edge_index, edge_attr, y, global_features)`.

`global_features` è salvato con shape `[1, 4]` per ogni grafo, così il batching PyG produce un tensore `[batch_size, 4]`.

## Feature Nodi

Ogni nodo/casella contiene:

- piece one-hot: pawn, knight, bishop, rook, queen, king;
- color;
- occupied;
- normalized row;
- normalized column;
- attacked_by_white;
- attacked_by_black;
- legal_mobility, calcolata per entrambi i colori senza modificare permanentemente la board;
- is_pinned;
- piece_value.

## Feature Archi

Ogni arco può codificare:

- legal_move;
- attack;
- defend;
- pin;
- check_line.

## Target

`TargetMove` è la prima mossa UCI della soluzione del puzzle Lichess, cioè `Moves[1]` dopo aver applicato `Moves[0]` alla FEN originale.

`move_to_idx` viene costruito esclusivamente sul training set, così la vocabulary delle classi dipende solo dai dati di train. Se `artifacts/move_to_idx.json` e `artifacts/idx_to_move.json` vengono versionati, servono a fissare una vocabulary stabile delle mosse tra generazioni del dataset, training e valutazione.

I target di validation/test non presenti nella vocabulary del training set vengono conteggiati come OOV durante la generazione PyG e non vengono aggiunti alla vocabulary, per evitare leakage.

## Dipendenze Principali

Le librerie principali usate dal progetto sono:

- Python 3;
- pandas;
- requests;
- tqdm;
- zstandard;
- python-chess;
- scikit-learn;
- torch;
- torch-geometric;
- streamlit;
- streamlit-agraph.

## Installazione

Creare e attivare un ambiente virtuale:

```bash
python3 -m venv venv
source venv/bin/activate
```

Installare le dipendenze principali:

```bash
pip install pandas requests tqdm zstandard python-chess scikit-learn torch torch-geometric streamlit streamlit-agraph
```

Nota: l'installazione di `torch` e `torch-geometric` può dipendere dalla versione di Python, dal sistema operativo e dall'eventuale supporto CUDA.

## Esecuzione

Pipeline completa:

```bash
python3 main.py
```

Debugger grafico:

```bash
streamlit run src/graph/debug/streamlit_graph_debugger.py
```

Il debugger richiede che siano già stati generati almeno i dataset finali e il move encoder.

## File Generati e Gitignore

I dati e gli artifact temporanei possono diventare grandi e dovrebbero essere ignorati in git quando non servono esplicitamente al versionamento:

```text
data/
data/pyg/
lib/
__pycache__/
*.pyc
.streamlit/
graph_visualization.html
```

Aggiungere inoltre eventuali file temporanei prodotti durante esperimenti, training o debug.

## Lavori Futuri

- validazione approfondita dei grafi;
- generazione dataset PyG completo;
- baseline MLP;
- baseline GCN;
- implementazione GAT;
- integrazione della libreria TimeGNN;
- generazione timing sintetici;
- studio timing vs no timing;
- valutazione per MateDepth;
- confronto con modelli LLM;
- stesura report finale.

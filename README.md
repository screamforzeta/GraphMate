# Progetto Damiani - Chess GNN

Repository Python per costruire grafi PyTorch Geometric da puzzle Lichess mate-in-1..5 e valutare una famiglia di modelli GNN/GAT per predire la prossima mossa corretta.

La fase GNN è congelata. La prossima fase è il benchmark held-out / LLM.

## Obiettivo

Il progetto studia:

- rappresentazione a grafo di posizioni chess;
- modelli no-timing e timing-aware;
- ablation timing sintetico vs no timing;
- futuro confronto con LLM su puzzle mate-in-n held-out.

## Stato Attuale

Completato:

- download puzzle Lichess `.csv.zst`;
- download streaming PGN Lichess con campionamento;
- preprocessing puzzle mate-in-1..5;
- parsing/cleaning partite;
- split train/validation/test;
- vocabulary mosse costruita solo sul train set;
- feature nodi, archi e globali;
- dataset PyG sharded;
- validator rappresentazioni;
- Model A/A1/A2/A3/A4/B congelati e documentati;
- terminal evaluation A3 vs A4;
- timing ablation Model B;
- Streamlit debugger/app con filtro MateDepth, graph view, puzzle interaction, model/result dashboard.

Non ancora completato:

- held-out classic puzzle benchmark;
- protocollo LLM;
- confronto finale GNN-vs-LLM.

## Graph Representation

Ogni posizione è un `torch_geometric.data.Data`:

```text
Data(x, edge_index, edge_attr, y, global_features)
```

- `x`: `[64, 15]`, una casella = un nodo;
- `edge_index`: sparse PyG, non matrice di adiacenza;
- `edge_attr`: `[E, 5]`, multilabel;
- `global_features`: `[1, 4]`.

Feature nodi:

- one-hot pezzo: pawn, knight, bishop, rook, queen, king;
- color;
- occupied;
- normalized row/column;
- attacked_by_white;
- attacked_by_black;
- legal_mobility;
- is_pinned;
- piece_value.

Feature archi:

- legal_move;
- attack;
- defend;
- pin;
- check_line.

Global features:

- side_to_move;
- is_check;
- `min(fullmove_number, 200) / 200`;
- `min(halfmove_clock, 100) / 100`.

Semantica Lichess:

```text
OriginalFEN -> apply Moves[0] -> solver FEN -> TargetMove = Moves[1]
```

I modelli predicono la prossima mossa, non l'intera linea Mate-in-N autonoma.

## Model Family

| Model | Sintesi | Stato |
|---|---|---|
| A | GAT no-timing fixed-vocabulary classifier | frozen historical baseline |
| A1 | best-legal inference mode su Model A | historical diagnostic mode |
| A2 | legal-masked fixed-vocabulary model | frozen |
| A3 | legal-candidate scorer no-timing | frozen official no-timing baseline |
| A4 | frozen A3 Top-5 + post-move GAT reranker | frozen terminal best GNN |
| B | A3 + synthetic timing encoder | frozen timing ablation |

Risultati principali:

- A3 test Top1: `67.5609756097561%`;
- A4 test end-to-end Top1: `85.4123112659698%`;
- A4 delta vs A3: `+17.8513` pp;
- B synthetic timing Top1: `65.9814%`, quindi sotto A3 per questa rappresentazione timing sintetica.

## Struttura

```text
src/
  data/             # download, preprocessing, timing data
  graph/            # node/edge/global features, PyG dataset, Streamlit app
  models/           # model_a e model_b architectures
  training/         # training loops A/A2/A3/A4/B
  inference/        # inference adapters
  evaluation/       # frozen evaluation and ablations
  validation/       # representation validator
  streamlit_app/    # Streamlit-independent UI helpers
  cli/              # python -m entrypoints

generic_info/
  models/           # definitive frozen model documentation
  architecture/     # repository and architecture notes
  reference/        # project specification PDF
```

## Comandi

Pipeline principale:

```bash
python3 main.py
```

Validazione rappresentazioni:

```bash
./venv/bin/python -m src.validation.representations \
  --csv-sample 1000 \
  --graph-sample 500 \
  --seed 42
```

Streamlit:

```bash
streamlit run src/graph/debug/streamlit_graph_debugger.py
```

Model A3 evaluation:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_a_vs_a2_vs_a3 \
  --device cuda \
  --batch-size 128 \
  --non-blocking \
  --amp
```

Model A4 terminal evaluator is already completed; do not rerun casually. The entrypoint is:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_a3_vs_a4 \
  --device cuda \
  --batch-size 128 \
  --non-blocking \
  --amp \
  --run-terminal-test
```

Model B timing ablation:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_b_timing_ablation \
  --device cuda \
  --batch-size 128 \
  --non-blocking \
  --amp
```

## Documentazione Chiave

- [Graph representation](generic_info/models/graph_representation.md)
- [Frozen model family](generic_info/models/model_family.md)
- [Experimental protocol](generic_info/models/experimental_protocol.md)
- [Pre-LLM project status](generic_info/pre_llm_project_status.md)
- [Project specification coverage](generic_info/project_specification_coverage.md)

## Gitignore / Artifact Policy

Dati generati e artifact pesanti non devono essere versionati:

- `data/`
- `data/pyg/`
- `data/pyg_puzzles_timing/`
- `lib/`
- `__pycache__/`
- `*.pyc`
- `.streamlit/`
- `graph_visualization.html`
- file temporanei/cache.

Se `artifacts/move_to_idx.json` e `artifacts/idx_to_move.json` sono versionati, fissano la vocabulary delle mosse train-only.

## Prossima Fase

La fase GNN è pronta per handoff. Rimangono:

- costruire held-out classic puzzle set;
- implementare protocollo LLM;
- confrontare GNN e LLM con metriche coerenti;
- finalizzare analisi statistica e report.

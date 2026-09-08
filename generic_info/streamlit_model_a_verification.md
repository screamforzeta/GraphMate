# Streamlit Model A Verification

## Scopo

Questo documento descrive l'estensione dell'app Streamlit esistente con la sezione:

```text
Puzzle Training & Model Verification
```

La sezione consente training umano sui puzzle, inference read-only del baseline congelato `MODEL_A_NO_TIMING_FROZEN_BASELINE` e diagnostica specifica sui puzzle Mate-in-1. Non esegue training, fine-tuning o model selection.

## Existing Streamlit Framework

| Area | Stato prima della task |
| --- | --- |
| Entrypoint | `src/graph/debug/streamlit_graph_debugger.py` |
| Comando | `streamlit run src/graph/debug/streamlit_graph_debugger.py` |
| Pagina esistente | Singola app standalone `Chess GNN Graph Debugger` |
| Dataset | Caricamento CSV puzzle, originariamente solo `data/final/puzzles/train.csv` |
| Graph builder | `src.graph.graph_builder.build_graph` |
| Scacchiera | `python-chess` + `chess.svg.board` |
| Frecce board | `chess.svg.Arrow` costruite da `edge_index` e `edge_attr` |
| Grafo interattivo | `streamlit_agraph` con nodi casella e edge tattiche |
| Edge relations | `legal_move`, `attack`, `defend`, `pin`, `check_line` |
| Controlli esistenti | indice puzzle, random puzzle, modalità Chessboard/Graph/Both, checkbox edge |
| Stato | `st.session_state["current_idx"]` |
| Output | UI debug, nessun file pipeline scritto |

Limiti preesistenti:

- split fisso al train set;
- nessuna inference del modello;
- nessun input mossa utente;
- nessun reveal controllato della soluzione;
- nessuna diagnostica OOV o Mate-in-1 batch.

## New Model A Functionality

La nuova sezione è integrata nello stesso entrypoint Streamlit e riusa:

- loader CSV;
- FEN trasformato `sample.FEN`;
- renderer `chess.svg.board`;
- edge overlay tattici esistenti;
- graph builder ufficiale;
- classe ufficiale `ChessGATNoTiming`.

Non è stato creato un secondo frontend.

## Data Source

La sidebar permette di selezionare:

- `train`;
- `validation`;
- `test`.

Il default è `test`, etichettato come uso qualitativo/diagnostico. La UI non deve essere usata per future scelte di modello, perché Model A è già congelato e il test set è esplorabile.

Filtri disponibili:

- Mate-in-1;
- theme, quando presente in `Themes`;
- rating range.

La posizione mostrata è sempre il FEN trasformato `FEN`, cioè la posizione solver dopo la setup move Lichess. Non viene usato `OriginalFEN` come posizione del puzzle.

## Model Loader

Il modulo `src/inference/chess_gat_inference.py` gestisce il caricamento read-only.

Discovery checkpoint:

1. `artifacts/convergence_training/chess_gat_no_timing/best.pt`;
2. `artifacts/checkpoints/chess_gat_no_timing_best.pt`;
3. `artifacts/adaptive_training/chess_gat_no_timing/best_overall.pt`.

Se il checkpoint finale non è disponibile, la UI mostra `checkpoint unavailable` oppure segnala il fallback locale caricato. Il path ufficiale atteso resta:

```text
artifacts/convergence_training/chess_gat_no_timing/best.pt
```

Il loader usa `@st.cache_resource`, seleziona CUDA se disponibile e imposta `model.eval()`.

## Graph Input Reuse

L'inference usa la stessa rappresentazione scientifica:

```python
Data(
    x=[64, 15],
    edge_index=[2, E],
    edge_attr=[E, 5],
    y=graph_level_move_target,
    global_features=[1, 4],
)
```

Per target OOV, il grafo può essere costruito con una classe dummy solo per ottenere `x`, `edge_index`, `edge_attr` e `global_features`. Il target resta marcato come OOV e non viene inventato un rank.

## Human Puzzle Training

Tab:

```text
Human Puzzle Training
```

Flusso:

1. visualizza board e metadata;
2. ground truth nascosta;
3. input UCI utente;
4. validazione sintassi/legalità con `python-chess`;
5. confronto con `TargetMove`;
6. reveal solution opzionale.

Stati possibili:

- `INVALID_SYNTAX`;
- `ILLEGAL_MOVE`;
- `LEGAL_BUT_WRONG`;
- `CORRECT`.

La soluzione mostra UCI, SAN quando disponibile e continuation della linea puzzle già presente in `Moves`.

## Model A Inference

Tab:

```text
Model A Inference
```

Il pulsante `Ask Model A` esegue:

```text
puzzle row -> official graph builder -> PyG Batch -> ChessGATNoTiming -> raw logits
```

Output:

- Top-1;
- Top-3;
- Top-5;
- target rank;
- raw Top-K table;
- SAN se la mossa è legale;
- legal diagnostic;
- OOV warning.

La tabella include:

| Rank | Move | SAN | Softmax Score | Legal | Ground Truth |
| ---: | --- | --- | ---: | --- | --- |

Il softmax è etichettato come score non calibrato. Model A non è calibrato probabilisticamente.

## Raw Ranking vs Legal Diagnostic

Model A non usa legal-move masking. Per questo:

- Top1/Top3/Top5 ufficiali sono calcolati sui 1,786 logits raw;
- le mosse illegali non vengono filtrate prima del ranking;
- il target rank è calcolato sul vettore raw completo;
- `Best legal move among shown Model A outputs` è solo diagnostica.

## OOV Semantics

La vocabulary è train-only. Alcuni target validation/test sono fuori vocabulary:

- validation: 8;
- test: 10.

Quando `TargetMove` è OOV:

- il puzzle resta visualizzabile;
- l'utente può risolverlo;
- Model A non può predire quella classe;
- la UI mostra `Ground Truth OOV`;
- il caso non è trattato come normale errore Top-1.

## Mate-in-1 Mode

Tab:

```text
Mate-in-1 Evaluation
```

La UI usa il metadata reale:

- `mateIn1` in `Themes`;
- oppure `MateDepth == 1`.

Per il puzzle corrente verifica anche con `python-chess`:

```python
board.push(TargetMove)
board.is_checkmate()
```

Se il metadata indica Mate-in-1 ma la mossa target non produce checkmate, viene mostrato un warning diagnostico senza alterare il dataset.

## Batch Mate-in-1 Evaluation

Il pulsante:

```text
Evaluate Model A on Mate-in-1 puzzles
```

esegue inference batched con PyG `DataLoader(batch_size=128)`.

Metriche diagnostiche:

- total Mate-in-1 rows;
- evaluable rows;
- OOV;
- Top1;
- Top3;
- Top5;
- illegal Top1;
- legal Top1;
- average target rank;
- median target rank;
- breakdown: Correct Top1, Wrong but legal Top1, Illegal Top1, OOV target.

Questa valutazione è una `diagnostic subgroup evaluation`, non un nuovo risultato ufficiale Model A.

## Session State

State gestito:

- `current_idx`;
- `selected_split`;
- `user_move`;
- `last_user_result`;
- `revealed_solution`;
- `model_result`;
- `session_counters`;
- `mate_eval_metrics`.

Il cambio puzzle resetta risposta utente, reveal e inference corrente, ma mantiene i contatori della sessione.

## Error Handling

Gestito in UI/helper:

- checkpoint mancante;
- fallback checkpoint locale;
- vocabulary mancante;
- target OOV;
- input UCI invalido;
- mossa illegale;
- filtro vuoto;
- failure inference;
- failure batch Mate-in-1.

## Limitazioni

- L'interazione mossa usa input UCI, non click sulla board.
- La board principale può mostrare overlay utente, Top-1 modello e ground truth; Top-3/Top-5 sono disponibili in tabella.
- La UI può esplorare il test set, quindi non deve guidare future ottimizzazioni manuali.
- Se manca il checkpoint ufficiale di convergence, i risultati locali dipendono dal fallback caricato.

## Comando

```bash
streamlit run src/graph/debug/streamlit_graph_debugger.py
```

```text
STREAMLIT_RUN_COMMAND = streamlit run src/graph/debug/streamlit_graph_debugger.py
```

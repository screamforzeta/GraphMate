# MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING

## Obiettivo

Model A3 è una variante sperimentale no-timing che mantiene un encoder GAT stile Model A, ma sostituisce il classificatore globale su `1786` classi con uno scorer esplicito sulle sole mosse legali della posizione.

La domanda scientifica è: un modello che confronta direttamente i candidati legali, invece di produrre logits sull'intera vocabulary, migliora il ranking della prima mossa corretta del puzzle?

## Differenze Rispetto A e A2

| Modello | Output | Legalità | Timing |
|---|---|---|---|
| Model A | logits globali sulla vocabulary train-only | diagnostica post-hoc | no |
| Model A2 | logits globali sulla vocabulary train-only | mask in loss/metriche | no |
| Model A3 | score variabile per mossa legale | nativa per costruzione | no |

A3 non usa la vocabulary globale come spazio di output. `move_to_idx.json` resta disponibile solo per compatibilità, audit e confronto con i baseline precedenti.

## Architettura

Encoder board:

```text
GATConv(15 -> 32, heads=4, edge_dim=5) -> 128
ELU
Dropout(0.30)
GATConv(128 -> 32, heads=4, edge_dim=5) -> 128
ELU
global_mean_pool(H)
concat(global_features) -> graph_context 132
```

Candidate representation:

```text
h_src 128
h_dst 128
graph_context 132
promotion_embedding 8
= 396
```

Scorer:

```text
Linear(396, 128)
ELU
Dropout(0.30)
Linear(128, 1)
```

Le promozioni sono distinte con embedding trainabile:

- none;
- queen;
- rook;
- bishop;
- knight.

## Target E Candidati

I candidati legali sono enumerati da `Data.fen` con `python-chess`.

Il target ufficiale è `Data.target_move`, cioè la prima mossa UCI della soluzione vista nella posizione post-setup del puzzle. Per ogni grafo A3 valida:

```text
target_move in legal_moves(Data.fen)
```

Se il target non è legale, la run fallisce con errore esplicito. I sample non vengono saltati silenziosamente.

## Loss

La loss è una cross entropy grouped/listwise:

```text
-log exp(score_target) / sum(exp(score_legal_moves_same_position))
```

I candidati di posizioni diverse non competono tra loro.

## Metriche

Metriche ufficiali:

- candidate CE / NLL loss;
- Top1;
- Top3;
- Top5;
- mean legal target rank;
- median legal target rank;
- illegal_top1_rate = 0.0 per costruzione.

Quando una posizione ha meno di `k` mosse legali, Top-k usa tutte le mosse disponibili senza candidati fittizi.

## Training

Comando principale:

```bash
./venv/bin/python -m src.cli.training.train_model_a3_legal_scorer \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp
```

Resume:

```bash
./venv/bin/python -m src.cli.training.train_model_a3_legal_scorer --resume --device cuda
```

Config fissa:

- optimizer Adam;
- learning rate `5e-4`;
- weight decay `1e-4`;
- dropout `0.30`;
- batch size `128`;
- seed `42`;
- ReduceLROnPlateau factor `0.5`, patience `3`, min LR `1e-6`;
- early stopping patience `12`;
- max epochs `300`;
- AMP abilitato su CUDA;
- test eseguito solo dopo training terminale.

## Artifact

Cartella dedicata:

```text
artifacts/model_a3_legal_move_scorer_no_timing/
  experiment_config.json
  controller_state.json
  history.json
  best.pt
  last.pt
  final_report.json
  final_report.md
```

## Note Di Compatibilità

Il dataset PyG attuale contiene già `fen` e `target_move`, quindi A3 può costruire candidati legali senza modificare gli split o la vocabulary. Gli OOV storici di validation/test esclusi dai file PyG non vengono recuperati automaticamente da questa prima implementazione, perché richiederebbero un evaluation frame separato costruito dal CSV sorgente.

Per un confronto scientifico equo con Model A e Model A2, usare lo shared test set già presente in `data/pyg/test`.

# MODEL_A2_LEGAL_MASK_NO_TIMING

## Hypothesis

`MODEL_A_NO_TIMING_FROZEN_BASELINE` usa una classificazione globale su 1,786 classi mossa senza vincolo di legalità. La post-hoc error analysis indica che una quota rilevante dei raw Top-1 è illegale e che il best-legal diagnostic aumenta Top1 rispetto al raw score.

La domanda sperimentale è:

```text
Quanto migliora la stessa GNN se durante training, validation e test competono solo classi legalmente ammesse nella posizione?
```

## Controlled Variables

Restano invariati:

- graph representation;
- node features `[64,15]`;
- edge features `[E,5]`;
- global features `[1,4]`;
- vocabulary train-only da 1,786 classi;
- train/validation/test split;
- backbone `ChessGATNoTiming`;
- GAT layers, heads, hidden dims, pooling e classifier;
- optimizer `Adam`;
- initial LR `5e-4`;
- weight decay `1e-4`;
- dropout `0.30`;
- batch size `128`;
- seed `42`;
- scheduler `ReduceLROnPlateau`;
- early stopping patience `12`;
- AMP/runtime policy.

L'unico cambiamento scientifico è:

```text
LEGAL MOVE MASKING
```

## Legal Mask Semantics

Per ogni grafo:

1. si legge il FEN solver già presente in `Data.fen`;
2. si ricostruisce `chess.Board(fen)`;
3. si enumerano `board.legal_moves`;
4. ogni mossa UCI viene convertita in classe se presente in `move_to_idx`;
5. si costruisce `legal_mask [B,1786]`.

La loss è:

```python
masked_logits = logits.masked_fill(~legal_mask, -1e4)
loss = CrossEntropyLoss(masked_logits, target)
```

Il valore `-1e4` è finito e più stabile di `-inf` in scenari AMP/fp16.

## Target Validation

Per ogni sample evaluable:

```text
legal_mask[target_idx] == True
```

Se il target non è ammesso, il training/eval solleva errore. Gli OOV mantengono la semantica Model A:

- train: atteso `0`;
- validation: atteso `8`;
- test: atteso `10`.

La vocabulary non viene ampliata.

## Training Protocol

Entrypoint:

```bash
./venv/bin/python -m src.cli.training.train_model_a2_legal_mask --device cuda
```

Artifact separati:

```text
artifacts/model_a2_legal_mask_no_timing/
```

Output:

- `experiment_config.json`;
- `controller_state.json`;
- `history.json`;
- `best.pt`;
- `last.pt`;
- `final_report.json`;
- `final_report.md`.

A2 viene addestrato da zero. Non carica i pesi di Model A.

## Official Metrics

Le metriche ufficiali A2 sono calcolate sui logits masked:

- masked loss;
- masked Top1;
- masked Top3;
- masked Top5.

Metriche raw conservate solo come diagnostica:

- raw Top1/3/5;
- raw illegal Top1 rate;
- masked illegal Top1 rate.

La masked illegal Top1 rate attesa è `0%` se il mask è corretto.

## Comparison Frame

Il futuro report finale dovrà confrontare:

| Metric | Model A Raw | Model A Best-Legal Diagnostic | Model A2 Legal-Masked |
| --- | ---: | ---: | ---: |
| Global Top1 | 39.62% | 49.83% | TBD |
| Top3 | 55.81% | N/A | TBD |
| Top5 | 62.75% | N/A | TBD |
| MateIn1 Top1 | 47.22% | 56.14% | TBD |
| Illegal Top1 rate | ~30% | 0% by filtering | TBD |

La domanda primaria è:

```text
A2 masked Top1 > Model A diagnostic best-legal Top1?
```

## Implementation Files

| File | Ruolo |
| --- | --- |
| `src/training/legal_mask.py` | legal indices, dense mask, masked CE |
| `src/training/model_a2_legal_mask.py` | trainer/controller A2 separato |
| `src/train_chess_gat_legal_mask.py` | CLI A2 |
| `tests/test_model_a2_legal_mask.py` | unit test mask/loss |

## Caveat

Questa è una ablation no-timing controllata. Non cambia Model A, non modifica TimeGNN e non ridefinisce le metriche frozen di Model A.

# Model A vs A2 Legal Evaluation

## Scopo

Questo evaluator completa il confronto diagnostico tra:

- `MODEL_A_NO_TIMING_FROZEN_BASELINE`;
- `MODEL_A2_LEGAL_MASK_NO_TIMING`.

Non esegue training, fine-tuning o checkpoint selection. È solo una post-hoc evaluation sul test set ufficiale.

## Checkpoint Richiesti

Model A:

```text
checkpoints/model_a/best.pt
```

Model A2:

```text
checkpoints/model_a2/best.pt
```

L'evaluator non usa fallback per produrre metriche ufficiali.

## Test Set

Usa:

- `data/pyg/test`;
- `data/final/puzzles/test.csv`;
- `resources/move_encoder/move_to_idx.json`.

Atteso:

- graph test evaluable: `8610`;
- OOV storico CSV: `10`.

## Metriche

Per Model A:

- raw Top1/Top3/Top5;
- raw illegal Top1 rate;
- best-legal Top1/Top3/Top5.

Per Model A2:

- masked Top1/Top3/Top5;
- masked illegal Top1 rate.

La metrica primaria non confronta A raw vs A2 raw. A2 raw resta solo diagnostica perché la policy ufficiale A2 è masked.

## Best-Legal Definition

Best-legal TopK:

1. prende i raw logits Model A;
2. costruisce la legal mask canonica da `src/training/legal_mask.py`;
3. rimuove le classi illegali;
4. calcola TopK sulle classi legali ordinate dai logits originali.

Non c'è softmax requirement: il ranking non cambia.

## Parity

Model A reference:

- Top1 `0.3961672474213732`;
- Top3 `0.5580720094022851`;
- Top5 `0.6275261325010993`;
- N `8610`.

Model A2 reference:

- masked Top1 `0.4925667828106852`;
- masked Top3 `0.7149825784802852`;
- masked Top5 `0.8117305459461146`;
- masked illegal Top1 rate `0.0`;
- N `8610`.

Il report salva `PASS/FAIL` per entrambe le parity.

## Subgroup

L'evaluator accumula nello stesso pass:

- global metrics;
- MateDepth: `mateIn1` ... categorie presenti;
- rating buckets: `<1200`, `1200-1599`, `1600-1999`, `2000-2399`, `2400+`;
- target legal rank medio/mediano e bucket.

## Output

Directory:

```text
artifacts/model_a_vs_a2_evaluation/
```

File:

- `summary.json`;
- `report.md`.

## Comando

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_a_vs_a2 \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp
```

## Guardrail

Model A e A2 restano frozen. Questa analisi serve a rispondere quantitativamente se A2 ordina meglio le mosse legali rispetto a Model A filtrato post-hoc, ma non deve essere usata per scegliere nuovi hyperparameter.

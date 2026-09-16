# Model A3 Post-Move Error Audit

Questo documento descrive l'audit diagnostico post-hoc per capire se gli errori residui di `MODEL_A3_LEGAL_MOVE_SCORER_NO_TIMING` siano spiegabili con informazioni disponibili dopo la candidate move.

## Scopo

L'audit non implementa A4 e non modifica A3. Usa il test set ufficiale solo per analisi descrittiva dei failure modes.

```text
POST_HOC_TEST_SET_ANALYSIS = YES
A4_IMPLEMENTED = NO
```

Se in futuro verra implementato A4, le decisioni di training e hyperparameter selection dovranno essere prese su validation, non iterando sul test set ufficiale.

## Metodologia

L'audit ricostruisce il ranking A3 ufficiale e verifica parity prima di produrre conclusioni.

Livelli analizzati:

- Level 0: ranking A3 originale e feature candidate gia implicite nella posizione iniziale.
- Level 1: feature deterministiche dopo aver applicato la candidate move.
- Level 2: statistiche aggregate su tutte le risposte legali dell'avversario, senza Stockfish.

Il focus e `Top5`, perche il target ufficiale e gia nella Top5 in circa `91.8583%` del test set. Gli errori con target oltre Top5 sono marcati come `TOP5_RERANKER_UNRECOVERABLE`.

## Feature Level 1

Level 1 include:

- capture, promotion, castling, check, checkmate;
- material delta immediato;
- opponent legal move count;
- checkmate/stalemate risultante;
- attacco/difesa del pezzo mosso;
- material balance post-move;
- mobilita legale e pezzi attaccati per entrambi i lati.

## Feature Level 2

Level 2 enumera tutte le risposte legali dell'avversario e calcola solo aggregati strutturali:

- numero risposte;
- frazione di risposte che catturano il pezzo candidato;
- frazione di risposte che danno check o mate;
- material balance dopo risposta;
- max/mean immediate material loss;
- numero di forcing responses.

Una forcing response e definita come risposta che da check, da checkmate, oppure cattura il pezzo candidato.

## Output

```text
artifacts/model_a3_postmove_audit/
  summary.json
  level1_feature_analysis.json
  level2_feature_analysis.json
  pairwise_target_vs_wrong.json
  bucket_analysis.json
  runtime.json
  report.md
```

## Comando

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_a3_postmove_audit \
  --device cuda \
  --batch-size 128 \
  --non-blocking \
  --amp
```

## Interpretazione

L'audit classifica l'evidenza in:

- `POST_MOVE_RERANKER_SUPPORTED`;
- `OPPONENT_RESPONSE_RERANKER_SUPPORTED`;
- `EVIDENCE_INCONCLUSIVE`.

La classificazione e descrittiva. Non va interpretata come nuovo risultato ufficiale e non sostituisce l'accuracy A3.

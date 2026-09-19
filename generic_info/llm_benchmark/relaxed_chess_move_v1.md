# relaxed_chess_move_v1

`relaxed_chess_move_v1` è una valutazione secondaria post-hoc delle risposte LLM già prodotte dai benchmark ufficiali.

La metrica primaria resta invariata:

- `strict_uci_v1`
- exact canonical Top-1
- singolo token UCI
- risultati ufficiali in `artifacts/llm_benchmark/official/...`

La metrica relaxed non sostituisce la strict. Serve a separare:

- fallimenti di formato;
- recupero di notazione scacchistica;
- legalità della mossa;
- correttezza rispetto al `TargetMove` canonico.

## Principio

Il parser usa solo:

- `raw_final_content`
- FEN corrente del puzzle

Non usa:

- target move per scegliere una interpretazione;
- MateDepth;
- rating;
- themes;
- solution line;
- future moves;
- Stockfish;
- GNN;
- legal moves nel prompt.

Regola chiave:

`PARSE FIRST, SCORE SECOND`

Il target viene letto solo dall'evaluator, dopo che il parser ha già determinato una mossa univoca oppure un fallimento.

## Output

La CLI post-hoc scrive artifact separati:

```text
artifacts/llm_benchmark/analysis/relaxed_chess_move_v1/<model_id>/
  predictions_relaxed.jsonl
  summary.json
  parser_coverage.json
  audit.json
```

Non sovrascrive gli artifact ufficiali.

## Stato

`RELAXED_PARSER_STATUS = PRE_FREEZE_PENDING_SERVER_AUDIT`

Lo status deve diventare `FROZEN` solo dopo il pre-freeze audit sui prediction ufficiali Qwen 4B/9B. Da quel momento, ogni modifica semantica dovrà usare una nuova versione, ad esempio `relaxed_chess_move_v2`.

## Regole Di Parsing

1. Safe outer normalization minimale
   - strip whitespace;
   - normalizzazione CR/LF;
   - rimozione di backtick markdown solo se racchiudono tutta la risposta;
   - rimozione di quote esterne solo se racchiudono tutta la risposta;
   - nessuna normalizzazione di notazione scacchistica.

2. `EXACT_UCI`
   - accetta UCI canonico come `e2e4`, `g1f3`, `e7e8q`;
   - se sintatticamente UCI ma illegale, registra `ILLEGAL_MOVE`;
   - non cerca UCI come substring dentro testo arbitrario.

3. `EXACT_SAN_RECHECK`
   - usa `python-chess` position-aware;
   - parse della risposta intera;
   - accetta solo SAN canonica;
   - esempi: `Nf3`, `exd5`, `Qh4#`, `e8=Q+`, `O-O`.

4. `PIECE_SOURCE_DESTINATION`
   - accetta solo una risposta intera del tipo `Nb8c6`, `Nb8-c6`, `Ra8-a7`;
   - richiede prefisso pezzo, source square e destination square;
   - verifica che il source contenga il pezzo indicato del side-to-move;
   - verifica che la mossa source-destination sia legale;
   - se il separatore è `x`, verifica che sia davvero una cattura;
   - non gestisce promozioni e non le indovina.

`TEXT_WRAPPED_MOVE` è disabilitato. Il parser non recupera prose come `The best move is Nf3`, non sceglie fra opzioni come `Nf3 or Qh5`, e non accetta substring interne come `garbageNf3garbage`, `bRc8` o `qf5`.

## Ambiguità

Il parser frozen non enumera opzioni da prose o substring. Se una risposta non corrisponde a una delle regole intere sopra:

- `parse_status = UNRECOVERABLE`
- `parsed_uci = null`
- nessuna scelta viene fatta.

## Summary

`summary.json` include:

- metriche strict originali;
- metriche relaxed;
- recuperi da strict parse error;
- delta accuracy relaxed-strict in punti percentuali;
- conteggi per metodo di parsing;
- conteggi per failure reason;
- accuracy per MateDepth;
- accuracy per rating bucket.

`parser_coverage.json` verifica che ogni record sia categorizzato:

- `categorized_records == total_records`
- `categorized_rate == 1.0`

100% categorized non significa 100% parsed: `UNRECOVERABLE`, `EMPTY` e `AMBIGUOUS` sono classificazioni valide.

## Comandi Server2

Audit/evaluation Qwen 4B:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_llm_relaxed \
  --predictions artifacts/llm_benchmark/official/next_move/qwen_3_5_4b/predictions.jsonl \
  --parser relaxed_chess_move_v1 \
  --output-dir artifacts/llm_benchmark/analysis/relaxed_chess_move_v1/qwen_3_5_4b
```

Audit/evaluation Qwen 9B:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_llm_relaxed \
  --predictions artifacts/llm_benchmark/official/next_move/qwen_3_5_9b/predictions.jsonl \
  --parser relaxed_chess_move_v1 \
  --output-dir artifacts/llm_benchmark/analysis/relaxed_chess_move_v1/qwen_3_5_9b
```

Freeze parser:

```bash
./venv/bin/python -m pytest tests/test_llm_relaxed_parsing.py -q
```

Pre-freeze audit finale:

```bash
./venv/bin/python -m src.cli.evaluation.audit_llm_relaxed_pre_freeze \
  --predictions artifacts/llm_benchmark/official/next_move/qwen_3_5_4b/predictions.jsonl \
  --predictions artifacts/llm_benchmark/official/next_move/qwen_3_5_9b/predictions.jsonl \
  --output-dir artifacts/llm_benchmark/analysis/relaxed_chess_move_v1/pre_freeze_audit
```

Il parser può essere considerato frozen solo se:

- test PASS;
- `parser_coverage.json` ha `categorized_rate = 1.0`;
- le recovery rules non sono state modificate sulla base della correctness.

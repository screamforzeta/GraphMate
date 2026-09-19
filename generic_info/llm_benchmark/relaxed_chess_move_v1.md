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

## Stages Di Parsing

1. Safe outer normalization
   - strip whitespace;
   - normalizzazione CR/LF;
   - rimozione di backtick markdown solo se racchiudono tutta la risposta;
   - rimozione di quote esterne solo se racchiudono tutta la risposta;
   - rimozione di punteggiatura testuale finale sicura.

2. Strict UCI
   - accetta UCI canonico come `e2e4`, `g1f3`, `e7e8q`;
   - se sintatticamente UCI ma illegale, registra `ILLEGAL_MOVE`;
   - non reinterpreta UCI illegale come SAN.

3. Exact SAN
   - usa `python-chess` position-aware;
   - accetta solo SAN canonica per evitare permissività eccessiva;
   - esempi: `Nf3`, `exd5`, `Qh4#`, `e8=Q+`, `O-O`.

4. Safe SAN normalization
   - `0-0 -> O-O`;
   - `0-0-0 -> O-O-O`;
   - rimozione annotation glyph finali: `!`, `?`, `!!`, `??`, `!?`, `?!`;
   - valida solo se la SAN risultante identifica una mossa legale unica.

5. Safe UCI normalization
   - `e2-e4 -> e2e4`;
   - `e2 e4 -> e2e4`;
   - `e7-e8=Q -> e7e8q`;
   - malformed coordinate capture esplicita, ad esempio `e4xd5`, solo se origine/destinazione producono una mossa legale unica.

6. Text wrapper recovery
   - recupera una risposta testuale solo se contiene esattamente una singola mossa legalmente interpretabile;
   - esempi recuperabili: `The best move is Nf3`, `My move: e2e4`;
   - esempi ambigui: `Nf3 or Qh5`, `I considered Nf3 but Qh5 is better`.

## Ambiguità

Se più di una mossa legale plausibile è presente:

- `parse_status = AMBIGUOUS`
- `parsed_uci = null`
- nessuna scelta viene fatta, anche se una candidata coincide con il target.

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

Il parser può essere considerato frozen solo se:

- test PASS;
- `parser_coverage.json` ha `categorized_rate = 1.0`;
- le recovery rules non sono state modificate sulla base della correctness.

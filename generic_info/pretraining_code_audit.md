# Pretraining Code Audit

Audit del codice del progetto prima della fase di training. `TimeGNN-main/` e stato letto solo per compatibilita, ma non va modificato.

## Executive Summary

Il progetto ha una pipeline chiara e quasi tutta eseguibile da root, con output ordinati in `data/` e `artifacts/`. Prima del training ci sono pero alcuni punti da sistemare o almeno validare con test:

| Severita | Area | Problema |
| --- | --- | --- |
| Alta | Target puzzle | `TargetMove = Moves[0]` e probabilmente semanticamente errato per il CSV puzzle Lichess: la prima mossa spesso e la mossa che porta alla posizione puzzle, mentre la soluzione inizia dalla seconda mossa. |
| Alta | Move vocabulary | `move_to_idx` e costruito solo su train, ma `pyg_dataset.py` prova a codificare val/test e scarta ogni target non visto nel train. Questo puo alterare validation/test e nascondere classi rare. |
| Alta | Graph semantics | `edge_attr` e documentato come multilabel, ma il codice crea edge separati e duplicati per ogni relazione. PyG supporta edge paralleli, ma il modello dovra gestirli consapevolmente. |
| Media-Alta | Node features | `legal_mobility` usa `board.legal_moves`, quindi calcola mobilita solo per il lato al tratto. I pezzi dell'altro colore ricevono sempre 0, anche se hanno pseudo/legal mobility dopo cambio turno. |
| Media-Alta | Edge pin | `extract_pin_edges()` usa `board.attackers(opponent, pinned_square)` per trovare il pinner: non garantisce di identificare il pezzo che crea il pin e puo generare falsi positivi. |
| Media | PyG batching | `global_features` ha shape `[4]`; PyG puo concatenare in `[batch_size * 4]` invece di produrre `[batch_size, 4]`. Per training graph-level conviene verificarlo o salvare `[1, 4]`. |
| Media | Download | File parziali non sono gestiti: se un download si interrompe, la pipeline vede il file esistente e lo considera valido. |
| Media | Performance | Feature extraction itera `board.legal_moves` 64 volte e rigenera board due volte per grafo; su scala grande sara un hotspot. |

## Scope

Moduli analizzati:

- `main.py`
- `src/download/download_puzzles.py`
- `src/download/download_games.py`
- `src/preprocess/preprocess_puzzles.py`
- `src/preprocess/parse_games.py`
- `src/preprocess/clean_games.py`
- `src/preprocess/clean_puzzles.py`
- `src/preprocess/prepare_games_dataset.py`
- `src/preprocess/prepare_puzzles_dataset.py`
- `src/graph/move_encoder.py`
- `src/graph/node_features.py`
- `src/graph/edge_features.py`
- `src/graph/graph_builder.py`
- `src/graph/pyg_dataset.py`
- `src/graph/debug/streamlit_graph_debugger.py`
- `src/utils.py`
- `generic_info/timegnn_info.md`
- lettura mirata di `TimeGNN-main/src/timegnn/`

## 1. Pipeline dati

### Stato

`main.py` esegue la pipeline in questo ordine:

1. download puzzle;
2. download sample partite;
3. preprocessing puzzle;
4. parsing partite;
5. cleaning partite;
6. cleaning puzzle;
7. split puzzle;
8. split partite;
9. move encoder;
10. test/manual run node features;
11. test/manual run edge features;
12. test/manual run graph builder;
13. generazione dataset PyG.

La separazione UI/pipeline e corretta: il debugger Streamlit non viene importato in `main.py`.

### Findings

| Severita | Finding | Dettaglio | Impatto |
| --- | --- | --- | --- |
| Media | `node_features.main()`, `edge_features.main()`, `graph_builder.main()` sono test/manual run dentro pipeline | In `main.py` vengono eseguiti come step della graph section, ma stampano test locali e `graph_builder.main()` dipende dal move encoder e da una mossa hardcoded. | La pipeline puo fallire se `f3e5` non e nella vocabulary train, e produce log non necessari. |
| Media | Idempotenza non uniforme | Molti step saltano se output esiste; `prepare_games_dataset.py`, `prepare_puzzles_dataset.py`, `move_encoder.py`, `pyg_dataset.py` invece sovrascrivono. | Riesecuzioni parziali possono lasciare output incoerenti tra loro. |
| Media | Path hardcoded relativi | Tutti i path sono relativi alla root repo. | Funziona da root, ma fallisce se eseguito da sottocartelle. README indica root, quindi accettabile per ora. |
| Bassa | Side effects a import | Molti moduli creano directory a import time con `OUTPUT_DIR.mkdir(...)`. | Non cambia la logica, ma gli import non sono puramente passivi. |

### Raccomandazioni pre-training

- Prima del training, evitare di chiamare test/manual `main()` nella pipeline principale o garantire che siano innocui.
- Aggiungere un controllo di coerenza pipeline: input esiste, output aggiornato, numero righe atteso.
- Centralizzare in futuro i path/config, ma non e necessario farlo prima della prima baseline se si esegue sempre da root.

## 2. Download

### `src/download/download_puzzles.py`

Punti positivi:

- usa `requests.get(..., stream=True)`;
- scrive chunk da 1 MB;
- usa progress bar con `content-length`.

Rischi:

| Severita | Finding | Impatto | Fix futuro |
| --- | --- | --- | --- |
| Media | Nessuna gestione file parziale | Un `.csv.zst` incompleto viene considerato valido solo perche esiste. | Scaricare in `.part`, rinominare solo a download completato. |
| Media | Nessun timeout/retry | Connessioni bloccate o instabili possono appendere la pipeline. | `timeout=(connect, read)` e retry/backoff. |
| Bassa | Nessuna validazione post-download | Corruzione compressione scoperta solo al preprocessing. | Test rapido decompress o checksum se disponibile. |

### `src/download/download_games.py`

Punti positivi:

- stream decompress con `zstandard`;
- non salva il file da circa 30GB;
- interrompe dopo `MAX_GAMES`.

Rischi:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Media | File parziale non gestito | Se l'interruzione avviene dopo apertura output, al run successivo `OUTPUT_PGN.exists()` fa skip. |
| Media | Game boundary euristico | Usa righe che finiscono con `1-0`, `0-1`, `1/2-1/2`. In PGN reali con commenti/annotazioni puo funzionare spesso, ma non e parser robusto. |
| Bassa | `CHUNK_SIZE` inutilizzato | Costante definita ma non usata. Non impatta comportamento. |
| Bassa | Nessun timeout/retry | Come sopra. |

## 3. Preprocessing / Cleaning

### Puzzle

`preprocess_puzzles.py` legge chunk da 100k righe e filtra temi `mateIn1..mateIn5`.

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Media | Accumulo chunk in RAM | `filtered_chunks` raccoglie tutti i chunk filtrati prima del `concat`. Su dataset completo puo crescere molto. |
| Bassa | `.apply(lambda...)` su `Themes` | O(N * numero temi), ma accettabile con 5 temi; il collo di bottiglia maggiore e I/O/decompressione. |
| Bassa | `extract_mate_theme()` dipende dall'ordine del set | `VALID_THEMES` e un set, quindi l'ordine non e garantito. Se un puzzle avesse piu temi mate, `MateTheme` potrebbe non essere deterministico. |

`clean_puzzles.py` valida FEN, rating, mate depth, mosse non vuote e deduplica per `FEN, Moves`.

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Media | `chess.Board(fen)` accetta FEN parseable ma non necessariamente posizioni "sane" | Per training puo bastare, ma test su posizioni illegali/king status sarebbe utile. |
| Bassa | Dedup su `FEN, Moves` | Corretto per duplicati esatti; puzzle stesso con stesso FEN ma soluzione diversa rimane. |

`prepare_puzzles_dataset.py` bilancia per `MateDepth`, shuffle e split stratificato.

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Alta | Target probabilmente sbagliato | `extract_target_move()` ritorna `moves[0]` (righe 52-70). Nel dataset puzzle Lichess, la prima mossa della colonna `Moves` e spesso la mossa giocata per raggiungere la posizione da presentare; la soluzione del giocatore inizia dalla seconda mossa. Va verificato con esempi reali prima del training. |
| Media | Bilanciamento prima dello split | Va bene per bilanciare MateDepth, ma va testato che duplicati o posizioni correlate non finiscano in split diversi. |
| Media | Non stratifica per `TargetMove` | Classi rare possono essere assenti dal train o presenti solo in val/test. |

### Games

`parse_games.py` usa `python-chess` per leggere PGN e genera metadata/mosse/timing.

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Media | Timing da clock non considera incremento | `compute_move_times()` fa `prev_clock - curr_clock`; in partite con incremento puo sottostimare/annullare tempi quando il clock cresce. |
| Media | Primo move time mancante | Il primo clock non ha clock precedente, quindi la lista move_times ha lunghezza `len(clock_values)-1`. |
| Bassa | `board.push(move)` non usato dopo parsing | Serve solo a mantenere board ma non viene letto. Non dannoso. |

`clean_games.py` valida risultati, Elo, tipo partita e legalita delle mosse.

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Media | Validazione mosse costosa | Ricostruisce ogni partita con python-chess, necessario ma potenzialmente lento. |
| Bassa | Tipo partita da `Event` string matching | Funziona se Event contiene Bullet/Blitz/Rapid; dipende dal formato Lichess. |

`prepare_games_dataset.py` crea feature `BaseTime`, `Increment`, `AverageElo`, `GameLengthCategory` e split per `GameType`.

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Bassa | `GameLengthCategory` non usato per stratificazione | Potrebbe essere utile in futuro per timing. |
| Bassa | Coerenza con puzzle dataset | Games dataset e puzzle dataset sono indipendenti; per timing sintetici serve un passaggio esplicito che li colleghi o usi statistiche aggregate. |

## 4. Target move / Move encoder

### Target Lichess

Finding critico:

- Codice: `src/preprocess/prepare_puzzles_dataset.py`, `extract_target_move()`, righe 52-70.
- Comportamento attuale: `TargetMove = str(Moves).split()[0]`.
- Rischio: nel formato puzzle Lichess la sequenza `Moves` include la mossa iniziale che porta alla posizione del puzzle; la prima mossa della soluzione richiesta al giocatore e tipicamente la seconda mossa. Se confermato sui dati reali, tutto il training apprenderebbe il target sbagliato.

Check consigliato prima di cambiare:

```python
import chess
import pandas as pd

df = pd.read_csv("data/processed/puzzles/mate_puzzles_clean.csv").head(20)
for row in df.itertuples():
    board = chess.Board(row.FEN)
    moves = row.Moves.split()
    print(row.PuzzleId, board.turn, moves[:3])
    print("moves[0] legal in FEN:", chess.Move.from_uci(moves[0]) in board.legal_moves)
    board.push(chess.Move.from_uci(moves[0]))
    print("moves[1] legal after setup:", len(moves) > 1 and chess.Move.from_uci(moves[1]) in board.legal_moves)
```

### Vocabulary

`move_encoder.py` costruisce `move_to_idx` solo da `data/final/puzzles/train.csv`, corretto per evitare leakage.

Rischi:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Alta | Val/test target out-of-vocabulary | `graph_builder.encode_target_move()` solleva `ValueError` se target non e nel train vocabulary; `pyg_dataset.py` cattura l'errore e scarta il grafo. Questo cambia la distribuzione di validation/test. |
| Media | Output space UCI pieno puo essere grande | Le mosse UCI includono from/to e promozione (`e7e8q`). Classificazione single-head su tutte le UCI viste puo produrre molte classi sparse. |
| Media | Promozioni | UCI promozione usa 5 caratteri; il codice string-based le supporta se presenti nel train, ma out-of-vocab resta probabile. |
| Bassa | Castling | In UCI e `e1g1`, `e1c1`, ecc.; supportato come stringa. |
| Bassa | En passant | In UCI e normale from/to; supportato come stringa. |

Alternative robuste da valutare dopo audit, senza cambiare ora:

- vocabulary UCI globale generata da tutte le mosse legali teoriche, evitando OOV;
- classificazione from-square + to-square (`64 x 64`);
- head separata per promozione;
- masking delle mosse legali a inference/training.

## 5. Node features

Feature attuali:

- one-hot piece type: corretto;
- color: `1.0` white, `0.0` black o empty;
- occupied: distingue empty da black;
- row/col normalizzati `[0,1]`;
- attacked_by_white / attacked_by_black;
- legal_mobility;
- is_pinned;
- piece_value.

Findings:

| Severita | Finding | Codice | Impatto |
| --- | --- | --- | --- |
| Media-Alta | `legal_mobility` solo lato al tratto | `compute_legal_mobility()` itera `board.legal_moves`, righe 216-237. | Tutti i pezzi del lato non al tratto hanno mobility 0. Questa e una feature asimmetrica; puo essere voluta, ma il nome suggerisce mobilita del pezzo in generale. |
| Media | Costo mobility | `extract_node_features()` chiama `compute_legal_mobility()` per ogni square, righe 334-337; ogni chiamata itera tutte le legal moves. | Hotspot in generazione grafi. Precomputare una mappa `from_square -> count` una volta per board riduce costo senza cambiare semantica. |
| Media | Feature numeriche non tutte normalizzate | attackers, mobility e piece_value sono raw. | Scale diverse possono influenzare training; batch norm o normalizzazione feature possono aiutare. |
| Bassa | `color=0` per nero e vuoto | `occupied` disambigua, quindi non e semanticamente ambiguo per il modello se usa entrambe. |
| Bassa | `attacked_by_*` include attacchi pseudo-legali | `board.attackers()` non equivale sempre a mosse legali; per attacchi tattici e spesso appropriato. |

## 6. Edge features

Feature documentate:

- `legal_move`;
- `attack`;
- `defend`;
- `pin`;
- `check_line`.

Findings:

| Severita | Finding | Codice | Impatto |
| --- | --- | --- | --- |
| Alta | Non e davvero multilabel per coppia `(src,dst)` | Ogni extractor chiama `add_edge()` separatamente; stessa coppia puo comparire piu volte con feature diverse. | `edge_attr` e multilabel per edge record, non per relazione unica. PyG supporta edge paralleli, ma modelli/analisi devono aspettarseli. |
| Media-Alta | Pin pinner non robusto | `extract_pin_edges()` usa `board.attackers(not piece.color, square)` e poi controlla solo `square_distance >= 1`, righe 318-339. | Un attaccante del pezzo pinned non e necessariamente il pezzo che lo pinna al re; possibili falsi positivi. |
| Media | `check_line` non e l'intera linea | `extract_check_line_edges()` aggiunge solo `checker -> king`, righe 390-410. | Nome/documentazione "check_line" puo far pensare a tutti i quadrati sulla linea di scacco. |
| Media | Attack su caselle vuote ignorato | `extract_attack_defend_edges()` salta target vuoti, righe 221-230. | Il grafo non contiene controllo di spazio su caselle vuote, solo relazioni con pezzi occupanti. Design legittimo, ma limita informazione tattica. |
| Media | Legal move edges solo lato al tratto | `board.legal_moves`, righe 157-172. | Coerente se si vuole rappresentare le mosse disponibili ora; non rappresenta mobilita/possibilita dell'avversario. |
| Media | Edge duplicati con legal+attack | Una cattura legale puo apparire sia come `legal_move` sia come `attack` su edge parallelo. | Se si voleva edge multilabel unico, serve merging. Se si accettano edge paralleli, va testato e documentato nel training. |

Compatibilita PyG:

- `edge_index` shape `[2, E]` e `edge_attr` shape `[E, 5]` sono compatibili con `GATConv(edge_dim=5)`.
- Edge paralleli sono tecnicamente supportati da PyG.
- `GCNConv` non usa `edge_attr` 2D come feature; accetta edge weight scalare.
- `TimeAwareGATConv` di TimeGNN si aspetta timing scalare come `edge_attr`, non edge feature multilabel.

## 7. Graph builder / PyG Data

`build_graph()` crea:

```python
Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y,
)
graph.global_features = global_features
graph.fen = fen
graph.target_move = target_move
```

Shape attese:

| Campo | Shape/tipo | Stato |
| --- | --- | --- |
| `x` | `[64, 15]`, float | OK |
| `edge_index` | `[2, E]`, long | OK se esiste almeno un edge |
| `edge_attr` | `[E, 5]`, float | OK per GAT edge_dim=5 |
| `y` | scalar long tensor | OK per graph-level classification |
| `global_features` | `[4]`, float | Da verificare in batching |

Findings:

| Severita | Finding | Dettaglio |
| --- | --- | --- |
| Media | `global_features` shape `[4]` in Batch | PyG concatena attributi tensor lungo dimensione default; su N grafi potrebbe diventare `[4N]`. Per usarlo come feature graph-level conviene testare o salvare `[1,4]`. |
| Media | Attributi stringa in `Data` | `fen`, `target_move`, `puzzle_id` sono utili per debug. PyG Batch puo mantenerli come lista o comportarsi diversamente a seconda versione. Testare serializzazione/batching. |
| Media | Classificazione graph-level non implementata | Oggetto dati e pronto, ma manca modello che faccia pooling + classifier su `y`. |
| Media | `torch.save` lista intera | `pyg_dataset.py` accumula tutti i grafi in lista e salva con `torch.save`. Per 5k OK; per dataset completo puo saturare RAM. |
| Bassa | Board ricostruita due volte | `build_graph()` crea `chess.Board(fen)`, poi `extract_node_features(fen)` e `extract_edge_features(fen)` ricreano board. Hotspot moderato. |

## 8. Compatibilita con TimeGNN

Riferimenti:

- `generic_info/timegnn_info.md`
- `TimeGNN-main/src/timegnn/models/gat_basic.py`
- `TimeGNN-main/src/timegnn/models/gat_time_decay.py`
- `TimeGNN-main/src/timegnn/models/gat_time_decay_status_emb.py`
- `TimeGNN-main/src/timegnn/models/prefix_gcn.py`
- `TimeGNN-main/src/timegnn/recipes/gat_outcome.py`

### A. Compatibilita immediata

| Componente | Compatibilita |
| --- | --- |
| PyG `Data.x` | Compatibile. |
| PyG `Data.edge_index` | Compatibile. |
| `edge_attr` con `DualGATModel(edge_dim=5)` | Tecnicamente compatibile. |
| `torch_geometric.loader.DataLoader` | Compatibile. |

### B. Compatibilita con wrapper/adattatore

| Componente TimeGNN | Cosa serve |
| --- | --- |
| `DualGATModel` | Aggiungere `event_ids` ai nodi e pooling graph-level. |
| `TimeAwareGATConv` | Passare timing edge-level `[E,1]`; non puo ricevere direttamente `edge_attr` tattico `[E,5]` come tempo. |
| `DualGATTimeAwareModel` | Aggiungere `event_ids`, `time`, pooling graph-level e head coerente con `move_to_idx`. |
| `TimeAwareETGATConv` | Aggiungere timing + edge type embedding; serve conversione edge multilabel -> tipo discreto o layer custom. |
| `global_features` | Concatenazione dopo pooling, perche i modelli TimeGNN non le consumano nativamente. |

### C. Incompatibilita strutturale

| Oggetto | Perche |
| --- | --- |
| Ricette `timegnn.gat_basic(...)`, `gat_time_decay(...)`, ecc. | Accettano DataFrame event-log e creano grafi sequenziali, non caricano i nostri `train_graphs.pt`. |
| Training loop `models/training.py` | Pensato per label token-level con padding `-1`; il nostro target e graph-level. |
| `PrefixGCNClassifier` diretto | Graph-level, ma `GCNConv` vuole edge weight scalare; il nostro `edge_attr` e `[E,5]`. |

Adattatore minimo no-timing:

```python
data.event_ids = torch.arange(data.num_nodes, dtype=torch.long).view(-1, 1)
data.edge_attr = data.edge_attr.float()
```

Poi:

```python
node_logits = model(batch)
graph_logits = global_mean_pool(node_logits, batch.batch)
loss = criterion(graph_logits, batch.y)
```

Adattatore minimo timing:

```python
data.event_ids = torch.arange(data.num_nodes, dtype=torch.long).view(-1, 1)
data.time = synthetic_edge_time.float().view(-1, 1)
```

## 9. Algebraic Decoupling

Il nostro formato facilita una separazione utile:

- topology: `edge_index`;
- node state: `x`;
- edge state: `edge_attr`;
- graph state: `global_features`;
- future temporal state: `time` o `edge_time_diff`.

Questo e coerente con l'idea dual-path di TimeGNN, anche se non esiste un modulo chiamato formalmente "Algebraic Decoupling". Per non ostacolare l'integrazione futura, conviene mantenere questi canali separati e non fondere timing dentro `x` o dentro `edge_attr` tattico senza una convenzione chiara.

Raccomandazione: introdurre timing come attributo separato (`data.time` o `data.edge_time_diff`) e lasciare `edge_attr` per le relazioni scacchistiche.

## 10. Prestazioni

| Hotspot | Gravita | Motivo | Ottimizzazione futura | Rischio semantico |
| --- | --- | --- | --- | --- |
| `compute_legal_mobility()` | Alta su larga scala | Itera tutte le legal moves per ognuna delle 64 caselle. | Precomputare counts una volta per board. | Basso se si mantiene `board.legal_moves`. |
| `count_attackers()` 128 volte per grafo | Media | 64 caselle x 2 colori. | Precomputare attack maps o accettare costo. | Medio se cambia definizione attacco. |
| `extract_pin_edges()` | Media | Scansione 64 caselle + attackers. | Calcolo pin piu diretto lungo raggi king-slider. | Medio, perche cambierebbe semantica. |
| `clean_games.is_valid_moves()` | Media | Replay completo partita per ogni game. | Parallelizzare o campionare validazione; tenere test completo per dataset finale. | Basso. |
| `pyg_dataset.py` lista in RAM | Alta per dataset grande | Tutti i grafi restano in memoria prima del save. | Dataset on-disk, shard `.pt`, `InMemoryDataset` controllato. | Basso. |
| `preprocess_puzzles.py` concat chunk | Media | Accumula chunk filtrati. | Scrittura append streaming CSV. | Basso. |
| Download PGN | Media | Streaming ok, ma senza resume. | `.part` + retry. | Basso. |

## 11. Human readability

Punti positivi:

- moduli separati per download, preprocess, graph features e dataset PyG;
- nomi funzione chiari;
- README e docstring ora aiutano l'onboarding;
- path output leggibili.

Punti da migliorare solo se portano beneficio concreto:

| Area | Osservazione |
| --- | --- |
| Config | Costanti hardcoded sparse in molti file. Una config centrale aiuterebbe esperimenti/training. |
| Funzioni lunghe | `clean_*`, `prepare_*`, `build_pyg_dataset()` sono leggibili ma difficili da testare a pezzi. |
| Magic numbers | `MAX_SAMPLES=5000`, rating range, normalizzazioni 200/100, sample 1000 games andrebbero documentati/configurati per esperimenti. |
| Commenti sezionali | Molti separatori rendono chiaro il flusso, ma possono diventare rumorosi. Non e un problema pre-training. |
| `src/utils.py` | Placeholder vuoto; ok, ma se resta inutilizzato puo essere rimosso in futuro. |

## 12. Test mancanti prima del training

Priorita alta:

- test TargetMove su esempi reali Lichess: verificare se target corretto e `Moves[0]` o `Moves[1]`;
- test che tutti i target val/test siano rappresentabili oppure conteggio OOV esplicito;
- test shape `Data`: `x=[64,15]`, `edge_index=[2,E]`, `edge_attr=[E,5]`, `y` scalar long;
- test batching PyG con `global_features` e attributi stringa;
- test `legal_mobility` su posizione con pezzi di entrambi i colori;
- test edge duplicati: contare coppie `(src,dst)` duplicate e decidere se e atteso.

Priorita media:

- test `attack` e `defend` su posizione costruita manualmente;
- test pin assoluto con pinner noto;
- test falso positivo pin;
- test check-line: singolo checker e doppio check;
- test promozioni UCI in encoder;
- test castling UCI in encoder;
- test en passant come target legale;
- test split leakage su `PuzzleId`, `FEN`, `Moves`;
- test serializzazione `torch.save`/`torch.load` dei grafi;
- test adattatore TimeGNN no-timing;
- test adattatore TimeGNN timing.

## 13. Piano di intervento consigliato

Senza modificare ancora codice, questo e l'ordine consigliato:

1. Verificare semanticamente `TargetMove` con 20-100 esempi Lichess reali.
2. Misurare OOV target val/test rispetto a `move_to_idx`.
3. Validare batching PyG di `global_features`.
4. Scrivere test node/edge feature su posizioni note.
5. Decidere se edge paralleli sono una scelta voluta o se serve merge multilabel.
6. Implementare baseline graph-level semplice MLP/GCN/GAT.
7. Solo dopo, integrare wrapper TimeGNN no-timing.
8. Aggiungere timing sintetici come attributo separato.
9. Integrare `TimeAwareGATConv`.
10. Fare ablation timing vs no-timing sugli stessi split.

## Conclusione pratica

La pipeline e una buona base per arrivare al training, ma non e ancora pronta per produrre metriche affidabili. I due blocchi da risolvere prima sono:

1. verificare e correggere, se necessario, la semantica del target Lichess;
2. decidere come gestire la vocabulary delle mosse e gli OOV in validation/test.

Subito dopo vanno testati shape/batching PyG, `global_features`, mobility e edge duplicati. La compatibilita con `TimeGNN-main/` e realistica ma richiede wrapper: i nostri grafi sono PyG, pero i modelli TimeGNN non consumano direttamente il nostro formato graph-level con `global_features` e `edge_attr` multilabel.

## Fix status

| Finding principale | Stato | Motivazione |
| --- | --- | --- |
| Semantica `TargetMove = Moves[0]` | FIXED | `prepare_puzzles_dataset.py` ora applica `Moves[0]` alla FEN originale, conserva `OriginalFEN`, salva `FEN` trasformata e usa `Moves[1]` come `TargetMove` solo se setup e target sono legali. |
| OOV val/test rispetto a train vocabulary | FIXED | `move_encoder.py` misura OOV su validation/test; `pyg_dataset.py` conta e logga OOV esplicitamente senza espandere la vocabulary train. |
| Edge paralleli invece di multilabel per `(src,dst)` | FIXED | `edge_features.py` aggrega deterministicamente le feature con un solo edge per coppia. |
| `legal_mobility` solo lato al tratto | FIXED | `node_features.py` calcola una mappa mobility per White e Black usando copie della board e la riusa per tutti i nodi. |
| Pin pinner non robusto | FIXED | `edge_features.py` identifica pin assoluti con scansione geometrica king -> pinned piece -> slider enemy. |
| `check_line` ambiguo | VERIFIED | Il nome resta invariato per compatibilita; la semantica e documentata come edge checker -> king, non come enumerazione di tutta la linea. |
| `global_features` non batch-safe | FIXED | `graph_builder.py` ora salva `global_features` con shape `[1, 4]`, cosi PyG batch produce `[batch_size, 4]`. |
| Demo/manual tests in `main.py` | FIXED | `main.py` orchestra solo download/preprocess/encoder/dataset PyG; test manuali restano eseguibili dai singoli moduli. |
| File parziali nei download | DEFERRED | Non corretto in questa fase per evitare cambi I/O piu ampi; resta consigliato `.part` + rename atomico + retry. |
| Dataset PyG tutto in RAM | DEFERRED | Mantenuto formato `.pt` corrente come richiesto; sharding/on-disk dataset rimandato a dataset su scala maggiore. |
| Compatibilita TimeGNN diretta | VERIFIED | Resta compatibilita parziale: i grafi sono PyG, ma serviranno wrapper per `event_ids`, timing separato, pooling graph-level e `global_features`. |

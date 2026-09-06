# Architectural Choices

Questo documento spiega le principali decisioni implementative del progetto. Il documento complementare [project_architecture.md](project_architecture.md) descrive invece come è organizzata la repository.

## 1. Semantica Dei Puzzle Lichess

**Decision**  
Conservare `OriginalFEN`, trasformare `FEN` applicando `Moves[0]`, e usare `Moves[1]` come `TargetMove`.

**Rationale**  
Nei puzzle Lichess la FEN originale precede la mossa setup dell'avversario. Il solver vede la posizione dopo quella mossa. Il target corretto per il modello è quindi la prima risposta del solver, non la setup move.

**Alternatives Considered**  
Usare direttamente la FEN raw Lichess o usare `Moves[0]` come target avrebbe addestrato il modello su una semantica sbagliata.

**Trade-offs**  
Serve conservare sia `OriginalFEN` sia `FEN` per debug e tracciabilità.

**Current Status**  
Implemented. La pipeline valida legalità della setup move e legalità di `TargetMove`.

## 2. Graph Representation

**Decision**  
Rappresentare ogni posizione come grafo a 64 nodi, uno per casella.

**Rationale**  
La scacchiera ha una topologia fissa e interpretabile. Usare nodi-casella consente di rappresentare sia pezzi occupanti sia caselle vuote rilevanti per mobilità, attacchi e linee tattiche.

**Alternatives Considered**  
Usare solo pezzi come nodi ridurrebbe il numero di nodi, ma perderebbe una rappresentazione esplicita delle caselle vuote e renderebbe meno naturale modellare mosse e raggi.

**Trade-offs**  
Il grafo ha sempre 64 nodi anche in posizioni con pochi pezzi. È semplice e stabile, ma include nodi vuoti.

**Current Status**  
Implemented: `x [64,15]`, `edge_index [2,E]`, `edge_attr [E,5]`, `global_features [1,4]`.

## 3. Node Features

**Decision**  
Usare feature esplicite e semplici: one-hot pezzo, colore, occupazione, coordinate normalizzate, attacchi, mobilità legale, pin e valore pezzo.

**Rationale**  
Il baseline no-timing deve partire da una rappresentazione chess-specific leggibile, senza delegare al modello la scoperta di concetti tattici elementari da zero.

**Alternatives Considered**  
Feature più minimali avrebbero aumentato il carico sul modello. Feature più ricche rischierebbero leakage o crescita non controllata del design space.

**Trade-offs**  
Alcune feature sono engineered e quindi incorporano conoscenza scacchistica. È accettabile per un baseline GNN tattico.

**Current Status**  
Implemented. Non esiste feature nodo `is_check`; `is_check` è graph-level.

## 4. Tactical Edge Attributes

**Decision**  
Usare `edge_attr [E,5]` multilabel binario: `legal_move`, `attack`, `defend`, `pin`, `check_line`.

**Rationale**  
Una stessa coppia di caselle può avere più significati contemporaneamente. Per esempio una mossa legale può anche essere un attacco. Il multilabel conserva questa informazione.

**Alternatives Considered**  
Un singolo edge type discreto avrebbe obbligato a scegliere una relazione dominante o a moltiplicare edge duplicate.

**Trade-offs**  
Serve aggregare le relazioni per evitare duplicate edge. La semantica è più ricca ma richiede validazione.

**Current Status**  
Implemented. `edge_attr` viene passato direttamente a `GATConv(edge_dim=5)`.

## 5. Sparse Edge Index

**Decision**  
Usare `edge_index` sparse COO PyTorch Geometric, non matrice densa 64x64.

**Rationale**  
PyG lavora naturalmente con COO sparse. Le relazioni tattiche sono molto meno dense di una matrice completa.

**Alternatives Considered**  
Una matrice 64x64 sarebbe semplice da visualizzare ma meno naturale per PyG e più costosa da batching.

**Trade-offs**  
Serve validare duplicate edge e coerenza `edge_attr`.

**Current Status**  
Implemented.

## 6. `check_line` Semantics

**Decision**  
Codificare `check_line` come relazione checker -> king.

**Rationale**  
Per il baseline serve segnalare quale pezzo dà scacco al re. Enumerare tutte le caselle lungo il raggio sarebbe una scelta diversa e più invasiva.

**Alternatives Considered**  
Rappresentare tutte le caselle intermedie lungo la linea di scacco.

**Trade-offs**  
La feature attuale è compatta ma non codifica esplicitamente ogni casella del raggio.

**Current Status**  
Implemented.

## 7. Global Features Dopo Pooling

**Decision**  
Concatenare `global_features [B,4]` dopo `global_mean_pool`.

**Rationale**  
Side-to-move, check status e contatori FEN sono proprietà della posizione intera, non di una singola casella. Concatenarli dopo il pooling evita di replicarli su ogni nodo.

**Alternatives Considered**  
Replicare feature globali su tutti i nodi. Sarebbe possibile ma meno pulito.

**Trade-offs**  
Il GAT message passing non vede direttamente le global features; le usa il classifier finale.

**Current Status**  
Implemented. Campi: side_to_move, is_check, fullmove_number normalized, halfmove_clock normalized.

## 8. Move Vocabulary Train-Only

**Decision**  
Costruire `move_to_idx` solo dal train set.

**Rationale**  
La vocabulary delle classi non deve dipendere da validation o test, altrimenti si introduce leakage.

**Alternatives Considered**  
Costruire la vocabulary su tutti gli split ridurrebbe OOV, ma userebbe informazione da validation/test.

**Trade-offs**  
Piccola quota di target OOV in validation/test.

**Current Status**  
Implemented. `num_classes=1786`; validation OOV `8/8620`; test OOV `10/8620`. Gli OOV vengono esclusi dai PyG split corrispondenti.

## 9. Sharded PyG Dataset

**Decision**  
Salvare il dataset PyG in shard invece di un unico `List[Data]`.

**Rationale**  
Il full dataset ha circa 69k grafi train. Lo sharding evita di materializzare tutto in memoria e rende più robusta la generazione.

**Alternatives Considered**  
Un unico `train_graphs.pt` era semplice ma nasceva da un limite debug di circa 5000 grafi e non scalava bene.

**Trade-offs**  
Serve manifest, indicizzazione globale e cache shard.

**Current Status**  
Implemented. `data/pyg/manifest.json`, shard da circa 1000 grafi, `ShardedPyGDataset`, LRU cache e shard-aware sampler.

## 10. Source Row Traceability

**Decision**  
Aggiungere `source_row_index` come metadata nel `Data`.

**Rationale**  
Serve tracciare ogni grafo verso la riga CSV sorgente per debug, error analysis e future metriche come MateDepth.

**Alternatives Considered**  
Mappa esterna nel manifest. Sarebbe valida ma più scomoda durante campionamenti e batching.

**Trade-offs**  
Il campo viene batcheato da PyG, ma non entra nel modello.

**Current Status**  
Implemented.

## 11. TimeGNN Come Vendor/Reference

**Decision**  
Trattare `TimeGNN-main/` come codice esterno/vendor/reference e non modificarlo.

**Rationale**  
TimeGNN è utile per studiare idee GAT/time-aware, ma le sue recipe high-level partono da event logs, mentre il nostro input è già PyG.

**Alternatives Considered**  
Usare direttamente le recipe TimeGNN. Questo richiederebbe adattare il progetto a una pipeline event-log non nativa.

**Trade-offs**  
Riusiamo concetti, non una recipe completa pronta.

**Current Status**  
Implemented as policy. `TimeGNN-main/` non viene modificato.

## 12. Chess-Specific GAT Wrapper

**Decision**  
Creare `ChessGATNoTiming` come wrapper/model chess-specific graph-level.

**Rationale**  
Le architetture TimeGNN principali sono orientate a output node-level/event-level. Il problema chess richiede:

```text
node embeddings
↓
graph pooling
↓
global board features
↓
classifier
↓
move logits
```

**Alternatives Considered**  
Adattare direttamente un classifier TimeGNN senza wrapper. Sarebbe meno chiaro e meno aderente al target graph-level.

**Trade-offs**  
Manteniamo un modello dedicato, ma dobbiamo gestire training e checkpoint nel progetto.

**Current Status**  
Implemented.

## 13. Model A Architecture

**Decision**  
Usare due layer `GATConv` edge-aware, 4 heads, 32 hidden per head, pooling medio globale e classifier MLP.

**Rationale / Project Reasoning**  
GAT permette attenzione su relazioni tattiche. Due layer sono un baseline controllato: abbastanza profondi da combinare informazioni locali, non così profondi da complicare overfitting/debug. Quattro head producono una rappresentazione da 128 dimensioni stabile e gestibile.

**Implemented Structure**

```text
x [64B,15]
edge_index [2,E]
edge_attr [E,5]

GATConv(in_channels=15, out_channels=32, heads=4, edge_dim=5)
→ [64B,128]
→ ELU
→ Dropout

GATConv(in_channels=128, out_channels=32, heads=4, edge_dim=5)
→ [64B,128]
→ ELU

global_mean_pool
→ [B,128]

concat global_features [B,4]
→ [B,132]

Linear(132,128)
→ ELU
→ Dropout
→ Linear(128,num_classes)
→ logits [B,1786]
```

**Alternatives Considered**  
GCN baseline, deeper GAT, larger hidden sizes, different pooling. These remain possible future comparisons.

**Trade-offs**  
Il modello è relativamente piccolo (`268,026` parametri) rispetto al numero di classi (`1786`), quindi la performance può essere limitata da target space e rappresentazione.

**Current Status**  
Implemented as Model A.

## 14. GAT Instead Of GCN For Model A

**Decision**  
Usare GAT come primo modello principale.

**Rationale / Project Reasoning**  
Le relazioni tattiche non hanno tutte la stessa importanza. Attention consente al modello di pesare edge diverse nel message passing.

**Alternatives Considered**  
GCN e MLP restano baseline future utili.

**Trade-offs**  
GAT è più costoso di GCN e più sensibile a hyperparameter/runtime.

**Current Status**  
Implemented.

## 15. No Event IDs In Model A

**Decision**  
Non usare `event_ids`.

**Rationale**  
In TimeGNN gli `event_ids` rappresentano eventi/log IDs e possono essere embedded. Nel nostro problema i nodi sono caselle 0..63. Si potevano adattare ID di casella, ma per il baseline sono state preferite feature esplicite della scacchiera.

**Alternatives Considered**  
Embedding della square identity.

**Trade-offs**  
Il modello riceve coordinate normalizzate invece di embedding discreti.

**Current Status**  
Implemented: `ChessGATNoTiming` non richiede `event_ids`.

## 16. No Timing In Model A

**Decision**  
Model A non usa timing, `edge_time` o feature temporali.

**Rationale**  
Serve un baseline no-timing pulito prima di introdurre una variabile temporale.

**Alternatives Considered**  
Inserire timing subito avrebbe reso meno chiaro il confronto.

**Trade-offs**  
Il modello attuale ignora completamente informazione temporale.

**Current Status**  
Implemented.

## 17. Model B Timing Blocker

**Decision**  
Non implementare ancora Model B timing-aware.

**Rationale**  
Una mossa ha naturalmente un tempo graph-level, mentre le tactical edges non hanno automaticamente un tempo edge-specific. Replicare lo stesso move time su tutte le edge probabilmente non cambia l'attenzione relativa in modo significativo.

**Alternatives Considered**  
Replicare timing su edge, aggiungerlo a global features, creare sequenze temporali multi-mossa.

**Trade-offs**  
Serve definire una semantica timing scientificamente difendibile prima di implementare.

**Current Status**  
READY_TO_IMPLEMENT_MODEL_B = NO  
TIMING_SEMANTICS_BLOCKER = YES

## 18. Standard Training

**Decision**  
Trainer standard con `CrossEntropyLoss`, `Adam`, validation loss per early stopping, checkpoint best e test dopo reload best.

**Rationale**  
È una baseline supervisionata chiara per classificazione multi-classe.

**Alternatives Considered**  
Scheduler, curriculum, progressive training. Sono stati aggiunti separatamente per non rendere opaco il trainer base.

**Trade-offs**  
Il trainer base può essere costoso sul full dataset se usato per molte prove.

**Current Status**  
Implemented. Default standard: batch `32`, lr `1e-3`, max epochs `30`, patience `5`, weight decay `0`, seed `42`. AMP è opzionale; scheduler assente nel trainer base.

## 19. Historical Adaptive Controller

**Decision**  
Mantenere il controller adaptive trial-based storico.

**Rationale**  
È utile per esperimenti piccoli e regressioni, ma non è ideale per molti full training sul dataset completo.

**Alternatives Considered**  
Sostituirlo completamente con progressive training.

**Trade-offs**  
Esistono due modalità: adaptive storico e progressive corrente.

**Current Status**  
Implemented and kept for backward compatibility.

## 20. Runtime Benchmark

**Decision**  
Creare `src/benchmark_chess_gat_training.py` per misurare throughput senza convergence training.

**Rationale**  
Prima di cambiare training runtime servivano misure su batch size, workers, AMP, data time e compute time.

**Measured RTX A2000 Reference**

- batch 32, workers 0, AMP false: circa `3531 graphs/s`;
- batch 128, workers 0, AMP true: circa `4758 graphs/s`;
- batch 256, workers 0, AMP true: circa `4904 graphs/s`.

**Trade-offs**  
Batch 256 è più veloce, ma solo circa 3% rispetto a batch 128. Batch 128 è stato scelto come compromesso operativo più conservativo.

**Current Status**  
Implemented. Non è una prova che batch 128 o AMP siano scientificamente ottimali.

## 21. `num_workers=0` Runtime

**Decision**  
Usare `num_workers=0` come configurazione operativa attuale.

**Rationale**  
Nel contesto osservato con Python 3.14 + multiprocessing DataLoader è stato riportato `ValueError: too many fds` con worker > 0.

**Alternatives Considered**  
Usare DataLoader multiprocessing con `persistent_workers` e `prefetch_factor`.

**Trade-offs**  
Meno parallelismo lato CPU, ma runtime più robusto.

**Current Status**  
Implemented as progressive runtime default. Non corretto in questa task.

## 22. Progressive Multi-Fidelity Training

**Decision**  
Usare pipeline `PILOT -> CONFIRMATION -> FULL -> FINAL TEST`.

**Rationale**  
Evitare `6 full trials x 69k graphs`, che sarebbe costoso e ridondante. Pilot scarta config deboli, Confirmation conferma le migliori, Full allena una sola configurazione sul dataset completo.

**Alternatives Considered**  
Continuare con adaptive trial-based full dataset. Scientificamente semplice, ma inefficiente.

**Trade-offs**  
La selezione usa subset deterministicamente campionati; una config scartata presto potrebbe comportarsi meglio sul full dataset. È il compromesso multi-fidelity.

**Current Status**  
Implemented in `src/training/progressive_controller.py`.

## 23. Deterministic Subsets

**Decision**  
Generare permutation deterministiche da seed; Pilot usa i primi N, Confirmation i primi M.

**Rationale**  
Stesso seed significa stessi subset. Pilot è sottoinsieme di Confirmation. Full usa tutto.

**Alternatives Considered**  
Prendere i primi N record CSV avrebbe potuto introdurre bias d'ordine.

**Trade-offs**  
Gli indici non sono salvati integralmente nel Markdown, ma sono tracciati con hash e metadata.

**Current Status**  
Implemented.

## 24. Fresh Model Between Stages

**Decision**  
Non trasferire pesi da Pilot a Confirmation e non trasferire pesi da Confirmation a Full.

**Rationale**  
Gli stage selezionano configurazioni, non fanno curriculum learning o pretraining implicito.

**Alternatives Considered**  
Continuare i pesi tra stage avrebbe ridotto costo, ma cambiato il significato scientifico del training.

**Trade-offs**  
Più costo rispetto a warm-start, ma confronto più pulito.

**Current Status**  
Implemented.

## 25. Ranking And Promotion

**Decision**  
Ranking: best validation loss, poi validation Top5, poi validation Top1.

**Rationale**  
La loss è l'obiettivo di ottimizzazione; Top5 e Top1 sono tie-breaker utili nel target space ampio.

**Alternatives Considered**  
Ranking per test accuracy. Non accettabile perché il test deve restare isolato.

**Trade-offs**  
La selection segue validation, non garantisce il migliore test score.

**Current Status**  
Implemented. Test non usato per ranking, promotion, scheduler, early stopping o tuning.

## 26. Full Stage Scheduler

**Decision**  
Usare `ReduceLROnPlateau` solo nello stage Full.

**Rationale**  
Dopo aver scelto una configurazione, il Full stage deve allenare un singolo modello fino a convergenza ragionevole senza restart continui.

**Defaults**  
`factor=0.5`, `patience=3`, `min_lr=1e-6`; early stopping Full `patience=8`.

**Trade-offs**  
Il trainer standard resta senza scheduler; la logica scheduler vive nella pipeline progressiva.

**Current Status**  
Implemented.

## 27. Checkpoint And Resume

**Decision**  
Nel Full stage salvare sia `best.pt` sia `last.pt`.

**Rationale**  
`best.pt` serve per final test; `last.pt` serve per resume coerente dopo interruzione.

**Resume State**  
`last.pt` include model, optimizer, scheduler, AMP scaler, epoch, early stopping, history e config. `controller_state.json` traccia stage completati e test finale.

**Trade-offs**  
Più artifact su disco, ma run overnight più sicuro.

**Current Status**  
Implemented.

## 28. Final Test Isolation

**Decision**  
Il test viene eseguito una sola volta dopo Full completato e usando `full/best.pt`.

**Rationale**  
Il test non deve influenzare ranking, promotion, early stopping, scheduler o tuning.

**Alternatives Considered**  
Valutare test a ogni stage. Non accettabile per fairness scientifica.

**Trade-offs**  
Meno visibilità durante lo sviluppo, ma valutazione più corretta.

**Current Status**  
Implemented. In resume, se `final_test_completed=true`, il test non viene rieseguito.

## 29. Runtime Time Budget

**Decision**  
Supportare `--max-runtime-hours`.

**Rationale**  
Il run full può essere lasciato overnight. Il controller controlla il limite a boundary sicuri e salva stato.

**Alternatives Considered**  
Kill esterno del processo. Rischia checkpoint incompleti o stato incoerente.

**Trade-offs**  
Il limite non predice perfettamente la fine di un epoch; termina al boundary sicuro successivo.

**Current Status**  
Implemented.

## 30. Progressive Training Command

Full run consigliato:

```bash
./venv/bin/python -m src.train_chess_gat_progressive \
  --device cuda \
  --batch-size 128 \
  --num-workers 0 \
  --pin-memory \
  --non-blocking \
  --amp \
  --pilot-train-graphs 12000 \
  --pilot-val-graphs 2000 \
  --pilot-max-trials 4 \
  --pilot-max-epochs 10 \
  --pilot-patience 3 \
  --confirmation-train-graphs 30000 \
  --confirmation-val-graphs 4000 \
  --confirmation-top-k 2 \
  --confirmation-max-epochs 15 \
  --confirmation-patience 4 \
  --full-max-epochs 60 \
  --full-patience 8 \
  --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 3 \
  --min-learning-rate 1e-6 \
  --max-runtime-hours 10 \
  --seed 42
```

Resume:

```bash
./venv/bin/python -m src.train_chess_gat_progressive --resume --device cuda
```

## 31. Old 5k Baseline Reference

**Decision**  
Conservare i numeri del vecchio baseline come riferimento descrittivo, non come tuning signal.

**Reference**

- train graphs: circa `5000`;
- best config: lr `5e-4`, dropout `0.30`, weight decay `1e-4`, batch `32`;
- test Top1 circa `5.27%`;
- test Top3 circa `10.19%`;
- test Top5 circa `13.78%`;
- best val loss circa `6.245944`.

**Trade-offs**  
Il nuovo esperimento cambia train size, batch size e runtime config. Non è una ablation isolata del solo dataset size.

**Current Status**  
Documented as historical reference.

## 32. Validation Utility

**Decision**  
Mantenere `src.validate_representations` come gate tecnico.

**Rationale**  
Il modello dipende dalla correttezza di semantica Lichess, vocabulary, shape PyG, edge multilabel e batching.

**Current Status**  
Implemented. Verdict corrente: `REPRESENTATION_VALID`.

## 33. Reset Utility

**Decision**  
`src.reset_data` elimina solo dati/artifact rigenerabili allowlisted.

**Rationale**  
Serve poter ripulire dati senza rischiare source, docs o vendor library.

**Current Status**  
Implemented. Non eseguito durante questo audit.

## 34. Future Work

**Planned / Not Implemented**

- Model B timing-aware;
- semantica timing robusta;
- synthetic timing;
- ablation timing/no timing;
- MateDepth evaluation;
- legal move masking;
- alternative target formulations;
- MLP baseline;
- GCN baseline;
- LLM comparison;
- final report.

Possibili alternative target future:

- full theoretical UCI vocabulary;
- from-square / to-square heads;
- promotion head;
- legal move masking.

Queste opzioni non sono implementate nello stato corrente.

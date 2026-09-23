# GraphMate

[English](README.md) | [Italiano](README.it.md)

GraphMate è un repository di ricerca su reti neurali a grafo per gli scacchi. Costruisce rappresentazioni PyTorch Geometric da puzzle Lichess mate-in-1..5, valuta una famiglia congelata di modelli GNN/GAT, li confronta con baseline LLM locali e misura la generalizzazione esterna su un benchmark congelato di composizioni classiche YACPDB.

La fase sperimentale è conclusa. La documentazione e l'analisi finale versionate servono a rendere verificabili i risultati congelati senza riaddestrare modelli o modificare artifact ufficiali.

## Domande Di Ricerca

RQ1 chiede in quali casi la GNN con timing offre una guida migliore rispetto alle baseline LLM selezionate. Il confronto letterale con timing è valutato su Lichess, dove le feature temporali esistono.

RQ2 chiede se il segnale temporale sintetico implementato migliori il modello GNN senza timing corrispondente. Il confronto controllato è Model B contro Model A3.

YACPDB classic è usato separatamente come benchmark di generalizzazione esterna per A3, A4 e LLM. Model B non è valutato su YACPDB perché il timing umano non è disponibile.

## Risultati Principali

| Scenario | Modello / Baseline | Top-1 |
| --- | --- | ---: |
| Test ufficiale Lichess | A3 legal-candidate scorer | 67.561% |
| Test ufficiale Lichess | B variante A3 con timing | 65.981% |
| Test ufficiale Lichess | A4 reranker post-move su Top-5 A3 | 85.412% |
| Lichess strict UCI | Qwen 3.5 4B | 0.070% |
| Lichess strict UCI | Qwen 3.5 9B | 0.116% |
| YACPDB classic | A3 | 11.0% |
| YACPDB classic | A4 | 15.5% |
| YACPDB classic | Qwen 3.5 4B | 0.0% |
| YACPDB classic | Qwen 3.5 9B | 0.5% |

L'ablazione controllata RQ2 non supporta un beneficio del timing: Model B è 1.580 punti percentuali sotto A3 nel confronto allineato Lichess (`n01=321`, `n10=457`, McNemar/binomiale esatto a due code `p=1.2238376595768834e-06`). La conclusione è limitata a questo esperimento: il segnale temporale sintetico implementato non ha migliorato le prestazioni rispetto alla baseline A3 senza timing.

La generalizzazione esterna cala fortemente da Lichess a YACPDB classic: A3 passa da 67.561% a 11.0%, A4 da 85.412% a 15.5%. L'analisi interpreta questo come shift di distribuzione, non come affermazione che un benchmark sia oggettivamente più difficile.

## Famiglia Di Modelli

| Modello | Descrizione | Stato |
| --- | --- | --- |
| A | Classificatore GAT senza timing a vocabolario globale | Baseline storica congelata |
| A1 | Modalità inferenziale best-legal sopra Model A | Diagnostica, senza checkpoint indipendente |
| A2 | Classificatore GAT a vocabolario globale con legal mask | Baseline congelata |
| A3 | Scorer di mosse legali candidate senza timing | Baseline ufficiale no-timing congelata |
| B | Variante di A3 con timing | Ablazione timing congelata |
| A4 | Reranker GNN post-move sulla Top-5 di A3 | Migliore GNN congelata |

## Rappresentazione A Grafo

Ogni posizione è rappresentata come `torch_geometric.data.Data` con 64 nodi, uno per casella.

Le feature dei nodi hanno dimensione 15: tipo pezzo one-hot (`pawn`, `knight`, `bishop`, `rook`, `queen`, `king`), colore, occupazione, riga, colonna, attaccata dal bianco, attaccata dal nero, mobilità legale, stato pinned e valore del pezzo.

Gli archi usano 5 etichette: legal move, attack, defend, pin e check line. Le feature globali sono lato al tratto, stato di scacco, numero di mossa normalizzato e halfmove clock normalizzato.

Per i puzzle Lichess, la posizione del solutore è ottenuta applicando `Moves[0]` a `OriginalFEN`; il target è `Moves[1]`. I modelli predicono la prossima mossa dalla posizione del solutore, non l'intera linea di matto in modo autonomo.

## Dati E Timing

La distribuzione principale di training e test è Lichess mate-in-1..5. Lo split finale è circa 68,958 posizioni di train, 8,620 di validation e 8,620 di test; alcune valutazioni ufficiali allineate usano 8,610 campioni terminali valutabili.

Model B usa feature di timing sintetiche condizionate dal rating: tempo della mossa precedente, tempo della mossa originale e indicatore `time_is_synthetic`. Non sono misure di riflessione umana.

Il benchmark esterno è `yacpdb_classic_v1`, un set congelato di 200 composizioni directmate YACPDB con 20 problemi per profondità da #1 a #10. Per le composizioni classiche si usa lo scoring su accepted key.

## Baseline LLM

Le baseline LLM principali sono esecuzioni locali di `qwen3.5:4b` e `qwen3.5:9b`. Il parsing strict UCI è la metrica primaria; il parsing relaxed è solo diagnostico. I tentativi GPT-OSS sono documentati ma esclusi dal confronto primario per comportamento di protocollo, reasoning e troncamento non adatto alla tabella finale strict.

## Struttura Del Repository

```text
src/                         # codice data, graph, model, training, inference, evaluation, analysis
tests/                       # test unitari e di integrazione
checkpoints/                 # checkpoint finali congelati distribuibili
resources/move_encoder/      # vocabolario mosse canonico versionato
generic_info/                # note tecniche pubbliche e report finali
generic_info/final_analysis/ # analisi finale canonica con tabelle, figure, summary
data/                        # root dati locale, in gran parte ignorata
artifacts/                   # output sperimentali locali, ignorati
TimeGNN-main/                # codice upstream esterno, tenuto separato
```

Vedi [generic_info/README.md](generic_info/README.md), [checkpoints/README.md](checkpoints/README.md) e [resources/README.md](resources/README.md).

## Installazione

```bash
python -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

L'installazione di PyTorch può dipendere dalla piattaforma, soprattutto con CUDA. Se `requirements.txt` non è adatto alla tua macchina, installa prima le wheel PyTorch/PyG corrette e poi le dipendenze rimanenti.

## Preparazione Dati

L'entry point storico della pipeline Lichess completa è:

```bash
./venv/bin/python main.py
```

La validazione delle rappresentazioni si ispeziona con:

```bash
./venv/bin/python -m src.validation.representations --help
```

I risultati finali congelati vanno riprodotti dagli artifact e checkpoint esistenti, non ricostruendo casualmente i dataset.

## Training

Gli entry point di training restano disponibili per riproducibilità ed estensioni:

```bash
./venv/bin/python -m src.cli.training.train_model_a3_legal_scorer --help
./venv/bin/python -m src.cli.training.train_model_b_timing_legal_scorer --help
./venv/bin/python -m src.cli.training.train_model_a4_postmove --help
```

I risultati pubblici usano i checkpoint congelati in [checkpoints/](checkpoints/). Non serve fare training per leggere l'analisi finale.

## Valutazione

Entry point rappresentativi:

```bash
./venv/bin/python -m src.cli.evaluation.evaluate_model_a_vs_a2_vs_a3 --help
./venv/bin/python -m src.cli.evaluation.evaluate_model_b_timing_ablation --help
./venv/bin/python -m src.cli.evaluation.evaluate_model_a3_vs_a4 --help
./venv/bin/python -m src.cli.evaluation.evaluate_classic_heldout --help
./venv/bin/python -m src.cli.evaluation.benchmark_llm_chess --help
```

Usa sempre `--help` prima di avviare una valutazione: alcuni comandi richiedono dati locali o percorsi in artifact ignorati da Git.

## Analisi Finale

L'analisi pubblica canonica è in [generic_info/final_analysis/](generic_info/final_analysis/):

- [final_experiment_summary.md](generic_info/final_analysis/final_experiment_summary.md) è il summary scientifico generato.
- [final_experiment_analysis.md](generic_info/final_analysis/final_experiment_analysis.md) spiega selezione delle sorgenti, statistiche, esclusioni e limiti.
- `data/` contiene tabelle e output statistici machine-readable.
- `figures/` contiene figure SVG e PNG.

Entry point di rigenerazione:

```bash
./venv/bin/python -m src.cli.analysis.build_final_experiment_report --help
```

## Checkpoint E Risorse

I checkpoint congelati sono documentati in [checkpoints/README.md](checkpoints/README.md). Il vocabolario canonico delle mosse è documentato in [resources/README.md](resources/README.md).

Non scrivere output di training in `checkpoints/`; usa directory locali ignorate per trial e artifact grezzi.

## Limiti

L'esperimento timing usa feature sintetiche, quindi non risponde alla domanda se tempi reali di riflessione umana aiuterebbero. I bucket YACPDB per profondità sono piccoli (`N=20`), quindi gli intervalli per profondità sono ampi. Il confronto LLM misura il comportamento strict di output mossa nel protocollo documentato e non va letto come benchmark generale di abilità scacchistica.

## Licenza E Citazione

Questo repository è rilasciato sotto [MIT License](LICENSE). Se usi GraphMate prima dell'aggiunta di metadati formali di citazione, cita il repository e il summary dell'analisi finale.

## Ringraziamenti

GraphMate usa sorgenti pubbliche di puzzle e composizioni scacchistiche, infrastruttura PyTorch Geometric e baseline locali con modelli aperti. Il codice upstream esterno in `TimeGNN-main/` è tenuto separato dall'implementazione GraphMate.

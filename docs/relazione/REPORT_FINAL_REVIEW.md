# Revisione finale della relazione GraphMate

## Ambito e file esaminati

L'entry point effettivo è generic_info/Rel_Pres/main.tex; il percorso usa
Rel_Pres, non rel_pres. La revisione ha incluso:

- main.tex, il frontespizio/abstract e tutti i capitoli da 1 a 9;
- le appendici A--F, le tabelle e i diagrammi richiamati dalle appendici;
- bibliography.bib, i file di build e il PDF prodotto;
- README, documentazione scientifica, configurazioni ufficiali, implementazioni
  dei modelli e delle feature, manifesti, analisi finali e artefatti congelati
  del repository, usati in sola lettura come fonti autorevoli.

Non sono stati eseguiti addestramenti o benchmark e non sono stati scaricati
dataset.

## Struttura prima e dopo

Prima della revisione il documento principale integrava i Capitoli 3--7; i file
dei Capitoli 1, 2, 8 e 9 erano vuoti o non integrati. La bibliografia non era
stampata e il riferimento al relativo file non coincideva con il nome presente.
Erano inoltre presenti sezioni metodologiche ripetute, label duplicate, conteggi
storici e note di lavorazione.

La struttura finale è:

1. Introduzione
2. Background e lavori correlati
3. Dataset e preparazione dei dati
4. Rappresentazione a grafo
5. Architetture dei modelli
6. Metodologia sperimentale
7. Risultati
8. Discussione
9. Conclusioni
10. Bibliografia
11. Appendici A--F

## Interventi editoriali e strutturali

- Creati e integrati i Capitoli 1, 2, 8 e 9 in italiano.
- Rifocalizzato il Capitolo 3 sulla pipeline e sui conteggi congelati, eliminando
  note storiche e TODO.
- Conservata nel Capitolo 4 la spiegazione concettuale del grafo, lasciando
  all'Appendice B lo schema implementativo completo.
- Reso il Capitolo 5 una progressione coerente da A ad A4, con rinvio
  all'Appendice A per configurazioni, dimensioni e checkpoint.
- Ridotto il Capitolo 6 a disegno sperimentale, protocolli, metriche e limiti,
  evitando di ripetere dati e architetture dei Capitoli 3--5.
- Lasciate nel Capitolo 7 le misure; interpretazione e risposte alle domande di
  ricerca sono state collocate nei Capitoli 8 e 9.
- Trasformate in rinvii le duplicazioni non necessarie tra corpo e appendici,
  mantenendo brevi richiami utili alla comprensione.
- Uniformati terminologia, gerarchia delle sezioni appendiciali, titoli e
  transizioni.

## Verifica scientifica e discrepanze

Le fonti autorevoli sono state il codice corrente e gli artefatti finali
congelati. Sono stati adottati e verificati i seguenti valori:

- 1.937.001 righe filtrate e 1.931.434 pulite;
- dataset bilanciato di 86.198 record;
- split CSV 68.958/8.620/8.620 e split PyG 68.958/8.612/8.610;
- OOV 0/8/10 e vocabolario di 1.786 mosse;
- 15 feature di nodo, 5 di arco e 4 globali;
- A1 solo inferenza, A2 con maschera di legalità, A3 con scoring delle mosse
  legali, B con timing sintetico, A4 con recupero Top-5 e reranking post-mossa;
- benchmark classico di 200 problemi: 193 verificati da Popeye, 188 con chiave
  unica, 5 con chiavi multiple e 7 timeout.

I precedenti conteggi locali/storici del Capitolo 3 sono stati sostituiti con i
manifesti congelati e la discrepanza è stata resa esplicita in questo audit.
Non sono stati modificati risultati verificati: A3 67,5610%, B 65,9814% e A4
85,4123% su Lichess; A3 11,0% e A4 15,5% su YACPDB. RQ1 è sostenuta, nel solo
protocollo Lichess congelato, a tutte le profondità considerate. RQ2 ha esito
negativo per il timing sintetico implementato, senza generalizzare
all'inutilità del timing reale. Non restano discrepanze scientifiche irrisolte.

## Tabelle, figure e impaginazione

Le tabelle critiche sono state ridisegnate con colonne flessibili o a larghezza
controllata; sono stati rimossi gli usi indiscriminati di resizebox. Sono state
corrette in particolare le tabelle di risultati classici, diagnostica LLM,
verifica Popeye, disponibilità YACPDB, training, checkpoint, split, timing e
inventario degli artefatti. Path e hash lunghi usano spezzatura controllata.

I diagrammi delle pipeline dati, grafo, benchmark classico e LLM sono stati
verificati e non richiedevano modifiche grafiche. Header, indice, caption,
numerazione, margini e page break sono stati uniformati. Due pagine di spill
accidentale e una chiusura di capitolo eccessivamente vuota sono state eliminate.
Gli header ora distinguono correttamente capitoli e appendici.

## Bibliografia

La bibliografia è stata integrata e contiene esclusivamente quattro fonti
verificate: GCN, GAT, TimeGNN e database pubblico Lichess. Tutte le quattro
chiavi citate esistono e sono stampate. Non sono stati inventati DOI, autori,
titoli o URL.

## Compilazione e QA

Il sistema disponibile non include latexmk, pdflatex, xelatex o biber; il
Makefile usa quindi Tectonic:

    make
    # equivalente a:
    tectonic -X compile main.tex --keep-logs --keep-intermediates

La compilazione termina con codice 0 e produce main.pdf, 65 pagine. Il log non
contiene errori fatali, file mancanti, overfull box, citazioni mancanti o
singoli riferimenti indefiniti. La verifica statica rileva 82 label univoche e
38 label richiamate, tutte definite.

Tutte le 65 pagine sono state renderizzate e ispezionate, con controlli
ingranditi su tabelle, hash, header e pagine finali. Non risultano elementi
tagliati, sovrapposti o esterni ai margini, figure illeggibili, tabelle
eccedenti, numerazione interrotta o grandi spazi bianchi ingiustificati.

## Avvisi e limiti residui

Restano 16 Underfull hbox in tabelle tecniche compatte e nel prompt verbatim:
sono spaziature locali, non overflow o perdita di contenuto. Tectonic segnala
inoltre l'uso del backend BibTeX di fallback di biblatex e conserva nel log un
avviso generico di rerun/riferimenti prodotto durante i passaggi interni. Il PDF
finale non contiene segnaposto; label, citazioni e bibliografia sono risolte.
Questi avvisi non hanno conseguenze visive o scientifiche.

I limiti scientifici restano quelli dichiarati nel testo: timing sintetico,
predizione della sola prima mossa, protocollo LLM circoscritto, benchmark
esterno di 200 problemi, ceiling Top-5 di A4 e dipendenza dai checkpoint e
budget congelati.

Non sono stati modificati file fuori da generic_info/Rel_Pres/. La directory
non tracciata .vscode/, visibile nello stato Git finale, non è stata creata né
modificata durante questa attività.

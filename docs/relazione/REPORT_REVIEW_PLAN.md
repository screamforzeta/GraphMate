# Piano di revisione finale della relazione GraphMate

## Ambito e fonti autorevoli

La revisione riguarda `generic_info/Rel_Pres/` (il percorso indicato inizialmente
come `generic_info/rel_pres/` non esiste: il nome effettivo usa le maiuscole).
L'entry point e `main.tex`. Sono stati letti i capitoli 0--9, le appendici A--F,
le relative tabelle e figure, la bibliografia e la documentazione scientifica
versionata. Per numeri e dettagli implementativi prevalgono configurazioni,
codice e artefatti finali congelati; le note esplorative o storiche non sono
trattate come risultati ufficiali.

## Problemi e interventi pianificati

| Posizione | Problema | Correzione | Azione | Motivo |
|---|---|---|---|---|
| `main.tex` | Mancano gli input dei Capitoli 8 e 9; titolo del Capitolo 2 incompleto rispetto alla struttura richiesta | Integrare tutti i nove capitoli nell'ordine previsto | mantenere e integrare | Senza gli input il documento non e completo |
| `main.tex` | Risorsa bibliografica configurata come `bibliografia.bib`, mentre esiste `bibliography.bib`; manca `\printbibliography` | Correggere il nome e stampare la bibliografia | correggere | Errore di compilazione e bibliografia assente |
| `main.tex` | Header con segnaposto e nomi degli studenti senza separatori di riga | Usare titolo breve reale e impaginare correttamente i nomi | correggere | Coerenza visiva e leggibilita |
| `main.tex` | Preambolo ridondante (`array` e `tabularx` duplicati) e supporto insufficiente per percorsi/URL lunghi | Consolidare i pacchetti e aggiungere impostazioni locali di spezzatura | unire | Ridurre rumore e prevenire overflow |
| Capitoli 1, 2, 8, 9 | File vuoti | Scrivere introduzione, background, discussione e conclusioni | creare | Completare la struttura scientifica richiesta |
| Capitolo 3 | Numeri locali/storici e TODO in conflitto con dataset e artefatti finali congelati | Adottare i conteggi ufficiali 68.958/8.620/8.620, 8/10 OOV e 68.958/8.612/8.610 grafi; documentare la discrepanza nel review finale | correggere e accorciare | Evitare che una working copy storica prevalga sulla provenienza congelata |
| Capitolo 3 | La sorgente ausiliaria di partite e i conteggi preliminari occupano molto spazio rispetto al ruolo finale | Mantenerne una sintesi e rinviare all'appendice per il timing sintetico | accorciare | Il modello B finale non usa tempi umani osservati |
| Capitolo 4 | Tabelle dettagliate duplicano l'Appendice B e usano le stesse label | Conservare la spiegazione intuitiva; rinviare all'appendice per lo schema esatto; rendere univoche le label residue | accorciare e unire | Evitare duplicazione e riferimenti ambigui |
| Capitolo 4 | TODO su shape legacy e persistenza locale | Descrivere solo il formato ufficiale verificato, registrando la provenienza nel review finale | rimuovere | Le note di lavorazione non appartengono al testo finale |
| Capitolo 5 | Confronto architetturale necessario ma in parte ripetuto nell'Appendice A | Mantenere la progressione concettuale e rinviare all'appendice per dimensioni, hash e iperparametri | mantenere e accorciare | Ripetizione necessaria per comprendere i modelli; dettagli riproducibili restano in appendice |
| Capitolo 6 | Ripete dataset, schema del grafo e architetture gia descritti nei Capitoli 3--5 | Rifocalizzare su domande di ricerca, disegno sperimentale, protocolli, metriche e statistica | unire e accorciare | Separare metodo di valutazione da implementazione |
| Capitolo 7 | La sezione finale interpreta e risponde gia alle RQ, anticipando Discussione e Conclusioni | Ridurla a sintesi fattuale dei risultati e spostare l'interpretazione nei Capitoli 8--9 | spostare e accorciare | Separare cosa e stato misurato dal suo significato |
| Capitolo 7 e Appendice D | Label `tab:classic-results` duplicata | Rinominare la label nel testo principale | correggere | Garantire riferimenti deterministici |
| Capitoli 4 e appendici | Label di tabelle su nodi, archi e feature globali duplicate | Usare prefisso `tab:main-*` nel capitolo e lasciare inalterate le label canoniche dell'appendice | correggere | Le appendici sono la specifica implementativa |
| Tabelle del testo | Alcune intestazioni sono troppo ampie per il formato A4 | Convertire le tabelle critiche a `tabularx`, abbreviare intestazioni equivalenti e consentire il ritorno a capo | ridisegnare | Evitare overfull box senza riduzione globale del font |
| Appendici | Tabelle tecniche con identificatori, path e hash potenzialmente non spezzabili | Verificare ogni tabella; usare colonne flessibili e comandi di spezzatura mirati | correggere se necessario | Conservare leggibilita e contenuto esatto |
| Bibliografia | File vuoto e assenza di citazioni | Inserire solo fonti verificate: GAT, GCN, TimeGNN e database Lichess | creare | Supportare il background senza riferimenti inventati |
| Intero documento | Terminologia mista (`repository`, `validation`, `strict`, `relaxed`) | Uniformare il testo italiano, mantenendo gli identificatori di protocollo quando tecnicamente necessari | correggere | Coerenza editoriale |
| Intero documento | Nessun build script e nessun PDF generato nel percorso della relazione | Aggiungere un `Makefile` minimale basato su `latexmk`, compilare e ispezionare log/PDF | creare e verificare | Rendere esplicito e ripetibile il processo di build |

## Classificazione della ripetizione

### A. Ripetizione necessaria per la comprensione

- Breve riepilogo della rappresentazione a grafo nel Capitolo 5 prima di
  descrivere l'encoder.
- Definizione sintetica di A3, B e A4 nel Capitolo 7 accanto ai risultati.
- Richiamo dei valori principali nei Capitoli 8 e 9 quando indispensabile per
  interpretare o rispondere alle domande di ricerca.

### B. Rinvii utili

- Dal Capitolo 3 all'Appendice C per conteggi, OOV e generazione del timing.
- Dal Capitolo 4 all'Appendice B per l'ordine esatto delle feature.
- Dal Capitolo 5 all'Appendice A per iperparametri, dimensioni e checkpoint.
- Dal Capitolo 6 alle Appendici D ed E per i protocolli YACPDB/Popeye e LLM.
- Dal testo alla Appendice F per i workflow riproducibili.

### C. Duplicazione non necessaria

- Ripetizione completa di dati, feature e architetture nel Capitolo 6.
- Tabelle identiche o quasi identiche tra Capitolo 4 e Appendice B.
- Risposte interpretative alle RQ alla fine del Capitolo 7 e poi nei futuri
  Capitoli 8 e 9.
- TODO editoriali e descrizioni di artefatti locali storici nel testo finale.

## Verifiche finali previste

La chiusura richiede: nove capitoli integrati, label univoche, citazioni risolte,
compilazione completa con i passaggi bibliografici, assenza di overflow
significativi nel log, conteggio pagine e ispezione visuale di tutte le pagine
renderizzate. Eventuali limiti dell'ambiente saranno riportati esplicitamente e
impediranno lo stato `COMPLETE` se non consentono una verifica richiesta.

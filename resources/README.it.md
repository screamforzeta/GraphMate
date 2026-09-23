# Risorse

[English](README.md) | [Italiano](README.it.md)

Questa directory contiene piccole risorse versionate richieste dai modelli GraphMate congelati.

## Move Encoder

`move_encoder/` contiene il vocabolario mosse canonico usato dai modelli a vocabolario globale e dal codice di compatibilità:

- `move_to_idx.json`
- `idx_to_move.json`
- `move_encoder_stats.json`

Questi file sono stati copiati dagli artifact sperimentali ignorati da Git, così il codice pubblico di valutazione può risolvere gli indici mossa senza dipendere da directory locali di output training.

Non rigenerare o modificare questi file quando ispezioni i risultati finali.

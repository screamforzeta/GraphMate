# Checkpoint Congelati

[English](README.md) | [Italiano](README.it.md)

Questa directory contiene solo i checkpoint finali congelati necessari per inferenza e valutazione locali riproducibili. Output di training, checkpoint intermedi, file `last.pt` e artifact sperimentali grezzi restano in percorsi `artifacts/` ignorati da Git.

| Modello | Ruolo | Percorso repository | SHA256 | Dimensione | Timing | Ruolo benchmark |
| --- | --- | --- | --- | --- | --- | --- |
| A | Baseline GAT senza timing a vocabolario globale | `checkpoints/model_a/best.pt` | `1a72d0b6f675b50c829d76c63343e7016a569c00f993f063b0d0b75061de9b70` | ~3.2 MB | no | Baseline Lichess |
| A2 | GAT a vocabolario globale con legal mask | `checkpoints/model_a2/best.pt` | `20a76ae5648ded7f8fe3eb835eddd5ec0a87c9043989c37a65b4e8b9dc6901c1` | ~3.1 MB | no | Baseline Lichess |
| A3 | Scorer di mosse legali candidate | `checkpoints/model_a3/best.pt` | `4efec653a451da7585f3663847c8dc5caaa8ebffadea677617e2f496e4253b80` | ~943 KB | no | Baseline Lichess e generalizzazione YACPDB |
| B | Variante A3 controllata con timing | `checkpoints/model_b/best.pt` | `b8bbad2420ce301582ad08377b1c80f56609513a8dae201600557e255b87d26f` | ~952 KB | timing sintetico | Confronto timing Lichess; non applicabile a YACPDB classic |
| A4 | Reranker GNN post-move su Top-5 A3 | `checkpoints/model_a4/best.pt` | `0cb73acf70487efa5c93f715a5a60c0aa09d64792d894301efaaaf47b2801e99` | ~1.1 MB | no | Reranking Lichess e generalizzazione YACPDB |

Gli hash sono stati verificati dopo la copia dagli artifact sperimentali ufficiali locali. Non indirizzare output di training in questa directory.

Model A1 è una modalità inferenziale sopra Model A, quindi non ha un checkpoint separato.

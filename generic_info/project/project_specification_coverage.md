# Project Specification Coverage

| Requirement | Status | Implementation | Evidence | Remaining work |
|---|---|---|---|---|
| Lichess puzzle dataset | COMPLETE | Download, cleaning, mateIn1-5 preprocessing, train/val/test split | `src/data/download/download_puzzles.py`, `src/data/preprocess/prepare_puzzles_dataset.py` | None for current phase |
| Elite game data | PARTIAL | Streaming PGN sampling and cleaning exists | `src/data/download/download_games.py`, `src/data/preprocess/prepare_games_dataset.py` | Larger real timing integration remains optional/future |
| Graph representation | COMPLETE | node15/edge5/global4 PyG representation | `src/graph/` and `generic_info/models/graph_representation.md` | None |
| No-timing GNN models | COMPLETE | A/A1/A2/A3/A4 frozen family | `src/models/model_a/`, `generic_info/models/model_family.md` | None for GNN phase |
| Timing model / ablation | COMPLETE for implemented synthetic timing | Model B timing-aware A3 variant and ablation | `src/models/model_b/`, `src/evaluation/model_b/` | Real human timing remains not proven |
| MateDepth analysis | COMPLETE for A3/A4 terminal reporting | A4 breakdown MateIn1-5 documented | `generic_info/models/model_family.md` | Extend to held-out/LLM later |
| GNN evaluation | COMPLETE for Lichess split | A3, A4, B frozen results documented | `generic_info/models/model_family.md` | Held-out classic puzzle evaluation not done |
| Held-out classic puzzles | NOT STARTED | No final held-out classic set in repo | Project spec | Build dataset and protocol |
| LLM comparison | NOT STARTED | No LLM baseline/inference protocol complete | Project spec | Implement fair benchmark |
| Visualization/web application | PARTIAL | Streamlit debugger supports graph, MateDepth filter, model registry, A4/B/result views | `src/graph/debug/streamlit_graph_debugger.py` | Full polished app and autonomous line solving remain future |
| Optional Stockfish/hybrid extensions | OUT OF SCOPE / OPTIONAL | Not introduced into primary model results | N/A | Keep separated if added later |


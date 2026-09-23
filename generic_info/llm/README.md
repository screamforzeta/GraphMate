# LLM Benchmark Documentation

This directory documents the local LLM comparison protocol used in the final GraphMate analysis.

Key documents:

- [llm_benchmark_protocol.md](llm_benchmark_protocol.md): benchmark protocol.
- [prompt_and_parser.md](prompt_and_parser.md): prompt format and strict parser.
- [relaxed_chess_move_v1.md](relaxed_chess_move_v1.md): diagnostic relaxed parser.
- [benchmark_reproducibility.md](benchmark_reproducibility.md): reproducibility notes.
- [contamination_limitations.md](contamination_limitations.md): limitations and contamination considerations.

Strict UCI parsing is the primary metric. Relaxed parsing is diagnostic and is not substituted into the primary result tables.

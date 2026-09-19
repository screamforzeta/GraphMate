# Benchmark Reproducibility

Every official run must record:

- run_id
- timestamp
- git commit and dirty state
- dataset id
- dataset fingerprint
- model registry id
- exact Ollama tag
- Ollama version
- prompt version and hash
- parser version
- generation options
- endpoint
- host/device information where available

Generation configuration:

```json
{
  "temperature": 0,
  "num_predict": 16
}
```

Qwen thinking setting:

```json
{
  "thinking_enabled": false
}
```

Qwen status: `RUNTIME_VALIDATED`.

GPT-OSS setting:

```json
{
  "thinking_mode": "low",
  "think": "low",
  "temperature": 0,
  "num_predict": 64
}
```

GPT-OSS status: `RUNTIME_VALIDATED`.

The first server2 smoke showed `num_predict=16` with default thinking enabled is invalid for all three models: generation ended with `done_reason=length`, non-empty truncated thinking, and empty final content. Qwen supports `think=false`; GPT-OSS supports reasoning levels and cannot fully disable reasoning. The frozen GPT-OSS protocol therefore uses the minimum supported reasoning effort, `think="low"`.

GPT-OSS calibration was run only on non-official fixtures before any official Lichess inference:

- candidate budgets: `[32, 64, 128, 256]`
- `32`: failed, `done_reason=length`, final content `e`, parse error
- `64`: succeeded, `done_reason=stop`, `eval_count=37`, final content `e2e4`
- selected budget: `64`

Chess correctness was not used to choose the budget.

The strict parser only sees final answer content, never thinking/reasoning content. Runs with thinking enabled and disabled are different benchmark configurations and cannot be resumed into each other.

Official execution must be resumable and must refuse unsafe resume when prompt hash, parser version, dataset fingerprint, model tag, or generation options differ.

Infrastructure failures are recorded separately from wrong chess moves.

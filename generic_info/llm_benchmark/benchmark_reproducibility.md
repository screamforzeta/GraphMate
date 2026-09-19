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

Thinking setting:

```json
{
  "thinking_enabled": false
}
```

Status: `PENDING_FINAL_SMOKE_VALIDATION`.

The first server2 smoke showed `num_predict=16` with default thinking enabled is invalid for all three models: generation ended with `done_reason=length`, non-empty truncated thinking, and empty final content. The candidate primary protocol explicitly requests thinking disabled while keeping `temperature=0` and `num_predict=16`.

The strict parser only sees final answer content, never thinking/reasoning content. Runs with thinking enabled and disabled are different benchmark configurations and cannot be resumed into each other.

Official execution must be resumable and must refuse unsafe resume when prompt hash, parser version, dataset fingerprint, model tag, or generation options differ.

Infrastructure failures are recorded separately from wrong chess moves.

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

Status: `PENDING_RUNTIME_VALIDATION`.

The three server2 models advertise `thinking`. Runtime smoke must verify whether `num_predict = 16` leaves enough budget for a final UCI answer. The strict parser only sees final answer content, never thinking/reasoning content.

Official execution must be resumable and must refuse unsafe resume when prompt hash, parser version, dataset fingerprint, model tag, or generation options differ.

Infrastructure failures are recorded separately from wrong chess moves.

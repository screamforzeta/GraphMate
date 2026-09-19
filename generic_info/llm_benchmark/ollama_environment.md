# Ollama Environment Audit

Audit date: 2026-09-19

The development VM must not contact Docker or Ollama directly. Runtime checks are executed manually on server2.

## Server2 Runtime

Endpoint:

```text
http://localhost:11436
```

Container:

```text
ollama-old-models
```

Hardware visible to Ollama:

- GPU 0: NVIDIA RTX A2000, 6138 MiB
- GPU 1: NVIDIA RTX A2000, 6138 MiB
- Driver: 595.84
- CUDA: 13.2

## Frozen Benchmark Registry

| Benchmark ID | Ollama tag | Digest | Parameters | Quantization | Context | Capabilities |
|---|---|---|---|---|---:|---|
| `qwen_3_5_4b` | `qwen3.5:4b` | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` | 4.7B | Q4_K_M | 262144 | vision, completion, tools, thinking |
| `qwen_3_5_9b` | `qwen3.5:9b` | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` | 9.7B | Q4_K_M | 262144 | vision, completion, tools, thinking |
| `gpt_oss_20b` | `gpt-oss:20b` | `17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7` | 20.9B | MXFP4 | 131072 | completion, tools, thinking |

Preflight must verify exact tag and digest. A digest mismatch blocks benchmark readiness.

## Server2 Commands

Preflight:

```bash
./venv/bin/python -m src.cli.evaluation.benchmark_llm_chess \
  --ollama-url http://localhost:11436 \
  --thinking-disabled \
  --preflight
```

Candidate runtime protocol:

```text
temperature = 0
num_predict = 16
thinking_enabled = false
status = PENDING_FINAL_SMOKE_VALIDATION
```

4B smoke:

```bash
./venv/bin/python -m src.cli.evaluation.benchmark_llm_chess \
  --ollama-url http://localhost:11436 \
  --model qwen_3_5_4b \
  --thinking-disabled \
  --runtime-smoke
```

9B smoke:

```bash
./venv/bin/python -m src.cli.evaluation.benchmark_llm_chess \
  --ollama-url http://localhost:11436 \
  --model qwen_3_5_9b \
  --thinking-disabled \
  --runtime-smoke
```

20B smoke:

```bash
./venv/bin/python -m src.cli.evaluation.benchmark_llm_chess \
  --ollama-url http://localhost:11436 \
  --model gpt_oss_20b \
  --thinking-disabled \
  --runtime-smoke
```

All models, sequential:

```bash
./venv/bin/python -m src.cli.evaluation.benchmark_llm_chess \
  --ollama-url http://localhost:11436 \
  --all-models \
  --thinking-disabled \
  --runtime-smoke
```

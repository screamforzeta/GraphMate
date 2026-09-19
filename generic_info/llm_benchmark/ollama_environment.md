# Ollama Environment Audit

Audit date: 2026-09-19

Docker access from the current user/session was denied:

```text
permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

Therefore the framework does not invent model tags. The model registry is generated only from the result of `ollama list` / `/api/tags` when the endpoint is reachable.

Expected model families:

- Qwen 3.5 4B
- Qwen 3.5 9B
- GPT-OSS 20B

Required future audit fields:

- container exists
- running/stopped state
- image name
- Ollama version
- endpoint/port
- GPU availability
- exact `ollama list`
- exact tags and sizes


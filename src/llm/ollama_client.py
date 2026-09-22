"""Small HTTP client for local Ollama benchmark calls."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import json
import os


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_GENERATION_OPTIONS = {
    "temperature": 0,
    "num_predict": 16,
}
DEFAULT_THINKING_ENABLED = False
GENERATION_CONFIG_STATUS = "FROZEN_MODEL_SPECIFIC_RUNTIME_VALIDATED"


def resolve_endpoint(cli_endpoint=None):
    """Resolve Ollama endpoint with CLI > environment > default precedence."""

    endpoint = (cli_endpoint or os.environ.get("OLLAMA_URL") or DEFAULT_OLLAMA_ENDPOINT).rstrip("/")
    for suffix in ("/api/generate", "/api/chat", "/api/tags", "/api/version", "/api"):
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
            break
    return endpoint.rstrip("/")


def extract_final_content_and_thinking(response):
    """Return final answer content and separate thinking text from Ollama JSON."""

    message = response.get("message") if isinstance(response, dict) else None
    if isinstance(message, dict):
        final_content = message.get("content")
        thinking = message.get("thinking") or message.get("reasoning")
    else:
        final_content = response.get("response") if isinstance(response, dict) else None
        thinking = None
    thinking = thinking or (response.get("thinking") if isinstance(response, dict) else None)
    return final_content or "", thinking


class OllamaClient:
    """Minimal Ollama HTTP client without cloud dependencies."""

    def __init__(self, endpoint=None, timeout=120):
        """Store endpoint and timeout for later requests."""

        self.endpoint = resolve_endpoint(endpoint)
        self.timeout = timeout

    def _json_request(self, path, payload=None):
        """Send one JSON request to the configured Ollama endpoint."""

        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.endpoint}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def version(self):
        """Return Ollama version metadata."""

        return self._json_request("/api/version")

    def list_models(self):
        """Return Ollama model list metadata."""

        return self._json_request("/api/tags")

    def generate(
        self,
        model,
        prompt,
        system,
        options=None,
        thinking_enabled=DEFAULT_THINKING_ENABLED,
        think=None,
    ):
        """Generate one deterministic response and return text plus metadata."""

        options = dict(DEFAULT_GENERATION_OPTIONS | (options or {}))
        think_value = bool(thinking_enabled) if think is None else think
        started = time.perf_counter()
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options,
            "think": think_value,
        }
        try:
            response = self._json_request("/api/generate", payload)
            error = None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            response = {}
            error = str(exc)
        elapsed = time.perf_counter() - started
        return {
            "response": extract_final_content_and_thinking(response)[0],
            "final_content": extract_final_content_and_thinking(response)[0],
            "thinking": extract_final_content_and_thinking(response)[1],
            "thinking_requested": think_value,
            "thinking_returned": bool(extract_final_content_and_thinking(response)[1]),
            "metadata": response,
            "request_payload": payload,
            "latency_seconds": elapsed,
            "error": error,
        }

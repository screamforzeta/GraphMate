"""Small HTTP client for local Ollama benchmark calls."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import json


DEFAULT_OLLAMA_ENDPOINT = "http://localhost:11434"
DEFAULT_GENERATION_OPTIONS = {
    "temperature": 0,
    "num_predict": 16,
}


class OllamaClient:
    """Minimal Ollama HTTP client without cloud dependencies."""

    def __init__(self, endpoint=DEFAULT_OLLAMA_ENDPOINT, timeout=120):
        """Store endpoint and timeout for later requests."""

        self.endpoint = endpoint.rstrip("/")
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

    def generate(self, model, prompt, system, options=None):
        """Generate one deterministic response and return text plus metadata."""

        options = dict(DEFAULT_GENERATION_OPTIONS | (options or {}))
        started = time.perf_counter()
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": options,
        }
        try:
            response = self._json_request("/api/generate", payload)
            error = None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            response = {}
            error = str(exc)
        elapsed = time.perf_counter() - started
        return {
            "response": response.get("response"),
            "metadata": response,
            "latency_seconds": elapsed,
            "error": error,
        }


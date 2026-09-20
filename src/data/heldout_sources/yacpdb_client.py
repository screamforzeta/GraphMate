"""Minimal YACPDB acquisition client.

This module only performs bounded query/fetch requests and raw response caching.
It does not normalize chess data, parse solutions, sample datasets, or run
model-dependent logic.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://yacpdb.org"
DEFAULT_USER_AGENT = "Progetto-Damiani-YACPDB-client/1.0 (+technical heldout acquisition)"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RECORDS = 3


class YacpdbClientError(RuntimeError):
    """Raised when a YACPDB request or response cannot be used safely."""


@dataclass(frozen=True)
class YacpdbResponse:
    """Raw YACPDB response plus request metadata."""

    method: str
    url: str
    status: int
    content_type: str
    body: bytes

    def json(self) -> dict[str, Any]:
        """Decode the response body as JSON."""

        try:
            payload = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise YacpdbClientError("YACPDB response is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise YacpdbClientError("YACPDB response root is not a JSON object")
        return payload


class YacpdbClient:
    """Small sequential client for YACPDB QL smoke acquisition."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
        retries: int = 1,
        opener: Any | None = None,
        sleep_seconds: float = 0.5,
    ) -> None:
        """Create a client with timeout, retry, and polite pacing settings."""

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_agent = user_agent
        self.retries = retries
        self.opener = opener or urllib.request
        self.sleep_seconds = sleep_seconds

    def ql_url(self, query: str, page: int = 1) -> str:
        """Build the public-undocumented QL endpoint URL."""

        if page < 1:
            raise ValueError("page must be >= 1")
        encoded_query = urllib.parse.urlencode({"q": query, "p": str(page)})
        return f"{self.base_url}/gateway/ql?{encoded_query}"

    def search(self, query: str, page: int = 1) -> YacpdbResponse:
        """Run one QL search request and return the raw response."""

        return self._get(self.ql_url(query, page))

    def fetch_by_id(self, problem_id: int | str) -> YacpdbResponse:
        """Fetch one problem by internal YACPDB ID via the QL endpoint."""

        problem_id_text = str(problem_id).strip()
        if not problem_id_text.isdigit():
            raise ValueError("problem_id must be an integer ID")
        return self.search(f"Id({problem_id_text})", page=1)

    def parse_entries(self, response: YacpdbResponse, max_records: int = DEFAULT_MAX_RECORDS) -> list[dict[str, Any]]:
        """Return bounded entries from a YACPDB JSON response."""

        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        payload = response.json()
        if not payload.get("success"):
            raise YacpdbClientError(str(payload.get("error") or "YACPDB query failed"))
        result = payload.get("result")
        if not isinstance(result, dict):
            raise YacpdbClientError("YACPDB response has no result object")
        entries = result.get("entries")
        if not isinstance(entries, list):
            raise YacpdbClientError("YACPDB response has no entries list")
        if len(entries) > max_records:
            raise YacpdbClientError(f"response has {len(entries)} entries, exceeding max_records={max_records}")
        return [entry for entry in entries if isinstance(entry, dict)]

    def cache_raw_response(
        self,
        response: YacpdbResponse,
        output_dir: Path,
        stem: str,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        """Write raw response and metadata files without normalizing content."""

        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / f"{stem}.json"
        metadata_path = output_dir / f"{stem}.metadata.json"
        if not overwrite and (raw_path.exists() or metadata_path.exists()):
            raise FileExistsError(f"Refusing to overwrite existing smoke artifact: {stem}")
        raw_path.write_bytes(response.body)
        payload = {
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
            "request_method": response.method,
            "endpoint": response.url,
            "status": response.status,
            "content_type": response.content_type,
            "source_client_reference": "FEN Tool src/app/tools/yacpdb.ts and YACPDB web static/js/assets.js",
        }
        if metadata:
            payload.update(metadata)
        metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return raw_path, metadata_path

    def _get(self, url: str) -> YacpdbResponse:
        """Perform a GET request with limited retry for transient failures."""

        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with self.opener.urlopen(request, timeout=self.timeout) as response:
                    status = int(getattr(response, "status", response.getcode()))
                    content_type = response.headers.get("Content-Type", "")
                    body = response.read()
                    if status >= 500:
                        raise YacpdbClientError(f"transient HTTP {status}")
                    if status >= 400:
                        raise YacpdbClientError(f"permanent HTTP {status}")
                    return YacpdbResponse("GET", url, status, content_type, body)
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    raise YacpdbClientError(f"permanent HTTP {exc.code}") from exc
                last_error = exc
            except (urllib.error.URLError, TimeoutError, YacpdbClientError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.sleep_seconds)
        raise YacpdbClientError(f"YACPDB request failed: {last_error}") from last_error


def parse_args() -> argparse.Namespace:
    """Parse diagnostic smoke CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fetch-id", "query"], required=True)
    parser.add_argument("--problem-id")
    parser.add_argument("--query")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument("--output-dir", default="data/heldout_classic/raw/yacpdb/smoke")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def main() -> int:
    """Run a bounded live smoke request and cache the raw response."""

    args = parse_args()
    client = YacpdbClient(timeout=args.timeout)
    if args.mode == "fetch-id":
        if not args.problem_id:
            raise SystemExit("--problem-id is required for fetch-id")
        response = client.fetch_by_id(args.problem_id)
        stem = f"fetch_id_{args.problem_id}"
        safe_parameters = {"problem_id": args.problem_id}
    else:
        if not args.query:
            raise SystemExit("--query is required for query")
        response = client.search(args.query, page=args.page)
        stem = f"query_page_{args.page}"
        safe_parameters = {"query": args.query, "page": args.page}
    entries = client.parse_entries(response, max_records=args.max_records)
    client.cache_raw_response(
        response,
        Path(args.output_dir),
        stem,
        metadata={
            "safe_request_parameters": safe_parameters,
            "record_count": len(entries),
            "max_records_guard": args.max_records,
        },
    )
    print(json.dumps({"record_count": len(entries), "output_dir": args.output_dir}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

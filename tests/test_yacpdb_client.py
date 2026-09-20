import json
import urllib.error
import urllib.request

import pytest

from src.data.heldout_sources.yacpdb_client import YacpdbClient, YacpdbClientError


class FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class FakeResponse:
    def __init__(self, body, status=200, content_type="application/json"):
        self.body = body
        self.status = status
        self.headers = FakeHeaders({"Content-Type": content_type})

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body


class FakeOpener:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def urlopen(self, request, timeout):
        self.requests.append((request, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def payload(entries=None, success=True):
    return json.dumps({"success": success, "result": {"entries": entries or [], "count": len(entries or [])}}).encode()


def test_search_response_parsing_and_request_construction():
    opener = FakeOpener([FakeResponse(payload([{"id": 26026}]))])
    client = YacpdbClient(opener=opener, retries=0, sleep_seconds=0)

    response = client.search('Stip("^#2$") AND Id(26026)', page=1)
    entries = client.parse_entries(response, max_records=3)

    assert entries == [{"id": 26026}]
    request, timeout = opener.requests[0]
    assert "gateway/ql?" in request.full_url
    assert "q=Stip" in request.full_url
    assert "p=1" in request.full_url
    assert request.headers["User-agent"].startswith("Progetto-Damiani")
    assert timeout == client.timeout


def test_fetch_by_id_uses_id_predicate():
    opener = FakeOpener([FakeResponse(payload([{"id": 26026}]))])
    client = YacpdbClient(opener=opener, retries=0, sleep_seconds=0)

    client.fetch_by_id(26026)

    assert "Id%2826026%29" in opener.requests[0][0].full_url


def test_timeout_then_retry_success():
    opener = FakeOpener([urllib.error.URLError("timed out"), FakeResponse(payload([{"id": 1}]))])
    client = YacpdbClient(opener=opener, retries=1, sleep_seconds=0)

    response = client.fetch_by_id(1)

    assert client.parse_entries(response) == [{"id": 1}]
    assert len(opener.requests) == 2


def test_permanent_error_does_not_retry():
    error = urllib.error.HTTPError("https://yacpdb.org/gateway/ql", 404, "not found", {}, None)
    opener = FakeOpener([error])
    client = YacpdbClient(opener=opener, retries=2, sleep_seconds=0)

    with pytest.raises(YacpdbClientError, match="permanent HTTP 404"):
        client.fetch_by_id(999999)

    assert len(opener.requests) == 1


def test_malformed_response_fails():
    opener = FakeOpener([FakeResponse(b"not-json")])
    client = YacpdbClient(opener=opener, retries=0, sleep_seconds=0)

    response = client.fetch_by_id(1)

    with pytest.raises(YacpdbClientError, match="not valid JSON"):
        client.parse_entries(response)


def test_unsuccessful_response_fails():
    body = json.dumps({"success": False, "error": "bad query"}).encode()
    opener = FakeOpener([FakeResponse(body)])
    client = YacpdbClient(opener=opener, retries=0, sleep_seconds=0)

    response = client.search("bad")

    with pytest.raises(YacpdbClientError, match="bad query"):
        client.parse_entries(response)


def test_max_record_guard():
    opener = FakeOpener([FakeResponse(payload([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]))])
    client = YacpdbClient(opener=opener, retries=0, sleep_seconds=0)

    response = client.search("Stip(\"^#1$\")", page=1)

    with pytest.raises(YacpdbClientError, match="exceeding max_records=3"):
        client.parse_entries(response, max_records=3)


def test_raw_cache_and_cache_hit(tmp_path):
    opener = FakeOpener([FakeResponse(payload([{"id": 26026}]))])
    client = YacpdbClient(opener=opener, retries=0, sleep_seconds=0)
    response = client.fetch_by_id(26026)

    raw_path, metadata_path = client.cache_raw_response(
        response,
        tmp_path,
        "fetch_id_26026",
        metadata={"safe_request_parameters": {"problem_id": "26026"}},
    )

    assert json.loads(raw_path.read_text())["result"]["entries"][0]["id"] == 26026
    metadata = json.loads(metadata_path.read_text())
    assert metadata["safe_request_parameters"] == {"problem_id": "26026"}
    with pytest.raises(FileExistsError):
        client.cache_raw_response(response, tmp_path, "fetch_id_26026")


def test_invalid_id_and_page_are_rejected():
    client = YacpdbClient(opener=FakeOpener([]), retries=0, sleep_seconds=0)

    with pytest.raises(ValueError):
        client.fetch_by_id("abc")
    with pytest.raises(ValueError):
        client.ql_url("Id(1)", page=0)

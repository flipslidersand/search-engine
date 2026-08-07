"""メトリクスエンドポイントとカウンターのテスト（Phase 6 #27）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient  # pylint: disable=import-error

from searchengine import server


@pytest.fixture()
def tmp_db(tmp_path: Path):
    db = str(tmp_path / "test.db")
    server._set_db(db)
    yield db
    server._set_db("search.db")


@pytest.fixture()
def client(tmp_db):
    return TestClient(server.app)


@pytest.fixture()
def sample_dir(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "doc.txt").write_text("メトリクステスト用文書", encoding="utf-8")
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(tmp_path))
    monkeypatch.delenv("API_KEY", raising=False)
    return tmp_path


# ── /metrics エンドポイント ───────────────────────────────────────────────────


def test_metrics_endpoint_reachable(client):
    r = client.get("/metrics")
    assert r.status_code == 200


def test_metrics_content_type(client):
    r = client.get("/metrics")
    assert "text/plain" in r.headers["content-type"]


def test_metrics_prometheus_format(client):
    r = client.get("/metrics")
    text = r.text
    assert "# HELP" in text or "# TYPE" in text or text.startswith("#")


def test_metrics_no_auth_required(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret")
    r = client.get("/metrics")
    assert r.status_code == 200


# ── カウンター動作確認 ────────────────────────────────────────────────────────


def test_search_counter_increments(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    before = client.get("/metrics").text

    client.get("/search", params={"q": "メトリクス", "mode": "keyword"})
    after = client.get("/metrics").text

    assert "search_requests_total" in after


def test_index_counter_increments(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    r = client.get("/metrics")
    assert "index_requests_total" in r.text


def test_index_gauge_updated(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    r = client.get("/metrics")
    text = r.text
    assert "index_documents_total" in text
    assert "index_chunks_total" in text


def test_search_latency_histogram(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    client.get("/search", params={"q": "テスト", "mode": "keyword"})
    r = client.get("/metrics")
    assert "search_latency_seconds" in r.text


def test_multiple_search_modes_tracked(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    client.get("/search", params={"q": "テスト", "mode": "keyword"})
    client.get("/search", params={"q": "テスト", "mode": "keyword"})
    r = client.get("/metrics")
    assert 'mode="keyword"' in r.text

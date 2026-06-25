"""FastAPI サーバーのエンドポイントテスト（Phase 3）。"""

import json
import tempfile
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
def sample_dir(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text(
        "機械学習はデータからパターンを学習する分野である。", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text(
        "ベクトル検索は意味的類似性に基づく検索手法である。", encoding="utf-8"
    )
    return tmp_path


# ── /health ───────────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ── / (Web UI) ────────────────────────────────────────────────────────────────


def test_web_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SearchEngine" in r.text


# ── /stats（空DB） ────────────────────────────────────────────────────────────


def test_stats_empty(client):
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["documents"] == 0
    assert data["chunks"] == 0
    assert "tokenizer" in data


# ── /index ────────────────────────────────────────────────────────────────────


def test_index(client, sample_dir):
    r = client.post("/index", json={"path": str(sample_dir)})
    assert r.status_code == 200
    data = r.json()
    assert data["indexed"] == 2
    assert data["documents"] == 2
    assert data["chunks"] >= 2


def test_index_nonexistent_path(client):
    r = client.post("/index", json={"path": "/nonexistent/path/xyz"})
    assert r.status_code == 400
    assert "パスが存在しません" in r.json()["detail"]


# ── /search ───────────────────────────────────────────────────────────────────


def test_search_after_index(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    r = client.get("/search", params={"q": "機械学習", "mode": "keyword"})
    assert r.status_code == 200
    data = r.json()
    assert data["total"] >= 1
    assert data["results"][0]["rank"] == 1
    assert "snippet" in data["results"][0]


def test_search_empty_query(client):
    r = client.get("/search", params={"q": "  "})
    assert r.status_code == 400


def test_search_no_results(client):
    r = client.get("/search", params={"q": "全くヒットしないはずのワード12345xyz"})
    assert r.status_code == 200
    assert r.json()["total"] == 0


def test_search_n_limit(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    r = client.get("/search", params={"q": "検索", "n": 1})
    assert r.status_code == 200
    assert len(r.json()["results"]) <= 1


def test_stats_after_index(client, sample_dir):
    client.post("/index", json={"path": str(sample_dir)})
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["documents"] == 2

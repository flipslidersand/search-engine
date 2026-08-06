"""FastAPI サーバーのエンドポイントテスト（Phase 3）。"""

import json
import os
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
def sample_dir(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "a.txt").write_text(
        "機械学習はデータからパターンを学習する分野である。", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text(
        "ベクトル検索は意味的類似性に基づく検索手法である。", encoding="utf-8"
    )
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(tmp_path))
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


def test_index_nonexistent_path(client, tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(tmp_path))
    r = client.post("/index", json={"path": str(tmp_path / "nonexistent_xyz")})
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


# ── /index セキュリティ: パストラバーサル対策 (#32) ──────────────────────────


def test_index_path_traversal_no_allowed_dirs(client, monkeypatch):
    monkeypatch.delenv("ALLOWED_INDEX_DIRS", raising=False)
    r = client.post("/index", json={"path": "/etc/passwd"})
    assert r.status_code == 403
    assert "ALLOWED_INDEX_DIRS" in r.json()["detail"]


def test_index_path_traversal_outside_allowed(client, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(allowed))
    r = client.post("/index", json={"path": "/etc/passwd"})
    assert r.status_code == 403
    assert "許可されていないパス" in r.json()["detail"]


def test_index_path_traversal_symlink_escape(client, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(allowed))
    # allowed/../../../etc/passwd のような相対パス
    evil = str(allowed) + "/../../../etc/passwd"
    r = client.post("/index", json={"path": evil})
    assert r.status_code == 403


def test_index_path_within_allowed(client, tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "doc.txt").write_text("test content", encoding="utf-8")
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(allowed))
    r = client.post("/index", json={"path": str(allowed)})
    assert r.status_code == 200
    assert r.json()["indexed"] == 1


def test_index_multiple_allowed_dirs(client, tmp_path, monkeypatch):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_b / "doc.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", f"{dir_a}:{dir_b}")
    r = client.post("/index", json={"path": str(dir_b)})
    assert r.status_code == 200


# ── API 認証: X-API-Key ヘッダー (#33) ───────────────────────────────────────


def test_api_key_not_set_allows_all(client, sample_dir, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    r = client.get("/search", params={"q": "test"})
    assert r.status_code == 200


def test_api_key_valid(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.get("/search", params={"q": "test"}, headers={"X-API-Key": "secret123"})
    assert r.status_code == 200


def test_api_key_invalid(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.get("/search", params={"q": "test"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401
    assert "Invalid API Key" in r.json()["detail"]


def test_api_key_missing_header(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.get("/search", params={"q": "test"})
    assert r.status_code == 401


def test_health_no_api_key_required(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.get("/health")
    assert r.status_code == 200


def test_api_key_index_protected(client, tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    monkeypatch.setenv("ALLOWED_INDEX_DIRS", str(tmp_path))
    r = client.post("/index", json={"path": str(tmp_path)})
    assert r.status_code == 401


def test_api_key_stats_protected(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    r = client.get("/stats")
    assert r.status_code == 401

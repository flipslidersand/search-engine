"""RAG パイプライン + /ask エンドポイントのテスト。

Ollama への HTTP 呼び出しは unittest.mock.patch で差し替える。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from searchengine import ingest, server
from searchengine.index import Index
from searchengine.llm import LLMResult, Message, OllamaClient
from searchengine.rag import AskResult, Source, ask


# ── フィクスチャ ──────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    db = str(tmp_path / "test.db")
    server._set_db(db)
    yield db
    server._set_db("search.db")


@pytest.fixture()
def indexed_db(tmp_db: str, tmp_path: Path) -> str:
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "RRF（Reciprocal Rank Fusion）は複数のランキングを統合する手法です。"
        "キーワード検索とベクトル検索の結果を組み合わせられます。",
        encoding="utf-8",
    )
    idx = Index(tmp_db)
    idx.add_document(str(doc), doc.read_text(encoding="utf-8"))
    idx.close()
    return tmp_db


@pytest.fixture()
def client(tmp_db: str):
    return TestClient(server.app)


@pytest.fixture()
def indexed_client(indexed_db: str):
    return TestClient(server.app)


def _mock_ollama_response(content: str = "テスト回答") -> MagicMock:
    """httpx.post のモックレスポンスを生成する。"""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": content},
        "model": "qwen2.5:7b",
    }
    return mock_resp


# ── OllamaClient 単体テスト ───────────────────────────────────────────────────


def test_ollama_client_chat():
    with patch("httpx.post", return_value=_mock_ollama_response("こんにちは")) as mock_post:
        client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:7b")
        result = client.chat([Message(role="user", content="挨拶して")])

    assert result.content == "こんにちは"
    assert result.model == "qwen2.5:7b"
    assert result.latency_ms >= 0
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert call_kwargs[0][0] == "http://localhost:11434/api/chat"
    payload = call_kwargs[1]["json"]
    assert payload["stream"] is False
    assert payload["model"] == "qwen2.5:7b"


def test_ollama_client_connection_error():
    import httpx as httpx_module

    with patch("httpx.post", side_effect=httpx_module.ConnectError("接続失敗")):
        client = OllamaClient(base_url="http://localhost:11434")
        with pytest.raises(httpx_module.ConnectError):
            client.chat([Message(role="user", content="test")])


# ── ask() 単体テスト ──────────────────────────────────────────────────────────


def test_ask_keyword_mode(indexed_db: str):
    with patch("httpx.post", return_value=_mock_ollama_response("RRF は順位統合手法です")):
        idx = Index(indexed_db)
        try:
            result = ask(idx, "RRF とは？", mode="keyword", top_k=3)
        finally:
            idx.close()

    assert isinstance(result, AskResult)
    assert result.answer == "RRF は順位統合手法です"
    assert result.model == "qwen2.5:7b"
    assert result.latency_ms >= 0
    assert len(result.sources) >= 0


def test_ask_hybrid_mode(indexed_db: str):
    with patch("httpx.post", return_value=_mock_ollama_response("ハイブリッド検索の回答")):
        idx = Index(indexed_db)
        try:
            result = ask(idx, "検索統合", mode="hybrid", top_k=3)
        finally:
            idx.close()

    assert result.answer == "ハイブリッド検索の回答"


def test_ask_sources_structure(indexed_db: str):
    with patch("httpx.post", return_value=_mock_ollama_response("回答")):
        idx = Index(indexed_db)
        try:
            result = ask(idx, "RRF", mode="keyword", top_k=5)
        finally:
            idx.close()

    for src in result.sources:
        assert isinstance(src, Source)
        assert isinstance(src.path, str)
        assert isinstance(src.chunk_index, int)
        assert isinstance(src.snippet, str)
        assert isinstance(src.score, float)


# ── /ask エンドポイントテスト ─────────────────────────────────────────────────


def test_ask_endpoint_success(indexed_client):
    with patch("httpx.post", return_value=_mock_ollama_response("エンドポイントの回答")):
        r = indexed_client.post(
            "/ask",
            json={"question": "RRF とは何ですか？", "mode": "keyword"},
        )

    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "エンドポイントの回答"
    assert data["model"] == "qwen2.5:7b"
    assert "sources" in data
    assert "latency_ms" in data


def test_ask_endpoint_ollama_error(indexed_client):
    import httpx as httpx_module

    with patch("httpx.post", side_effect=httpx_module.ConnectError("Ollama 未起動")):
        r = indexed_client.post(
            "/ask",
            json={"question": "テスト", "mode": "keyword"},
        )

    assert r.status_code == 500


def test_ask_endpoint_empty_question(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code in (400, 422)


def test_ask_endpoint_custom_model(indexed_client):
    with patch("httpx.post", return_value=_mock_ollama_response("カスタムモデルの回答")) as mock_post:
        r = indexed_client.post(
            "/ask",
            json={"question": "テスト", "mode": "keyword", "model": "llama3:8b"},
        )

    assert r.status_code == 200
    call_payload = mock_post.call_args[1]["json"]
    assert call_payload["model"] == "llama3:8b"


def test_ask_endpoint_custom_ollama_url(indexed_client):
    with patch("httpx.post", return_value=_mock_ollama_response("回答")) as mock_post:
        r = indexed_client.post(
            "/ask",
            json={
                "question": "テスト",
                "mode": "keyword",
                "ollama_url": "http://192.168.68.59:11434",
            },
        )

    assert r.status_code == 200
    called_url = mock_post.call_args[0][0]
    assert "192.168.68.59" in called_url

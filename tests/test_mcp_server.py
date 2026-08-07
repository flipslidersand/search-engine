"""Phase 4: MCP サーバーのユニットテスト（Issue #24）。

実際の stdio 通信は行わず、handle_call_tool を直接呼び出してテストする。
asyncio.run() で同期テストから非同期ハンドラを呼び出す。
Ollama / RemoteEmbedder 依存は環境依存として is_error の有無を検証する。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp import types

from searchengine.mcp_server import TOOLS, create_server

# ── ヘルパー ──────────────────────────────────────────────────────────────────


def call_tool(name: str, args: dict) -> dict:
    """MCP tools/call を同期的に呼び出す（ハンドラは params モデルを受け取る）。"""
    server = create_server()
    handler = server.get_request_handler("tools/call")
    params = types.CallToolRequestParams(name=name, arguments=args)
    result: types.CallToolResult = asyncio.run(handler.handler(None, params))
    assert not result.is_error, f"Tool error: {result.content[0].text}"
    return json.loads(result.content[0].text)


def call_tool_raw(name: str, args: dict) -> types.CallToolResult:
    """is_error を含む生の CallToolResult を返す。"""
    server = create_server()
    handler = server.get_request_handler("tools/call")
    params = types.CallToolRequestParams(name=name, arguments=args)
    return asyncio.run(handler.handler(None, params))


# ── フィクスチャ ──────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture()
def indexed_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    doc = tmp_path / "sample.md"
    doc.write_text("# Hello\n\nThis is a test document about Python.", encoding="utf-8")

    from searchengine.index import Index

    idx = Index(db_path)
    idx.add_document(str(doc), doc.read_text(), doc_type="markdown")
    return db_path


# ── tools/list ────────────────────────────────────────────────────────────────


def test_tools_list():
    """4 つのツールが登録されていること。"""
    names = {t.name for t in TOOLS}
    assert names == {"search", "ask", "index", "stats"}


def test_tools_have_required_fields():
    """各ツールに description と input_schema があること。"""
    for tool in TOOLS:
        assert tool.description
        assert tool.input_schema


# ── stats ─────────────────────────────────────────────────────────────────────


def test_stats_empty_db(tmp_db: str):
    """空の DB で stats を呼ぶと documents=0 / chunks=0 が返る。"""
    result = call_tool("stats", {"db": tmp_db})
    assert result["documents"] == 0
    assert result["chunks"] == 0
    assert result["db"] == tmp_db


def test_stats_after_index(indexed_db: str):
    """インデックス後は documents >= 1。"""
    result = call_tool("stats", {"db": indexed_db})
    assert result["documents"] >= 1


# ── index ─────────────────────────────────────────────────────────────────────


def test_index_single_file(tmp_path: Path, tmp_db: str):
    """Markdown ファイルを index すると indexed=1 が返る。"""
    doc = tmp_path / "doc.md"
    doc.write_text("# Test\n\nHello world.", encoding="utf-8")

    result = call_tool("index", {"path": str(doc), "db": tmp_db})
    assert result["indexed"] == 1
    assert result["documents"] == 1


def test_index_unknown_extension(tmp_path: Path, tmp_db: str):
    """未対応拡張子のファイルは indexed=0。"""
    doc = tmp_path / "file.xyz"
    doc.write_text("some content", encoding="utf-8")

    result = call_tool("index", {"path": str(doc), "db": tmp_db})
    assert result["indexed"] == 0


# ── search ────────────────────────────────────────────────────────────────────


def test_search_keyword(indexed_db: str):
    """keyword モードで検索すると results が返る。"""
    result = call_tool(
        "search", {"query": "Python", "mode": "keyword", "db": indexed_db}
    )
    assert result["mode"] == "keyword"
    assert result["query"] == "Python"
    assert isinstance(result["results"], list)


def test_search_empty_db(tmp_db: str):
    """空の DB では results=[] で is_error にならない。"""
    result = call_tool("search", {"query": "hello", "mode": "keyword", "db": tmp_db})
    assert result["results"] == []


def test_search_result_fields(indexed_db: str):
    """ヒット結果に必須フィールドが含まれること。"""
    result = call_tool("search", {"query": "test", "mode": "keyword", "db": indexed_db})
    for hit in result["results"]:
        assert "rank" in hit
        assert "score" in hit
        assert "path" in hit
        assert "snippet" in hit


# ── エラーケース ──────────────────────────────────────────────────────────────


def test_unknown_tool_returns_error():
    """未知のツール名は is_error=True。"""
    result = call_tool_raw("nonexistent", {})
    assert result.is_error
    payload = json.loads(result.content[0].text)
    assert "error" in payload


# ── ask ───────────────────────────────────────────────────────────────────────


def test_ask_returns_result_or_error(indexed_db: str):
    """Ollama 有無に関わらず、answer または error を含む JSON が返ること。"""
    result = call_tool_raw(
        "ask", {"question": "What is this doc about?", "db": indexed_db}
    )
    payload = json.loads(result.content[0].text)
    if result.is_error:
        assert "error" in payload
    else:
        assert "answer" in payload
        assert "sources" in payload
        assert "model" in payload

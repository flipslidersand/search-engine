"""MCP サーバー — search-engine を Claude Code から直接利用可能にする（Phase 4）。

ツール一覧:
  search  — キーワード / ベクトル / ハイブリッド検索
  ask     — RAG クエリ（検索 + Ollama 生成）
  index   — ファイル／ディレクトリをインデックス
  stats   — DB 統計を返す

起動:
  python -m searchengine.mcp_server

環境変数:
  SEARCH_DB     — SQLite DB パス (default: search.db)
  OLLAMA_URL    — Ollama エンドポイント (default: http://localhost:11434)
  OLLAMA_MODEL  — モデル名 (default: qwen2.5:7b)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from mcp.server import Server
from mcp import types

from .index import Index
from . import hybrid, ingest, query
from .query import parse as parse_query
from .rag import ask as rag_ask

_DB_PATH = os.environ.get("SEARCH_DB", "search.db")
_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


def _make_text(text: str) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=text)]


def _open_index(db: str | None = None, *, want_vector: bool = False) -> Index:
    from .embedder import create_embedder

    db_path = db or _DB_PATH
    embedder = create_embedder() if want_vector else None
    return Index(db_path, embedder=embedder)


TOOLS = [
    types.Tool(
        name="search",
        description=(
            "ローカル文書を検索する。mode に応じてキーワード / ベクトル / ハイブリッドを切り替え。"
            "結果は rank / score / path / snippet のリスト。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索クエリ"},
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "vector", "hybrid"],
                    "default": "hybrid",
                    "description": "検索モード",
                },
                "top_k": {"type": "integer", "default": 5, "description": "取得件数"},
                "db": {"type": "string", "description": "DB パス（省略時は SEARCH_DB 環境変数）"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="ask",
        description=(
            "RAG クエリ: 文書を検索して Ollama でまとめた回答を返す。"
            "answer / sources / model / latency_ms を含む。"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "質問文"},
                "mode": {
                    "type": "string",
                    "enum": ["keyword", "vector", "hybrid"],
                    "default": "hybrid",
                },
                "model": {
                    "type": "string",
                    "default": "qwen2.5:7b",
                    "description": "Ollama モデル名",
                },
                "top_k": {"type": "integer", "default": 5},
                "db": {"type": "string"},
            },
            "required": ["question"],
        },
    ),
    types.Tool(
        name="index",
        description="ファイルまたはディレクトリをインデックスする。indexed / documents / chunks を返す。",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "インデックス対象のファイルまたはディレクトリ"},
                "vector": {
                    "type": "boolean",
                    "default": False,
                    "description": "ベクトル埋め込みも生成するか",
                },
                "db": {"type": "string"},
            },
            "required": ["path"],
        },
    ),
    types.Tool(
        name="stats",
        description="DB の統計情報（文書数・チャンク数・使用トークナイザー）を返す。",
        input_schema={
            "type": "object",
            "properties": {
                "db": {"type": "string"},
            },
            "required": [],
        },
    ),
]


def create_server() -> Server:
    server = Server("search-engine")

    async def handle_list_tools(_req: types.ListToolsRequest) -> types.ListToolsResult:
        return types.ListToolsResult(tools=TOOLS)

    async def handle_call_tool(req: types.CallToolRequest) -> types.CallToolResult:
        name = req.params.name
        args = req.params.arguments or {}

        try:
            if name == "search":
                mode: Literal["keyword", "vector", "hybrid"] = args.get("mode", "hybrid")
                want_vector = mode in ("vector", "hybrid")
                idx = _open_index(args.get("db"), want_vector=want_vector)
                top_k: int = int(args.get("top_k", 5))
                parsed = parse_query(args["query"])

                if mode == "keyword":
                    raw_hits = [(h, h.score) for h in idx.search(parsed.fts, limit=top_k, filters=parsed.filters)]
                elif mode == "vector":
                    raw_hits = [(h, h.score) for h in idx.vector_search(parsed.raw, limit=top_k, filters=parsed.filters)]
                else:
                    fused = hybrid.search(idx, parsed, limit=top_k)
                    raw_hits = [(f.hit, f.rrf) for f in fused]

                result = {
                    "mode": mode,
                    "query": args["query"],
                    "total": len(raw_hits),
                    "results": [
                        {
                            "rank": i + 1,
                            "score": round(score, 4),
                            "path": h.path,
                            "chunk_index": h.chunk_index,
                            "snippet": h.snippet,
                        }
                        for i, (h, score) in enumerate(raw_hits)
                    ],
                }

            elif name == "ask":
                mode = args.get("mode", "hybrid")
                idx = _open_index(args.get("db"), want_vector=True)
                res = rag_ask(
                    idx,
                    args["question"],
                    mode=mode,
                    top_k=int(args.get("top_k", 5)),
                    ollama_url=_OLLAMA_URL,
                    model=args.get("model", _OLLAMA_MODEL),
                )
                result = {
                    "answer": res.answer,
                    "model": res.model,
                    "latency_ms": res.latency_ms,
                    "sources": [
                        {
                            "path": s.path,
                            "chunk_index": s.chunk_index,
                            "snippet": s.snippet,
                            "score": round(s.score, 4),
                        }
                        for s in res.sources
                    ],
                }

            elif name == "index":
                want_vector = bool(args.get("vector", False))
                idx = _open_index(args.get("db"), want_vector=want_vector)
                indexed = 0
                for file_path in ingest.iter_files(args["path"]):
                    doc_type = ingest.detect_type(file_path)
                    if doc_type is None:
                        continue
                    text = ingest.read_text(file_path)
                    idx.add_document(file_path, text, doc_type=doc_type)
                    indexed += 1
                st = idx.stats()
                embedder_name = type(idx.embedder).__name__ if idx.embedder else None
                result = {
                    "indexed": indexed,
                    "documents": st["documents"],
                    "chunks": st["chunks"],
                    "embedder": embedder_name,
                }

            elif name == "stats":
                idx = _open_index(args.get("db"))
                st = idx.stats()
                result = {
                    "documents": st["documents"],
                    "chunks": st["chunks"],
                    "db": args.get("db") or _DB_PATH,
                }

            else:
                return types.CallToolResult(
                    content=_make_text(json.dumps({"error": f"unknown tool: {name}"})),
                    is_error=True,
                )

            return types.CallToolResult(content=_make_text(json.dumps(result, ensure_ascii=False)))

        except Exception as exc:  # pylint: disable=broad-except
            return types.CallToolResult(
                content=_make_text(json.dumps({"error": str(exc), "tool": name})),
                is_error=True,
            )

    server.add_request_handler("tools/list", types.ListToolsRequest, handle_list_tools)
    server.add_request_handler("tools/call", types.CallToolRequest, handle_call_tool)

    return server


if __name__ == "__main__":
    from mcp.server.stdio import stdio_server

    server = create_server()

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())

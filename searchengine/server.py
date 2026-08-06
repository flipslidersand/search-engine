"""FastAPI 検索 API サーバー（Phase 3）。

起動:
    uvicorn searchengine.server:app --reload
    python -m searchengine.server [--db search.db] [--host 0.0.0.0] [--port 8000]

エンドポイント:
    GET  /           → Web UI
    GET  /health     → ヘルスチェック
    POST /index      → ファイルをインデックス
    GET  /search     → 検索（keyword / vector / hybrid）
    GET  /stats      → DB 統計
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query  # pylint: disable=import-error
from fastapi.responses import FileResponse, HTMLResponse  # pylint: disable=import-error
from pydantic import BaseModel  # pylint: disable=import-error

from . import hybrid, ingest, query, tokenizer
from .index import Index
from .schema_gen.api import router as schema_router

# ── DB パス（起動時に差し替え可） ────────────────────────────────────────────
_DB_PATH: str = os.environ.get("SEARCH_DB", "search.db")


def _set_db(path: str) -> None:
    global _DB_PATH
    _DB_PATH = path


# ── FastAPI アプリ ────────────────────────────────────────────────────────────
app = FastAPI(
    title="SearchEngine",
    description="BM25 + ベクトル ハイブリッド検索 API",
    version="0.4.0",
)
app.include_router(schema_router)

# ── スキーマ ──────────────────────────────────────────────────────────────────


class IndexRequest(BaseModel):
    path: str
    vector: bool = False
    db: str | None = None


class IndexResponse(BaseModel):
    indexed: int
    documents: int
    chunks: int
    tokenizer: str
    embedder: str | None = None


class SearchHit(BaseModel):
    rank: int
    score: float
    path: str
    chunk_index: int
    snippet: str


class SearchResponse(BaseModel):
    mode: str
    query: str
    total: int
    results: list[SearchHit]


class StatsResponse(BaseModel):
    documents: int
    chunks: int
    tokenizer: str
    db: str


class AskSource(BaseModel):
    path: str
    chunk_index: int
    snippet: str
    score: float


class AskRequest(BaseModel):
    question: str
    mode: Literal["keyword", "vector", "hybrid"] = "hybrid"
    model: str = "qwen2.5:7b"
    top_k: int = 5
    ollama_url: str | None = None
    db: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[AskSource]
    model: str
    latency_ms: int


# ── ヘルパー ──────────────────────────────────────────────────────────────────


def _open_index(db: str | None, *, want_vector: bool = False) -> Index:
    db_path = db or _DB_PATH
    embedder = None
    if want_vector:
        from .embedder import Embedder

        embedder = Embedder()
    return Index(db_path, embedder=embedder)


# ── エンドポイント ────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/index", response_model=IndexResponse)
def index_files(req: IndexRequest) -> IndexResponse:
    """指定パス（ファイルまたはディレクトリ）をインデックスに追加する。"""
    target = Path(req.path)
    if not target.exists():
        raise HTTPException(status_code=400, detail=f"パスが存在しません: {req.path}")

    idx = _open_index(req.db, want_vector=req.vector)
    count = 0
    try:
        for path in ingest.iter_files(str(target)):
            text = ingest.read_text(path)
            idx.add_document(
                path,
                text,
                doc_type=ingest.detect_type(path) or "text",
                mtime=os.path.getmtime(path),
            )
            count += 1
        s = idx.stats()
    finally:
        idx.close()

    return IndexResponse(
        indexed=count,
        documents=s["documents"],
        chunks=s["chunks"],
        tokenizer=tokenizer.backend(),
        embedder=idx.embedder.backend if idx.embedder else None,
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="検索クエリ"),
    mode: Literal["keyword", "vector", "hybrid"] = Query("keyword"),
    n: int = Query(10, ge=1, le=100),
    db: str | None = Query(None),
) -> SearchResponse:
    """クエリを検索して上位 n 件を返す。"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="クエリが空です")

    want_vector = mode in ("vector", "hybrid")
    idx = _open_index(db, want_vector=want_vector)
    parsed = query.parse(q)

    try:
        if mode == "keyword":
            hits = [(h, h.score) for h in idx.search(parsed.fts, limit=n, filters=parsed.filters)]
        elif mode == "vector":
            hits = [
                (h, h.score) for h in idx.vector_search(parsed.raw, limit=n, filters=parsed.filters)
            ]
        else:
            fused = hybrid.search(idx, parsed, limit=n)
            hits = [(f.hit, f.rrf) for f in fused]
    finally:
        idx.close()

    results = [
        SearchHit(
            rank=i + 1,
            score=round(score, 6),
            path=h.path,
            chunk_index=h.chunk_index,
            snippet=h.snippet,
        )
        for i, (h, score) in enumerate(hits)
    ]
    return SearchResponse(mode=mode, query=q, total=len(results), results=results)


@app.get("/stats", response_model=StatsResponse)
def stats(db: str | None = Query(None)) -> StatsResponse:
    """インデックスの統計情報を返す。"""
    idx = _open_index(db)
    try:
        s = idx.stats()
    finally:
        idx.close()
    return StatsResponse(
        documents=s["documents"],
        chunks=s["chunks"],
        tokenizer=tokenizer.backend(),
        db=db or _DB_PATH,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """RAG: 検索 → Ollama で回答生成。"""
    import os
    from . import rag as rag_module

    want_vector = req.mode in ("vector", "hybrid")
    idx = _open_index(req.db, want_vector=want_vector)
    try:
        result = rag_module.ask(
            idx,
            req.question,
            mode=req.mode,
            top_k=req.top_k,
            ollama_url=req.ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            model=req.model,
        )
    finally:
        idx.close()

    return AskResponse(
        answer=result.answer,
        sources=[
            AskSource(
                path=s.path,
                chunk_index=s.chunk_index,
                snippet=s.snippet,
                score=round(s.score, 6),
            )
            for s in result.sources
        ],
        model=result.model,
        latency_ms=result.latency_ms,
    )


@app.get("/", response_class=HTMLResponse)
def web_ui() -> HTMLResponse:
    """シングルページ検索 UI を返す。"""
    html_path = Path(__file__).parent / "web" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>SearchEngine</h1><p>web/index.html が見つかりません。</p>")


# ── エントリポイント ──────────────────────────────────────────────────────────


def _cli() -> None:
    p = argparse.ArgumentParser(prog="python -m searchengine.server")
    p.add_argument("--db", default="search.db")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    _set_db(args.db)

    import uvicorn  # pylint: disable=import-error

    uvicorn.run(
        "searchengine.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    _cli()

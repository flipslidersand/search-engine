"""schema-gen FastAPI ルーター（Phase 5）。

エンドポイント:
    POST /schema/analyze       → ファイル解析 + ストア保存
    GET  /schema/search        → カラム名・型・テーブル名で検索
    GET  /schema/list          → 保存済みスキーマ一覧
    GET  /schema/{table_name}  → スキーマ詳細取得
    DELETE /schema/{table_name} → スキーマ削除
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .analyzer import analyze, ColumnInfo
from .ingest import load_file, load_sheets
from .reporter import report_sql, report_er
from .requirements_doc import render_requirements_md
from .store import SchemaStore, SchemaMeta, SchemaDetail, ColumnHit

_SCHEMA_DB: str = os.environ.get("SCHEMA_DB", "schemas.db")

router = APIRouter(prefix="/schema", tags=["schema-gen"])


def _store() -> SchemaStore:
    return SchemaStore(_SCHEMA_DB)


# ── スキーマ ──────────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    file_path: str | None = None
    sheets_url: str | None = None
    sheet: str | None = None
    credentials: str | None = None
    table: str | None = None
    include_sql: bool = True
    include_er: bool = False
    include_req: bool = False
    save: bool = True


class ColumnInfoOut(BaseModel):
    name: str
    inferred_type: str
    null_pct: float
    unique_pct: float
    pk_candidate: bool
    sample_values: list[str]


class AnalyzeResponse(BaseModel):
    table_name: str
    filename: str
    row_count: int
    columns: list[ColumnInfoOut]
    sql: str | None = None
    er_md: str | None = None
    req_md: str | None = None
    saved: bool = False


class SchemaMetaOut(BaseModel):
    id: int
    table_name: str
    filename: str
    row_count: int
    col_count: int
    saved_at: str


class ColumnHitOut(BaseModel):
    table_name: str
    filename: str
    col_name: str
    col_type: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    field: str | None
    total: int
    results: list[ColumnHitOut]


# ── ヘルパー ──────────────────────────────────────────────────────────────────


def _col_out(c: ColumnInfo) -> ColumnInfoOut:
    return ColumnInfoOut(
        name=c.name,
        inferred_type=c.inferred_type,
        null_pct=c.null_pct,
        unique_pct=c.unique_pct,
        pk_candidate=c.pk_candidate,
        sample_values=c.sample_values[:3],
    )


# ── エンドポイント ────────────────────────────────────────────────────────────


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_file(req: AnalyzeRequest) -> AnalyzeResponse:
    """CSV / Excel / Google Sheets を解析してスキーマを返す。save=true で DB に保存。"""
    if req.file_path:
        if not Path(req.file_path).exists():
            raise HTTPException(status_code=400, detail=f"ファイルが存在しません: {req.file_path}")
        rows, filename = load_file(req.file_path, req.sheet)
        stem = Path(req.file_path).stem
    elif req.sheets_url:
        rows, source_id = load_sheets(req.sheets_url, req.sheet, req.credentials)
        filename = f"sheets:{source_id[:12]}…"
        stem = source_id[:20]
    else:
        raise HTTPException(status_code=400, detail="file_path または sheets_url を指定してください")

    if not rows:
        raise HTTPException(status_code=422, detail="データが空です")

    table_name = req.table or stem.replace(" ", "_").replace("-", "_")
    columns = analyze(rows)

    sql = report_sql(table_name, columns) if req.include_sql else None
    er_md = report_er(table_name, columns) if req.include_er else None
    req_md = render_requirements_md(filename, table_name, columns, len(rows)) if req.include_req else None

    saved = False
    if req.save:
        store = _store()
        store.save(table_name, filename, columns, len(rows),
                   sql=sql or "", er_md=er_md or "", req_md=req_md or "")
        saved = True

    return AnalyzeResponse(
        table_name=table_name,
        filename=filename,
        row_count=len(rows),
        columns=[_col_out(c) for c in columns],
        sql=sql,
        er_md=er_md,
        req_md=req_md,
        saved=saved,
    )


@router.get("/search", response_model=SearchResponse)
def search_schemas(
    q: str = Query(..., description="検索クエリ（カラム名・型・テーブル名）"),
    field: Literal["col_name", "col_type", "table_name"] | None = Query(
        None, description="検索対象フィールド（省略時は全フィールド）"
    ),
    n: int = Query(20, ge=1, le=100),
) -> SearchResponse:
    """保存済みスキーマをカラム名・型・テーブル名で全文検索する。"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="クエリが空です")
    try:
        hits = _store().search_columns(q, field=field, limit=n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return SearchResponse(
        query=q,
        field=field,
        total=len(hits),
        results=[ColumnHitOut(**h.__dict__) for h in hits],
    )


@router.get("/list", response_model=list[SchemaMetaOut])
def list_schemas(limit: int = Query(50, ge=1, le=500)) -> list[SchemaMetaOut]:
    """保存済みスキーマの一覧を返す。"""
    return [SchemaMetaOut(**m.__dict__) for m in _store().list_schemas(limit)]


@router.get("/{table_name}", response_model=AnalyzeResponse)
def get_schema(table_name: str) -> AnalyzeResponse:
    """テーブル名でスキーマ詳細を取得する。"""
    detail = _store().get(table_name)
    if not detail:
        raise HTTPException(status_code=404, detail=f"スキーマが見つかりません: {table_name}")
    return AnalyzeResponse(
        table_name=detail.table_name,
        filename=detail.filename,
        row_count=detail.row_count,
        columns=[ColumnInfoOut(**c) for c in detail.columns],
        sql=detail.sql or None,
        er_md=detail.er_md or None,
        req_md=detail.req_md or None,
        saved=True,
    )


@router.delete("/{table_name}")
def delete_schema(table_name: str) -> dict:
    """保存済みスキーマを削除する。"""
    if not _store().delete(table_name):
        raise HTTPException(status_code=404, detail=f"スキーマが見つかりません: {table_name}")
    return {"deleted": table_name}

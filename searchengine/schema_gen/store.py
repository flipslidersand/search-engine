"""解析済みスキーマの SQLite ストア。

スキーマ解析結果を保存し、カラム名・型・テーブル名で全文検索できる。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SchemaMeta:
    id: int
    table_name: str
    filename: str
    row_count: int
    col_count: int
    saved_at: str


@dataclass
class SchemaDetail(SchemaMeta):
    columns: list[dict]
    sql: str
    er_md: str
    req_md: str


@dataclass
class ColumnHit:
    table_name: str
    filename: str
    col_name: str
    col_type: str
    snippet: str
    score: float


_DDL = """
CREATE TABLE IF NOT EXISTS schemas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    table_name  TEXT NOT NULL UNIQUE,
    filename    TEXT NOT NULL,
    row_count   INTEGER NOT NULL,
    columns_json TEXT NOT NULL,
    sql_text    TEXT NOT NULL DEFAULT '',
    er_md       TEXT NOT NULL DEFAULT '',
    req_md      TEXT NOT NULL DEFAULT '',
    saved_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS schema_columns (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_id INTEGER NOT NULL REFERENCES schemas(id) ON DELETE CASCADE,
    table_name TEXT NOT NULL,
    col_name  TEXT NOT NULL,
    col_type  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS columns_fts USING fts5(
    table_name,
    col_name,
    col_type,
    content='schema_columns',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS schema_columns_ai AFTER INSERT ON schema_columns BEGIN
    INSERT INTO columns_fts(rowid, table_name, col_name, col_type)
    VALUES (new.id, new.table_name, new.col_name, new.col_type);
END;

CREATE TRIGGER IF NOT EXISTS schema_columns_ad AFTER DELETE ON schema_columns BEGIN
    INSERT INTO columns_fts(columns_fts, rowid, table_name, col_name, col_type)
    VALUES ('delete', old.id, old.table_name, old.col_name, old.col_type);
END;
"""


class SchemaStore:
    def __init__(self, db_path: str | Path = "schemas.db"):
        self._db = str(db_path)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.executescript(_DDL)

    def save(
        self,
        table_name: str,
        filename: str,
        columns: list,  # list[ColumnInfo]
        row_count: int,
        sql: str = "",
        er_md: str = "",
        req_md: str = "",
    ) -> int:
        cols_json = json.dumps(
            [
                {
                    "name": c.name,
                    "inferred_type": c.inferred_type,
                    "null_pct": c.null_pct,
                    "unique_pct": c.unique_pct,
                    "pk_candidate": c.pk_candidate,
                    "sample_values": c.sample_values[:3],
                }
                for c in columns
            ],
            ensure_ascii=False,
        )
        with self._conn() as conn:
            # 既存スキーマ削除（cascade で schema_columns も削除 → FTS delete trigger 発火）
            conn.execute("DELETE FROM schemas WHERE table_name = ?", (table_name,))

            cur = conn.execute(
                """
                INSERT INTO schemas
                    (table_name, filename, row_count, columns_json, sql_text, er_md, req_md)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (table_name, filename, row_count, cols_json, sql, er_md, req_md),
            )
            schema_id = cur.lastrowid

            # カラム行を挿入（insert trigger が FTS に反映）
            for c in columns:
                conn.execute(
                    "INSERT INTO schema_columns (schema_id, table_name, col_name, col_type) "
                    "VALUES (?, ?, ?, ?)",
                    (schema_id, table_name, c.name, c.inferred_type),
                )

        return schema_id or 0

    def list_schemas(self, limit: int = 50) -> list[SchemaMeta]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, table_name, filename, row_count, "
                "json_array_length(columns_json) col_count, saved_at "
                "FROM schemas ORDER BY saved_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [SchemaMeta(**dict(r)) for r in rows]

    def get(self, table_name: str) -> SchemaDetail | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, table_name, filename, row_count, "
                "json_array_length(columns_json) col_count, saved_at, "
                "columns_json, sql_text sql, er_md, req_md "
                "FROM schemas WHERE table_name = ?",
                (table_name,),
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["columns"] = json.loads(d.pop("columns_json"))
        return SchemaDetail(**d)

    def search_columns(
        self,
        q: str,
        field: str | None = None,
        limit: int = 20,
    ) -> list[ColumnHit]:
        """カラム名・型・テーブル名で全文検索する。

        field: 'col_name' | 'col_type' | 'table_name' | None（全フィールド）
        """
        valid_fields = ("col_name", "col_type", "table_name")
        if field and field not in valid_fields:
            raise ValueError(f"field must be one of {valid_fields}, got {field!r}")

        fts_q = f"{field}:{q}" if field else q

        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT sc.table_name, s.filename,
                       sc.col_name, sc.col_type,
                       snippet(columns_fts, 1, '<b>', '</b>', '…', 10) snip,
                       bm25(columns_fts) score
                FROM columns_fts
                JOIN schema_columns sc ON sc.id = columns_fts.rowid
                JOIN schemas s ON s.id = sc.schema_id
                WHERE columns_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_q, limit),
            ).fetchall()
        return [
            ColumnHit(
                table_name=r["table_name"],
                filename=r["filename"],
                col_name=r["col_name"],
                col_type=r["col_type"],
                snippet=r["snip"],
                score=round(abs(r["score"]), 6),
            )
            for r in rows
        ]

    def delete(self, table_name: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM schemas WHERE table_name = ?", (table_name,))
        return cur.rowcount > 0

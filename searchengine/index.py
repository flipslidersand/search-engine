"""SQLite FTS5 によるインデックス（設計書 §2.4, §2.5, §2.8）。

- chunks_fts: FTS5 仮想テーブル（BM25スコアリング内蔵）
- documents : 文書メタデータ（パス・種別・タイムスタンプ・内容ハッシュ）
- インクリメンタル更新: 内容ハッシュで変更検知し、変更分のみ DELETE→INSERT。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass

from . import chunker, tokenizer
from .vector_store import VectorStore


@dataclass
class Hit:
    doc_id: str
    path: str
    chunk_index: int
    snippet: str
    score: float  # 小さいほど良い（FTS5 bm25 はマイナス側が高関連）


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    doc_type   TEXT,
    mtime      REAL,
    hash       TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,                 -- 形態素分割済みテキスト（検索対象）
    original,                -- スニペット表示用の原文
    doc_id UNINDEXED,
    chunk_index UNINDEXED,
    tokenize = 'unicode61'
);
"""


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_id(path: str) -> str:
    return _hash(os.path.abspath(path))[:16]


class Index:
    def __init__(self, db_path: str, embedder=None) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        # embedder を渡すとベクトル索引も同 DB に構築（設計書 §2.6）
        self.embedder = embedder
        self.vectors = VectorStore(self.conn) if embedder is not None else None

    def close(self) -> None:
        self.conn.close()

    # --- インクリメンタル更新 (§2.5) ---
    def add_document(
        self, path: str, text: str, doc_type: str = "text", mtime: float | None = None
    ) -> str:
        doc_id = _doc_id(path)
        h = _hash(text)
        row = self.conn.execute("SELECT hash FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row and row[0] == h:
            return doc_id  # 変更なし → スキップ

        # 旧チャンクを削除してから再投入（更新 = DELETE→INSERT）
        self.conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        if self.vectors is not None:
            self.vectors.delete_doc(doc_id)
        chunks = chunker.chunk_text(text)
        for ch in chunks:
            self.conn.execute(
                "INSERT INTO chunks_fts (content, original, doc_id, chunk_index)"
                " VALUES (?, ?, ?, ?)",
                (tokenizer.tokenized_text(ch.text), ch.text, doc_id, ch.index),
            )
        if self.vectors is not None and chunks:
            vecs = self.embedder.encode([c.text for c in chunks])
            for ch, vec in zip(chunks, vecs):
                self.vectors.upsert(doc_id, ch.index, vec)
        self.conn.execute(
            "INSERT INTO documents (doc_id, path, doc_type, mtime, hash)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(doc_id) DO UPDATE SET"
            " path=excluded.path, doc_type=excluded.doc_type,"
            " mtime=excluded.mtime, hash=excluded.hash",
            (doc_id, path, doc_type, mtime, h),
        )
        self.conn.commit()
        return doc_id

    def remove_document(self, path: str) -> None:
        doc_id = _doc_id(path)
        self.conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        if self.vectors is not None:
            self.vectors.delete_doc(doc_id)
        self.conn.commit()

    # --- フィールド検索フィルタ (§2.9) ---
    def _filter_sql(self, filters: dict | None) -> tuple[str, list]:
        if not filters:
            return "", []
        clauses, params = [], []
        if "type" in filters:
            clauses.append("d.doc_type = ?")
            params.append(filters["type"])
        if "path" in filters:
            clauses.append("d.path LIKE ?")
            params.append(f"%{filters['path']}%")
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    def allowed_doc_ids(self, filters: dict | None) -> set[str] | None:
        """フィルタに合致する doc_id 集合（None = フィルタ無し）。"""
        if not filters:
            return None
        where, params = self._filter_sql(filters)
        sql = "SELECT doc_id FROM documents d WHERE 1=1" + where
        return {r[0] for r in self.conn.execute(sql, params).fetchall()}

    # --- 検索 (§2.7 BM25) ---
    def search(self, fts_query: str, limit: int = 10, filters: dict | None = None) -> list[Hit]:
        where, fparams = self._filter_sql(filters)
        sql = (
            "SELECT f.doc_id, d.path, f.chunk_index,"
            " snippet(chunks_fts, 1, '[', ']', ' … ', 12) AS snip,"
            " bm25(chunks_fts) AS score"
            " FROM chunks_fts f JOIN documents d ON d.doc_id = f.doc_id"
            " WHERE chunks_fts MATCH ?" + where + " ORDER BY score LIMIT ?"
        )
        rows = self.conn.execute(sql, [fts_query, *fparams, limit]).fetchall()
        return [
            Hit(doc_id=r[0], path=r[1], chunk_index=r[2], snippet=r[3], score=r[4]) for r in rows
        ]

    # --- ベクトル検索 (§2.6) ---
    def vector_search(self, text: str, limit: int = 10, filters: dict | None = None) -> list[Hit]:
        if self.vectors is None:
            raise RuntimeError("Index was created without an embedder")
        qvec = self.embedder.encode([text], mode="search")[0]
        allowed = self.allowed_doc_ids(filters)
        vhits = self.vectors.search(qvec, limit=limit, allowed_docs=allowed)
        return [self._to_hit(v.doc_id, v.chunk_index, v.score) for v in vhits]

    def _to_hit(self, doc_id: str, chunk_index: int, score: float) -> Hit:
        row = self.conn.execute(
            "SELECT d.path, f.original FROM chunks_fts f"
            " JOIN documents d ON d.doc_id = f.doc_id"
            " WHERE f.doc_id = ? AND f.chunk_index = ?",
            (doc_id, chunk_index),
        ).fetchone()
        path = row[0] if row else "?"
        snippet = (row[1][:120] + " …") if row and len(row[1]) > 120 else (row[1] if row else "")
        return Hit(doc_id=doc_id, path=path, chunk_index=chunk_index, snippet=snippet, score=score)

    def stats(self) -> dict:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        return {"documents": docs, "chunks": chunks}

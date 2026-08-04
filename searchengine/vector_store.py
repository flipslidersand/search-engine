"""ベクトル索引（設計書 §2.6）。

## 構成

- `VectorStoreProtocol` — 抽象インターフェース（typing.Protocol）
- `SqliteVectorStore`    — SQLite BLOB + numpy ブルートフォース（デフォルト）
- `QdrantVectorStore`   — Qdrant クライアント（Phase 2 実装: #44）

## 切り替え

`QDRANT_URL` 環境変数が設定されている場合は `QdrantVectorStore`、
未設定の場合は `SqliteVectorStore` を使う（`create_vector_store()` ファクトリで切り替え）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

VECTOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS vectors (
    doc_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    PRIMARY KEY (doc_id, chunk_index)
);
"""


@dataclass
class VHit:
    doc_id: str
    chunk_index: int
    score: float  # コサイン類似（大きいほど良い）


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """ベクトルストアの共通インターフェース。"""

    def upsert(self, doc_id: str, chunk_index: int, vec: np.ndarray) -> None: ...

    def delete_doc(self, doc_id: str) -> None: ...

    def search(
        self,
        query_vec: np.ndarray,
        limit: int = 10,
        allowed_docs: set[str] | None = None,
    ) -> list[VHit]: ...

    def close(self) -> None: ...


class SqliteVectorStore:
    """SQLite BLOB にベクトルを永続化し、numpy でブルートフォース検索する。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.conn.executescript(VECTOR_SCHEMA)

    def upsert(self, doc_id: str, chunk_index: int, vec: np.ndarray) -> None:
        blob = np.asarray(vec, dtype=np.float32).tobytes()
        self.conn.execute(
            "INSERT INTO vectors (doc_id, chunk_index, vec) VALUES (?, ?, ?)"
            " ON CONFLICT(doc_id, chunk_index) DO UPDATE SET vec=excluded.vec",
            (doc_id, chunk_index, blob),
        )

    def delete_doc(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM vectors WHERE doc_id = ?", (doc_id,))

    def search(
        self,
        query_vec: np.ndarray,
        limit: int = 10,
        allowed_docs: set[str] | None = None,
    ) -> list[VHit]:
        rows = self.conn.execute("SELECT doc_id, chunk_index, vec FROM vectors").fetchall()
        keys, mats = [], []
        for doc_id, ci, blob in rows:
            if allowed_docs is not None and doc_id not in allowed_docs:
                continue
            keys.append((doc_id, ci))
            mats.append(np.frombuffer(blob, dtype=np.float32))
        if not mats:
            return []
        mat = np.vstack(mats)
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0:
            q = q / qn
        sims = mat @ q
        order = np.argsort(-sims)[:limit]
        return [VHit(keys[i][0], keys[i][1], float(sims[i])) for i in order]

    def close(self) -> None:
        pass  # conn は Index が管理するため、ここでは閉じない


# 後方互換エイリアス（既存コードへの影響を防ぐ）
VectorStore = SqliteVectorStore

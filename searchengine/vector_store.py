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


# ── Qdrant コレクション仕様（#42 確定） ──────────────────────────────────────
# vector_size: MINIPC e5-small embedding svc (port 9092) の出力次元
# distance:    Cosine（正規化済みベクトルの内積と等価）
# payload:     { source: str, chunk_id: str, text: str, doc_type: str }
QDRANT_COLLECTION = "search-engine-docs"
QDRANT_VECTOR_SIZE = 384
QDRANT_DISTANCE = "Cosine"

class QdrantVectorStore:
    """Qdrant をバックエンドとするベクトルストア。"""

    def __init__(
        self,
        url: str,
        collection: str = QDRANT_COLLECTION,
        vector_size: int = QDRANT_VECTOR_SIZE,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # pylint: disable=import-error
            from qdrant_client.models import Distance, VectorParams  # pylint: disable=import-error
        except ImportError:
            raise ImportError(
                "Qdrant 連携には追加パッケージが必要です:\n"
                "  pip install qdrant-client"
            )

        self._client = QdrantClient(url=url)
        self._collection = collection
        self._vector_size = vector_size
        self._Distance = Distance
        self._VectorParams = VectorParams
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams  # pylint: disable=import-error

        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(self, doc_id: str, chunk_index: int, vec: np.ndarray) -> None:
        from qdrant_client.models import PointStruct  # pylint: disable=import-error

        point_id = abs(hash(f"{doc_id}:{chunk_index}")) % (2**63)
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vec.tolist(),
                    payload={"doc_id": doc_id, "chunk_index": chunk_index},
                )
            ],
        )

    def delete_doc(self, doc_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # pylint: disable=import-error

        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
            ),
        )

    def search(
        self,
        query_vec: np.ndarray,
        limit: int = 10,
        allowed_docs: set[str] | None = None,
    ) -> list[VHit]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny  # pylint: disable=import-error

        query_filter = None
        if allowed_docs is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=list(allowed_docs)),
                    )
                ]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vec.tolist(),
            limit=limit,
            query_filter=query_filter,
        )
        return [
            VHit(
                doc_id=r.payload["doc_id"],
                chunk_index=r.payload["chunk_index"],
                score=r.score,
            )
            for r in results
        ]

    def close(self) -> None:
        self._client.close()


# ── ファクトリ ────────────────────────────────────────────────────────────────


def create_vector_store(conn: sqlite3.Connection) -> SqliteVectorStore | QdrantVectorStore:
    """QDRANT_URL 環境変数の有無でバックエンドを自動選択する。"""
    import os

    qdrant_url = os.environ.get("QDRANT_URL")
    if qdrant_url:
        collection = os.environ.get("QDRANT_COLLECTION", QDRANT_COLLECTION)
        return QdrantVectorStore(url=qdrant_url, collection=collection)
    return SqliteVectorStore(conn)


# 後方互換エイリアス（既存コードへの影響を防ぐ）
VectorStore = SqliteVectorStore

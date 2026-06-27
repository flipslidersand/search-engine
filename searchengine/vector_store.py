"""ベクトル索引（設計書 §2.6）。

ベクトルは Index と同じ SQLite DB の `vectors` テーブルに BLOB 永続化する。
検索は numpy ブルートフォース（コサイン類似）。コーパスが大きくなれば
hnswlib / FAISS(IVF/PQ) に差し替える（設計書 §6）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

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


class VectorStore:
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

    def _load(self, allowed_docs: set[str] | None = None):
        rows = self.conn.execute("SELECT doc_id, chunk_index, vec FROM vectors").fetchall()
        keys, mats = [], []
        for doc_id, ci, blob in rows:
            if allowed_docs is not None and doc_id not in allowed_docs:
                continue
            keys.append((doc_id, ci))
            mats.append(np.frombuffer(blob, dtype=np.float32))
        if not mats:
            return [], None
        return keys, np.vstack(mats)

    def search(
        self, query_vec: np.ndarray, limit: int = 10, allowed_docs: set[str] | None = None
    ) -> list[VHit]:
        keys, mat = self._load(allowed_docs)
        if mat is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        # ベクトルは格納時に正規化済み前提。コサイン = 内積。
        sims = mat @ q
        order = np.argsort(-sims)[:limit]
        return [VHit(keys[i][0], keys[i][1], float(sims[i])) for i in order]

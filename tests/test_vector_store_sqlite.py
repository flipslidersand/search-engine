"""SqliteVectorStore のユニットテスト (#47)。"""

from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest

from searchengine.vector_store import (
    SqliteVectorStore,
    VectorStore,
    VectorStoreProtocol,
    create_vector_store,
)


@pytest.fixture()
def store():
    conn = sqlite3.connect(":memory:")
    s = SqliteVectorStore(conn)
    yield s
    conn.close()


def _vec(dim: int = 4, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ── Protocol 準拠 ─────────────────────────────────────────────────────────────


def test_protocol_conformance(store):
    assert isinstance(store, VectorStoreProtocol)


def test_alias():
    assert VectorStore is SqliteVectorStore


# ── upsert / search ───────────────────────────────────────────────────────────


def test_upsert_and_search(store):
    v = _vec(seed=1)
    store.upsert("doc1", 0, v)
    store.conn.commit()

    hits = store.search(v, limit=5)
    assert len(hits) == 1
    assert hits[0].doc_id == "doc1"
    assert hits[0].chunk_index == 0
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_top_k(store):
    for i in range(5):
        store.upsert("doc", i, _vec(seed=i))
    store.conn.commit()

    hits = store.search(_vec(seed=0), limit=3)
    assert len(hits) == 3


def test_search_empty(store):
    hits = store.search(_vec(), limit=5)
    assert hits == []


def test_upsert_overwrite(store):
    v1 = _vec(seed=1)
    v2 = _vec(seed=2)
    store.upsert("doc1", 0, v1)
    store.upsert("doc1", 0, v2)
    store.conn.commit()

    hits = store.search(v2, limit=1)
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


# ── delete_doc ────────────────────────────────────────────────────────────────


def test_delete_doc(store):
    store.upsert("doc1", 0, _vec(seed=1))
    store.upsert("doc2", 0, _vec(seed=2))
    store.conn.commit()

    store.delete_doc("doc1")
    store.conn.commit()

    hits = store.search(_vec(seed=1), limit=10)
    doc_ids = {h.doc_id for h in hits}
    assert "doc1" not in doc_ids
    assert "doc2" in doc_ids


# ── allowed_docs フィルタ ─────────────────────────────────────────────────────


def test_allowed_docs_filter(store):
    store.upsert("docA", 0, _vec(seed=1))
    store.upsert("docB", 0, _vec(seed=1))
    store.conn.commit()

    hits = store.search(_vec(seed=1), limit=10, allowed_docs={"docA"})
    assert all(h.doc_id == "docA" for h in hits)


# ── ファクトリ ────────────────────────────────────────────────────────────────


def test_factory_returns_sqlite_without_qdrant_url():
    os.environ.pop("QDRANT_URL", None)
    conn = sqlite3.connect(":memory:")
    store = create_vector_store(conn)
    assert isinstance(store, SqliteVectorStore)
    conn.close()

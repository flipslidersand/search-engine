"""QdrantVectorStore のモックテスト (#48)。"""

from __future__ import annotations

import os
import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import searchengine.vector_store as vs_mod
from searchengine.vector_store import (
    QdrantVectorStore,
    VectorStoreProtocol,
    create_vector_store,
)


def _vec(dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_client_mock(existing_collections: list[str] | None = None):
    """QdrantClient のモックを返す。"""
    client = MagicMock()

    col_mock = MagicMock()
    col_mock.name = "search-engine-docs"
    collections_result = MagicMock()
    col_mocks = []
    for n in (existing_collections or []):
        m = MagicMock()
        m.name = n
        col_mocks.append(m)
    collections_result.collections = col_mocks
    client.get_collections.return_value = collections_result

    hit = MagicMock()
    hit.payload = {"doc_id": "doc1", "chunk_index": 0}
    hit.score = 0.95
    client.search.return_value = [hit]

    return client


from contextlib import contextmanager


@contextmanager
def _qdrant_patches(client_mock):
    """qdrant-client 未インストール環境でも QdrantVectorStore をテストできるよう全シンボルをパッチ。"""
    mm = MagicMock
    with patch.multiple(
        "searchengine.vector_store",
        _HAS_QDRANT=True,
        QdrantClient=MagicMock(return_value=client_mock),
        VectorParams=mm(),
        Distance=mm(),
        Filter=mm(side_effect=lambda must: {"must": must}),
        FieldCondition=mm(side_effect=lambda key, match: {"key": key, "match": match}),
        MatchAny=mm(side_effect=lambda any: {"any": any}),
        MatchValue=mm(side_effect=lambda value: {"value": value}),
        PointStruct=mm(side_effect=lambda id, vector, payload: MagicMock(id=id, vector=vector, payload=payload)),
    ):
        yield


@pytest.fixture()
def qdrant_store():
    client_mock = _make_client_mock()
    with _qdrant_patches(client_mock):
        store = QdrantVectorStore(url="http://localhost:6333")
        store._client = client_mock
        yield store


# ── Protocol 準拠 ─────────────────────────────────────────────────────────────


def test_protocol_conformance(qdrant_store):
    assert isinstance(qdrant_store, VectorStoreProtocol)


# ── コレクション自動作成 ──────────────────────────────────────────────────────


def _make_store(existing_collections=None):
    client_mock = _make_client_mock(existing_collections=existing_collections or [])
    with _qdrant_patches(client_mock):
        store = QdrantVectorStore(url="http://localhost:6333")
    store._client = client_mock
    return store, client_mock


def test_collection_created_when_missing():
    _, client_mock = _make_store(existing_collections=[])
    client_mock.create_collection.assert_called_once()


def test_collection_not_recreated_when_exists():
    _, client_mock = _make_store(existing_collections=["search-engine-docs"])
    client_mock.create_collection.assert_not_called()


# ── upsert ────────────────────────────────────────────────────────────────────


def test_upsert_calls_qdrant(qdrant_store):
    qdrant_store.upsert("doc1", 0, _vec())
    qdrant_store._client.upsert.assert_called_once()
    call_kwargs = qdrant_store._client.upsert.call_args[1]
    assert call_kwargs["collection_name"] == "search-engine-docs"
    points = call_kwargs["points"]
    assert len(points) == 1
    assert points[0].payload["doc_id"] == "doc1"
    assert points[0].payload["chunk_index"] == 0


# ── delete_doc ────────────────────────────────────────────────────────────────


def test_delete_doc_calls_qdrant(qdrant_store):
    qdrant_store.delete_doc("doc1")
    qdrant_store._client.delete.assert_called_once()
    call_kwargs = qdrant_store._client.delete.call_args[1]
    assert call_kwargs["collection_name"] == "search-engine-docs"


# ── search ────────────────────────────────────────────────────────────────────


def test_search_returns_vhits(qdrant_store):
    hits = qdrant_store.search(_vec(), limit=5)
    assert len(hits) == 1
    assert hits[0].doc_id == "doc1"
    assert hits[0].chunk_index == 0
    assert hits[0].score == pytest.approx(0.95)


def test_search_with_allowed_docs(qdrant_store):
    qdrant_store.search(_vec(), limit=5, allowed_docs={"doc1", "doc2"})
    call_kwargs = qdrant_store._client.search.call_args[1]
    assert call_kwargs["query_filter"] is not None


def test_search_without_filter(qdrant_store):
    qdrant_store.search(_vec(), limit=5, allowed_docs=None)
    call_kwargs = qdrant_store._client.search.call_args[1]
    assert call_kwargs["query_filter"] is None


# ── ファクトリ ────────────────────────────────────────────────────────────────


def test_factory_returns_qdrant_with_url():
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    try:
        client_mock = _make_client_mock()
        conn = sqlite3.connect(":memory:")
        with _qdrant_patches(client_mock):
            store = create_vector_store(conn)
        assert isinstance(store, QdrantVectorStore)
        conn.close()
    finally:
        del os.environ["QDRANT_URL"]


# ── ImportError ───────────────────────────────────────────────────────────────


def test_import_error_without_qdrant_client():
    with patch("searchengine.vector_store._HAS_QDRANT", False):
        with pytest.raises(ImportError, match="pip install qdrant-client"):
            QdrantVectorStore(url="http://localhost:6333")

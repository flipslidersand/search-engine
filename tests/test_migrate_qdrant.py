"""migrate_to_qdrant.py のテスト (#49)。"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.migrate_to_qdrant import _load_sqlite_vectors, migrate


def _make_test_db(tmp_path: Path, n: int = 5, dim: int = 384) -> str:
    db = str(tmp_path / "test.db")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE vectors (doc_id TEXT, chunk_index INTEGER, vec BLOB,"
        " PRIMARY KEY(doc_id, chunk_index))"
    )
    rng = np.random.default_rng(0)
    for i in range(n):
        v = rng.random(dim).astype(np.float32)
        conn.execute("INSERT INTO vectors VALUES (?, ?, ?)", (f"doc{i}", 0, v.tobytes()))
    conn.commit()
    conn.close()
    return db


# ── _load_sqlite_vectors ──────────────────────────────────────────────────────


def test_load_returns_correct_count(tmp_path):
    db = _make_test_db(tmp_path, n=7)
    records = _load_sqlite_vectors(db)
    assert len(records) == 7


def test_load_returns_correct_shape(tmp_path):
    db = _make_test_db(tmp_path, n=3, dim=384)
    records = _load_sqlite_vectors(db)
    for doc_id, chunk_index, vec in records:
        assert isinstance(doc_id, str)
        assert isinstance(chunk_index, int)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32


def test_load_empty_db(tmp_path):
    db = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db)
    conn.close()
    records = _load_sqlite_vectors(db)
    assert records == []


# ── migrate / dry-run ─────────────────────────────────────────────────────────


def test_dry_run_returns_count_without_writing(tmp_path):
    db = _make_test_db(tmp_path, n=5)

    with patch("scripts.migrate_to_qdrant.QdrantVectorStore") as mock_store_cls:
        count = migrate(
            db_path=db,
            qdrant_url="http://localhost:6333",
            collection="test-col",
            dry_run=True,
            batch_size=10,
        )

    assert count == 5
    mock_store_cls.assert_not_called()


def test_migrate_calls_upsert(tmp_path):
    db = _make_test_db(tmp_path, n=3)
    mock_store = MagicMock()

    with patch("scripts.migrate_to_qdrant.QdrantVectorStore", return_value=mock_store):
        count = migrate(
            db_path=db,
            qdrant_url="http://localhost:6333",
            collection="test-col",
            dry_run=False,
            batch_size=10,
        )

    assert count == 3
    assert mock_store.upsert.call_count == 3
    mock_store.close.assert_called_once()


def test_migrate_empty_db_returns_zero(tmp_path):
    db = str(tmp_path / "empty.db")
    conn = sqlite3.connect(db)
    conn.close()

    count = migrate(
        db_path=db,
        qdrant_url="http://localhost:6333",
        collection="test-col",
        dry_run=False,
        batch_size=10,
    )
    assert count == 0


def test_migrate_batch_size_respected(tmp_path):
    db = _make_test_db(tmp_path, n=10)
    mock_store = MagicMock()

    with patch("scripts.migrate_to_qdrant.QdrantVectorStore", return_value=mock_store):
        migrate(
            db_path=db,
            qdrant_url="http://localhost:6333",
            collection="test-col",
            dry_run=False,
            batch_size=3,
        )

    assert mock_store.upsert.call_count == 10

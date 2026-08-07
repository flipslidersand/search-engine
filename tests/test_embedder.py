"""Embedder / RemoteEmbedder のテスト (#21)。

RemoteEmbedder の HTTP 呼び出しは unittest.mock.patch で差し替える。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from searchengine.embedder import (
    Embedder,
    EmbedderProtocol,
    RemoteEmbedder,
    create_embedder,
)

_TEST_URL = "http://localhost:9092"
_TEST_API_KEY = "test-key-123"


# ── ヘルパー ──────────────────────────────────────────────────────────────────


def _make_resp(vector: list[float], dim: int = 4) -> MagicMock:
    """httpx.post のモックレスポンス。"""
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "vector": vector,
        "dim": dim,
        "model": "intfloat/multilingual-e5-base",
        "collection": "search-engine",
        "mode": "index",
    }
    return m


def _unit_vec(dim: int = 4, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float64)
    v = v / np.linalg.norm(v)
    return v.tolist()


@contextmanager
def _remote_embedder(dim: int = 4, api_key: str | None = None):
    """httpx.post をモックした RemoteEmbedder を生成するコンテキスト。"""
    vec = _unit_vec(dim)
    resp = _make_resp(vec, dim=dim)
    with patch("httpx.post", return_value=resp):
        embedder = RemoteEmbedder(
            base_url=_TEST_URL,
            collection="search-engine",
            api_key=api_key,
        )
    yield embedder, vec


# ── EmbedderProtocol 準拠 ─────────────────────────────────────────────────────


def test_local_embedder_protocol_conformance():
    e = Embedder()
    assert isinstance(e, EmbedderProtocol)


def test_remote_embedder_protocol_conformance():
    vec = _unit_vec()
    resp = _make_resp(vec)
    with patch("httpx.post", return_value=resp):
        e = RemoteEmbedder(_TEST_URL)
    assert isinstance(e, EmbedderProtocol)


# ── Embedder (ローカル) ───────────────────────────────────────────────────────


def test_local_embedder_returns_ndarray():
    e = Embedder()
    vecs = e.encode(["hello", "world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, e.dim)
    assert vecs.dtype == np.float32


def test_local_embedder_mode_ignored():
    e = Embedder()
    v_idx = e.encode(["test"], mode="index")
    v_srch = e.encode(["test"], mode="search")
    np.testing.assert_array_equal(v_idx, v_srch)


def test_local_embedder_backend_is_string():
    e = Embedder()
    assert isinstance(e.backend, str)
    assert len(e.backend) > 0


# ── RemoteEmbedder ────────────────────────────────────────────────────────────


def test_remote_embedder_dim_from_service():
    vec = _unit_vec(768)
    resp = _make_resp(vec, dim=768)
    with patch("httpx.post", return_value=resp):
        e = RemoteEmbedder(_TEST_URL)
    assert e.dim == 768


def test_remote_embedder_encode_single():
    with _remote_embedder(dim=4) as (e, expected_vec):
        vec = _unit_vec(4)
        resp = _make_resp(vec)
        with patch("httpx.post", return_value=resp):
            result = e.encode(["hello world"])

    assert result.shape == (1, 4)
    assert result.dtype == np.float32


def test_remote_embedder_encode_batch():
    vec = _unit_vec(4)
    resp = _make_resp(vec)
    with patch("httpx.post", return_value=resp) as mock_post:
        e_resp = _make_resp(vec)
        mock_post.return_value = e_resp

        with patch("httpx.post", return_value=_make_resp(_unit_vec(4))) as _:
            pass

        with patch("httpx.post", return_value=_make_resp(vec)):
            e = RemoteEmbedder(_TEST_URL, collection="search-engine")

        with patch("httpx.post", return_value=resp) as mock_encode:
            result = e.encode(["text1", "text2", "text3"])

    assert result.shape == (3, 4)


def test_remote_embedder_passes_mode_to_service():
    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)) as mock_init:
        e = RemoteEmbedder(_TEST_URL, collection="search-engine")

    with patch("httpx.post", return_value=_make_resp(vec)) as mock_post:
        e.encode(["query text"], mode="search")

    payload = mock_post.call_args[1]["json"]
    assert payload["mode"] == "search"


def test_remote_embedder_passes_api_key_header():
    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)):
        e = RemoteEmbedder(_TEST_URL, api_key=_TEST_API_KEY)

    with patch("httpx.post", return_value=_make_resp(vec)) as mock_post:
        e.encode(["text"])

    headers = mock_post.call_args[1]["headers"]
    assert headers.get("X-API-Key") == _TEST_API_KEY


def test_remote_embedder_no_api_key_no_header():
    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)):
        e = RemoteEmbedder(_TEST_URL, api_key=None)

    with patch("httpx.post", return_value=_make_resp(vec)) as mock_post:
        e.encode(["text"])

    headers = mock_post.call_args[1].get("headers", {})
    assert "X-API-Key" not in headers


def test_remote_embedder_http_error_propagates():
    import httpx

    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)):
        e = RemoteEmbedder(_TEST_URL)

    with patch("httpx.post", side_effect=httpx.ConnectError("接続失敗")):
        with pytest.raises(httpx.ConnectError):
            e.encode(["text"])


def test_remote_embedder_backend_contains_url():
    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)):
        e = RemoteEmbedder(_TEST_URL, collection="search-engine")
    assert "localhost" in e.backend
    assert "search-engine" in e.backend


# ── create_embedder ───────────────────────────────────────────────────────────


def test_create_embedder_returns_local_without_url():
    env = {k: v for k, v in os.environ.items() if k != "EMBEDDING_URL"}
    env.pop("EMBEDDING_URL", None)
    with patch.dict(os.environ, env, clear=True):
        e = create_embedder()
    assert isinstance(e, Embedder)


def test_create_embedder_returns_remote_with_url():
    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)):
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_URL": _TEST_URL,
                "EMBEDDING_COLLECTION": "search-engine",
                "EMBEDDING_API_KEY": _TEST_API_KEY,
            },
        ):
            e = create_embedder()
    assert isinstance(e, RemoteEmbedder)
    assert "localhost" in e.backend


def test_create_embedder_passes_collection_from_env():
    vec = _unit_vec(4)
    with patch("httpx.post", return_value=_make_resp(vec)):
        with patch.dict(
            os.environ,
            {
                "EMBEDDING_URL": _TEST_URL,
                "EMBEDDING_COLLECTION": "custom-col",
            },
            clear=False,
        ):
            e = create_embedder()
    assert isinstance(e, RemoteEmbedder)
    assert "custom-col" in e.backend

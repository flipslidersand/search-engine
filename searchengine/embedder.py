"""文ベクトル化（設計書 §2.6）。

## 実装

- `Embedder`         — ローカル (sentence-transformers or ハッシュフォールバック)
- `RemoteEmbedder`   — HTTP 経由で MINIPC embedding-svc を呼び出す (Phase 3)
- `EmbedderProtocol` — duck-typing 用 Protocol
- `create_embedder()` — 環境変数で Embedder を選択するファクトリ

## 切り替え

`EMBEDDING_URL` が設定されている場合は `RemoteEmbedder`、
未設定の場合は `Embedder`（ローカル）を使う。
"""

from __future__ import annotations

import hashlib
import os
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from . import tokenizer


def _stable_hash(token: str) -> int:
    """プロセス間で安定なハッシュ（組込 hash() はシード変動するため不可）。"""
    return int.from_bytes(
        hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big"
    )


_FALLBACK_DIM = 256
_DEFAULT_MODEL = "cl-nagoya/ruri-base"

try:  # pragma: no cover - 環境依存
    from sentence_transformers import (
        SentenceTransformer,
    )  # pylint: disable=import-error

    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class EmbedderProtocol(Protocol):
    dim: int

    @property
    def backend(self) -> str: ...

    def encode(
        self,
        texts: list[str],
        mode: Literal["index", "search"] = "index",
    ) -> np.ndarray: ...


# ── ローカル Embedder ──────────────────────────────────────────────────────────


class Embedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = None
        self._backend = "fallback(hashing)"
        self.dim = _FALLBACK_DIM
        if _HAS_ST:
            try:
                self._model = SentenceTransformer(model_name)
                self.dim = int(self._model.get_sentence_embedding_dimension())
                self._backend = f"sentence-transformers:{model_name}"
            except Exception:
                self._model = None  # ダウンロード失敗等 → フォールバック

    @property
    def backend(self) -> str:
        return self._backend

    def encode(
        self,
        texts: list[str],
        mode: Literal["index", "search"] = "index",
    ) -> np.ndarray:
        if self._model is not None:  # pragma: no cover - 環境依存
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._hash_vec(t) for t in texts])

    def _hash_vec(self, text: str) -> np.ndarray:
        """特徴ハッシュ（符号トリック付き）→ L2正規化。"""
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenizer.tokenize(text):
            h = _stable_hash(tok)
            idx = (h >> 1) % self.dim
            sign = 1.0 if (h & 1) else -1.0
            v[idx] += sign
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v


# ── RemoteEmbedder ────────────────────────────────────────────────────────────

_DEFAULT_COLLECTION = "search-engine"
_DEFAULT_TIMEOUT = 30.0


class _BatchUnsupported(Exception):
    """embedding-svc がバッチエンドポイント (/embed/batch) 非対応を示す内部シグナル。"""


class RemoteEmbedder:
    """MINIPC embedding-svc (port 9092) を呼び出す HTTP Embedder。

    バッチ優先 (#66):

    - `POST /embed/batch {"collection", "texts": [...], "mode"}`
      → `{"vectors": [[...], ...], "dim": int, "model": str}`
      を最初に試み、1リクエストで全チャンクをベクトル化する。
    - サービスが未対応 (404/405 または "vectors" 欠落) の場合は
      `POST /embed {"collection", "text", "mode"}` の直列送信にフォールバックし、
      以降はバッチを再試行しない（`_batch_supported = False` をキャッシュ）。
    """

    def __init__(
        self,
        base_url: str,
        *,
        collection: str = _DEFAULT_COLLECTION,
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        import httpx  # pylint: disable=import-error

        self._base_url = base_url.rstrip("/")
        self._collection = collection
        self._timeout = timeout
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["X-API-Key"] = api_key

        self._backend = f"remote:{base_url}:{collection}"
        # None=未判定 / True=バッチ対応 / False=非対応（直列送信）
        self._batch_supported: bool | None = None
        self.dim: int = self._discover_dim(httpx)

    def _discover_dim(self, httpx_mod) -> int:
        """サービスに trial encode を送って dim を取得する。"""
        try:
            resp = httpx_mod.post(
                f"{self._base_url}/embed",
                json={"collection": self._collection, "text": "test", "mode": "index"},
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            return int(resp.json()["dim"])
        except Exception:
            return 768  # e5-base デフォルト

    @property
    def backend(self) -> str:
        return self._backend

    def encode(
        self,
        texts: list[str],
        mode: Literal["index", "search"] = "index",
        *,
        retries: int = 2,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # バッチ対応が判明していない / 対応済みならバッチを試みる (#66)
        if self._batch_supported is not False:
            try:
                vecs = self._encode_batch(texts, mode, retries=retries)
                self._batch_supported = True
                return np.asarray(vecs, dtype=np.float32)
            except _BatchUnsupported:
                self._batch_supported = False  # 以降は直列送信

        return self._encode_serial(texts, mode, retries=retries)

    def _request_with_retry(self, url: str, payload: dict, retries: int):
        """指定 URL へ POST し、transient 失敗はリトライする。最終例外は呼び出し側へ送出。"""
        import httpx  # pylint: disable=import-error
        import logging
        import time

        logger = logging.getLogger(__name__)
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = httpx.post(
                    url, json=payload, headers=self._headers, timeout=self._timeout
                )
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as e:
                # 4xx はクライアント側エラーでリトライしても直らない → 即送出
                if e.response is not None and 400 <= e.response.status_code < 500:
                    raise
                last_exc = e
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_exc = e
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        logger.warning(
            "RemoteEmbedder request failed after %d retries: %s", retries, last_exc
        )
        assert last_exc is not None
        raise last_exc

    def _encode_batch(
        self, texts: list[str], mode: str, *, retries: int
    ) -> list[list[float]]:
        """/embed/batch を1リクエストで叩く。非対応なら `_BatchUnsupported` を送出。"""
        import httpx  # pylint: disable=import-error

        try:
            resp = self._request_with_retry(
                f"{self._base_url}/embed/batch",
                {"collection": self._collection, "texts": texts, "mode": mode},
                retries,
            )
        except httpx.HTTPStatusError as e:
            if e.response is not None and e.response.status_code in (404, 405):
                raise _BatchUnsupported() from e
            raise

        data = resp.json()
        vectors = data.get("vectors")
        if vectors is None:
            raise _BatchUnsupported()  # 旧サービス（単一 /embed のみ）
        if len(vectors) != len(texts):
            raise ValueError(
                f"/embed/batch returned {len(vectors)} vectors for {len(texts)} texts"
            )
        return vectors

    def _encode_serial(
        self, texts: list[str], mode: str, *, retries: int
    ) -> np.ndarray:
        """1テキスト1リクエストの直列送信（バッチ非対応サービス向けフォールバック）。"""
        vecs = []
        for text in texts:
            resp = self._request_with_retry(
                f"{self._base_url}/embed",
                {"collection": self._collection, "text": text, "mode": mode},
                retries,
            )
            vecs.append(resp.json()["vector"])
        return np.asarray(vecs, dtype=np.float32)


# ── FallbackEmbedder ──────────────────────────────────────────────────────────


class FallbackEmbedder:
    """Remote → LocalST → FeatureHash の順でフォールバックする Embedder。

    EMBEDDING_URL が設定されている場合にのみ生成される。
    Remote が失敗した場合はローカル Embedder にフォールバックして処理を継続する。
    """

    def __init__(self, remote: RemoteEmbedder, local: Embedder) -> None:
        import logging

        self._remote = remote
        self._local = local
        self._using_remote = True
        self._logger = logging.getLogger(__name__)
        # dim は remote 優先
        self.dim = remote.dim

    @property
    def backend(self) -> str:
        active = "remote" if self._using_remote else "local"
        return f"fallback({active}):remote={self._remote.backend},local={self._local.backend}"

    def encode(
        self,
        texts: list[str],
        mode: Literal["index", "search"] = "index",
    ) -> np.ndarray:
        if self._using_remote:
            try:
                return self._remote.encode(texts, mode)
            except Exception as e:
                self._logger.warning(
                    "RemoteEmbedder unavailable (%s), falling back to local embedder", e
                )
                self._using_remote = False
        return self._local.encode(texts, mode)


# ── ファクトリ ────────────────────────────────────────────────────────────────


def create_embedder() -> FallbackEmbedder | Embedder:
    """環境変数に応じて Embedder を選択する。

    EMBEDDING_URL が設定されている場合は FallbackEmbedder（Remote → Local フォールバック）、
    未設定の場合は ローカル Embedder を返す。
    """
    url = os.environ.get("EMBEDDING_URL", "").strip()
    if url:
        remote = RemoteEmbedder(
            base_url=url,
            collection=os.environ.get("EMBEDDING_COLLECTION", _DEFAULT_COLLECTION),
            api_key=os.environ.get("EMBEDDING_API_KEY"),
        )
        local = Embedder()
        return FallbackEmbedder(remote, local)
    return Embedder()

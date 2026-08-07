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
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


_FALLBACK_DIM = 256
_DEFAULT_MODEL = "cl-nagoya/ruri-base"

try:  # pragma: no cover - 環境依存
    from sentence_transformers import SentenceTransformer  # pylint: disable=import-error

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


class RemoteEmbedder:
    """MINIPC embedding-svc (port 9092) を呼び出す HTTP Embedder。

    POST /embed  {"collection": ..., "text": ..., "mode": "index"|"search"}
    → {"vector": [...], "dim": int, "model": str}
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
    ) -> np.ndarray:
        import httpx  # pylint: disable=import-error

        vecs = []
        for text in texts:
            resp = httpx.post(
                f"{self._base_url}/embed",
                json={"collection": self._collection, "text": text, "mode": mode},
                headers=self._headers,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            vecs.append(resp.json()["vector"])
        return np.asarray(vecs, dtype=np.float32)


# ── ファクトリ ────────────────────────────────────────────────────────────────


def create_embedder() -> Embedder | RemoteEmbedder:
    """環境変数に応じて Embedder を選択する。

    EMBEDDING_URL が設定されている場合は RemoteEmbedder、
    未設定の場合は ローカル Embedder を返す。
    """
    url = os.environ.get("EMBEDDING_URL", "").strip()
    if url:
        return RemoteEmbedder(
            base_url=url,
            collection=os.environ.get("EMBEDDING_COLLECTION", _DEFAULT_COLLECTION),
            api_key=os.environ.get("EMBEDDING_API_KEY"),
        )
    return Embedder()

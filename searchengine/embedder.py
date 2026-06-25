"""文ベクトル化（設計書 §2.6）。

sentence-transformers があれば日本語埋め込みモデルを使う。
無ければ numpy による「特徴ハッシュ + 形態素トークン」フォールバックで動作する。
フォールバックは語彙重複ベースの近似であり、真の意味的類似ではない点に注意
（Phase 2 のパイプライン検証用プレースホルダ。実運用ではモデルを導入する）。
"""

from __future__ import annotations

import hashlib

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

    def encode(self, texts: list[str]) -> np.ndarray:
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

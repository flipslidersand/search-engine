"""日本語トークナイザ。

形態素解析（fugashi）を主とし、未導入環境では
正規表現分割 + CJK文字 bigram のフォールバックで動作する。
設計書 §2.3 のハイブリッド方針（形態素 + N-gram 補完）に対応。
"""

from __future__ import annotations

import re
import unicodedata

# fugashi があれば使う。無ければフォールバック。
try:
    import fugashi  # type: ignore  # pylint: disable=import-error

    _TAGGER = fugashi.Tagger()
    _HAS_FUGASHI = True
except Exception:  # pragma: no cover - 環境依存
    _TAGGER = None
    _HAS_FUGASHI = False

# CJK（漢字・ひらがな・カタカナ）判定用
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")
# ラテン語・数字の語
_WORD = re.compile(r"[A-Za-z0-9_]+")

# ごく軽量なストップワード（設計書 §2.3）
_STOPWORDS = {"の", "は", "が", "を", "に", "へ", "と", "で", "も", "や", "た", "し"}


def normalize(text: str) -> str:
    """NFKC正規化 + 改行・連続空白の整理（設計書 §2.1）。"""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _char_bigrams(token: str) -> list[str]:
    return [token[i : i + 2] for i in range(len(token) - 1)]


def _fallback_tokens(text: str) -> list[str]:
    """fugashi 非依存の分割: ラテン語はそのまま、CJK連は bigram 化。"""
    tokens: list[str] = []
    for chunk in re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]+", text):
        if _WORD.fullmatch(chunk):
            tokens.append(chunk.lower())
        elif _CJK.search(chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
            else:
                tokens.extend(_char_bigrams(chunk))
    return tokens


def _mecab_tokens(text: str) -> list[str]:
    """形態素 + 隣接 bigram（複合語・未知語対策, 設計書 §2.3）。"""
    words = [w.surface for w in _TAGGER(text) if w.surface.strip()]
    bigrams = [words[i] + words[i + 1] for i in range(len(words) - 1)]
    return words + bigrams


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """テキストをトークン列に変換。順序保持の重複排除あり。"""
    text = normalize(text)
    if not text:
        return []
    raw = _mecab_tokens(text) if _HAS_FUGASHI else _fallback_tokens(text)
    if drop_stopwords:
        raw = [t for t in raw if t not in _STOPWORDS]
    return list(dict.fromkeys(raw))  # 順序保持 dedup


def tokenized_text(text: str) -> str:
    """FTS5 投入用に空白区切り文字列へ（設計書 §2.4）。"""
    return " ".join(tokenize(text))


def backend() -> str:
    return "fugashi" if _HAS_FUGASHI else "fallback(regex+bigram)"

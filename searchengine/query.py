"""最小クエリパーサー（設計書 §2.9）。

ユーザー入力を FTS5 の MATCH 構文へ変換する。
対応構文:
  - フレーズ検索  : "機械学習 モデル"
  - ブーリアン    : AND / OR / NOT（大文字小文字非依存）
  - フィールド検索: type:code  path:sample （検索前にフィルタとして抽出）
  - それ以外の語は形態素分割して AND 結合
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import tokenizer

_PHRASE = re.compile(r'"([^"]+)"')
_BOOL = {"and": "AND", "or": "OR", "not": "NOT"}
# フィールド検索: フレーズ外の `key:value`（value はクォート可）
_FIELD = re.compile(r'(?<!\S)(type|path):("[^"]+"|\S+)')
_FIELDS = {"type", "path"}


@dataclass
class ParsedQuery:
    fts: str  # FTS5 MATCH 文字列
    raw: str = ""  # フィールド除去後の自然文（ベクトル検索用）
    filters: dict[str, str] = field(default_factory=dict)


def parse(user_query: str) -> ParsedQuery:
    """フィールドフィルタを抽出し、残りを FTS5 クエリ + 自然文へ変換。"""
    filters: dict[str, str] = {}

    def _grab(m: re.Match) -> str:
        filters[m.group(1)] = m.group(2).strip('"')
        return " "

    remaining = _FIELD.sub(_grab, user_query)
    raw = remaining.replace('"', " ")
    for op in (" AND ", " OR ", " NOT ", " and ", " or ", " not "):
        raw = raw.replace(op, " ")
    raw = " ".join(raw.split())
    return ParsedQuery(fts=to_fts_query(remaining), raw=raw, filters=filters)


def _quote_terms(text: str) -> list[str]:
    """生テキストを形態素分割し、各語を FTS5 用にクォート。"""
    out = []
    for tok in tokenizer.tokenize(text):
        # FTS5 の特殊文字を避けるため二重引用符で包む
        out.append('"' + tok.replace('"', "") + '"')
    return out


def to_fts_query(user_query: str) -> str:
    """ユーザークエリ → FTS5 MATCH 文字列。"""
    parts: list[str] = []
    pos = 0
    for m in _PHRASE.finditer(user_query):
        # フレーズ前の通常部分を処理
        before = user_query[pos : m.start()]
        parts.extend(_emit_tokens(before))
        # フレーズはトークン化して NEAR 的に連結（順序保持の "a b" 形式）
        phrase_tokens = tokenizer.tokenize(m.group(1))
        if phrase_tokens:
            parts.append('"' + " ".join(phrase_tokens) + '"')
        pos = m.end()
    parts.extend(_emit_tokens(user_query[pos:]))

    # 空なら全ヒット回避のため非マッチ語を返す
    if not [p for p in parts if p not in ("AND", "OR", "NOT")]:
        return '""'
    return _join(parts)


def _emit_tokens(text: str) -> list[str]:
    result: list[str] = []
    for word in text.split():
        low = word.lower()
        if low in _BOOL:
            result.append(_BOOL[low])
        else:
            result.extend(_quote_terms(word))
    return result


def _join(parts: list[str]) -> str:
    """演算子が無い隣接語は暗黙 AND で連結。"""
    out: list[str] = []
    for i, p in enumerate(parts):
        if i > 0 and p not in ("AND", "OR", "NOT") and out[-1] not in ("AND", "OR", "NOT"):
            out.append("AND")
        out.append(p)
    return " ".join(out)

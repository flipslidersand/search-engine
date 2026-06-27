"""ハイブリッド検索 = キーワード(BM25) × ベクトル を RRF で統合（設計書 §2.7）。

Reciprocal Rank Fusion は順位のみを使うため、BM25 とコサイン類似という
スケールの異なるスコアを重み調整なしで統合できる。
"""

from __future__ import annotations

from dataclasses import dataclass

from .index import Hit, Index
from .query import ParsedQuery


@dataclass
class Fused:
    hit: Hit
    rrf: float


def _key(h: Hit) -> tuple[str, int]:
    return (h.doc_id, h.chunk_index)


def rrf_merge(keyword: list[Hit], vector: list[Hit], k: int = 60, limit: int = 10) -> list[Fused]:
    """両ランキングを RRF スコアで統合して降順に返す。"""
    scores: dict[tuple[str, int], float] = {}
    hits: dict[tuple[str, int], Hit] = {}
    for ranking in (keyword, vector):
        for rank, h in enumerate(ranking):
            key = _key(h)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            hits.setdefault(key, h)  # 表示は先に出た方（通常キーワード）を採用
    fused = [Fused(hit=hits[kk], rrf=sc) for kk, sc in scores.items()]
    fused.sort(key=lambda f: f.rrf, reverse=True)
    return fused[:limit]


def search(index: Index, parsed: ParsedQuery, limit: int = 10, pool: int = 50) -> list[Fused]:
    """ハイブリッド検索の入口。pool は各手法から集める候補数。"""
    keyword = index.search(parsed.fts, limit=pool, filters=parsed.filters)
    vector: list[Hit] = []
    if index.vectors is not None and parsed.raw:
        vector = index.vector_search(parsed.raw, limit=pool, filters=parsed.filters)
    return rrf_merge(keyword, vector, limit=limit)

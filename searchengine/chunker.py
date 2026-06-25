"""チャンキング（設計書 §2.2）。

長文の局所内容を検索可能にするため、文書をチャンク単位に分割する。
段落・見出し境界を優先し、無ければトークン数で分割。
オーバーラップを設けて境界での文脈断絶を防ぐ。

注: トークン長は簡易に「空白区切り語数」で近似する（Phase 1 MVP）。
本番では tokenizer の出力長やモデルの tokenizer に合わせる。
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SIZE = 512
DEFAULT_OVERLAP = 64


@dataclass
class Chunk:
    index: int
    text: str


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_text(text: str, size: int = DEFAULT_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """段落優先 + トークン数フォールバックで分割。"""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append(Chunk(index=len(chunks), text=" ".join(buf)))

    for para in _split_paragraphs(text) or [text]:
        words = para.split()
        if not words:
            continue
        # 段落がチャンクサイズに収まるならそのまま積む
        if len(buf) + len(words) <= size:
            buf.extend(words)
            continue
        # バッファを確定し、長い段落はトークン窓でスライド分割
        flush()
        buf = []
        if len(words) <= size:
            buf = words
            continue
        step = size - overlap
        for start in range(0, len(words), step):
            window = words[start : start + size]
            if not window:
                break
            chunks.append(Chunk(index=len(chunks), text=" ".join(window)))
            if start + size >= len(words):
                break
    flush()

    # 全体が空ならフォールバックで1チャンク
    if not chunks and text.strip():
        chunks.append(Chunk(index=0, text=text.strip()))
    return chunks

"""RAG パイプライン: 検索 → プロンプト組み立て → Ollama 生成。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

from . import hybrid, query
from .index import Index
from .llm import LLMResult, Message, OllamaClient

_DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

_SYSTEM_PROMPT = """\
あなたは検索エンジンと連携した AI アシスタントです。
提供された参考文書のみを根拠に質問に答えてください。
文書に情報がない場合は「提供された文書には情報が見つかりませんでした」と答えてください。
"""

_USER_TEMPLATE = """\
[参考文書]
{context}

[質問]
{question}
"""


@dataclass
class Source:
    path: str
    chunk_index: int
    snippet: str
    score: float


@dataclass
class AskResult:
    answer: str
    sources: list[Source]
    model: str
    latency_ms: int


def ask(
    index: Index,
    question: str,
    *,
    mode: Literal["keyword", "vector", "hybrid"] = "hybrid",
    top_k: int = 5,
    ollama_url: str = _DEFAULT_OLLAMA_URL,
    model: str = _DEFAULT_MODEL,
) -> AskResult:
    """RAG パイプラインのエントリポイント。

    1. hybrid / keyword / vector 検索で上位 top_k チャンクを取得
    2. チャンクを結合してプロンプトを組み立て
    3. Ollama に送信して回答を生成
    """
    t0 = time.monotonic()

    parsed = query.parse(question)

    if mode == "keyword":
        hits = index.search(parsed.fts, limit=top_k, filters=parsed.filters)
        sources = [Source(path=h.path, chunk_index=h.chunk_index, snippet=h.snippet, score=h.score) for h in hits]
    elif mode == "vector":
        hits = index.vector_search(parsed.raw, limit=top_k, filters=parsed.filters)
        sources = [Source(path=h.path, chunk_index=h.chunk_index, snippet=h.snippet, score=h.score) for h in hits]
    else:
        fused = hybrid.search(index, parsed, limit=top_k)
        sources = [Source(path=f.hit.path, chunk_index=f.hit.chunk_index, snippet=f.hit.snippet, score=f.rrf) for f in fused]

    context = "\n\n---\n\n".join(
        f"[{i+1}] {s.path}\n{s.snippet}" for i, s in enumerate(sources)
    )

    client = OllamaClient(base_url=ollama_url, model=model)
    result: LLMResult = client.chat([
        Message(role="system", content=_SYSTEM_PROMPT),
        Message(role="user", content=_USER_TEMPLATE.format(context=context, question=question)),
    ])

    latency_ms = int((time.monotonic() - t0) * 1000)
    return AskResult(
        answer=result.content,
        sources=sources,
        model=result.model,
        latency_ms=latency_ms,
    )

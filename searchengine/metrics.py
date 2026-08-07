"""Prometheus メトリクス定義（設計書 Phase 6）。

メトリクス一覧:
  search_requests_total   Counter   mode, status   検索リクエスト数
  search_latency_seconds  Histogram mode           検索レイテンシ
  index_requests_total    Counter   status         /index リクエスト数
  index_documents_total   Gauge     —              インデックス文書数
  index_chunks_total      Gauge     —              インデックスチャンク数
  rag_requests_total      Counter   status         /ask リクエスト数
  rag_latency_seconds     Histogram —              RAG レイテンシ
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import (  # pylint: disable=import-error
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

if _HAS_PROMETHEUS:
    SEARCH_REQUESTS = Counter(
        "search_requests_total",
        "Total search requests",
        ["mode", "status"],
    )
    SEARCH_LATENCY = Histogram(
        "search_latency_seconds",
        "Search request latency",
        ["mode"],
        buckets=_LATENCY_BUCKETS,
    )
    INDEX_REQUESTS = Counter(
        "index_requests_total",
        "Total /index requests",
        ["status"],
    )
    INDEX_DOCUMENTS = Gauge(
        "index_documents_total",
        "Number of indexed documents",
    )
    INDEX_CHUNKS = Gauge(
        "index_chunks_total",
        "Number of indexed chunks",
    )
    RAG_REQUESTS = Counter(
        "rag_requests_total",
        "Total /ask requests",
        ["status"],
    )
    RAG_LATENCY = Histogram(
        "rag_latency_seconds",
        "RAG (/ask) request latency",
        buckets=_LATENCY_BUCKETS,
    )


@contextmanager
def track_search(mode: str) -> Generator[None, None, None]:
    if not _HAS_PROMETHEUS:
        yield
        return
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        SEARCH_REQUESTS.labels(mode=mode, status=status).inc()
        SEARCH_LATENCY.labels(mode=mode).observe(elapsed)


@contextmanager
def track_index() -> Generator[None, None, None]:
    if not _HAS_PROMETHEUS:
        yield
        return
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        INDEX_REQUESTS.labels(status=status).inc()


@contextmanager
def track_rag() -> Generator[None, None, None]:
    if not _HAS_PROMETHEUS:
        yield
        return
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed = time.perf_counter() - start
        RAG_REQUESTS.labels(status=status).inc()
        RAG_LATENCY.observe(elapsed)


def update_index_gauges(documents: int, chunks: int) -> None:
    if not _HAS_PROMETHEUS:
        return
    INDEX_DOCUMENTS.set(documents)
    INDEX_CHUNKS.set(chunks)


def metrics_output() -> tuple[bytes, str]:
    """Prometheus テキスト形式のメトリクスを返す。"""
    if not _HAS_PROMETHEUS:
        return b"# prometheus-client not installed\n", "text/plain"
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

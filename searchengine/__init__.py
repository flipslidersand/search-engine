"""自作検索エンジン（BM25 + ベクトル + ハイブリッド + Web API）。

Phase 1: SQLite FTS5 + 形態素解析 + BM25
Phase 2: ベクトル検索 + RRF ハイブリッド
Phase 3: FastAPI サーバー + Web UI
"""

__version__ = "0.3.0"

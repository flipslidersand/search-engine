# search-engine

BM25 + ベクトル ハイブリッド検索 + RAG（Retrieval-Augmented Generation）REST API。

SQLite FTS5 によるキーワード検索と [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) によるベクトル検索を RRF（Reciprocal Rank Fusion）で統合し、Ollama 連携で自然言語 Q&A を提供する。MCP サーバーとして Claude Code から直接呼び出すことも可能。

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│  Client (curl / Claude Code MCP / Browser)              │
└──────────────┬──────────────────────────────────────────┘
               │ HTTP
┌──────────────▼──────────────────────────────────────────┐
│  FastAPI (server.py)                                    │
│  POST /index   GET /search   POST /ask   GET /metrics   │
└──┬────────────┬────────────────────┬────────────────────┘
   │            │                    │
   ▼            ▼                    ▼
ingest.py    index.py            rag.py
              │  ├─ FTS5 (BM25)   │  ├─ ask()
              │  ├─ vector_store  │  └─ OllamaClient
              │  └─ hybrid (RRF)  │
              ▼                   ▼
          embedder.py         llm.py
          ├─ RemoteEmbedder ──▶ MINIPC :9092 (e5-base dim=768)
          └─ Embedder (fallback: ST / hash)

Storage: SQLite (FTS5 + BLOB vectors) / Qdrant (optional)
```

---

## 実装済み機能

| Phase | 機能                                      | 実装                                              |
| ----- | ----------------------------------------- | ------------------------------------------------- |
| 1     | BM25 キーワード検索 (FTS5)                | `index.py`, `query.py`                            |
| 2     | ベクトル検索 + RRF ハイブリッド           | `embedder.py`, `hybrid.py`                        |
| 3     | REST API サーバー                         | `server.py` (FastAPI)                             |
| 4     | RAG — `/ask` エンドポイント + Ollama 連携 | `rag.py`, `llm.py`                                |
| 5     | Qdrant バックエンド + 移行スクリプト      | `vector_store.py`, `scripts/migrate_to_qdrant.py` |
| 6     | RemoteEmbedder (MINIPC e5-base)           | `embedder.py`                                     |
| 7     | MCP サーバー (Claude Code 連携)           | `mcp_server.py`                                   |
| 8     | Prometheus メトリクス `/metrics`          | `metrics.py`                                      |

---

## クイックスタート

### 依存インストール

```bash
pip install -r requirements.txt
```

### インデックス構築

```bash
# キーワードのみ（numpy だけで動作）
python3 -m searchengine.cli index ./sample_docs --db search.db

# ベクトル索引も同時構築（EMBEDDING_URL 推奨）
EMBEDDING_URL=http://192.168.68.63:9092 \
  python3 -m searchengine.cli index ./sample_docs --db search.db --vector
```

### 検索 (CLI)

```bash
# キーワード検索（BM25）
python3 -m searchengine.cli search "RAG 検索" --db search.db

# ハイブリッド検索（BM25 + ベクトル + RRF）
python3 -m searchengine.cli search "意味的類似" --db search.db --mode hybrid

# フィールド検索 + ブーリアン
python3 -m searchengine.cli search '"ベクトル検索" AND BM25 type:markdown'
```

### サーバー起動

```bash
cp .env.example .env   # 設定を編集
uvicorn searchengine.server:app --reload --port 8000
```

---

## API リファレンス

全エンドポイントは `X-API-Key` ヘッダーで認証（`API_KEY` 未設定時はスキップ）。

### `POST /index` — ドキュメントをインデックス

```bash
curl -X POST http://localhost:8000/index \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/docs", "vector": true}'
```

**レスポンス**: `{"indexed": 5, "documents": 12, "chunks": 84, "tokenizer": "...", "embedder": "..."}`

パストラバーサル対策: `ALLOWED_INDEX_DIRS` に許可ディレクトリを `:` 区切りで設定すること（必須）。

### `GET /search` — 検索

```bash
curl "http://localhost:8000/search?q=RAG+検索&mode=hybrid&n=5" \
  -H "X-API-Key: $API_KEY"
```

| パラメータ | 型                      | デフォルト | 説明       |
| ---------- | ----------------------- | ---------- | ---------- |
| `q`        | string                  | 必須       | 検索クエリ |
| `mode`     | keyword\|vector\|hybrid | `keyword`  | 検索モード |
| `n`        | int (1-100)             | `10`       | 取得件数   |

### `POST /ask` — RAG 回答生成

```bash
curl -X POST http://localhost:8000/ask \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "BM25 と RRF の違いは？", "mode": "hybrid", "top_k": 5}'
```

**レスポンス**: `{"answer": "...", "sources": [...], "model": "qwen2.5:7b", "latency_ms": 1234}`

### `GET /metrics` — Prometheus メトリクス

```bash
curl http://localhost:8000/metrics
```

認証不要。Prometheus テキスト形式で `search_requests_total`、`search_latency_seconds`、`rag_latency_seconds` などを返す。

---

## 環境変数

| 変数                   | デフォルト                      | 説明                                         |
| ---------------------- | ------------------------------- | -------------------------------------------- |
| `SEARCH_DB`            | `search.db`                     | SQLite DB パス                               |
| `API_KEY`              | (未設定=認証スキップ)           | `X-API-Key` 認証キー                         |
| `ALLOWED_INDEX_DIRS`   | (必須)                          | `/index` が許可するディレクトリ（`:`区切り） |
| `OLLAMA_URL`           | `http://localhost:11434`        | Ollama サーバーURL                           |
| `OLLAMA_MODEL`         | `qwen2.5:7b`                    | RAG で使用するモデル                         |
| `EMBEDDING_URL`        | (未設定=ローカルフォールバック) | RemoteEmbedder エンドポイント                |
| `EMBEDDING_COLLECTION` | `search-engine`                 | embedding-svc のコレクション名               |
| `EMBEDDING_API_KEY`    | (空)                            | embedding-svc の API キー                    |
| `QDRANT_URL`           | (未設定=SQLite)                 | Qdrant サーバーURL                           |
| `QDRANT_COLLECTION`    | `search-engine-docs`            | Qdrant コレクション名                        |
| `QDRANT_VECTOR_SIZE`   | `768`                           | ベクトル次元数（e5-base=768）                |

---

## MCP サーバー（Claude Code 連携）

Claude Code から直接 `search` / `ask` / `index` / `stats` ツールを呼び出せる。

セットアップ手順は [docs/mcp-setup.md](docs/mcp-setup.md) を参照。

```json
{
  "mcpServers": {
    "search-engine": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "searchengine.mcp_server"],
      "cwd": "/path/to/search-engine",
      "env": {
        "SEARCH_DB": "/path/to/search.db",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

---

## Docker

```bash
cp .env.example .env   # DEPLOY_HOST 等を設定
docker compose up -d
```

Qdrant も同時起動する場合:

```yaml
# docker-compose.yml の qdrant サービスを有効化して
docker compose --profile qdrant up -d
```

---

## 開発セットアップ

```bash
# テスト実行
python3 -m pytest tests/ -q

# 精度向上（任意）
pip install "fugashi[unidic-lite]"    # MeCab 形態素解析
pip install sentence-transformers     # ローカルベクトル化

# Qdrant ベクトルストア（任意）
pip install qdrant-client

# SQLite → Qdrant 移行
python3 scripts/migrate_to_qdrant.py \
  --db search.db \
  --qdrant-url http://localhost:6333
```

---

## テスト

```bash
python3 -m pytest tests/ -q
# 159 passed, 3 skipped
```

| テストファイル                | 対象                       | テスト数 |
| ----------------------------- | -------------------------- | -------- |
| `test_search.py`              | FTS5・BM25・クエリパーサー | 34       |
| `test_server.py`              | FastAPI エンドポイント     | 47       |
| `test_rag.py`                 | RAG パイプライン + /ask    | 10       |
| `test_vector_store_sqlite.py` | SqliteVectorStore          | 9        |
| `test_vector_store_qdrant.py` | QdrantVectorStore (モック) | 11       |
| `test_migrate_qdrant.py`      | 移行スクリプト             | 7        |
| `test_embedder.py`            | Embedder / RemoteEmbedder  | 16       |
| `test_metrics.py`             | Prometheus メトリクス      | 9        |
| `test_mcp_server.py`          | MCP サーバー               | 11       |

---

## ロードマップ

- [ ] #28 README 詳細化 ← **現在**
- [ ] RemoteEmbedder バッチ API 対応（現在はテキスト1件ずつ）
- [ ] Grafana ダッシュボード設定ファイル
- [ ] 評価指標 (Precision@K / MRR / nDCG) 計測スクリプト
- [ ] GitHub Actions CI パイプライン

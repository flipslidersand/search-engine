# search-engine

自作検索エンジン。設計書は [docs/search-engine-design.md](../../docs/search-engine-design.md)。

## Phase 1 + 2 — 実装済み

SQLite FTS5 + 形態素解析 + チャンキング + BM25 + クエリパーサーに加え、
**ベクトル検索 + RRF ハイブリッド + フィールド検索**。
**stdlib + numpy のみで動作**（fugashi / sentence-transformers が無くてもフォールバックで動く）。

| モジュール        | 役割                                                 | 設計書   |
| ----------------- | ---------------------------------------------------- | -------- |
| `tokenizer.py`    | 形態素解析（fugashi）/ フォールバック分割            | §2.3     |
| `chunker.py`      | 段落優先チャンキング + オーバーラップ                | §2.2     |
| `index.py`        | FTS5・インクリメンタル更新・BM25・ベクトル・フィルタ | §2.4–2.8 |
| `embedder.py`     | 文ベクトル化（ST）/ 特徴ハッシュ・フォールバック     | §2.6     |
| `vector_store.py` | ベクトル永続化（SQLite BLOB）+ numpy コサイン検索    | §2.6     |
| `hybrid.py`       | RRF によるキーワード × ベクトル統合                  | §2.7     |
| `query.py`        | フレーズ / AND・OR・NOT / フィールド検索             | §2.9     |
| `ingest.py`       | Markdown / テキスト / コード取込                     | §2.1     |
| `cli.py`          | index / search / stats コマンド                      | §5       |

## 使い方

```bash
cd ACTIVE/search-engine

# インデックス作成（--vector でベクトル索引も同時構築）
python3 -m searchengine.cli index sample_docs --db /tmp/search.db --vector

# キーワード検索（暗黙 AND）
python3 -m searchengine.cli search "機械学習 モデル" --db /tmp/search.db

# ベクトル検索（意味的類似）
python3 -m searchengine.cli search "ニューラルネットワーク 画像" --db /tmp/search.db --mode vector

# ハイブリッド検索（RRF 統合）
python3 -m searchengine.cli search "ハイブリッド検索" --db /tmp/search.db --mode hybrid

# フィールド検索 + フレーズ + ブーリアン
python3 -m searchengine.cli search '"ベクトル検索" OR BM25 type:markdown' --db /tmp/search.db

# 統計
python3 -m searchengine.cli stats --db /tmp/search.db
```

## テスト

```bash
python3 tests/test_search.py        # stdlib + numpy のみ
# または pytest tests/
```

## 精度を上げる（任意）

```bash
pip install "fugashi[unidic-lite]"      # 形態素解析を有効化
pip install sentence-transformers       # 真の意味的ベクトル検索を有効化
```

導入すると `tokenizer.backend()` / `embedder.backend` がフォールバックから
実モデルへ切り替わる（コード変更不要）。

## 次のフェーズ

- **Phase 3**: 自動分類 + 重要語抽出 + クエリ拡張 + 評価指標（Precision@K / MRR / nDCG）
- **Phase 4**: グラフ可視化 + 時系列追跡
- **Phase 5**: 逆引き（引用関係グラフ）

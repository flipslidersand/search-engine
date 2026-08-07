# MCP サーバー セットアップガイド（Phase 4）

search-engine を Claude Code の MCP サーバーとして登録し、
`search` / `ask` / `index` / `stats` ツールを直接呼び出せるようにする。

## 前提条件

- Python 3.12+
- `pip install -r requirements.txt` 済み
- SQLite DB にドキュメントがインデックス済み

## 1. `.claude/settings.json` に登録

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
        "OLLAMA_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "qwen2.5:7b"
      }
    }
  }
}
```

### 環境変数

| 変数           | デフォルト               | 説明                  |
| -------------- | ------------------------ | --------------------- |
| `SEARCH_DB`    | `search.db`              | SQLite DB パス        |
| `OLLAMA_URL`   | `http://localhost:11434` | Ollama エンドポイント |
| `OLLAMA_MODEL` | `qwen2.5:7b`             | RAG で使用するモデル  |

## 2. Claude Code を再起動

MCP サーバーは起動時にのみ読み込まれる。設定変更後は Claude Code を再起動すること。

## 3. 利用可能なツール

### `search` — ドキュメント検索

```
search(query="Python 非同期処理", mode="hybrid", top_k=5)
```

| パラメータ | 型                      | 必須 | 説明                             |
| ---------- | ----------------------- | ---- | -------------------------------- |
| `query`    | string                  | ✓    | 検索クエリ                       |
| `mode`     | keyword\|vector\|hybrid | —    | 検索モード（デフォルト: hybrid） |
| `top_k`    | int                     | —    | 取得件数（デフォルト: 5）        |
| `db`       | string                  | —    | DB パス（省略時は SEARCH_DB）    |

### `ask` — RAG クエリ

```
ask(question="Rustのライフタイムとは？", mode="hybrid")
```

| パラメータ | 型                      | 必須 | 説明            |
| ---------- | ----------------------- | ---- | --------------- |
| `question` | string                  | ✓    | 質問文          |
| `mode`     | keyword\|vector\|hybrid | —    | 検索モード      |
| `model`    | string                  | —    | Ollama モデル名 |
| `top_k`    | int                     | —    | 参照チャンク数  |
| `db`       | string                  | —    | DB パス         |

### `index` — ドキュメントをインデックス

```
index(path="/home/user/docs", vector=false)
```

| パラメータ | 型     | 必須 | 説明                                              |
| ---------- | ------ | ---- | ------------------------------------------------- |
| `path`     | string | ✓    | ファイルまたはディレクトリ                        |
| `vector`   | bool   | —    | ベクトル埋め込みも生成するか（デフォルト: false） |
| `db`       | string | —    | DB パス                                           |

対応拡張子: `.md` `.txt` `.py` `.js` `.ts` `.json` `.rs`

### `stats` — DB 統計

```
stats()
```

文書数・チャンク数・DB パスを返す。

## 4. permissions 設定（必要な場合）

Claude Code が確認ダイアログを出す場合は `.claude/settings.json` に追加:

```json
{
  "permissions": {
    "allow": [
      "mcp__search-engine__search",
      "mcp__search-engine__ask",
      "mcp__search-engine__index",
      "mcp__search-engine__stats"
    ]
  }
}
```

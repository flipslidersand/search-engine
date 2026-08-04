#!/usr/bin/env python3
"""SQLite ベクトルストア → Qdrant 移行スクリプト。

使い方:
    python scripts/migrate_to_qdrant.py --db search.db --qdrant-url http://192.168.68.63:6333
    python scripts/migrate_to_qdrant.py --db search.db --dry-run   # 書き込みなし確認
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

# searchengine パッケージを参照できるようにプロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from searchengine.vector_store import (
    QDRANT_COLLECTION,
    QDRANT_VECTOR_SIZE,
    QdrantVectorStore,
)


def _load_sqlite_vectors(db_path: str) -> list[tuple[str, int, np.ndarray]]:
    """SQLite の vectors テーブルから全チャンクを読み込む。"""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT doc_id, chunk_index, vec FROM vectors").fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    result = []
    for doc_id, chunk_index, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32).copy()
        result.append((doc_id, chunk_index, vec))
    return result


def migrate(
    db_path: str,
    qdrant_url: str,
    collection: str,
    dry_run: bool,
    batch_size: int,
) -> int:
    records = _load_sqlite_vectors(db_path)
    total = len(records)

    if total == 0:
        print("ベクトルデータが見つかりません。インデックスを --vector 付きで作成しましたか？")
        return 0

    print(f"移行対象: {total} チャンク  →  {qdrant_url}/{collection}")
    if dry_run:
        print("[DRY-RUN] 書き込みはスキップします")
        return total

    store = QdrantVectorStore(url=qdrant_url, collection=collection)
    migrated = 0
    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        for doc_id, chunk_index, vec in batch:
            store.upsert(doc_id, chunk_index, vec)
        migrated += len(batch)
        print(f"  {migrated}/{total} 完了", end="\r", flush=True)
    store.close()

    print(f"\n✅ 移行完了: {migrated} チャンク")
    return migrated


def main() -> None:
    p = argparse.ArgumentParser(description="SQLite → Qdrant ベクトル移行")
    p.add_argument("--db", required=True, help="SQLite DB パス")
    p.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    p.add_argument("--collection", default=QDRANT_COLLECTION, help="コレクション名")
    p.add_argument("--batch-size", type=int, default=100, help="バッチサイズ")
    p.add_argument("--dry-run", action="store_true", help="書き込みなしで件数のみ確認")
    args = p.parse_args()

    if not Path(args.db).exists():
        print(f"エラー: DB が見つかりません: {args.db}", file=sys.stderr)
        sys.exit(1)

    migrate(
        db_path=args.db,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

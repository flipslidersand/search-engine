"""CLI（設計書 §5 Phase 1 + §2.6/2.7 Phase 2）。

使い方:
    python -m searchengine.cli index <パス> [--db search.db] [--vector]
    python -m searchengine.cli search "<クエリ>" [--mode keyword|vector|hybrid]
    python -m searchengine.cli stats [--db search.db]
"""

from __future__ import annotations

import argparse
import os
import sys

from . import hybrid, ingest, query, tokenizer
from .index import Index


def _make_index(args: argparse.Namespace, *, want_vector: bool) -> Index:
    embedder = None
    if want_vector:
        from .embedder import Embedder  # numpy 依存。必要時のみ import

        embedder = Embedder()
    return Index(args.db, embedder=embedder)


def cmd_index(args: argparse.Namespace) -> int:
    idx = _make_index(args, want_vector=args.vector)
    count = 0
    for path in ingest.iter_files(args.target):
        text = ingest.read_text(path)
        idx.add_document(
            path,
            text,
            doc_type=ingest.detect_type(path) or "text",
            mtime=os.path.getmtime(path),
        )
        count += 1
    s = idx.stats()
    vec_info = f" | embedder={idx.embedder.backend}" if idx.embedder else ""
    idx.close()
    print(f"indexed {count} file(s) | tokenizer={tokenizer.backend()}{vec_info}")
    print(f"documents={s['documents']} chunks={s['chunks']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    want_vector = args.mode in ("vector", "hybrid")
    idx = _make_index(args, want_vector=want_vector)
    parsed = query.parse(args.query)

    if args.mode == "keyword":
        rows = [(h, h.score) for h in idx.search(parsed.fts, limit=args.n, filters=parsed.filters)]
    elif args.mode == "vector":
        rows = [
            (h, h.score)
            for h in idx.vector_search(parsed.raw, limit=args.n, filters=parsed.filters)
        ]
    else:  # hybrid (RRF)
        rows = [(f.hit, f.rrf) for f in hybrid.search(idx, parsed, limit=args.n)]
    idx.close()

    meta = f"mode={args.mode} fts={parsed.fts!r}"
    if parsed.filters:
        meta += f" filters={parsed.filters}"
    if not rows:
        print(f"(no results) {meta}")
        return 0
    print(meta)
    for i, (h, score) in enumerate(rows, 1):
        rel = os.path.relpath(h.path)
        print(f"{i:2}. [{score:8.4f}] {rel}  (chunk {h.chunk_index})")
        print(f"      {h.snippet}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    idx = Index(args.db)
    s = idx.stats()
    idx.close()
    print(f"documents={s['documents']} chunks={s['chunks']} tokenizer={tokenizer.backend()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # --db はグローバル/サブコマンド両方の位置で受けられるよう親パーサに置く
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", default="search.db", help="SQLite DB path")

    p = argparse.ArgumentParser(prog="searchengine", parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", parents=[common], help="ファイル/ディレクトリをインデックス")
    pi.add_argument("target")
    pi.add_argument("--vector", action="store_true", help="ベクトル索引も構築")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", parents=[common], help="検索")
    ps.add_argument("query")
    ps.add_argument("-n", type=int, default=10, help="件数")
    ps.add_argument(
        "--mode",
        choices=["keyword", "vector", "hybrid"],
        default="keyword",
        help="検索方式（hybrid は RRF 統合）",
    )
    ps.set_defaults(func=cmd_search)

    pst = sub.add_parser("stats", parents=[common], help="統計")
    pst.set_defaults(func=cmd_stats)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

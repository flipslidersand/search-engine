#!/usr/bin/env python3
"""
schema-gen CLI — CSV/Excel 構造解析 → 要件定義・DBスキーマ生成

使い方:
  python -m searchengine.schema_gen.cli <file> [options]

オプション:
  --sheet <name>   Excel シート名（省略時はアクティブシート）
  --table <name>   SQL テーブル名（省略時はファイル名から推定）
  --out <dir>      出力ディレクトリ（省略時は標準出力）
  --format md|sql|both  出力形式（デフォルト: both）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from searchengine.schema_gen.ingest import load_file
from searchengine.schema_gen.analyzer import analyze
from searchengine.schema_gen.reporter import report_markdown, report_sql


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    filepath = args[0]
    sheet = args[args.index("--sheet") + 1] if "--sheet" in args else None
    out_dir = Path(args[args.index("--out") + 1]) if "--out" in args else None
    fmt = args[args.index("--format") + 1] if "--format" in args else "both"

    stem = Path(filepath).stem
    table_name = args[args.index("--table") + 1] if "--table" in args else stem.replace(" ", "_").replace("-", "_")

    rows, filename = load_file(filepath, sheet)
    if not rows:
        print("ERROR: データが空です", file=sys.stderr)
        sys.exit(1)

    columns = analyze(rows)

    md = report_markdown(filename, columns, len(rows)) if fmt in ("md", "both") else None
    sql = report_sql(table_name, columns) if fmt in ("sql", "both") else None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        if md:
            (out_dir / f"{stem}-report.md").write_text(md, encoding="utf-8")
            print(f"  → {out_dir}/{stem}-report.md")
        if sql:
            (out_dir / f"{stem}-schema.sql").write_text(sql, encoding="utf-8")
            print(f"  → {out_dir}/{stem}-schema.sql")
    else:
        if md:
            print(md)
        if sql:
            print("\n---\n")
            print(sql)


if __name__ == "__main__":
    main()

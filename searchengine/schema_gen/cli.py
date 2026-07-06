#!/usr/bin/env python3
"""
schema-gen CLI — CSV/Excel/Google Sheets 構造解析 → 要件定義・DBスキーマ生成

使い方:
  python -m searchengine.schema_gen.cli <file>          # CSV/Excel
  python -m searchengine.schema_gen.cli --sheets <url>  # Google Sheets

オプション:
  --sheet <name>        Excel/Sheets シート名（省略時は最初のシート）
  --sheets <url|id>     Google Sheets URL またはスプレッドシート ID
  --credentials <path>  認証ファイル（service_account.json or client_secret.json）
  --table <name>        SQL テーブル名（省略時はファイル名から推定）
  --out <dir>           出力ディレクトリ（省略時は標準出力）
  --format md|sql|both  出力形式（デフォルト: both）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from searchengine.schema_gen.ingest import load_file, load_sheets
from searchengine.schema_gen.analyzer import analyze
from searchengine.schema_gen.reporter import report_markdown, report_sql


def _get(args: list[str], flag: str, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return

    sheet = _get(args, "--sheet")
    out_dir = Path(_get(args, "--out")) if _get(args, "--out") else None
    fmt = _get(args, "--format", "both")
    sheets_url = _get(args, "--sheets")
    credentials = _get(args, "--credentials")

    if sheets_url:
        rows, source_id = load_sheets(sheets_url, sheet, credentials)
        stem = source_id[:20]
        filename = f"sheets:{source_id[:12]}…"
    else:
        if not args or args[0].startswith("--"):
            print("ERROR: ファイルパスまたは --sheets を指定してください", file=sys.stderr)
            sys.exit(1)
        filepath = args[0]
        rows, filename = load_file(filepath, sheet)
        stem = Path(filepath).stem

    table_name = _get(args, "--table") or stem.replace(" ", "_").replace("-", "_")

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

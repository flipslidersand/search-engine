"""解析結果を Markdown / SQL / ER図 で出力する。"""
from __future__ import annotations
from .analyzer import ColumnInfo
from .er import detect_fk_candidates, detect_normalization_issues, render_mermaid, render_sql_with_fk


def report_markdown(filename: str, columns: list[ColumnInfo], row_count: int) -> str:
    lines = [
        f"# 構造解析レポート: `{filename}`",
        "",
        f"- 行数: {row_count:,}",
        f"- カラム数: {len(columns)}",
        "",
        "## カラム一覧",
        "",
        "| カラム名 | 推定型 | NULL率 | 一意率 | PK候補 | サンプル値 |",
        "|---|---|---|---|---|---|",
    ]

    for c in columns:
        pk = "✅" if c.pk_candidate else ""
        samples = ", ".join(c.sample_values[:3])
        lines.append(
            f"| `{c.name}` | {c.inferred_type} | {c.null_pct}% | {c.unique_pct}% | {pk} | {samples} |"
        )

    pk_candidates = [c for c in columns if c.pk_candidate]
    high_null = [c for c in columns if c.null_pct >= 30]

    lines += ["", "## 所見"]
    if pk_candidates:
        names = ", ".join(f"`{c.name}`" for c in pk_candidates)
        lines.append(f"- **主キー候補**: {names}（NULL なし・完全一意）")
    if high_null:
        names = ", ".join(f"`{c.name}` ({c.null_pct}%)" for c in high_null)
        lines.append(f"- **NULL 率 30% 超**: {names} — 任意項目として扱うか要確認")

    nullable_cols = [c for c in columns if c.null_pct > 0 and not c.pk_candidate]
    if nullable_cols:
        lines.append(f"- **任意項目（NULL あり）**: {len(nullable_cols)} カラム")

    lines += [
        "",
        "## 要件定義たたき台",
        "",
        "### データの目的・用途",
        "> （クライアントへのヒアリング内容を記入）",
        "",
        "### 主な操作",
        "- [ ] 登録",
        "- [ ] 検索 / 絞り込み",
        "- [ ] 更新",
        "- [ ] 削除",
        "- [ ] 一覧表示 / ページング",
        "",
        "### 追加ヒアリング項目",
    ]

    for c in high_null:
        lines.append(f"- `{c.name}` が空のケースはどんな場合か？（NULL率 {c.null_pct}%）")

    if not pk_candidates:
        lines.append("- 主キー（一意識別子）はどのカラムか？現状 NULL なし・完全一意のカラムなし")

    return "\n".join(lines)


def report_er(table_name: str, columns: list[ColumnInfo]) -> str:
    """Mermaid ER図 + 正規化提案を含む Markdown を生成する。"""
    relations = detect_fk_candidates(columns, table_name)
    hints = detect_normalization_issues(columns)

    lines = [
        f"## ER図（Mermaid）",
        "",
        "```mermaid",
        render_mermaid(table_name, columns, relations),
        "```",
    ]

    if relations:
        lines += ["", "### FK 候補（自動検出）", ""]
        for r in relations:
            lines.append(f"- `{r.from_col}` → `{r.ref_table}.{r.ref_col}`（推定）")
        lines.append("")
        lines.append("> ⚠️ FK 先テーブルはファイルから自動推定したスタブです。実際のテーブル名・カラム名を確認してください。")

    if hints:
        lines += ["", "### 正規化提案", ""]
        for h in hints:
            lines.append(f"**{h.issue}**")
            lines.append(f"→ {h.suggestion}")
            lines.append("")

    return "\n".join(lines)


def report_sql(table_name: str, columns: list[ColumnInfo]) -> str:
    pk_candidates = [c for c in columns if c.pk_candidate]
    lines = [f"CREATE TABLE {table_name} ("]
    col_lines = []

    for c in columns:
        null_clause = "NOT NULL" if c.null_pct == 0.0 else "NULL"
        col_lines.append(f"    {_quote(c.name)} {c.inferred_type} {null_clause}")

    if pk_candidates:
        pk_cols = ", ".join(_quote(c.name) for c in pk_candidates[:1])
        col_lines.append(f"    PRIMARY KEY ({pk_cols})")

    lines.append(",\n".join(col_lines))
    lines.append(");")
    return "\n".join(lines)


def _quote(name: str) -> str:
    return f'"{name}"' if not name.isidentifier() or name[0].isdigit() else name

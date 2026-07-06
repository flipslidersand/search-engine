"""ER図・正規化提案モジュール。

単一テーブル解析から:
  1. FK 候補カラムを名前パターンで検出
  2. 繰り返しグループ（非正規化）を検出して正規化提案
  3. Mermaid erDiagram 形式で出力
  4. FK 制約付き SQL CREATE TABLE を生成
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from .analyzer import ColumnInfo


# *_id, *_code, *_no, *_num などを FK 候補とみなすパターン
_FK_SUFFIX_RE = re.compile(
    r"(_id|_code|_no|_num|_key|_ref|id$|コード$|番号$|_cd$)",
    re.IGNORECASE,
)

# カラム名から推定されるリファレンステーブル名
def _guess_ref_table(col_name: str) -> str:
    """order_id → orders, customer_code → customers"""
    name = re.sub(r"(_id|_code|_no|_num|_key|_ref|_cd)$", "", col_name, flags=re.IGNORECASE)
    name = re.sub(r"(コード|番号)$", "", name)
    # snake_case の最後の単語を除く
    parts = name.split("_")
    base = parts[-1] if parts else name
    # 複数形化（簡易）
    if base and not base.endswith("s"):
        base += "s"
    return base or name


@dataclass
class FKRelation:
    from_col: str
    ref_table: str
    ref_col: str = "id"


@dataclass
class NormalizationHint:
    issue: str
    suggestion: str
    columns: list[str] = field(default_factory=list)


def detect_fk_candidates(columns: list[ColumnInfo], table_name: str) -> list[FKRelation]:
    """FK 候補カラムを検出する。主キー単独カラムは除外。"""
    pk_cols = {c.name for c in columns if c.pk_candidate}
    relations = []
    for c in columns:
        if c.name in pk_cols:
            continue
        if _FK_SUFFIX_RE.search(c.name):
            ref_table = _guess_ref_table(c.name)
            if ref_table != table_name:
                relations.append(FKRelation(from_col=c.name, ref_table=ref_table))
    return relations


def detect_normalization_issues(columns: list[ColumnInfo]) -> list[NormalizationHint]:
    """非正規化の兆候を検出して正規化提案を返す。"""
    hints = []

    # パターン1: 連番サフィックス（item_1, item_2, item_3 → 繰り返しグループ）
    groups: dict[str, list[str]] = {}
    for c in columns:
        m = re.match(r"^(.+?)_?(\d+)$", c.name)
        if m:
            base, num = m.group(1), m.group(2)
            groups.setdefault(base, []).append(c.name)

    for base, cols in groups.items():
        if len(cols) >= 2:
            hints.append(NormalizationHint(
                issue=f"繰り返しグループ: `{'`, `'.join(cols)}`",
                suggestion=f"`{base}` を別テーブルに分離し、1対多リレーションに正規化する",
                columns=cols,
            ))

    # パターン2: 高 NULL 率カラムが多い → 別テーブル候補
    high_null = [c for c in columns if c.null_pct >= 50]
    if len(high_null) >= 3:
        names = [c.name for c in high_null]
        hints.append(NormalizationHint(
            issue=f"NULL率 50% 超のカラムが {len(high_null)} 件集中",
            suggestion="オプション属性を別テーブル（EAV or サブタイプ）に分離する検討を推奨",
            columns=names,
        ))

    # パターン3: 同一プレフィックスのカラム群 → 埋め込みオブジェクト候補
    prefix_groups: dict[str, list[str]] = {}
    for c in columns:
        parts = c.name.split("_")
        if len(parts) >= 2:
            prefix_groups.setdefault(parts[0], []).append(c.name)

    for prefix, cols in prefix_groups.items():
        if len(cols) >= 3:
            hints.append(NormalizationHint(
                issue=f"プレフィックス `{prefix}_` が {len(cols)} カラムに集中",
                suggestion=f"`{prefix}` エンティティとして別テーブル化するか、JSON カラムへの集約を検討",
                columns=cols,
            ))

    return hints


def render_mermaid(
    table_name: str,
    columns: list[ColumnInfo],
    relations: list[FKRelation],
) -> str:
    """Mermaid erDiagram 形式で ER 図を生成する。"""
    lines = ["erDiagram"]

    # メインテーブル
    lines.append(f"    {table_name} {{")
    for c in columns:
        col_type = c.inferred_type.split("(")[0]  # VARCHAR(50) → VARCHAR
        pk_marker = " PK" if c.pk_candidate else ""
        fk_marker = " FK" if any(r.from_col == c.name for r in relations) else ""
        nullable = "" if c.null_pct == 0.0 else "?"
        lines.append(f"        {col_type}{nullable} {_mermaid_id(c.name)}{pk_marker}{fk_marker}")
    lines.append("    }")

    # 参照先テーブル（スタブ）＋リレーション
    seen_refs: set[str] = set()
    for r in relations:
        if r.ref_table not in seen_refs:
            seen_refs.add(r.ref_table)
            lines.append(f"    {r.ref_table} {{")
            lines.append(f"        INTEGER id PK")
            lines.append(f"    }}")
        lines.append(
            f"    {r.ref_table} ||--o{{ {table_name} : \"{r.from_col}\""
        )

    return "\n".join(lines)


def render_sql_with_fk(
    table_name: str,
    columns: list[ColumnInfo],
    relations: list[FKRelation],
) -> str:
    """FK 制約付き CREATE TABLE SQL を生成する。"""
    pk_candidates = [c for c in columns if c.pk_candidate]
    fk_cols = {r.from_col for r in relations}

    lines = [f"CREATE TABLE {table_name} ("]
    col_lines = []

    for c in columns:
        null_clause = "NOT NULL" if c.null_pct == 0.0 else "NULL"
        col_lines.append(f"    {_sql_id(c.name)} {c.inferred_type} {null_clause}")

    if pk_candidates:
        pk_col = _sql_id(pk_candidates[0].name)
        col_lines.append(f"    PRIMARY KEY ({pk_col})")

    for r in relations:
        col_lines.append(
            f"    FOREIGN KEY ({_sql_id(r.from_col)}) REFERENCES {r.ref_table} ({r.ref_col})"
        )

    lines.append(",\n".join(col_lines))
    lines.append(");")
    return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    """Mermaid 識別子として使える形式に変換する。"""
    return re.sub(r"[^\w]", "_", name)


def _sql_id(name: str) -> str:
    return f'"{name}"' if not name.isidentifier() or name[0].isdigit() else name

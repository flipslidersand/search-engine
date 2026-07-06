"""カラム構造解析: 型推定・NULL率・重複率・主キー候補を算出する。"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime


_DATE_PATTERNS = [
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}$",
    r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$",
    r"^\d{4}年\d{1,2}月\d{1,2}日$",
]


def _is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() in ("", "NULL", "null", "None", "N/A", "na", "NA", "-"):
        return True
    return False


def _infer_type(values: list) -> str:
    """非 null 値のリストから型を推定する。"""
    non_null = [v for v in values if not _is_null(v)]
    if not non_null:
        return "TEXT"

    samples = [str(v).strip() for v in non_null[:200]]

    # bool
    bool_set = {"true", "false", "yes", "no", "0", "1", "はい", "いいえ"}
    if all(s.lower() in bool_set for s in samples):
        return "BOOLEAN"

    # integer
    def is_int(s):
        return re.fullmatch(r"-?\d{1,15}", s.replace(",", "")) is not None

    if all(is_int(s) for s in samples):
        # 桁数でBIGINTを使い分ける
        max_val = max(abs(int(s.replace(",", ""))) for s in samples)
        return "BIGINT" if max_val > 2_147_483_647 else "INTEGER"

    # float
    def is_float(s):
        try:
            float(s.replace(",", ""))
            return True
        except ValueError:
            return False

    if all(is_float(s) for s in samples):
        return "NUMERIC"

    # date
    if any(re.match(p, s) for s in samples[:20] for p in _DATE_PATTERNS):
        if all(any(re.match(p, s) for p in _DATE_PATTERNS) for s in samples):
            return "DATE"

    # text length → VARCHAR か TEXT か
    lengths = [len(s) for s in samples]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    if max_len <= 255:
        return f"VARCHAR({min(255, max(max_len * 2, 50))})"
    return "TEXT"


@dataclass
class ColumnInfo:
    name: str
    inferred_type: str
    total: int
    null_count: int
    unique_count: int
    sample_values: list = field(default_factory=list)

    @property
    def null_pct(self) -> float:
        return round(self.null_count / self.total * 100, 1) if self.total else 0.0

    @property
    def unique_pct(self) -> float:
        non_null = self.total - self.null_count
        return round(self.unique_count / non_null * 100, 1) if non_null else 0.0

    @property
    def pk_candidate(self) -> bool:
        return self.null_pct == 0.0 and self.unique_pct == 100.0


def analyze(rows: list[dict]) -> list[ColumnInfo]:
    """行リストを解析してカラム情報リストを返す。"""
    if not rows:
        return []

    columns = list(rows[0].keys())
    total = len(rows)
    results = []

    for col in columns:
        values = [row.get(col) for row in rows]
        null_count = sum(1 for v in values if _is_null(v))
        non_null_vals = [v for v in values if not _is_null(v)]
        unique_count = len(set(str(v) for v in non_null_vals))
        inferred = _infer_type(non_null_vals)

        # サンプル値（重複排除・最大5件）
        seen: set = set()
        samples = []
        for v in non_null_vals:
            s = str(v).strip()
            if s not in seen:
                seen.add(s)
                samples.append(s)
            if len(samples) >= 5:
                break

        results.append(ColumnInfo(
            name=col,
            inferred_type=inferred,
            total=total,
            null_count=null_count,
            unique_count=unique_count,
            sample_values=samples,
        ))

    return results

"""schema_gen Phase 1 テスト"""
import csv
import io
import tempfile
from pathlib import Path

import pytest

from searchengine.schema_gen.analyzer import analyze, _infer_type, _is_null
from searchengine.schema_gen.reporter import report_markdown, report_sql
from searchengine.schema_gen.ingest import load_csv


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class TestIsNull:
    def test_none(self):
        assert _is_null(None)

    def test_empty_string(self):
        assert _is_null("")

    def test_null_string(self):
        assert _is_null("NULL")

    def test_na(self):
        assert _is_null("N/A")

    def test_valid_value(self):
        assert not _is_null("hello")
        assert not _is_null(0)
        assert not _is_null("0")


class TestInferType:
    def test_integer(self):
        assert _infer_type(["1", "2", "3"]) == "INTEGER"

    def test_bigint(self):
        assert _infer_type(["9999999999"]) == "BIGINT"

    def test_numeric(self):
        assert _infer_type(["1.5", "2.3"]) == "NUMERIC"

    def test_boolean(self):
        assert _infer_type(["true", "false"]) == "BOOLEAN"

    def test_date(self):
        assert _infer_type(["2024-01-15", "2024-02-01"]) == "DATE"

    def test_varchar(self):
        result = _infer_type(["hello", "world"])
        assert result.startswith("VARCHAR")

    def test_text_long(self):
        long_val = ["x" * 300]
        assert _infer_type(long_val) == "TEXT"

    def test_empty_returns_text(self):
        assert _infer_type([]) == "TEXT"


class TestAnalyze:
    def test_basic(self):
        rows = [
            {"id": "1", "name": "Alice", "age": "30"},
            {"id": "2", "name": "Bob", "age": "25"},
        ]
        cols = analyze(rows)
        assert len(cols) == 3
        id_col = next(c for c in cols if c.name == "id")
        assert id_col.inferred_type == "INTEGER"
        assert id_col.null_pct == 0.0
        assert id_col.pk_candidate

    def test_null_detection(self):
        rows = [
            {"id": "1", "memo": "ok"},
            {"id": "2", "memo": ""},
            {"id": "3", "memo": None},
        ]
        cols = analyze(rows)
        memo = next(c for c in cols if c.name == "memo")
        assert memo.null_count == 2
        assert memo.null_pct == pytest.approx(66.7, abs=0.1)
        assert not memo.pk_candidate

    def test_empty_rows(self):
        assert analyze([]) == []

    def test_sample_values_deduped(self):
        rows = [{"x": str(i % 3)} for i in range(20)]
        cols = analyze(rows)
        assert len(cols[0].sample_values) <= 5


class TestReporters:
    def _cols(self):
        rows = [
            {"customer_id": "1", "name": "Alice", "note": ""},
            {"customer_id": "2", "name": "Bob", "note": "VIP"},
        ]
        return analyze(rows)

    def test_markdown_has_table(self):
        cols = self._cols()
        md = report_markdown("test.csv", cols, 2)
        assert "| カラム名 |" in md
        assert "customer_id" in md
        assert "PK候補" in md

    def test_sql_create_table(self):
        cols = self._cols()
        sql = report_sql("customers", cols)
        assert "CREATE TABLE customers" in sql
        assert "customer_id" in sql
        assert "PRIMARY KEY" in sql

    def test_sql_null_clause(self):
        cols = self._cols()
        sql = report_sql("t", cols)
        assert "NOT NULL" in sql
        assert "NULL" in sql


class TestIngestCSV:
    def test_load_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        rows = [{"a": "1", "b": "hello"}, {"a": "2", "b": "world"}]
        _write_csv(rows, p)
        loaded, name = load_csv(p), "test.csv"
        assert len(loaded) == 2
        assert loaded[0]["a"] == "1"

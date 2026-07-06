"""er.py のテスト"""
import pytest
from searchengine.schema_gen.analyzer import analyze
from searchengine.schema_gen.er import (
    detect_fk_candidates,
    detect_normalization_issues,
    render_mermaid,
    render_sql_with_fk,
    _guess_ref_table,
)


def _make_cols(rows):
    return analyze(rows)


class TestGuessRefTable:
    def test_order_id(self):
        assert _guess_ref_table("order_id") == "orders"

    def test_customer_code(self):
        assert _guess_ref_table("customer_code") == "customers"

    def test_product_no(self):
        assert _guess_ref_table("product_no") == "products"


class TestDetectFKCandidates:
    def test_detects_id_columns(self):
        # customer_id は重複あり（非一意）→ PK 除外 → FK 候補になる
        rows = [
            {"order_id": str(i), "customer_id": str(i % 3 + 1), "total": str(i * 100)}
            for i in range(1, 6)
        ]
        cols = _make_cols(rows)
        fks = detect_fk_candidates(cols, "orders")
        fk_cols = [f.from_col for f in fks]
        assert "customer_id" in fk_cols

    def test_pk_excluded(self):
        rows = [{"id": str(i), "name": f"user{i}"} for i in range(1, 6)]
        cols = _make_cols(rows)
        fks = detect_fk_candidates(cols, "users")
        # id はPKなので除外
        assert not any(f.from_col == "id" for f in fks)

    def test_same_table_excluded(self):
        rows = [{"user_id": str(i), "name": f"u{i}"} for i in range(1, 6)]
        cols = _make_cols(rows)
        fks = detect_fk_candidates(cols, "users")
        # users テーブル自身への参照は除外
        assert not any(f.ref_table == "users" for f in fks)


class TestDetectNormalizationIssues:
    def test_repeating_group(self):
        rows = [{"id": "1", "item_1": "A", "item_2": "B", "item_3": "C"}]
        cols = _make_cols(rows)
        hints = detect_normalization_issues(cols)
        assert any("繰り返し" in h.issue for h in hints)

    def test_high_null_concentration(self):
        rows = [
            {"id": str(i), "a": None, "b": None, "c": None, "d": "x"}
            for i in range(1, 11)
        ]
        cols = _make_cols(rows)
        hints = detect_normalization_issues(cols)
        assert any("NULL率" in h.issue for h in hints)

    def test_prefix_group(self):
        rows = [{"id": "1", "addr_city": "Tokyo", "addr_zip": "100", "addr_pref": "Tokyo"}]
        cols = _make_cols(rows)
        hints = detect_normalization_issues(cols)
        assert any("addr" in h.issue for h in hints)


class TestRenderMermaid:
    def test_contains_erdiagram(self):
        rows = [{"order_id": str(i), "customer_id": str(i * 10)} for i in range(1, 4)]
        cols = _make_cols(rows)
        from searchengine.schema_gen.er import FKRelation
        fks = [FKRelation(from_col="customer_id", ref_table="customers")]
        mermaid = render_mermaid("orders", cols, fks)
        assert "erDiagram" in mermaid
        assert "orders" in mermaid
        assert "customers" in mermaid
        assert "customer_id" in mermaid

    def test_pk_marker(self):
        rows = [{"id": str(i), "name": f"u{i}"} for i in range(1, 6)]
        cols = _make_cols(rows)
        mermaid = render_mermaid("users", cols, [])
        assert "PK" in mermaid


class TestRenderSQLWithFK:
    def test_fk_constraint(self):
        rows = [{"order_id": str(i), "customer_id": str(i * 10)} for i in range(1, 4)]
        cols = _make_cols(rows)
        from searchengine.schema_gen.er import FKRelation
        fks = [FKRelation(from_col="customer_id", ref_table="customers")]
        sql = render_sql_with_fk("orders", cols, fks)
        assert "FOREIGN KEY" in sql
        assert "customers" in sql

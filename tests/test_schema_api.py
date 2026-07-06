"""schema-gen API のテスト（store + FastAPI エンドポイント）"""
import json
import pytest
from pathlib import Path


# ── SchemaStore ───────────────────────────────────────────────────────────────

class TestSchemaStore:
    @pytest.fixture
    def store(self, tmp_path):
        from searchengine.schema_gen.store import SchemaStore
        return SchemaStore(tmp_path / "test.db")

    @pytest.fixture
    def sample_cols(self):
        from searchengine.schema_gen.analyzer import analyze
        rows = [
            {"order_id": str(i), "customer_id": str(i % 3 + 1), "total": str(i * 100)}
            for i in range(1, 6)
        ]
        return analyze(rows)

    def test_save_and_list(self, store, sample_cols):
        store.save("orders", "orders.csv", sample_cols, 5, sql="CREATE TABLE orders ();")
        metas = store.list_schemas()
        assert len(metas) == 1
        assert metas[0].table_name == "orders"
        assert metas[0].col_count == 3

    def test_upsert_replaces(self, store, sample_cols):
        store.save("orders", "old.csv", sample_cols, 5)
        store.save("orders", "new.csv", sample_cols, 10)
        metas = store.list_schemas()
        assert len(metas) == 1
        assert metas[0].filename == "new.csv"
        assert metas[0].row_count == 10

    def test_get_returns_detail(self, store, sample_cols):
        store.save("orders", "orders.csv", sample_cols, 5, sql="SELECT 1")
        detail = store.get("orders")
        assert detail is not None
        assert detail.sql == "SELECT 1"
        assert isinstance(detail.columns, list)
        assert detail.columns[0]["name"] == "order_id"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_search_col_name(self, store, sample_cols):
        store.save("orders", "orders.csv", sample_cols, 5)
        hits = store.search_columns("customer_id", field="col_name")
        assert len(hits) > 0
        assert any(h.col_name == "customer_id" for h in hits)

    def test_search_col_type(self, store, sample_cols):
        store.save("orders", "orders.csv", sample_cols, 5)
        hits = store.search_columns("INTEGER", field="col_type")
        assert len(hits) > 0

    def test_search_table_name(self, store, sample_cols):
        store.save("orders", "orders.csv", sample_cols, 5)
        hits = store.search_columns("orders", field="table_name")
        assert len(hits) > 0
        assert all(h.table_name == "orders" for h in hits)

    def test_search_invalid_field_raises(self, store):
        with pytest.raises(ValueError):
            store.search_columns("x", field="invalid_field")

    def test_delete(self, store, sample_cols):
        store.save("orders", "orders.csv", sample_cols, 5)
        assert store.delete("orders") is True
        assert store.get("orders") is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("nonexistent") is False


# ── FastAPI エンドポイント ───────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEMA_DB", str(tmp_path / "test.db"))
    import importlib
    import searchengine.schema_gen.api as api_mod
    # SCHEMA_DB を再読込
    api_mod._SCHEMA_DB = str(tmp_path / "test.db")

    from fastapi.testclient import TestClient
    from searchengine.server import app
    return TestClient(app)


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text(
        "order_id,customer_id,total\n1,10,1000\n2,20,2000\n3,10,1500\n4,30,500\n5,20,300\n",
        encoding="utf-8",
    )
    return str(csv_path)


class TestAnalyzeEndpoint:
    def test_analyze_csv(self, client, sample_csv):
        res = client.post("/schema/analyze", json={"file_path": sample_csv})
        assert res.status_code == 200
        data = res.json()
        assert data["table_name"] == "orders"
        assert data["row_count"] == 5
        assert len(data["columns"]) == 3

    def test_analyze_saves_by_default(self, client, sample_csv):
        client.post("/schema/analyze", json={"file_path": sample_csv})
        res = client.get("/schema/list")
        assert res.status_code == 200
        assert len(res.json()) == 1

    def test_analyze_no_save(self, client, sample_csv):
        client.post("/schema/analyze", json={"file_path": sample_csv, "save": False})
        res = client.get("/schema/list")
        assert res.json() == []

    def test_analyze_includes_sql(self, client, sample_csv):
        res = client.post("/schema/analyze", json={"file_path": sample_csv, "include_sql": True})
        assert "CREATE TABLE" in res.json()["sql"]

    def test_analyze_missing_file(self, client):
        res = client.post("/schema/analyze", json={"file_path": "/nonexistent/file.csv"})
        assert res.status_code == 400

    def test_analyze_no_source(self, client):
        res = client.post("/schema/analyze", json={})
        assert res.status_code == 400


class TestSearchEndpoint:
    def test_search_returns_hits(self, client, sample_csv):
        client.post("/schema/analyze", json={"file_path": sample_csv})
        res = client.get("/schema/search?q=customer_id&field=col_name")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] > 0
        assert data["results"][0]["col_name"] == "customer_id"

    def test_search_empty_query(self, client):
        res = client.get("/schema/search?q=")
        assert res.status_code == 400

    def test_search_no_results(self, client, sample_csv):
        client.post("/schema/analyze", json={"file_path": sample_csv})
        res = client.get("/schema/search?q=nonexistent_column_xyz")
        assert res.status_code == 200
        assert res.json()["total"] == 0


class TestGetAndDeleteEndpoints:
    def test_get_schema(self, client, sample_csv):
        client.post("/schema/analyze", json={"file_path": sample_csv})
        res = client.get("/schema/orders")
        assert res.status_code == 200
        assert res.json()["table_name"] == "orders"

    def test_get_missing(self, client):
        res = client.get("/schema/nonexistent")
        assert res.status_code == 404

    def test_delete_schema(self, client, sample_csv):
        client.post("/schema/analyze", json={"file_path": sample_csv})
        res = client.delete("/schema/orders")
        assert res.status_code == 200
        assert client.get("/schema/orders").status_code == 404

    def test_delete_missing(self, client):
        res = client.delete("/schema/nonexistent")
        assert res.status_code == 404

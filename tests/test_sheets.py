"""sheets.py のユニットテスト（Google API をモックして実行）"""
from unittest.mock import MagicMock, patch
import pytest

from searchengine.schema_gen.sheets import extract_spreadsheet_id, load_sheet


class TestExtractSpreadsheetId:
    def test_url(self):
        url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms/edit"
        assert extract_spreadsheet_id(url) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"

    def test_bare_id(self):
        sid = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
        assert extract_spreadsheet_id(sid) == sid

    def test_url_with_gid(self):
        url = "https://docs.google.com/spreadsheets/d/ABCD1234/edit#gid=0"
        assert extract_spreadsheet_id(url) == "ABCD1234"


class TestLoadSheet:
    def _mock_service(self, values):
        service = MagicMock()
        (
            service.spreadsheets()
            .get()
            .execute.return_value
        ) = {"sheets": [{"properties": {"title": "Sheet1"}}]}
        (
            service.spreadsheets()
            .values()
            .get()
            .execute.return_value
        ) = {"values": values}
        return service

    @patch("searchengine.schema_gen.sheets._build_service")
    def test_basic(self, mock_build):
        values = [
            ["id", "name", "age"],
            ["1", "Alice", "30"],
            ["2", "Bob", "25"],
        ]
        mock_build.return_value = self._mock_service(values)
        rows = load_sheet("FAKE_ID")
        assert len(rows) == 2
        assert rows[0]["id"] == "1"
        assert rows[0]["name"] == "Alice"

    @patch("searchengine.schema_gen.sheets._build_service")
    def test_short_row_padded(self, mock_build):
        values = [
            ["a", "b", "c"],
            ["1", "2"],       # c が欠損
        ]
        mock_build.return_value = self._mock_service(values)
        rows = load_sheet("FAKE_ID")
        assert rows[0]["c"] is None

    @patch("searchengine.schema_gen.sheets._build_service")
    def test_empty_sheet(self, mock_build):
        svc = self._mock_service([])
        (
            svc.spreadsheets()
            .values()
            .get()
            .execute.return_value
        ) = {"values": []}
        mock_build.return_value = svc
        rows = load_sheet("FAKE_ID")
        assert rows == []

    def test_import_error_raised(self):
        with patch.dict("sys.modules", {
            "googleapiclient": None,
            "googleapiclient.discovery": None,
            "google.oauth2": None,
            "google.auth": None,
        }):
            with pytest.raises((ImportError, Exception)):
                load_sheet("FAKE_ID")

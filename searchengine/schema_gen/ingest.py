"""CSV / Excel ファイルを読み込み、行リスト (list[dict]) を返す。"""
import csv
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_excel(path: Path, sheet: str | None = None) -> list[dict]:
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl が必要です: pip install openpyxl")

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    return [dict(zip(headers, row)) for row in rows[1:]]


def load_sheets(url_or_id: str, sheet: str | None = None, credentials: str | None = None) -> tuple[list[dict], str]:
    """Google Sheets を読み込んで (行リスト, スプレッドシートID) を返す。"""
    from .sheets import load_sheet, extract_spreadsheet_id
    rows = load_sheet(url_or_id, sheet, credentials)
    return rows, extract_spreadsheet_id(url_or_id)


def load_file(path: str | Path, sheet: str | None = None) -> tuple[list[dict], str]:
    """ファイルを読み込んで (行リスト, ファイル名) を返す。"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return load_csv(p), p.name
    if suffix in (".xlsx", ".xls", ".xlsm"):
        return load_excel(p, sheet), p.name
    raise ValueError(f"非対応フォーマット: {suffix} (csv/xlsx のみ)")

"""schema_gen — CSV/Excel 構造解析 → DBスキーマ・要件定義書生成"""
from .ingest import load_file
from .analyzer import analyze
from .reporter import report_markdown, report_sql

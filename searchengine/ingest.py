"""文書取込（設計書 §2.1）。

Phase 1 は Markdown / テキスト / コードのプレーン読み込みを対象とする。
PDF・メールは Phase 1 では拡張ポイントとして関数を用意のみ。
"""

from __future__ import annotations

import os

TEXT_EXTS = {
    ".md": "markdown",
    ".txt": "text",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".json": "code",
    ".rs": "code",
}


def detect_type(path: str) -> str | None:
    return TEXT_EXTS.get(os.path.splitext(path)[1].lower())


def read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def iter_files(root: str):
    """対応拡張子のファイルを再帰列挙。"""
    if os.path.isfile(root):
        if detect_type(root):
            yield root
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # ノイズディレクトリを除外
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__"}]
        for name in filenames:
            full = os.path.join(dirpath, name)
            if detect_type(full):
                yield full

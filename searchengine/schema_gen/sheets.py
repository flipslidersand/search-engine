"""Google Sheets → 行リスト変換。

依存パッケージ（任意）:
  pip install google-api-python-client google-auth-oauthlib

認証方法:
  1. サービスアカウント: --credentials service_account.json
  2. OAuth2: --credentials client_secret.json（初回ブラウザ認証）
  3. 環境変数: GOOGLE_APPLICATION_CREDENTIALS
"""
from __future__ import annotations

import os
import re
from pathlib import Path


_SPREADSHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")


def extract_spreadsheet_id(url_or_id: str) -> str:
    """URL またはスプレッドシート ID から ID を取得する。"""
    m = _SPREADSHEET_ID_RE.search(url_or_id)
    return m.group(1) if m else url_or_id


def _build_service(credentials_path: str | None):
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account
        from google.auth import default as google_default
        import google.auth.transport.requests
    except ImportError:
        raise ImportError(
            "Google Sheets 連携には追加パッケージが必要です:\n"
            "  pip install google-api-python-client google-auth-oauthlib"
        )

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    if credentials_path:
        p = Path(credentials_path)
        if not p.exists():
            raise FileNotFoundError(f"認証ファイルが見つかりません: {credentials_path}")
        import json
        cred_data = json.loads(p.read_text())
        cred_type = cred_data.get("type", "")

        if cred_type == "service_account":
            creds = service_account.Credentials.from_service_account_file(
                str(p), scopes=SCOPES
            )
        else:
            # OAuth2 client secret → ブラウザ認証フロー
            creds = _oauth2_flow(str(p), SCOPES)
    else:
        # GOOGLE_APPLICATION_CREDENTIALS 環境変数 or ADC
        env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if env_path:
            creds = service_account.Credentials.from_service_account_file(
                env_path, scopes=SCOPES
            )
        else:
            creds, _ = google_default(scopes=SCOPES)

    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def _oauth2_flow(client_secret_path: str, scopes: list[str]):
    """OAuth2 ブラウザ認証フロー（トークンをローカルキャッシュ）。"""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.oauth2.credentials import Credentials
        import google.auth.transport.requests
    except ImportError:
        raise ImportError("pip install google-auth-oauthlib")

    token_path = Path(client_secret_path).parent / "token.json"
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes)
        if creds.valid:
            return creds
        if creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
            token_path.write_text(creds.to_json())
            return creds

    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, scopes)
    creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json())
    return creds


def load_sheet(
    url_or_id: str,
    sheet_name: str | None = None,
    credentials_path: str | None = None,
) -> list[dict]:
    """Google Sheets を読み込み、行リスト (list[dict]) を返す。

    Args:
        url_or_id: スプレッドシートの URL またはスプレッドシート ID。
        sheet_name: シート名（省略時は最初のシート）。
        credentials_path: サービスアカウント or OAuth2 client_secret.json のパス。

    Returns:
        1行目をヘッダーとして使った行リスト。
    """
    service = _build_service(credentials_path)
    spreadsheet_id = extract_spreadsheet_id(url_or_id)

    # シート名解決
    if sheet_name is None:
        meta = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
        sheet_name = meta["sheets"][0]["properties"]["title"]

    range_name = f"{sheet_name}"
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
        .execute()
    )
    values = result.get("values", [])
    if not values:
        return []

    headers = [str(h) if h else f"col_{i}" for i, h in enumerate(values[0])]
    rows = []
    for row in values[1:]:
        # 短い行を None でパディング
        padded = row + [None] * (len(headers) - len(row))
        rows.append(dict(zip(headers, padded)))
    return rows

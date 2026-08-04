#!/usr/bin/env bash
# MINIPC (192.168.68.63) へのデプロイスクリプト。
# 使い方: ./deploy.sh [--host <ip>] [--port <port>]
set -euo pipefail

HOST="${DEPLOY_HOST:?DEPLOY_HOST を設定してください (例: export DEPLOY_HOST=192.168.1.10)}"
REMOTE_USER="${DEPLOY_USER:-dev-nodee}"
REMOTE_DIR="/home/${REMOTE_USER}/services/search-engine"
LOCAL_PORT=8010

while [[ $# -gt 0 ]]; do
  case $1 in
    --host) HOST="$2"; shift 2 ;;
    --port) LOCAL_PORT="$2"; shift 2 ;;
    *) echo "不明なオプション: $1"; exit 1 ;;
  esac
done

echo "▶ デプロイ先: ${REMOTE_USER}@${HOST}:${REMOTE_DIR}"

# リモートにディレクトリを作成
ssh "${REMOTE_USER}@${HOST}" "mkdir -p ${REMOTE_DIR}"

# ファイルを転送（.venv, __pycache__ は除外）
rsync -az --delete \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.db' \
  --exclude='.pytest_cache/' \
  . "${REMOTE_USER}@${HOST}:${REMOTE_DIR}/"

# リモートで docker compose 再起動
ssh "${REMOTE_USER}@${HOST}" bash <<REMOTE
  cd ${REMOTE_DIR}
  docker compose pull 2>/dev/null || true
  docker compose up -d --build
  echo "✅ 起動完了 — http://${HOST}:${LOCAL_PORT}"
REMOTE

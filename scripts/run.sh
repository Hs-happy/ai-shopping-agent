#!/usr/bin/env bash
# 一键启动（macOS/Linux）
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/bin/python" ]; then
  echo "[1/4] 创建虚拟环境..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f "data/shop.db" ]; then
  echo "[2/4] 生成 5 万条 SKU..."
  .venv/bin/python -m app.data.generator --count 50000
  echo "[3/4] 清洗入库..."
  .venv/bin/python -m app.data.etl
fi

echo "[4/4] 启动服务：http://127.0.0.1:8771"
exec .venv/bin/python -m app.main

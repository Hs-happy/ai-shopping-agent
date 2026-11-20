# 一键启动（Windows）：建虚拟环境 → 装依赖 → 造数 → 起服务
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "[1/4] 创建虚拟环境..." -ForegroundColor Cyan
    python -m venv .venv
    & ".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
}

if (-not (Test-Path "data\shop.db")) {
    Write-Host "[2/4] 生成 5 万条 SKU..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m app.data.generator --count 50000
    Write-Host "[3/4] 清洗入库..." -ForegroundColor Cyan
    & ".venv\Scripts\python.exe" -m app.data.etl
}

Write-Host "[4/4] 启动服务：http://127.0.0.1:8771" -ForegroundColor Green
& ".venv\Scripts\python.exe" -m app.main

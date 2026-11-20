"""一个最小但真实的 MCP Server（stdio 传输 + JSON-RPC 2.0）。

没有依赖官方 SDK，是为了把协议讲清楚：
- 握手：initialize → notifications/initialized
- 发现：tools/list
- 调用：tools/call

传输层就是"一行一个 JSON"，日志一律写 stderr，stdout 只留给协议数据。
工具本身查 SQLite 里的库存，并用 sku_id 的哈希稳定地分配仓库与时效。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import time
from typing import Any

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "inventory-mcp", "version": "1.0.0"}
DB_PATH = os.environ.get("SHOP_DB", "data/shop.db")
FAKE_LATENCY_MS = float(os.environ.get("MCP_FAKE_LATENCY_MS", "0"))

WAREHOUSES = ("华东仓", "华南仓", "华北仓", "西南仓")

TOOLS = [
    {
        "name": "query_inventory",
        "description": "查询商品实时库存（仓库、可售数量、预计发货时效），最多 20 个 SKU",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sku_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}
            },
            "required": ["sku_ids"],
        },
    },
    {
        "name": "get_promotion",
        "description": "查询商品当前可用的促销活动与优惠券",
        "inputSchema": {
            "type": "object",
            "properties": {"sku_id": {"type": "string"}},
            "required": ["sku_id"],
        },
    },
]


def _bucket(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=3)
    conn.row_factory = sqlite3.Row
    return conn


def query_inventory(args: dict[str, Any]) -> dict[str, Any]:
    sku_ids = [str(s) for s in (args.get("sku_ids") or [])][:20]
    if not sku_ids:
        return {"items": [], "note": "sku_ids 为空"}
    marks = ",".join("?" * len(sku_ids))
    rows = _conn().execute(f"SELECT sku_id, title, price, stock FROM products WHERE sku_id IN ({marks})", sku_ids).fetchall()
    found = {r["sku_id"]: r for r in rows}
    items = []
    for sku_id in sku_ids:
        row = found.get(sku_id)
        if not row:
            items.append({"sku_id": sku_id, "available": False, "reason": "商品不存在"})
            continue
        bucket = _bucket(sku_id)
        stock = int(row["stock"])
        items.append(
            {
                "sku_id": sku_id,
                "title": row["title"],
                "available": stock > 0,
                "stock": stock,
                "warehouse": WAREHOUSES[bucket % len(WAREHOUSES)],
                "eta_days": 1 + bucket % 3,
                "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
    return {"items": items}


def get_promotion(args: dict[str, Any]) -> dict[str, Any]:
    sku_id = str(args.get("sku_id") or "")
    row = _conn().execute("SELECT sku_id, price FROM products WHERE sku_id=?", (sku_id,)).fetchone()
    if not row:
        return {"sku_id": sku_id, "available": False, "reason": "商品不存在"}
    price = float(row["price"])
    if price >= 5000:
        coupon, desc = 300, "满 5000 减 300"
    elif price >= 2000:
        coupon, desc = 100, "满 2000 减 100"
    else:
        coupon, desc = 20, "满 500 减 20"
    return {"sku_id": sku_id, "coupon": coupon, "promotion": desc, "stackable": _bucket(sku_id) % 2 == 0}


HANDLERS = {"query_inventory": query_inventory, "get_promotion": get_promotion}


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        result: Any = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        }
    elif method in ("notifications/initialized", "notifications/cancelled"):
        return None
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if not handler:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"未知工具: {name}"}}
        if FAKE_LATENCY_MS:
            time.sleep(FAKE_LATENCY_MS / 1000.0)
        try:
            payload = handler(arguments)
            result = {"content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}], "isError": False}
        except Exception as exc:  # noqa: BLE001
            result = {"content": [{"type": "text", "text": f"工具执行失败: {exc}"}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"不支持的方法: {method}"}}

    if req_id is None:
        return None
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def main() -> None:
    print(f"[mcp-server] 启动，db={DB_PATH}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print("[mcp-server] 非法 JSON", file=sys.stderr, flush=True)
            continue
        response = handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

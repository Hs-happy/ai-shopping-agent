"""MCP 客户端 + 库存工具服务（带超时熔断与降级）。

设计要点：
- MCP server 常驻子进程，避免每次调用都付 Python 启动开销（否则 100ms 超时毫无意义）；
- 调用超时/进程异常 → 标记 degraded，回退到商品库里的静态库存；
- 每次调用都写 tool_logs，方便统计成功率与 P99。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import BASE_DIR, Settings, get_settings
from app.store.db import Database

logger = logging.getLogger(__name__)

TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "query_inventory",
            "description": "查询候选商品的实时库存与发货时效，用于确认是否可下单",
            "parameters": {
                "type": "object",
                "properties": {"sku_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
                "required": ["sku_ids"],
            },
        },
    }
]


class MCPError(RuntimeError):
    pass


class MCPClient:
    def __init__(self, module: str = "app.tools.mcp_server", db_path: Path | None = None,
                 fake_latency_ms: int = 0, python: str | None = None):
        self.module = module
        self.db_path = Path(db_path or get_settings().db_path)
        self.fake_latency_ms = fake_latency_ms
        self.python = python or sys.executable
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._started = False

    @property
    def alive(self) -> bool:
        return bool(self._proc and self._proc.returncode is None and self._started)

    async def start(self) -> None:
        if self.alive:
            return
        # 并发请求可能同时触发启动，必须串行化，否则会起出两个进程、读循环互相抢 stdout
        async with self._start_lock:
            if self.alive:
                return
            env = os.environ.copy()
            env["SHOP_DB"] = str(self.db_path)
            env["MCP_FAKE_LATENCY_MS"] = str(self.fake_latency_ms)
            self._proc = await asyncio.create_subprocess_exec(
                self.python, "-m", self.module,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=str(BASE_DIR),
                env=env,
            )
            self._reader_task = asyncio.create_task(self._read_loop())
            try:
                await self._request("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"roots": {"listChanged": False}},
                    "clientInfo": {"name": "shopping-agent", "version": "0.1.0"},
                }, timeout=8.0)
                await self._notify("notifications/initialized", {})
            except Exception:
                await self._stop_process()
                raise
            self._started = True
            logger.info("MCP server 已连接 (pid=%s)", self._proc.pid if self._proc else "-")

    async def _stop_process(self) -> None:
        self._started = False
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._proc and self._proc.returncode is None:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        self._proc = None

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, payload: dict[str, Any]) -> None:
        if not self._proc or self._proc.stdin is None or self._proc.returncode is not None:
            self._started = False
            raise MCPError("MCP 进程不可用")
        async with self._write_lock:
            try:
                self._proc.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                await self._proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, AttributeError, RuntimeError) as exc:
                self._started = False
                raise MCPError(f"MCP 写入失败: {exc}") from exc

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        try:
            while True:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                try:
                    message = json.loads(line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                req_id = message.get("id")
                future = self._pending.pop(req_id, None) if req_id is not None else None
                if future and not future.done():
                    if "error" in message:
                        future.set_exception(MCPError(str(message["error"])))
                    else:
                        future.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP 读循环退出: %s", exc)
        finally:
            self._started = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(MCPError("MCP 连接已断开"))
            self._pending.clear()

    async def _request(self, method: str, params: dict[str, Any], timeout: float) -> Any:
        self._next_id += 1
        req_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        await self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(req_id, None)
            raise MCPError(f"调用 {method} 超时({timeout * 1000:.0f}ms)") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if not self.alive:
            await self.start()
        result = await self._request("tools/call", {"name": name, "arguments": arguments}, timeout_s)
        if not result:
            raise MCPError("空响应")
        if result.get("isError"):
            text = (result.get("content") or [{}])[0].get("text", "工具返回错误")
            raise MCPError(text)
        text = (result.get("content") or [{}])[0].get("text", "{}")
        return json.loads(text)

    async def close(self) -> None:
        await self._stop_process()


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    degraded: bool = False
    error: str = ""
    source: str = "mcp"


class InventoryService:
    def __init__(self, settings: Settings | None = None, db: Database | None = None):
        self.settings = settings or get_settings()
        self.db = db or Database(self.settings.db_path)
        self.client = MCPClient(
            module=self.settings.mcp_server_module,
            db_path=self.settings.db_path,
            fake_latency_ms=getattr(self.settings, "mcp_fake_latency_ms", 0),
        )
        self.warm = False

    async def ensure_ready(self) -> None:
        if self.warm:
            return
        try:
            await self.client.start()
            self.warm = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("MCP server 启动失败，库存工具将降级: %s", exc)
            self.warm = False

    async def query_inventory(self, sku_ids: list[str], session_id: str = "-") -> ToolResult:
        timeout_s = max(self.settings.tool_timeout_ms, 10) / 1000.0
        started = time.perf_counter()
        result = ToolResult(tool="query_inventory", ok=False)
        try:
            if not self.warm:
                await self.ensure_ready()
            data = await self.client.call_tool("query_inventory", {"sku_ids": sku_ids}, timeout_s)
            result.ok = True
            result.data = data
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            result.degraded = True
            result.source = "db_fallback"
            result.data = self._fallback(sku_ids)
            self.warm = self.client.alive
        result.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        try:
            self.db.log_tool(
                time.strftime("%Y-%m-%dT%H:%M:%S"), session_id, result.tool,
                {"sku_ids": sku_ids}, result.ok, result.latency_ms, result.error,
            )
        except Exception:  # noqa: BLE001 - 日志失败不影响主链路
            pass
        return result

    def _fallback(self, sku_ids: list[str]) -> dict[str, Any]:
        rows = self.db.get_products(sku_ids)
        items = []
        for sku_id in sku_ids:
            doc = rows.get(sku_id)
            if not doc:
                items.append({"sku_id": sku_id, "available": False, "reason": "商品不存在"})
                continue
            items.append({
                "sku_id": sku_id,
                "title": doc["title"],
                "available": int(doc["stock"]) > 0,
                "stock": int(doc["stock"]),
                "warehouse": "未知(降级)",
                "eta_days": 3,
                "source": "db_fallback",
            })
        return {"items": items, "degraded": True}

    async def close(self) -> None:
        await self.client.close()
        self.warm = False

"""SQLite 存储：商品主数据 + 会话状态。

选 SQLite 是因为它零部署、单文件、读性能足够，适合 demo 与单机部署；
商品检索走内存索引（BM25 + 向量），SQLite 只负责点查与持久化。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku_id     TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    brand      TEXT NOT NULL,
    category   TEXT NOT NULL,
    price      REAL NOT NULL,
    stock      INTEGER NOT NULL DEFAULT 0,
    rating     REAL NOT NULL DEFAULT 0,
    sales      INTEGER NOT NULL DEFAULT 0,
    specs      TEXT NOT NULL DEFAULT '{}',
    tags       TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    shop       TEXT NOT NULL DEFAULT '',
    search_text TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_products_category ON products(category);
CREATE INDEX IF NOT EXISTS idx_products_brand ON products(brand);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tool_logs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    session_id TEXT,
    tool       TEXT NOT NULL,
    args       TEXT,
    ok         INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    error      TEXT
);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.init_schema()

    # ------------------------------------------------------------------ 连接
    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def init_schema(self) -> None:
        self.connect().executescript(SCHEMA)
        self.connect().commit()

    # ------------------------------------------------------------------ 商品
    def replace_products(self, rows: list[dict[str, Any]]) -> None:
        conn = self.connect()
        with conn:
            conn.execute("DELETE FROM products")
            conn.executemany(
                """INSERT OR REPLACE INTO products
                   (sku_id, title, brand, category, price, stock, rating, sales,
                    specs, tags, description, shop, search_text)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        r["sku_id"], r["title"], r["brand"], r["category"], r["price"],
                        r["stock"], r["rating"], r["sales"],
                        json.dumps(r["specs"], ensure_ascii=False),
                        json.dumps(r["tags"], ensure_ascii=False),
                        r["description"], r["shop"], r["search_text"],
                    )
                    for r in rows
                ],
            )

    def count_products(self) -> int:
        return int(self.connect().execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def iter_products(self) -> Iterable[dict[str, Any]]:
        for row in self.connect().execute("SELECT * FROM products"):
            yield self._row_to_dict(row)

    def get_products(self, sku_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not sku_ids:
            return {}
        marks = ",".join("?" * len(sku_ids))
        rows = self.connect().execute(f"SELECT * FROM products WHERE sku_id IN ({marks})", sku_ids)
        return {row["sku_id"]: self._row_to_dict(row) for row in rows}

    def get_product(self, sku_id: str) -> dict[str, Any] | None:
        row = self.connect().execute("SELECT * FROM products WHERE sku_id=?", (sku_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def facet_counts(self) -> dict[str, Any]:
        conn = self.connect()
        cats = conn.execute("SELECT category, COUNT(*) c FROM products GROUP BY category").fetchall()
        brands = conn.execute("SELECT brand, COUNT(*) c FROM products GROUP BY brand ORDER BY c DESC").fetchall()
        price = conn.execute("SELECT MIN(price), MAX(price) FROM products").fetchone()
        return {
            "categories": {r["category"]: r["c"] for r in cats},
            "brands": {r["brand"]: r["c"] for r in brands},
            "price_range": [price[0], price[1]],
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        data["specs"] = json.loads(data["specs"])
        data["tags"] = json.loads(data["tags"])
        return data

    # ------------------------------------------------------------------ 会话
    def load_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.connect().execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row:
            return None
        return {"session_id": row["session_id"], "state": json.loads(row["state"]), "updated_at": row["updated_at"]}

    def save_session(self, session_id: str, state: dict[str, Any]) -> None:
        import time

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        conn = self.connect()
        with conn:
            conn.execute(
                """INSERT INTO sessions(session_id, created_at, updated_at, state) VALUES (?,?,?,?)
                   ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, state=excluded.state""",
                (session_id, now, now, json.dumps(state, ensure_ascii=False)),
            )

    # ------------------------------------------------------------------ 工具日志
    def log_tool(self, ts: str, session_id: str, tool: str, args: dict, ok: bool, latency_ms: float, error: str = "") -> None:
        conn = self.connect()
        with conn:
            conn.execute(
                "INSERT INTO tool_logs(ts, session_id, tool, args, ok, latency_ms, error) VALUES (?,?,?,?,?,?,?)",
                (ts, session_id, tool, json.dumps(args, ensure_ascii=False), int(ok), latency_ms, error),
            )

    def tool_stats(self) -> list[dict[str, Any]]:
        rows = self.connect().execute(
            """SELECT tool, COUNT(*) n, AVG(latency_ms) avg_ms, AVG(ok) success_rate
               FROM tool_logs GROUP BY tool ORDER BY n DESC"""
        ).fetchall()
        return [dict(r) for r in rows]

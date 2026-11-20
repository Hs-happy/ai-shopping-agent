"""打印 MCP 工具调用统计（成功率 / 平均耗时 / 最近失败原因）。

用法：python scripts/tool_stats.py [--limit 5]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--clear", action="store_true", help="清空历史工具日志（联调/演示前重置统计）")
    args = parser.parse_args()

    conn = sqlite3.connect(get_settings().db_path)
    conn.row_factory = sqlite3.Row
    if args.clear:
        with conn:
            conn.execute("DELETE FROM tool_logs")
        print("tool_logs 已清空")
        return

    print("== 汇总（按结果）")
    for row in conn.execute("SELECT ok, COUNT(*) n, MIN(ts) first, MAX(ts) last FROM tool_logs GROUP BY ok"):
        print(dict(row))
    print("\n== 最近失败")
    for row in conn.execute(
        "SELECT ts, tool, latency_ms, error FROM tool_logs WHERE ok=0 ORDER BY id DESC LIMIT ?", (args.limit,)
    ):
        print(dict(row))
    print("\n== 最近成功")
    for row in conn.execute(
        "SELECT ts, tool, latency_ms FROM tool_logs WHERE ok=1 ORDER BY id DESC LIMIT ?", (args.limit,)
    ):
        print(dict(row))


if __name__ == "__main__":
    main()

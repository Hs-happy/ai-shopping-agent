"""Agent 运行入口：装配依赖、执行状态机、维护会话。"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.agent.graph import build_graph
from app.agent.nodes import AgentDeps
from app.agent.state import new_state
from app.config import Settings, get_settings
from app.core.logging import session_id_var, trace_id_var
from app.llm.client import LLMClient
from app.retrieval.engine import SearchEngine
from app.store.db import Database
from app.tools.inventory import InventoryService

logger = logging.getLogger(__name__)


class ShoppingAgent:
    def __init__(self, settings: Settings | None = None, engine: SearchEngine | None = None):
        self.settings = settings or get_settings()
        self.engine: SearchEngine = engine or SearchEngine(self.settings).ensure_ready()
        self.db: Database = self.engine.db
        self.llm = LLMClient(self.settings)
        self.inventory = InventoryService(self.settings, self.db)
        self.graph = build_graph(
            AgentDeps(engine=self.engine, llm=self.llm, inventory=self.inventory, settings=self.settings)
        )

    async def start(self) -> None:
        await self.inventory.ensure_ready()

    async def close(self) -> None:
        await self.inventory.close()

    # ------------------------------------------------------------------ run
    async def run(self, session_id: str, query: str) -> dict[str, Any]:
        record = self.db.load_session(session_id) or {}
        state_data = record.get("state") or {}
        history = state_data.get("history") or []
        slots = state_data.get("slots") or {}

        trace_id = f"{session_id}-{int(time.time() * 1000)}"
        trace_id_var.set(trace_id)
        session_id_var.set(session_id)

        started = time.perf_counter()
        initial = new_state(session_id, query, history, slots)
        final = await self.graph.ainvoke(initial, {"recursion_limit": 15})
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        answer = final.get("answer") or ""
        new_history = history + [{"role": "user", "content": query}, {"role": "assistant", "content": answer[:500]}]
        new_history = new_history[-self.settings.max_turns_memory * 2 :]
        self.db.save_session(
            session_id,
            {
                "history": new_history,
                "slots": final.get("slots") or {},
                "last_query": query,
                "last_answer": answer,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        final["timings"] = {**(final.get("timings") or {}), "total_ms": elapsed_ms}
        logger.info(
            "turn done",
            extra={"extra_fields": {
                "latency_ms": elapsed_ms,
                "confidence": final.get("confidence"),
                "degraded": final.get("degraded"),
                "rewrites": final.get("rewrite_count", 0),
            }},
        )
        return final

    # ------------------------------------------------------------------ session
    def session_snapshot(self, session_id: str) -> dict[str, Any]:
        record = self.db.load_session(session_id) or {}
        return record.get("state") or {}

    def reset_session(self, session_id: str) -> None:
        self.db.save_session(session_id, {"history": [], "slots": {}})

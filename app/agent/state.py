"""Agent 状态定义。

LangGraph 的节点只读写这个 TypedDict：每个节点返回部分更新，
框架负责合并。状态本身保持 JSON 可序列化，便于落库与回放。
"""

from __future__ import annotations

import time
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # 会话
    session_id: str
    trace_id: str
    history: list[dict[str, str]]

    # 本轮输入
    query: str
    search_query: str
    slots: dict[str, Any]
    intent: str

    # 检索
    candidates: list[dict[str, Any]]
    confidence: float
    retrieval: dict[str, Any]
    rewrite_count: int
    relaxed: list[str]

    # 工具
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    degraded: bool

    # 输出
    answer: str
    clarify: str | None
    timings: dict[str, float]


def new_state(session_id: str, query: str, history: list[dict[str, str]] | None = None,
              slots: dict[str, Any] | None = None) -> AgentState:
    return AgentState(
        session_id=session_id,
        trace_id=f"{session_id}-{int(time.time() * 1000)}",
        history=history or [],
        query=query,
        slots=slots or {},
        rewrite_count=0,
        candidates=[],
        tool_calls=[],
        tool_results=[],
        degraded=False,
        timings={},
    )

"""Agent 运行期事件。

对话链路的每个阶段（抽槽、检索、工具、生成）都会往同一个 sink 发事件，
HTTP 层把 sink 接到 SSE，测试里接到内存列表，CLI 里接到 NullSink。
这样业务代码不依赖 FastAPI / SSE，也不依赖 LangGraph 的流式 API。
"""

from __future__ import annotations

from contextvars import ContextVar
from enum import Enum
from typing import Any, Protocol


class EventType(str, Enum):
    META = "meta"            # 链路元信息（trace_id / session_id / model）
    SLOTS = "slots"          # 槽位抽取结果
    INTENT = "intent"        # 意图识别
    TOOL_CALL = "tool_call"  # 工具调用开始
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"  # 召回/融合/重排统计
    REWRITE = "rewrite"      # 低置信度改写
    DELTA = "delta"          # 流式 token
    PRODUCTS = "products"    # 结构化商品卡片
    CLARIFY = "clarify"      # 需要用户补充信息
    DONE = "done"
    ERROR = "error"


class EventSink(Protocol):
    async def emit(self, type: EventType, data: dict[str, Any]) -> None: ...


class NullSink:
    async def emit(self, type: EventType, data: dict[str, Any]) -> None:  # noqa: A002
        return None


class CollectSink:
    """测试用：把所有事件收进列表。"""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def emit(self, type: EventType, data: dict[str, Any]) -> None:  # noqa: A002
        self.events.append({"type": type.value, **data})

    def of(self, type: EventType) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == type.value]


class QueueSink:
    """HTTP 层用：把事件推进 asyncio 队列，由 SSE 生成器消费。"""

    def __init__(self, queue) -> None:
        self.queue = queue

    async def emit(self, type: EventType, data: dict[str, Any]) -> None:  # noqa: A002
        await self.queue.put({"type": type.value, **data})


current_sink: ContextVar[EventSink] = ContextVar("event_sink", default=NullSink())


def set_sink(sink: EventSink):
    return current_sink.set(sink)


def reset_sink(token) -> None:
    current_sink.reset(token)


async def emit(type: EventType, **data: Any) -> None:  # noqa: A002
    await current_sink.get().emit(type, data)

"""对话接口：SSE 流式 + 一次性返回。

SSE 事件协议（前端按 event 名分发）：
  meta      链路元信息
  slots     槽位解析结果
  intent    意图
  retrieval 双路召回/融合/重排统计与候选
  rewrite   低置信度改写
  tool_call / tool_result  MCP 工具
  delta     流式文本增量
  products  结构化商品卡片
  clarify   需要用户补充
  done      结束（含总耗时、置信度、是否降级）
  error     异常
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.events import QueueSink, reset_sink, set_sink

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _agent(request: Request):
    return request.app.state.agent


async def _event_stream(request: Request, session_id: str, query: str) -> AsyncIterator[str]:
    agent = _agent(request)
    queue: asyncio.Queue = asyncio.Queue()
    token = set_sink(QueueSink(queue))

    async def run() -> None:
        try:
            final = await agent.run(session_id, query)
            await queue.put(
                {
                    "type": "done",
                    "session_id": session_id,
                    "answer": final.get("answer"),
                    "slots": final.get("slots"),
                    "confidence": final.get("confidence"),
                    "degraded": final.get("degraded"),
                    "timings": final.get("timings"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("对话执行失败")
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run())
    yield _sse("meta", {"session_id": session_id, "model": agent.llm.model if agent.llm.enabled else "mock"})
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            event = item.pop("type")
            yield _sse(event, item)
    finally:
        reset_sink(token)
        if not task.done():
            task.cancel()


@router.post("/stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    session_id = payload.session_id or uuid.uuid4().hex[:12]
    return StreamingResponse(
        _event_stream(request, session_id, payload.query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/stream")
async def chat_stream_get(request: Request, q: str, session_id: str = "default") -> StreamingResponse:
    return StreamingResponse(
        _event_stream(request, session_id, q),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("")
async def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    agent = _agent(request)
    final = await agent.run(payload.session_id, payload.query)
    return {
        "session_id": payload.session_id,
        "answer": final.get("answer"),
        "slots": final.get("slots"),
        "products": final.get("candidates"),
        "confidence": final.get("confidence"),
        "degraded": final.get("degraded"),
        "timings": final.get("timings"),
    }


@router.get("/session/{session_id}")
async def get_session(session_id: str, request: Request) -> dict[str, Any]:
    return {"session_id": session_id, **(_agent(request).session_snapshot(session_id) or {})}


@router.delete("/session/{session_id}")
async def reset_session(session_id: str, request: Request) -> dict[str, Any]:
    _agent(request).reset_session(session_id)
    return {"ok": True, "session_id": session_id}


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    agent = _agent(request)
    if agent is None:
        raise HTTPException(status_code=503, detail="agent 未就绪")
    return {
        "ok": True,
        "llm": agent.llm.model if agent.llm.enabled else "mock",
        "embedding": getattr(agent.engine.embedder, "name", "unknown"),
        "docs": len(agent.engine.docs),
        "mcp": agent.inventory.warm,
    }

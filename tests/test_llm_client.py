"""用 httpx MockTransport 验证 OpenAI 兼容链路：JSON 模式、Function Calling、流式。

不需要真实 API Key——这里校验的是"协议层没写错"：
请求体里的 response_format / tools / stream 字段，以及响应的解析。
"""

from __future__ import annotations

import json

import httpx

from app.config import Settings
from app.llm.client import LLMClient


def _handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        chunks = [
            {"choices": [{"delta": {"content": "你好"}}]},
            {"choices": [{"delta": {"content": "，我是小导"}}]},
        ]
        text = "".join(f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=text.encode("utf-8"), headers={"content-type": "text/event-stream"})
    if body.get("tools"):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool", "object": "chat.completion", "created": 0, "model": "mock",
                "choices": [{
                    "index": 0, "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant", "content": None,
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "query_inventory", "arguments": "{\"sku_ids\": [\"SKU1\"]}"},
                        }],
                    },
                }],
            },
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-json", "object": "chat.completion", "created": 0, "model": "mock",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "{\"category\": \"手机\", \"price_max\": 5000}"},
            }],
        },
    )


def _client() -> LLMClient:
    settings = Settings(llm_mode="openai", llm_api_key="test-key", llm_base_url="https://example.invalid/v1")
    transport = httpx.MockTransport(_handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://example.invalid/v1")
    return LLMClient(settings, http_client=http_client)


async def test_chat_json_parses_structured_output():
    client = _client()
    data = await client.chat_json([{"role": "user", "content": "5000以内拍照好的手机"}])
    assert data == {"category": "手机", "price_max": 5000}


async def test_function_calling_returns_tool_calls():
    client = _client()
    calls = await client.call_with_tools(
        [{"role": "user", "content": "查一下库存"}],
        [{"type": "function", "function": {"name": "query_inventory", "parameters": {"type": "object"}}}],
    )
    assert calls == [{"name": "query_inventory", "arguments": {"sku_ids": ["SKU1"]}}]


async def test_stream_yields_incremental_text():
    client = _client()
    parts = []
    async for token in client.stream([{"role": "user", "content": "你好"}]):
        parts.append(token)
    assert "".join(parts) == "你好，我是小导"

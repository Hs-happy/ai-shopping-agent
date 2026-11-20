"""LLM 客户端：OpenAI 兼容协议，封装超时、JSON 模式与流式输出。

没有配置 key 时 enabled=False，Agent 各节点会自动走确定性规则分支，
保证"没有模型也能完整跑通链路"，这对本地开发和 CI 很重要。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, AsyncIterator

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里抠出第一个 JSON 对象，兼容 ```json 围栏和前后废话。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


class LLMClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = None
        if self.settings.llm_enabled:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                timeout=self.settings.llm_timeout_s,
                max_retries=1,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self.settings.llm_model

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        if not self._client:
            raise LLMError("LLM 未启用")
        kwargs: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.settings.llm_max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            raise LLMError(str(exc)) from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if not self._client:
            raise LLMError("LLM 未启用")
        try:
            stream = await self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                temperature=self.settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:  # noqa: BLE001
            raise LLMError(str(exc)) from exc

    async def chat_json(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> dict[str, Any] | None:
        try:
            text = await self.chat(messages, temperature=temperature, json_mode=True, max_tokens=400)
            return extract_json(text)
        except LLMError as exc:
            logger.warning("LLM JSON 调用失败: %s", exc)
            return None

    async def call_with_tools(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Function Calling：让模型自己决定调哪个工具。返回 [{name, arguments}]。"""
        if not self._client:
            return []
        try:
            resp = await self._client.chat.completions.create(
                model=self.settings.llm_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=200,
            )
            message = resp.choices[0].message
            calls = []
            for call in message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                calls.append({"name": call.function.name, "arguments": arguments})
            return calls
        except Exception as exc:  # noqa: BLE001
            logger.warning("Function Calling 失败: %s", exc)
            return []

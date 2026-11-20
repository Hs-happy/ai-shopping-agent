"""六个 Agent 节点。

链路：understand → retrieve →(低置信度)→ rewrite → retrieve → plan → tool → respond

每个节点都是 async 函数，只返回"状态增量"，并把过程事件写给 EventSink，
HTTP 层再把这些事件转成 SSE。节点里不做任何 HTTP/SSE 相关的事。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.agent import prompts
from app.agent.slots import Slots, extract_by_rules, merge_slots
from app.agent.state import AgentState
from app.config import Settings
from app.core.events import EventType, emit
from app.data.taxonomy import CATEGORIES, CATEGORY_ALIASES
from app.llm.client import LLMClient
from app.retrieval.engine import SearchEngine
from app.tools.inventory import TOOL_SCHEMA, InventoryService

logger = logging.getLogger(__name__)

KNOWN_ATTRIBUTES = {
    "拍照", "续航", "游戏", "轻薄", "降噪", "高刷", "快充", "音质", "办公", "学生",
    "大内存", "防水", "运动", "健康", "长续航", "4K", "大屏", "性价比", "旗舰", "送礼",
    "户外", "视频", "便携",
}
KNOWN_BRANDS = {b for cat in CATEGORIES.values() for brand in cat.brands for b in (brand.name,)}

ATTR_EXPANSION = {
    "拍照": "拍照 影像 大底 长焦",
    "续航": "续航 大电池 长续航",
    "游戏": "游戏 电竞 高刷 性能",
    "轻薄": "轻薄 便携 重量轻",
    "降噪": "降噪 ANC 隔音",
    "高刷": "高刷 高刷新率 120Hz",
    "快充": "快充 充电快",
    "音质": "音质 音效 低频",
    "办公": "办公 商务 接口",
    "学生": "学生 宿舍 性价比",
    "视频": "视频 4K Vlog",
    "大屏": "大屏 大尺寸",
}
SCENE_EXPANSION = {
    "送礼": "送礼 礼物 礼盒",
    "学生": "学生 宿舍 性价比",
    "商务": "商务 办公 通勤",
    "家庭": "家用 客厅",
    "户外": "户外 露营 防水",
}


@dataclass
class AgentDeps:
    engine: SearchEngine
    llm: LLMClient
    inventory: InventoryService
    settings: Settings


# ---------------------------------------------------------------- 工具函数
def _detect_intent(query: str) -> str:
    text = query.lower()
    if any(k in text for k in ("对比", "比较", "哪个好", "选哪个", "pk")):
        return "compare"
    if any(k in text for k in ("退货", "退款", "保修", "售后", "发票", "换货", "物流", "发货")):
        return "aftersales"
    if len(text) <= 6 and any(k in text for k in ("你好", "在吗", "hello", "hi", "谢谢")):
        return "chitchat"
    return "recommend"


def _slots_from_llm(data: dict[str, Any] | None) -> Slots | None:
    if not data:
        return None
    slots = Slots()
    category = data.get("category")
    if isinstance(category, str):
        if category in CATEGORIES:
            slots.category = category
        else:
            for alias, cat in CATEGORY_ALIASES.items():
                if category in alias or alias in category:
                    slots.category = cat
                    break
    brand = data.get("brand")
    if isinstance(brand, str) and brand in KNOWN_BRANDS:
        slots.brand = brand
    for field in ("price_min", "price_max"):
        value = data.get(field)
        if isinstance(value, (int, float)) and 0 < value < 200_000:
            setattr(slots, field, float(value))
    attrs = data.get("attributes") or []
    if isinstance(attrs, list):
        slots.attributes = [a for a in attrs if isinstance(a, str) and a in KNOWN_ATTRIBUTES][:3]
    scene = data.get("scene")
    if isinstance(scene, str) and scene in SCENE_EXPANSION:
        slots.scene = scene
    excludes = data.get("exclude_brands") or []
    if isinstance(excludes, list):
        slots.exclude_brands = [b for b in excludes if b in KNOWN_BRANDS]
    intent = data.get("intent")
    if intent in ("recommend", "compare", "aftersales", "chitchat"):
        slots.intent = intent
    return slots


def _rule_rewrite(query: str, slots: dict[str, Any]) -> str:
    bits = [query]
    for attr in slots.get("attributes") or []:
        bits.append(ATTR_EXPANSION.get(attr, attr))
    if slots.get("scene"):
        bits.append(SCENE_EXPANSION.get(slots["scene"], ""))
    if slots.get("category"):
        bits.append(slots["category"])
    return " ".join(b for b in bits if b).strip()


def _stream_chunks(text: str, size: int = 8):
    for i in range(0, len(text), size):
        yield text[i : i + size]


def _template_answer(slots: Slots, cards: list[dict[str, Any]], relaxed: list[str],
                     degraded: bool, missing: list[str]) -> str:
    if not cards:
        head = f"按「{slots.describe()}」没找到合适的商品。"
        if missing:
            head += "你把品类或预算再具体一点，我重新帮你筛？"
        return head

    lines = [f"按「{slots.describe()}」筛了一轮，给你挑了 {min(len(cards), 3)} 款："]
    for idx, card in enumerate(cards[:3], start=1):
        price = int(card["price"])
        reasons = "，".join(card.get("reasons", [])[:2]) or "综合配置均衡"
        specs = "、".join(f"{k}{v}" for k, v in list(card.get("specs", {}).items())[:2])
        stock = card.get("stock_live", card.get("stock"))
        stock_text = "有货" if (stock or 0) > 0 else "暂时缺货"
        lines.append(f"{idx}. **{card['title']}**（¥{price}，{stock_text}）\n   {reasons}；{specs}。")

    notes: list[str] = []
    if "price_widened" in relaxed:
        notes.append("严格卡预算的款不多，我把价格浮动 20% 以内的也放进来了。")
    if "brand" in relaxed or "category" in relaxed:
        notes.append("你要的条件在库里没有完全对上的，这里按相近方向推荐。")
    if degraded:
        notes.append("实时库存接口刚超时了，上面的库存来自商品库快照。")
    if notes:
        lines.append("")
        lines.extend(notes)

    question = "你更看重哪一点？我可以按拍照、续航或手感再排一版。"
    if not slots.price_max:
        question = "预算大概多少？定了预算我就能缩到 3 款以内。"
    elif not slots.brand:
        question = "有偏好的品牌吗？没有的话我按口碑排。"
    lines.append("")
    lines.append(question)
    return "\n".join(lines)


# ---------------------------------------------------------------- 节点
def make_nodes(deps: AgentDeps):
    settings = deps.settings

    async def understand(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        query = (state.get("query") or "").strip()
        history = state.get("history") or []
        prev = Slots(**(state.get("slots") or {}))
        rules = extract_by_rules(query)

        if query in ("重置", "重新开始", "重来", "清空"):
            slots = Slots()
        else:
            slots = merge_slots(prev, rules)
            # 品类切换时，上一个品类的需求点（如"降噪"）不该继续生效
            if rules.category and prev.category and rules.category != prev.category:
                slots = Slots(category=rules.category, price_min=prev.price_min, price_max=prev.price_max)
                slots = merge_slots(slots, rules)

        # LLM 通道：补语义槽位；硬字段仍以正则为准
        if deps.llm.enabled:
            data = await deps.llm.chat_json(prompts.understand_messages(query, history, slots.to_dict()))
            llm_slots = _slots_from_llm(data)
            if llm_slots:
                slots = merge_slots(slots, llm_slots)
                for field in ("category", "brand", "price_min", "price_max"):
                    value = getattr(rules, field)
                    if value is not None:
                        setattr(slots, field, value)

        intent = _detect_intent(query)
        if slots.intent in ("compare", "aftersales", "chitchat"):
            intent = slots.intent

        # 多轮微调：用户不一定会重新报预算，"再便宜点/太贵了/加点预算"要能落在槽位上
        if any(k in query for k in ("便宜", "降点", "低一点", "太贵", "贵了")):
            if slots.price_max:
                slots.price_max = float(round(slots.price_max * 0.72, -1))
            elif slots.price_min:
                slots.price_min = float(round(slots.price_min * 0.8, -1))
        elif any(k in query for k in ("贵一点", "好一点", "升级", "加点预算", "提高预算")):
            if slots.price_max:
                slots.price_max = float(round(slots.price_max * 1.35, -1))

        await emit(
            EventType.SLOTS,
            slots=slots.to_dict(),
            summary=slots.describe(),
            missing=slots.missing(),
            source="llm+rule" if deps.llm.enabled else "rule",
        )
        await emit(EventType.INTENT, intent=intent)
        timings = {**(state.get("timings") or {}), "understand_ms": round((time.perf_counter() - started) * 1000, 2)}
        return {"slots": slots.to_dict(), "intent": intent, "timings": timings}

    async def retrieve(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        slots = state.get("slots") or {}
        query = state.get("search_query") or state["query"]
        result = deps.engine.search(query, slots, top_k=settings.final_k)
        cards = [c.card() for c in result.hits]
        await emit(
            EventType.RETRIEVAL,
            query=query,
            confidence=result.confidence,
            relaxed=result.relaxed,
            stats=result.stats,
            candidates=[
                {"sku_id": c["sku_id"], "title": c["title"], "score": c["score"],
                 "bm25_rank": c["recall"]["bm25_rank"], "vector_rank": c["recall"]["vector_rank"]}
                for c in cards
            ],
        )
        timings = {**(state.get("timings") or {}), "retrieve_ms": round((time.perf_counter() - started) * 1000, 2)}
        return {
            "candidates": cards,
            "confidence": result.confidence,
            "retrieval": result.stats,
            "relaxed": result.relaxed,
            "search_query": query,
            "timings": timings,
        }

    async def rewrite(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        slots = state.get("slots") or {}
        query = state["query"]
        new_query = ""
        if deps.llm.enabled:
            try:
                text = await deps.llm.chat(prompts.rewrite_messages(query, slots), temperature=0.2, max_tokens=80)
                new_query = " ".join((text or "").strip().splitlines())[:80]
            except Exception as exc:  # noqa: BLE001
                logger.warning("改写失败，走规则改写: %s", exc)
        if not new_query:
            new_query = _rule_rewrite(query, slots)
        await emit(
            EventType.REWRITE,
            original=query,
            rewritten=new_query,
            reason=f"检索置信度低({state.get('confidence', 0):.2f})",
            round=state.get("rewrite_count", 0) + 1,
        )
        timings = {**(state.get("timings") or {}), "rewrite_ms": round((time.perf_counter() - started) * 1000, 2)}
        return {"search_query": new_query, "rewrite_count": state.get("rewrite_count", 0) + 1, "timings": timings}

    async def plan(state: AgentState) -> dict[str, Any]:
        cards = state.get("candidates") or []
        if not cards:
            return {"tool_calls": []}
        sku_ids = [c["sku_id"] for c in cards[:3]]
        calls: list[dict[str, Any]] = []
        if deps.llm.enabled and settings.enable_function_calling:
            messages = [
                {"role": "system", "content": "你是导购助手，需要确认推荐商品是否真的有货。需要时调用库存工具。"},
                {"role": "user", "content": f"用户需求：{state['query']}\n候选 SKU：{sku_ids}"},
            ]
            try:
                calls = await deps.llm.call_with_tools(messages, TOOL_SCHEMA)
                calls = [c for c in calls if c.get("name") == "query_inventory"]
                if calls:
                    arguments = calls[0].get("arguments") or {}
                    ids = [str(i) for i in (arguments.get("sku_ids") or []) if str(i) in sku_ids] or sku_ids
                    calls = [{"name": "query_inventory", "arguments": {"sku_ids": ids[:3]}, "source": "llm"}]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Function Calling 决策失败，走规则: %s", exc)
        if not calls:
            calls = [{"name": "query_inventory", "arguments": {"sku_ids": sku_ids}, "source": "rule"}]
        return {"tool_calls": calls}

    async def tool(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        degraded = False
        for call in state.get("tool_calls") or []:
            arguments = call.get("arguments") or {}
            await emit(EventType.TOOL_CALL, tool=call["name"], arguments=arguments, source=call.get("source", "rule"))
            tool_result = await deps.inventory.query_inventory(arguments.get("sku_ids") or [], state.get("session_id", "-"))
            degraded = degraded or tool_result.degraded
            results.append(
                {
                    "tool": tool_result.tool,
                    "ok": tool_result.ok,
                    "latency_ms": tool_result.latency_ms,
                    "degraded": tool_result.degraded,
                    "error": tool_result.error,
                    "data": tool_result.data,
                }
            )
            await emit(
                EventType.TOOL_RESULT,
                tool=tool_result.tool,
                ok=tool_result.ok,
                latency_ms=tool_result.latency_ms,
                degraded=tool_result.degraded,
                error=tool_result.error,
                items=(tool_result.data.get("items") or [])[:3],
            )

        cards = [dict(c) for c in (state.get("candidates") or [])]
        live: dict[str, dict[str, Any]] = {}
        for res in results:
            for item in res["data"].get("items") or []:
                live[item.get("sku_id")] = item
        for card in cards:
            info = live.get(card["sku_id"])
            if info:
                card["stock_live"] = info.get("stock")
                card["available"] = info.get("available")
                card["warehouse"] = info.get("warehouse")
                card["eta_days"] = info.get("eta_days")
        timings = {**(state.get("timings") or {}), "tool_ms": round((time.perf_counter() - started) * 1000, 2)}
        return {"tool_results": results, "degraded": degraded, "candidates": cards, "timings": timings}

    async def respond(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        slots = Slots(**(state.get("slots") or {}))
        cards = state.get("candidates") or []
        confidence = float(state.get("confidence") or 0.0)
        relaxed = state.get("relaxed") or []
        degraded = bool(state.get("degraded"))
        intent = state.get("intent", "recommend")
        history = state.get("history") or []
        clarify: str | None = None

        if intent == "chitchat":
            answer = "我是小导，专门帮你挑 3C 数码。说个品类、预算或者用途，我就能开始推荐，比如「5000 以内拍照好的手机」。"
        elif intent == "aftersales":
            answer = "售后问题我帮你转人工客服：订单页点「申请售后」即可，7 天内支持无理由，保修凭发票。需要我先帮你查一下同款的其他购买建议吗？"
        elif confidence < 0.25 or (not slots.category and not slots.brand and confidence < settings.confidence_threshold):
            # 连品类都没识别出来时，给一堆随机商品不如直接问清楚
            clarify = "你想看哪类产品？手机 / 耳机 / 笔记本 / 相机都行，顺便说下预算，我就能直接给你排。"
            answer = clarify
        else:
            if deps.llm.enabled:
                messages_fn = prompts.compare_messages if intent == "compare" else None
                messages = (
                    messages_fn(state["query"], cards)
                    if messages_fn
                    else prompts.respond_messages(state["query"], slots.to_dict(), cards, state.get("tool_results") or [], history)
                )
                buffer = ""
                try:
                    async for token in deps.llm.stream(messages):
                        buffer += token
                        await emit(EventType.DELTA, text=token)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("模型生成失败，回退模板: %s", exc)
                    buffer = ""
                if buffer.strip():
                    answer = buffer.strip()
                else:
                    answer = _template_answer(slots, cards, relaxed, degraded, slots.missing())
                    for chunk in _stream_chunks(answer):
                        await emit(EventType.DELTA, text=chunk)
            else:
                answer = _template_answer(slots, cards, relaxed, degraded, slots.missing())
                for chunk in _stream_chunks(answer):
                    await emit(EventType.DELTA, text=chunk)
                    await asyncio.sleep(0)

        if clarify:
            await emit(EventType.CLARIFY, question=clarify, missing=slots.missing())
        if cards and not clarify:
            await emit(EventType.PRODUCTS, products=cards[:4])

        timings = {**(state.get("timings") or {}), "respond_ms": round((time.perf_counter() - started) * 1000, 2)}
        return {"answer": answer, "clarify": clarify, "timings": timings}

    return {
        "understand": understand,
        "retrieve": retrieve,
        "rewrite": rewrite,
        "plan": plan,
        "tool": tool,
        "respond": respond,
    }

"""Prompt 模板。

三个环节用到模型：需求解析、Query 改写、答案生成。
每个 prompt 都尽量约束输出格式，避免下游解析失败；解析失败时 Agent 会自动降级到规则分支。
"""

from __future__ import annotations

import json
from typing import Any

from app.data.taxonomy import CATEGORIES

CATEGORY_LIST = "、".join(CATEGORIES.keys())
ATTRIBUTE_LIST = "拍照、续航、游戏、轻薄、降噪、高刷、快充、音质、办公、学生、大内存、防水、运动、健康、长续航、4K、大屏、性价比、旗舰、送礼、户外、视频、便携"

UNDERSTAND_SYSTEM = f"""你是电商导购系统的需求解析器。从用户最新一句话中抽取购物条件，只输出 JSON，不要解释。

字段：
- category: 只能取 [{CATEGORY_LIST}]，无法判断填 null
- brand: 品牌中文名，无法判断填 null
- price_min / price_max: 数字（单位元），用户说"5000以内"就填 price_max=5000，"3000左右"填 price_max=3500
- attributes: 从 [{ATTRIBUTE_LIST}] 里选，最多 3 个，没有给空数组
- scene: 送礼/学生/商务/家庭/户外，没有填 null
- exclude_brands: 用户明确不要的品牌
- intent: recommend（选购）/ compare（对比）/ aftersales（售后）/ chitchat（闲聊）

注意：用户这句话如果是补充上一条需求（比如"再便宜点""要华为的"），只输出这次提到的字段。
输出示例：{{"category": "手机", "brand": null, "price_min": null, "price_max": 5000, "attributes": ["拍照"], "scene": null, "exclude_brands": [], "intent": "recommend"}}"""

REWRITE_SYSTEM = """你是电商搜索的 Query 改写器。用户原始需求检索效果不好，请改写成一串更适合关键词检索的中文短语。

要求：
- 保留品类、品牌、价格区间、核心卖点；
- 补充同义词与常见规格词（例如"拍照好"→"拍照 影像 大底 长焦"）；
- 输出一行纯文本，不要解释，不要标点符号堆砌，不超过 40 字。"""

RESPOND_SYSTEM = """你是电商导购助手"小导"。根据检索到的真实商品回答用户，语气自然、具体、不夸张。

硬性要求：
1. 第一句话复述你理解到的需求（品类、预算、核心诉求）；
2. 推荐 2~3 款，每款必须写出真实标题关键词、价格，并用一句话说明为什么适合；
3. 只能使用提供给你的商品信息，禁止编造参数、库存、优惠；
4. 如果结果不完全匹配（例如价格超预算、品类放宽），必须直说；
5. 结尾用一句简短的话追问一个关键信息，帮助缩小范围；
6. 总长度不超过 220 字，用简短分段，不要用表格。"""

COMPARE_SYSTEM = """你是电商导购助手。用户想对比商品，请基于提供的商品参数做客观对比：
先给一句结论，再分 2~3 点说差异（价格/参数/适用人群），最后给一句购买建议。不超过 200 字，禁止编造参数。"""


def understand_messages(query: str, history: list[dict[str, str]], current_slots: dict[str, Any]) -> list[dict[str, str]]:
    context = ""
    if history:
        recent = history[-4:]
        context = "最近对话：\n" + "\n".join(f"{m['role']}: {m['content']}" for m in recent) + "\n"
    known = json.dumps({k: v for k, v in current_slots.items() if v}, ensure_ascii=False)
    return [
        {"role": "system", "content": UNDERSTAND_SYSTEM},
        {"role": "user", "content": f"{context}已知条件：{known}\n用户最新一句：{query}"},
    ]


def rewrite_messages(query: str, slots: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"原始需求：{query}\n已解析条件：{json.dumps(slots, ensure_ascii=False)}"},
    ]


def _product_block(candidates: list[dict[str, Any]], limit: int = 4) -> str:
    lines: list[str] = []
    for idx, card in enumerate(candidates[:limit], start=1):
        specs = "，".join(f"{k}{v}" for k, v in list(card.get("specs", {}).items())[:4])
        lines.append(
            f"{idx}. {card['title']} | 价格 {int(card['price'])} 元 | 评分 {card['rating']} | "
            f"库存 {card['stock']} | 标签 {'/'.join(card['tags'][:3])} | {specs}"
        )
    return "\n".join(lines)


def respond_messages(query: str, slots: dict[str, Any], candidates: list[dict[str, Any]],
                     tool_results: list[dict[str, Any]], history: list[dict[str, str]]) -> list[dict[str, str]]:
    history_text = ""
    if history:
        history_text = "最近对话：\n" + "\n".join(
            f"{m['role']}: {m['content'][:120]}" for m in history[-4:]
        ) + "\n\n"
    stock_note = ""
    if tool_results:
        stock_note = "实时库存工具返回：\n" + json.dumps(tool_results, ensure_ascii=False) + "\n\n"
    return [
        {"role": "system", "content": RESPOND_SYSTEM},
        {
            "role": "user",
            "content": (
                f"{history_text}用户需求：{query}\n"
                f"解析出的条件：{json.dumps(slots, ensure_ascii=False)}\n\n"
                f"检索到的商品：\n{_product_block(candidates)}\n\n{stock_note}"
                "请给出推荐。"
            ),
        },
    ]


def compare_messages(query: str, candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COMPARE_SYSTEM},
        {"role": "user", "content": f"用户需求：{query}\n候选商品：\n{_product_block(candidates, 3)}"},
    ]

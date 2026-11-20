"""检索评测集的构造。

评测集不是手写死标准答案，而是"约束 → 相关集"：
给定品类/品牌/价格/需求点，商品库里所有满足硬约束、且标签里带有所需卖点的商品
都算相关。这样数据集会随着商品库自动更新，也能覆盖长尾。

需求点用"标签"判定（严格口径），而不是标题里出现"主摄"这种宽泛文本命中，
否则几乎所有手机都会被算成"拍照好"，指标会虚高到 1.0、失去区分度。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ATTRS_BY_CATEGORY: dict[str, list[str]] = {
    "手机": ["拍照", "续航", "游戏", "轻薄", "快充"],
    "笔记本电脑": ["轻薄", "游戏", "办公", "学生"],
    "平板电脑": ["办公", "大屏", "便携"],
    "无线耳机": ["降噪", "音质", "运动"],
    "智能手表": ["运动", "健康", "长续航"],
    "相机": ["拍照", "视频"],
    "蓝牙音箱": ["音质", "户外", "大功率"],
    "机械键盘": ["办公", "游戏", "无线"],
    "游戏鼠标": ["游戏", "轻量化", "无线"],
    "显示器": ["游戏", "4K", "办公"],
}

TEMPLATES = (
    ("{attr}好的{category}，预算{price}元", lambda c, a, p, b: {"category": c, "attributes": [a], "price_max": p}),
    ("{price}以内的{category}有什么推荐", lambda c, a, p, b: {"category": c, "price_max": p}),
    ("想买个{brand}的{category}", lambda c, a, p, b: {"category": c, "brand": b}),
    ("{category}，要{attr}，{price}元左右", lambda c, a, p, b: {"category": c, "attributes": [a], "price_max": p}),
    ("学生党用{category}，{price}元封顶", lambda c, a, p, b: {"category": c, "price_max": p, "scene": "学生"}),
    ("{brand}的{category}，要{attr}", lambda c, a, p, b: {"category": c, "brand": b, "attributes": [a]}),
)


def _price_quantiles(docs: list[dict[str, Any]], category: str) -> list[int]:
    prices = sorted(d["price"] for d in docs if d["category"] == category)
    if not prices:
        return [3000, 5000, 8000]
    return [int(round(prices[int(len(prices) * q)] / 100) * 100) for q in (0.35, 0.55, 0.75)]


def relevant_ids(docs: list[dict[str, Any]], slots: dict[str, Any]) -> set[str]:
    attrs = list(slots.get("attributes") or [])
    out: set[str] = set()
    for doc in docs:
        if slots.get("category") and doc["category"] != slots["category"]:
            continue
        if slots.get("brand") and doc["brand"] != slots["brand"]:
            continue
        if slots.get("price_max") and doc["price"] > slots["price_max"]:
            continue
        if slots.get("price_min") and doc["price"] < slots["price_min"]:
            continue
        if doc["brand"] in (slots.get("exclude_brands") or []):
            continue
        if any(attr not in doc["tags"] for attr in attrs):
            continue
        out.add(doc["sku_id"])
    return out


def _narrow(pool: list[dict[str, Any]], base: dict[str, Any], attrs: list[str],
            rng: random.Random) -> dict[str, Any] | None:
    """把约束逐步收紧，让相关集落在 [5, 120] 这个可评测的区间。

    相关集几千条时，任何 10 条都"正确"，指标恒为 1.0，没有信息量；
    真正能区分召回质量的是"小相关集 + 长尾约束"的查询。
    """
    kept = [d for d in pool if (not base.get("brand") or d["brand"] == base["brand"])]
    if len(kept) < 5:
        return None
    if len(kept) <= 120:
        return base

    for attr in rng.sample(attrs, k=min(2, len(attrs))):
        candidate = {**base, "attributes": [attr]}
        rel = relevant_ids(pool, candidate)
        if 5 <= len(rel) <= 120:
            return candidate
        if len(rel) > 120:
            prices = sorted(d["price"] for d in pool if d["sku_id"] in rel)
            for p0, p1 in ((0.15, 0.45), (0.3, 0.6), (0.45, 0.8)):
                low, high = prices[int(len(prices) * p0)], prices[int(len(prices) * p1)]
                narrowed = {**candidate, "price_min": low, "price_max": high}
                if 5 <= len(relevant_ids(pool, narrowed)) <= 120:
                    return narrowed
    return None


def _render(slots: dict[str, Any]) -> str:
    parts: list[str] = []
    if slots.get("brand"):
        parts.append(slots["brand"])
    parts.append(slots["category"])
    if slots.get("price_min") and slots.get("price_max"):
        parts.append(f"{int(slots['price_min'])}到{int(slots['price_max'])}元")
    elif slots.get("price_max"):
        parts.append(f"{int(slots['price_max'])}元以内")
    if slots.get("attributes"):
        parts.append("要" + "、".join(slots["attributes"]))
    return "，".join(parts) + "，有什么推荐"


SEMANTIC_QUERIES: list[tuple[str, dict[str, Any]]] = [
    ("想买个能拍月亮拍演唱会的手机", {"category": "手机", "attributes": ["拍照"]}),
    ("宿舍里打游戏不卡的本子", {"category": "笔记本电脑", "attributes": ["游戏"]}),
    ("地铁通勤想把噪音隔掉", {"category": "无线耳机", "attributes": ["降噪"]}),
    ("给爸妈买块能测心率的表", {"category": "智能手表", "attributes": ["健康"]}),
    ("出门拍 vlog 用的小相机", {"category": "相机", "attributes": ["视频"]}),
    ("躺床上追剧想要大一点的屏幕", {"category": "平板电脑", "attributes": ["大屏"]}),
    ("经常出差想要背着轻的电脑", {"category": "笔记本电脑", "attributes": ["轻薄"]}),
    ("跑步的时候戴不容易掉的耳机", {"category": "无线耳机", "attributes": ["运动"]}),
    ("写代码想要手感好的键盘", {"category": "机械键盘", "attributes": ["办公"]}),
    ("打游戏要屏幕跟得上手", {"category": "显示器", "attributes": ["游戏"]}),
]


def _narrow_semantic(pool: list[dict[str, Any]], slots: dict[str, Any]) -> dict[str, Any] | None:
    """给语义查询加一个价格带，把相关集压到可评测的规模。

    语义查询的难点在"需求点没被说出来"，不在价格；加价格约束既能保持自然，
    又能让 recall@10 有意义（相关集几千条时怎么排都是满分）。
    """
    attrs = slots.get("attributes") or []
    tagged = [d for d in pool if all(a in d["tags"] for a in attrs)]
    if len(tagged) < 20:
        return None
    prices = sorted(d["price"] for d in tagged)
    step = max(1, len(prices) // 8)
    for i in range(0, len(prices) - 60, step):
        low, high = prices[i], prices[i + 60]
        if low < 50 or high - low < 50:  # 区间过窄或价格异常时跳过
            continue
        candidate = {**slots, "price_min": low, "price_max": high}
        size = len(relevant_ids(pool, candidate))
        if 15 <= size <= 120:
            return candidate
    return None


def build_dataset(docs: list[dict[str, Any]], seed: int = 7, per_category: int = 5,
                  min_relevant: int = 5) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        by_category.setdefault(doc["category"], []).append(doc)

    dataset: list[dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0
    for category, attrs in ATTRS_BY_CATEGORY.items():
        pool = by_category.get(category, [])
        if not pool:
            continue
        brands = sorted({d["brand"] for d in pool})
        for i in range(per_category):
            for attempt in range(10):
                base: dict[str, Any] = {"category": category}
                if (i + attempt) % 2 == 0:
                    base["brand"] = rng.choice(brands)
                slots = _narrow(pool, base, attrs, rng)
                if not slots:
                    continue
                rel = relevant_ids(pool, slots)
                if len(rel) < min_relevant:
                    continue
                key = json.dumps(slots, sort_keys=True, ensure_ascii=False)
                if key in seen:
                    continue
                seen.add(key)
                idx += 1
                dataset.append({"id": f"q{idx:03d}", "query": _render(slots), "slots": slots,
                                "relevant": len(rel), "kind": "keyword"})
                break

    # 语义型查询：query 里不出现卖点词，只能靠向量/同义词理解
    for query, base in SEMANTIC_QUERIES:
        pool = by_category.get(base["category"], [])
        if not pool:
            continue
        slots = _narrow_semantic(pool, base)
        if not slots:
            continue
        rel = relevant_ids(pool, slots)
        if len(rel) < min_relevant:
            continue
        key = json.dumps(slots, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        idx += 1
        text = f"{query}，{int(slots['price_min'])} 到 {int(slots['price_max'])} 元"
        dataset.append({"id": f"q{idx:03d}", "query": text, "slots": slots,
                        "relevant": len(rel), "kind": "semantic"})
    return dataset


def save_dataset(dataset: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

"""槽位定义与抽取。

双通道设计：
- 正则通道：价格、品牌、品类、否定词这类"硬信息"，规则稳定、可解释、不花钱；
- LLM 通道：把"拍照好""适合学生"这类模糊表述归一到标准需求点，并补出正则漏掉的槽位。

合并策略：数值与品牌以正则为准（避免模型把 5000 说成 5499），
语义属性取并集，否定约束只增不减。
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.data.taxonomy import CATEGORIES, CATEGORY_ALIASES

CATEGORY_LOOKUP: list[tuple[str, str]] = []
for _cat, _aliases in CATEGORY_ALIASES.items():
    for _alias in {_cat, *_aliases}:
        CATEGORY_LOOKUP.append((_alias.lower(), _cat))
CATEGORY_LOOKUP.sort(key=lambda item: -len(item[0]))

BRAND_LOOKUP: list[tuple[str, str]] = []
for _cat in CATEGORIES.values():
    for _brand in _cat.brands:
        for _alias in {_brand.name, *_brand.aliases}:
            BRAND_LOOKUP.append((_alias.lower(), _brand.name))
BRAND_LOOKUP = sorted({(a.lower(), b) for a, b in BRAND_LOOKUP}, key=lambda item: -len(item[0]))

ATTRIBUTE_LEXICON: dict[str, tuple[str, ...]] = {
    "拍照": ("拍照", "影像", "相机好", "拍人", "拍夜景", "出片", "摄影"),
    "续航": ("续航", "电池大", "耐用", "电量"),
    "快充": ("快充", "充电快", "闪充"),
    "游戏": ("游戏", "打游戏", "电竞", "王者", "原神", "帧率"),
    "轻薄": ("轻薄", "轻便", "便携", "不重", "重量轻", "小巧"),
    "降噪": ("降噪", "隔音", "ANC"),
    "高刷": ("高刷", "高刷新率", "120Hz", "144Hz", "165Hz"),
    "音质": ("音质", "音效", "听歌", "低音", "hifi"),
    "办公": ("办公", "工作", "商务", "写文档"),
    "学生": ("学生", "学生党", "上课", "网课", "宿舍"),
    "大内存": ("大内存", "内存大", "多开", "不卡"),
    "防水": ("防水", "防泼溅", "游泳"),
    "运动": ("运动", "跑步", "健身", "骑行"),
    "健康": ("健康", "心率", "血氧", "睡眠"),
    "长续航": ("长续航", "续航久", "不用天天充"),
    "4K": ("4k", "高清", "高分"),
    "大屏": ("大屏", "屏幕大"),
    "性价比": ("性价比", "便宜", "实惠", "预算有限"),
    "旗舰": ("旗舰", "顶配", "高端"),
    "送礼": ("送礼", "送人", "礼物", "生日", "情人节"),
    "户外": ("户外", "露营", "防水防尘"),
}

PRICE_RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|~|到|至)\s*(\d+(?:\.\d+)?)\s*(万|w|k|千)?")
PRICE_BUDGET_RE = re.compile(
    r"(?:预算|价格|价位)?\s*(\d+(?:\.\d+)?)\s*(万|w|k|千)?\s*(?:元|块)?\s*(以内|以下|之内|左右|上下|出头|封顶)"
)
PRICE_PREFIX_RE = re.compile(r"(?:预算|价格)\s*(?:在|是|为)?\s*(\d+(?:\.\d+)?)\s*(万|w|k|千)?")
UNIT_MULTIPLIER = {"万": 10000, "w": 10000, "k": 1000, "千": 1000}


class Slots(BaseModel):
    category: str | None = None
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    attributes: list[str] = Field(default_factory=list)
    scene: str | None = None
    exclude_brands: list[str] = Field(default_factory=list)
    intent: str = "recommend"
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def queryable(self) -> dict[str, Any]:
        return self.model_dump(include={"category", "brand", "price_min", "price_max", "attributes", "scene", "exclude_brands"})

    def missing(self) -> list[str]:
        missing: list[str] = []
        if not self.category:
            missing.append("category")
        if not self.brand and not self.price_max and not self.attributes:
            missing.append("preference")
        return missing

    def describe(self) -> str:
        bits = []
        if self.brand:
            bits.append(self.brand)
        if self.category:
            bits.append(self.category)
        if self.price_max and self.price_min:
            bits.append(f"{int(self.price_min)}-{int(self.price_max)}元")
        elif self.price_max:
            bits.append(f"{int(self.price_max)}元以内")
        elif self.price_min:
            bits.append(f"{int(self.price_min)}元以上")
        if self.attributes:
            bits.append("、".join(self.attributes))
        if self.exclude_brands:
            bits.append("排除" + "、".join(self.exclude_brands))
        return " · ".join(bits) if bits else "还没有明确条件"


def extract_by_rules(text: str) -> Slots:
    lowered = text.lower()
    slots = Slots()

    for alias, category in CATEGORY_LOOKUP:
        if alias in lowered:
            slots.category = category
            break

    for alias, brand in BRAND_LOOKUP:
        if alias in lowered:
            slots.brand = brand
            break

    range_match = PRICE_RANGE_RE.search(lowered)
    if range_match:
        mult = UNIT_MULTIPLIER.get((range_match.group(3) or "").lower(), 1)
        low, high = float(range_match.group(1)) * mult, float(range_match.group(2)) * mult
        if 10 <= low < high:
            slots.price_min, slots.price_max = low, high
    if slots.price_max is None:
        budget_match = PRICE_BUDGET_RE.search(lowered) or PRICE_PREFIX_RE.search(lowered)
        if budget_match:
            mult = UNIT_MULTIPLIER.get((budget_match.group(2) or "").lower(), 1)
            value = float(budget_match.group(1)) * mult
            suffix = budget_match.group(3) if budget_match.re is PRICE_BUDGET_RE else None
            if value >= 10:
                if suffix in ("左右", "上下"):
                    slots.price_min = round(value * 0.7, 0)
                    slots.price_max = round(value * 1.2, 0)
                else:
                    slots.price_max = value

    for attr, phrases in ATTRIBUTE_LEXICON.items():
        if any(p.lower() in lowered for p in phrases):
            if attr not in slots.attributes:
                slots.attributes.append(attr)

    if any(p in text for p in ("不要", "别买", "除了", "不想要")):
        for alias, brand in BRAND_LOOKUP:
            idx = lowered.find(alias)
            if idx == -1:
                continue
            window = lowered[max(0, idx - 6) : idx]
            if any(p in window for p in ("不要", "别买", "除了", "不想要")):
                slots.exclude_brands.append(brand)
                if slots.brand == brand:
                    slots.brand = None

    if any(p in text for p in ("送礼", "送人", "礼物", "生日", "情人节")):
        slots.scene = "送礼"
    elif any(p in text for p in ("学生", "宿舍", "网课")):
        slots.scene = "学生"
    elif any(p in text for p in ("出差", "通勤", "商务")):
        slots.scene = "商务"
    elif any(p in text for p in ("家用", "家庭", "客厅")):
        slots.scene = "家庭"

    slots.attributes = list(dict.fromkeys(slots.attributes))[:5]
    slots.exclude_brands = list(dict.fromkeys(slots.exclude_brands))
    return slots


def merge_slots(old: Slots, new: Slots) -> Slots:
    """新槽位覆盖硬字段，语义属性取并集；价格后出现的覆盖先出现的。"""
    merged = old.model_copy(deep=True)
    if new.category:
        merged.category = new.category
    if new.brand:
        merged.brand = new.brand
    if new.price_min is not None:
        merged.price_min = new.price_min
    if new.price_max is not None:
        merged.price_max = new.price_max
    if new.price_min is None and new.price_max is None:
        pass
    if new.scene:
        merged.scene = new.scene
    for attr in new.attributes:
        if attr not in merged.attributes:
            merged.attributes.append(attr)
    merged.attributes = merged.attributes[:6]
    for brand in new.exclude_brands:
        if brand not in merged.exclude_brands:
            merged.exclude_brands.append(brand)
    if merged.brand in merged.exclude_brands:
        merged.brand = None
    return merged

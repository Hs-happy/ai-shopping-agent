"""重排。

两档实现：
- FeatureReranker：把词面匹配、语义相似、约束满足、商品质量做成可解释的线性特征，
  零依赖、毫秒级，默认用它；
- CrossEncoderReranker：fastembed 加载 bge-reranker-base 做 query-passage 交互打分，
  效果更好但需要下载模型，RERANK_MODE=cross_encoder 时启用。

注意：特征重排用的是"绝对分"而不是"批内归一化分"。
批内归一化会让每次查询的 top1 都拿到 1.0，重排分失去区分度，
置信度也就永远很高、触发不了改写——这是很多 RAG demo 的通病。
"""

from __future__ import annotations

import math
from typing import Any, Protocol

from app.retrieval.types import Candidate

_ATTR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "拍照": ("拍照", "影像", "相机", "像素", "主摄", "长焦"),
    "续航": ("续航", "电池", "mAh"),
    "游戏": ("游戏", "电竞", "帧率", "性能", "显卡"),
    "轻薄": ("轻薄", "重量", "便携", "kg"),
    "降噪": ("降噪", "ANC", "噪音"),
    "高刷": ("高刷", "刷新率", "Hz"),
    "快充": ("快充", "充电", "W"),
    "办公": ("办公", "商务", "office"),
    "学生": ("学生", "网课", "性价比"),
    "音质": ("音质", "单元", "调音"),
    "大内存": ("内存", "运存", "GB"),
    "防水": ("防水", "IPX", "IP68"),
    "运动": ("运动", "跑步", "健身"),
    "健康": ("健康", "心率", "血氧"),
    "长续航": ("长续航", "续航", "天"),
    "4K": ("4K", "3840", "2160"),
    "大屏": ("大屏", "尺寸", "英寸"),
    "性价比": ("性价比", "便宜", "实惠"),
    "旗舰": ("旗舰", "顶配", "高端"),
    "送礼": ("送礼", "礼盒", "限定"),
    "户外": ("户外", "防水", "防尘"),
    "视频": ("视频", "4K", "8K", "Vlog"),
    "便携": ("便携", "小巧", "重量"),
    "办公": ("办公", "商务", "接口"),
}

CONSTRAINT_WEIGHTS = {"category": 0.45, "brand": 0.20, "price": 0.20, "attributes": 0.15}


def _price_fit(price: float, low: float | None, high: float | None) -> float:
    if low and price < low:
        return math.exp(-((low - price) / max(low, 1.0)) * 2.0)
    if high and price > high:
        return math.exp(-((price - high) / max(high, 1.0)) * 2.0)
    return 1.0


def _attribute_hits(doc: dict[str, Any], attrs: list[str]) -> tuple[int, int]:
    """需求点命中：优先看人工/规则打的标签，标签没有再退回标题文本。

    只看规格字段名会误判——"降噪深度: 无主动降噪"里也含"降噪"，
    会让所有耳机都被算成"降噪耳机"。
    """
    if not attrs:
        return 0, 0
    tags = set(doc.get("tags") or [])
    title = doc.get("title", "").lower()
    hit = 0
    for attr in attrs:
        if attr in tags:
            hit += 1
            continue
        keys = _ATTR_SYNONYMS.get(attr, (attr,))
        if any(key.lower() in title for key in keys):
            hit += 1
    return hit, len(attrs)


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Candidate], slots: dict[str, Any], top_k: int) -> list[Candidate]: ...


class FeatureReranker:
    name = "feature"

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {"lexical": 0.28, "semantic": 0.22, "constraint": 0.32, "quality": 0.18}

    @staticmethod
    def _lexical(bm25_score: float) -> float:
        """BM25 原始分做饱和映射：10 分左右对应 0.63，20 分对应 0.86。"""
        return 1.0 - math.exp(-max(bm25_score, 0.0) / 10.0)

    @staticmethod
    def _semantic(vec_score: float) -> float:
        """余弦相似度映射：hashing/bge 的分布不同，这里按经验把 0.12~0.57 拉到 0~1。"""
        return min(1.0, max(0.0, (vec_score - 0.12) / 0.45))

    def rerank(self, query: str, candidates: list[Candidate], slots: dict[str, Any], top_k: int) -> list[Candidate]:
        if not candidates:
            return []
        brand = (slots.get("brand") or "").strip()
        category = (slots.get("category") or "").strip()
        low, high = slots.get("price_min"), slots.get("price_max")
        attrs = list(slots.get("attributes") or [])
        negatives = set(slots.get("exclude_brands") or [])
        has_constraint = bool(brand or category or low or high or attrs)

        for cand in candidates:
            doc = cand.doc
            price_fit = _price_fit(float(doc["price"]), low, high)
            category_fit = 1.0 if (not category or doc["category"] == category) else 0.0
            brand_fit = 1.0 if (not brand or doc["brand"] == brand) else 0.0
            attr_hit, attr_total = _attribute_hits(doc, attrs)
            attr_fit = (attr_hit / attr_total) if attr_total else 0.5

            parts: list[tuple[float, float]] = []
            if category:
                parts.append((CONSTRAINT_WEIGHTS["category"], category_fit))
            if brand:
                parts.append((CONSTRAINT_WEIGHTS["brand"], brand_fit))
            if low or high:
                parts.append((CONSTRAINT_WEIGHTS["price"], price_fit))
            if attrs:
                parts.append((CONSTRAINT_WEIGHTS["attributes"], attr_fit))
            constraint = sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else 0.5

            quality = 0.6 * (float(doc["rating"]) / 5.0) + 0.4 * min(
                1.0, math.log1p(doc["sales"]) / math.log1p(99_999)
            )
            lexical = self._lexical(cand.bm25_score)
            semantic = self._semantic(cand.vec_score)
            score = (
                self.weights["lexical"] * lexical
                + self.weights["semantic"] * semantic
                + (self.weights["constraint"] if has_constraint else 0.0) * constraint
                + self.weights["quality"] * quality
            )
            if not has_constraint:
                score += 0.16  # 无约束时给个中性基线，避免"什么都没匹配"也拿高分
            if doc["brand"] in negatives:
                score -= 0.5
            if int(doc.get("stock") or 0) <= 0:
                score -= 0.05

            reasons: list[str] = []
            if category and category_fit:
                reasons.append(f"品类匹配 {category}")
            if brand and brand_fit:
                reasons.append(f"品牌匹配 {brand}")
            if (low or high) and price_fit > 0.99:
                reasons.append(f"¥{int(doc['price'])} 在预算内")
            if attr_total and attr_hit:
                reasons.append(f"满足 {attr_hit}/{attr_total} 个需求点")
            if float(doc["rating"]) >= 4.6:
                reasons.append("高评分")
            if int(doc.get("stock") or 0) > 0:
                reasons.append("有货")

            cand.features = {
                "lexical": round(lexical, 4),
                "semantic": round(semantic, 4),
                "constraint": round(constraint, 4),
                "quality": round(quality, 4),
            }
            cand.rerank_score = round(score, 6)
            cand.reasons = reasons[:4]

        candidates.sort(key=lambda c: -c.rerank_score)
        return candidates[:top_k]


class CrossEncoderReranker:
    name = "bge-reranker"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)
        self._fallback = FeatureReranker()

    def rerank(self, query: str, candidates: list[Candidate], slots: dict[str, Any], top_k: int) -> list[Candidate]:
        if not candidates:
            return []
        docs = [f"{c.doc['title']} {' '.join(c.doc.get('tags') or [])}" for c in candidates]
        scores = list(self._model.rerank(query, docs))
        lo, hi = min(scores), max(scores)
        span = (hi - lo) or 1.0
        for cand, score in zip(candidates, scores):
            cand.rerank_score = 0.75 * ((score - lo) / span) + 0.25 * cand.features.get("constraint", 0.5)
            cand.reasons = cand.reasons[:2]
        candidates.sort(key=lambda c: -c.rerank_score)
        return candidates[:top_k]


def build_reranker(mode: str = "feature") -> Reranker:
    if mode == "cross_encoder":
        try:
            return CrossEncoderReranker()
        except Exception:  # noqa: BLE001 - 模型没装就退回特征重排，不影响服务
            return FeatureReranker()
    return FeatureReranker()

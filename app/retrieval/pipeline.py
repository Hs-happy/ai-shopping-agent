"""检索流水线：双路召回 → RRF 融合 → 截断 → 重排 → 约束过滤 → 置信度。

置信度不是拍脑袋的常数，而是三件事的加权：
1. 头部绝对分：top1 的重排分；
2. 头部区分度：top1 与 top2 的差距，差距小说明召回模糊；
3. 约束覆盖：品类/品牌/价格/需求点这些硬条件在前三条里的满足比例。
低于阈值就触发 Query 改写与二次检索（由 Agent 层决定）。
"""

from __future__ import annotations

import time
from typing import Any

from app.config import Settings
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import rrf
from app.retrieval.rerank import Reranker
from app.retrieval.types import Candidate, SearchResult
from app.retrieval.vector import VectorIndex


def build_query_text(query: str, slots: dict[str, Any]) -> str:
    parts = [query.strip()]
    if slots.get("brand"):
        parts.append(str(slots["brand"]))
    if slots.get("category"):
        parts.append(str(slots["category"]))
    for attr in slots.get("attributes") or []:
        parts.append(str(attr))
    if slots.get("scene"):
        parts.append(str(slots["scene"]))
    return " ".join(p for p in parts if p)


class RetrievalPipeline:
    def __init__(self, bm25: BM25Index, vector: VectorIndex, reranker: Reranker, settings: Settings):
        self.bm25 = bm25
        self.vector = vector
        self.reranker = reranker
        self.settings = settings

    # ------------------------------------------------------------------ search
    def search(self, query: str, slots: dict[str, Any] | None = None, top_k: int | None = None) -> SearchResult:
        slots = slots or {}
        top_k = top_k or self.settings.final_k
        query_text = build_query_text(query, slots)
        stats: dict[str, Any] = {"query_text": query_text, "relaxed": []}
        started = time.perf_counter()

        t0 = time.perf_counter()
        bm25_hits = self.bm25.search(query_text, self.settings.recall_k)
        bm25_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        vec_hits = self.vector.search(query_text, self.settings.recall_k)
        vec_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fused = rrf(
            [bm25_hits, vec_hits],
            k=self.settings.rrf_k,
            weights=[self.settings.rrf_weight_bm25, self.settings.rrf_weight_vector],
        )
        candidates: list[Candidate] = []
        bm25_rank = {idx: rank for rank, (idx, _) in enumerate(bm25_hits, 1)}
        bm25_score = dict(bm25_hits)
        vec_rank = {idx: rank for rank, (idx, _) in enumerate(vec_hits, 1)}
        vec_score = dict(vec_hits)
        # 只被 BM25 召回的候选没有向量分，重排时会吃亏；这里给融合候选补算余弦相似度
        missing = [idx for idx, _ in fused if idx not in vec_score]
        if missing and len(self.vector.matrix):
            qvec = self.vector.embedder.embed(query_text)
            for doc_idx in missing:
                vec_score[doc_idx] = float(self.vector.matrix[doc_idx] @ qvec)
        docs = self._docs
        for doc_idx, score in fused:
            candidates.append(
                Candidate(
                    doc=docs[doc_idx],
                    bm25_rank=bm25_rank.get(doc_idx),
                    bm25_score=bm25_score.get(doc_idx, 0.0),
                    vec_rank=vec_rank.get(doc_idx),
                    vec_score=vec_score.get(doc_idx, 0.0),
                    rrf_score=score,
                )
            )
        fusion_ms = (time.perf_counter() - t0) * 1000

        # 关键：硬约束必须在大池子上先过滤再截断。
        # 如果先截断 30 条再过滤，很可能池子里一条符合预算的都没有，
        # 结果只能整体放宽——这就是"预算 2700 却推荐 6700 手机"的根因。
        candidates, relaxed = self._filter(candidates, slots)
        stats["relaxed"] = relaxed
        filtered_count = len(candidates)
        candidates = candidates[: self.settings.rerank_top_n]

        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query_text, candidates, slots, top_k)
        rerank_ms = (time.perf_counter() - t0) * 1000

        confidence = self._confidence(reranked, slots)
        stats.update(
            {
                "bm25_hits": len(bm25_hits),
                "vector_hits": len(vec_hits),
                "fused": filtered_count,
                "returned": len(reranked),
                "latency": {
                    "bm25_ms": round(bm25_ms, 2),
                    "vector_ms": round(vec_ms, 2),
                    "fusion_ms": round(fusion_ms, 2),
                    "rerank_ms": round(rerank_ms, 2),
                    "total_ms": round((time.perf_counter() - started) * 1000, 2),
                },
                "confidence": round(confidence, 4),
                "reranker": self.reranker.name,
            }
        )
        return SearchResult(query=query_text, hits=reranked, confidence=confidence, stats=stats, relaxed=relaxed)

    def bind_docs(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    # ------------------------------------------------------------------ filter
    def _filter(self, candidates: list[Candidate], slots: dict[str, Any]) -> tuple[list[Candidate], list[str]]:
        relaxed: list[str] = []
        negatives = set(slots.get("exclude_brands") or [])
        if negatives:
            candidates = [c for c in candidates if c.doc["brand"] not in negatives]

        brand = (slots.get("brand") or "").strip()
        if brand:
            kept = [c for c in candidates if c.doc["brand"] == brand]
            if kept:
                candidates = kept
            else:
                relaxed.append("brand")

        category = (slots.get("category") or "").strip()
        if category:
            kept = [c for c in candidates if c.doc["category"] == category]
            if kept:
                candidates = kept
            else:
                relaxed.append("category")

        low, high = slots.get("price_min"), slots.get("price_max")
        if low or high:
            def in_budget(cand: Candidate, slack: float = 1.0) -> bool:
                price = float(cand.doc["price"])
                if low and price < low * 0.9:
                    return False
                if high and price > high * slack:
                    return False
                return True

            kept = [c for c in candidates if in_budget(c)]
            if not kept:
                kept = [c for c in candidates if in_budget(c, 1.2)]
                relaxed.append("price_widened" if kept else "price")
            if kept:
                candidates = kept
        return candidates, relaxed

    # ------------------------------------------------------------------ confidence
    @staticmethod
    def _confidence(hits: list[Candidate], slots: dict[str, Any]) -> float:
        if not hits:
            return 0.0
        top = hits[0].rerank_score
        second = hits[1].rerank_score if len(hits) > 1 else top * 0.5
        margin = max(0.0, top - second) / max(top, 1e-6)
        head = hits[:3]

        checks: list[bool] = []
        if slots.get("category"):
            checks.append(all(c.doc["category"] == slots["category"] for c in head))
        if slots.get("brand"):
            checks.append(all(c.doc["brand"] == slots["brand"] for c in head))
        if slots.get("price_max"):
            checks.append(all(float(c.doc["price"]) <= slots["price_max"] for c in head))
        if slots.get("price_min"):
            checks.append(all(float(c.doc["price"]) >= slots["price_min"] for c in head))
        coverage = (sum(checks) / len(checks)) if checks else 0.5
        confidence = 0.55 * top + 0.15 * margin + 0.30 * coverage
        # 一个约束都没解析出来时，检索本质是在猜，不能给高置信度
        has_constraint = bool(
            slots.get("category") or slots.get("brand")
            or slots.get("price_min") or slots.get("price_max")
            or slots.get("attributes")
        )
        if not has_constraint:
            confidence *= 0.72
        return round(confidence, 4)

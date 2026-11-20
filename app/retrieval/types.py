"""检索层的公共数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Candidate:
    doc: dict[str, Any]
    bm25_rank: int | None = None
    bm25_score: float = 0.0
    vec_rank: int | None = None
    vec_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def card(self) -> dict[str, Any]:
        doc = self.doc
        return {
            "sku_id": doc["sku_id"],
            "title": doc["title"],
            "brand": doc["brand"],
            "category": doc["category"],
            "price": doc["price"],
            "rating": doc["rating"],
            "sales": doc["sales"],
            "stock": doc["stock"],
            "tags": doc["tags"],
            "specs": dict(list(doc["specs"].items())[:6]),
            "shop": doc.get("shop", ""),
            "score": round(self.rerank_score, 4),
            "reasons": self.reasons,
            "recall": {
                "bm25_rank": self.bm25_rank,
                "vector_rank": self.vec_rank,
                "rrf": round(self.rrf_score, 5),
            },
        }


@dataclass(slots=True)
class SearchResult:
    query: str
    hits: list[Candidate]
    confidence: float
    stats: dict[str, Any] = field(default_factory=dict)
    relaxed: list[str] = field(default_factory=list)

    @property
    def top(self) -> Candidate | None:
        return self.hits[0] if self.hits else None

    def cards(self) -> list[dict[str, Any]]:
        return [h.card() for h in self.hits]

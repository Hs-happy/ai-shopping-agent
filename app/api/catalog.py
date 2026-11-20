"""商品与系统信息接口。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, Query

router = APIRouter(prefix="/api", tags=["catalog"])


@router.get("/products/{sku_id}")
async def get_product(sku_id: str, request: Request) -> dict[str, Any]:
    agent = request.app.state.agent
    doc = agent.engine.get_product(sku_id)
    if not doc:
        raise HTTPException(status_code=404, detail="商品不存在")
    return doc


@router.get("/search")
async def search(request: Request, q: str = Query(min_length=1), top_k: int = 6) -> dict[str, Any]:
    agent = request.app.state.agent
    result = agent.engine.search(q, {}, top_k=top_k)
    return {"query": q, "confidence": result.confidence, "stats": result.stats, "products": result.cards()}


@router.get("/meta")
async def meta(request: Request) -> dict[str, Any]:
    agent = request.app.state.agent
    return {
        "app": agent.settings.app_name,
        "llm": {"enabled": agent.llm.enabled, "model": agent.llm.model, "base_url": agent.settings.llm_base_url},
        "retrieval": agent.engine.stats(),
        "facets": agent.db.facet_counts(),
        "tools": agent.db.tool_stats(),
        "thresholds": {
            "confidence": agent.settings.confidence_threshold,
            "tool_timeout_ms": agent.settings.tool_timeout_ms,
            "rrf_k": agent.settings.rrf_k,
            "recall_k": agent.settings.recall_k,
        },
    }


@router.post("/admin/rebuild-index")
async def rebuild(request: Request) -> dict[str, Any]:
    agent = request.app.state.agent
    agent.engine.rebuild()
    return {"ok": True, "stats": agent.engine.stats()}

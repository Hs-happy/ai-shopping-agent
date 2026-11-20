"""FastAPI 入口。

启动时装配：SQLite 商品库 → 检索索引（有缓存则秒加载）→ LangGraph 状态机 → MCP 库存子进程。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent.runner import ShoppingAgent
from app.api import catalog, chat
from app.config import settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("启动中：加载商品库与索引...")
    agent = ShoppingAgent(settings)
    await agent.start()
    app.state.agent = agent
    logger.info(
        "就绪",
        extra={"extra_fields": {
            "docs": len(agent.engine.docs),
            "llm": agent.llm.model if agent.llm.enabled else "mock",
            "embedding": getattr(agent.engine.embedder, "name", "-"),
            "mcp": agent.inventory.warm,
        }},
    )
    try:
        yield
    finally:
        await agent.close()


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat.router)
app.include_router(catalog.router)


@app.exception_handler(Exception)
async def unhandled(request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    logger.exception("未捕获异常: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


app.mount("/", StaticFiles(directory=str(settings.web_dir), html=True), name="web")


def main() -> None:
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()

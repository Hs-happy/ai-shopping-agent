"""全局配置。

所有可调参数集中在这里，通过环境变量 / .env 覆盖。
设计原则：
- 任何外部依赖（LLM、Embedding、ES/Milvus）都有离线兜底实现，保证 demo 随时可跑；
- 检索、重排、降级相关的阈值全部配置化，方便做 A/B 与评测。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

LLMMode = Literal["auto", "mock", "openai"]
EmbeddingMode = Literal["auto", "hashing", "openai", "fastembed"]
RerankMode = Literal["feature", "cross_encoder"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- 基础 ----------
    app_name: str = "全渠道智能导购 Agent"
    env: str = "dev"
    host: str = "127.0.0.1"
    port: int = 8771
    log_level: str = "INFO"

    # ---------- 数据与存储 ----------
    data_dir: Path = BASE_DIR / "data"
    db_path: Path = BASE_DIR / "data" / "shop.db"
    products_jsonl: Path = BASE_DIR / "data" / "products_clean.jsonl"
    raw_jsonl: Path = BASE_DIR / "data" / "skus_raw.jsonl"
    index_dir: Path = BASE_DIR / "data" / "index"
    web_dir: Path = BASE_DIR / "web"

    # ---------- LLM ----------
    llm_mode: LLMMode = "auto"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_timeout_s: float = 40.0
    llm_max_tokens: int = 1024

    # ---------- Embedding ----------
    embedding_mode: EmbeddingMode = "auto"
    embedding_dim: int = 512
    embed_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embed_api_key: str = ""
    embed_model: str = "text-embedding-v3"
    fastembed_model: str = "BAAI/bge-small-zh-v1.5"

    # ---------- 检索 ----------
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_field_boost: float = 2.0
    recall_k: int = 200
    rrf_k: int = 60
    rrf_weight_bm25: float = 1.0
    rrf_weight_vector: float = 0.6
    rerank_top_n: int = 30
    final_k: int = 6

    # ---------- 对话 ----------
    confidence_threshold: float = 0.42
    max_rewrite: int = 1
    max_turns_memory: int = 12
    session_ttl_hours: int = 24
    enable_function_calling: bool = True

    # ---------- 工具 ----------
    tool_timeout_ms: int = 100
    mcp_server_module: str = "app.tools.mcp_server"
    mcp_fake_latency_ms: int = 0

    # 评测
    eval_path: Path = BASE_DIR / "data" / "eval_queries.json"

    @property
    def llm_enabled(self) -> bool:
        if self.llm_mode == "mock":
            return False
        if self.llm_mode == "openai":
            return True
        return bool(self.llm_api_key)

    @property
    def resolved_rerank_mode(self) -> RerankMode:
        return "cross_encoder" if os.getenv("RERANK_MODE", "feature") == "cross_encoder" else "feature"

    def model_post_init(self, __context) -> None:  # noqa: D401
        """从常见厂商的独立环境变量里兜底取 key，并把 base_url 对齐。"""
        if not self.llm_api_key:
            for name, base, model in (
                ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-chat"),
                ("DASHSCOPE_API_KEY", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
                ("MOONSHOT_API_KEY", "https://api.moonshot.cn/v1", "moonshot-v1-8k"),
                ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4o-mini"),
            ):
                key = os.getenv(name, "").strip()
                if key:
                    self.llm_api_key = key
                    self.llm_base_url = base
                    self.llm_model = model
                    break
        if not self.embed_api_key:
            self.embed_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir):
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

"""检索引擎：装配 SQLite 商品库 + BM25 + 向量索引 + 重排，并负责索引落盘。

索引文件的版本由 meta.json 里的 (条数, embedding 后端, 维度) 决定，
任何一项变化都会触发重建，避免"换了 embedding 模型但沿用旧向量"这种脏数据问题。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.llm.embeddings import build_embedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.rerank import build_reranker
from app.retrieval.types import SearchResult
from app.retrieval.vector import VectorIndex
from app.store.db import Database

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, settings: Settings | None = None, db: Database | None = None):
        self.settings = settings or get_settings()
        self.db = db or Database(self.settings.db_path)
        self.embedder = build_embedder(
            self.settings.embedding_mode,
            self.settings.embedding_dim,
            base_url=self.settings.embed_base_url,
            api_key=self.settings.embed_api_key,
            model=self.settings.embed_model,
            fastembed_model=self.settings.fastembed_model,
        )
        self.bm25 = BM25Index(self.settings.bm25_k1, self.settings.bm25_b)
        self.vector = VectorIndex(self.embedder)
        self.reranker = build_reranker(self.settings.resolved_rerank_mode)
        self.pipeline = RetrievalPipeline(self.bm25, self.vector, self.reranker, self.settings)
        self.docs: list[dict[str, Any]] = []
        self.docs_by_id: dict[str, dict[str, Any]] = {}
        self.build_info: dict[str, Any] = {}
        self.ready = False

    # ------------------------------------------------------------------ paths
    @property
    def bm25_path(self) -> Path:
        return self.settings.index_dir / "bm25.pkl"

    @property
    def vector_path(self) -> Path:
        return self.settings.index_dir / "vectors.npy"

    @property
    def meta_path(self) -> Path:
        return self.settings.index_dir / "meta.json"

    # ------------------------------------------------------------------ load
    def ensure_ready(self, force: bool = False) -> "SearchEngine":
        if self.ready and not force:
            return self
        from app.core.text import init_tokenizer

        init_tokenizer()  # jieba 首次加载词典要 300~500ms，放启动阶段，别让第一条请求买单
        started = time.perf_counter()
        self.docs = list(self.db.iter_products())
        if not self.docs:
            raise RuntimeError("商品库为空，请先执行 python -m app.data.generator 与 python -m app.data.etl")
        self.docs_by_id = {d["sku_id"]: d for d in self.docs}

        meta = self._read_meta()
        reusable = (
            not force
            and meta
            and meta.get("count") == len(self.docs)
            and meta.get("embedder") == getattr(self.embedder, "name", "")
            and meta.get("dim") == int(self.embedder.dim)
            and self.bm25_path.exists()
            and self.vector_path.exists()
        )
        if reusable:
            self.bm25 = BM25Index.load(self.bm25_path)
            self.vector = VectorIndex(self.embedder)
            self.vector.load(self.vector_path, self.bm25.sku_ids)
            self.build_info = {**meta, "loaded_from_cache": True}
        else:
            self.bm25 = BM25Index(self.settings.bm25_k1, self.settings.bm25_b)
            self.bm25.build(self.docs)
            self.vector = VectorIndex(self.embedder)
            self.vector.build(self.docs)
            self.bm25.save(self.bm25_path)
            self.vector.save(self.vector_path)
            self._write_meta()
            self.build_info = {**self._read_meta(), "loaded_from_cache": False}

        self.pipeline = RetrievalPipeline(self.bm25, self.vector, self.reranker, self.settings)
        self.pipeline.bind_docs(self.docs)
        self.ready = True
        self.build_info["load_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("检索索引就绪: %s", self.stats())
        return self

    def rebuild(self) -> "SearchEngine":
        return self.ensure_ready(force=True)

    # ------------------------------------------------------------------ search
    def search(self, query: str, slots: dict[str, Any] | None = None, top_k: int | None = None) -> SearchResult:
        if not self.ready:
            self.ensure_ready()
        return self.pipeline.search(query, slots, top_k)

    def get_product(self, sku_id: str) -> dict[str, Any] | None:
        return self.docs_by_id.get(sku_id) or self.db.get_product(sku_id)

    def stats(self) -> dict[str, Any]:
        return {
            "docs": len(self.docs),
            "bm25": self.bm25.stats() if self.ready else {},
            "vector": self.vector.stats() if self.ready else {},
            "reranker": self.reranker.name,
            "cache": self.build_info.get("loaded_from_cache"),
            "build": {k: v for k, v in self.build_info.items() if k not in ("bm25", "vector")},
        }

    # ------------------------------------------------------------------ io
    def _read_meta(self) -> dict[str, Any] | None:
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_meta(self) -> None:
        meta = {
            "count": len(self.docs),
            "embedder": getattr(self.embedder, "name", ""),
            "dim": int(self.embedder.dim),
            "bm25_seconds": round(getattr(self.bm25, "build_seconds", 0), 2),
            "vector_seconds": round(getattr(self.vector, "build_seconds", 0), 2),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bm25": self.bm25.stats(),
            "vector": self.vector.stats(),
        }
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine().ensure_ready()
    return _engine

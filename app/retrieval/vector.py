"""向量召回：numpy 矩阵 + 内积（向量已归一化，等价于 cosine）。

5 万条 × 512 维 float32 ≈ 100MB，全量点积在单机上 3~5ms，
比维护 Milvus 更适合"单机可跑"的目标；接口与 Milvus 版保持一致，
换成 Milvus 只需要替换 search 实现。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from app.llm.embeddings import Embedder


class VectorIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.matrix: np.ndarray = np.zeros((0, embedder.dim), dtype=np.float32)
        self.sku_ids: list[str] = []

    def build(self, docs: list[dict[str, Any]], batch_size: int = 512) -> None:
        started = time.perf_counter()
        texts = [
            f"{d['title']} {' '.join(d.get('tags') or [])} {(d.get('description') or '')[:120]}"
            for d in docs
        ]
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunks.append(self.embedder.embed_batch(texts[start : start + batch_size]))
        self.matrix = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, self.embedder.dim), np.float32)
        self.sku_ids = [d["sku_id"] for d in docs]
        self.build_seconds = time.perf_counter() - started

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        if not len(self.matrix):
            return []
        q = self.embedder.embed(query)
        scores = self.matrix @ q
        top_k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.matrix)

    def load(self, path: str | Path, sku_ids: list[str]) -> None:
        self.matrix = np.load(path).astype(np.float32)
        self.sku_ids = sku_ids

    def stats(self) -> dict[str, Any]:
        return {
            "docs": len(self.sku_ids),
            "dim": int(self.matrix.shape[1]) if len(self.matrix) else 0,
            "backend": getattr(self.embedder, "name", "unknown"),
            "mem_mb": round(self.matrix.nbytes / 1024 / 1024, 1),
        }

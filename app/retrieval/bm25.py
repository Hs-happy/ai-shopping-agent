"""BM25 倒排索引（自研，带字段加权）。

为什么不用 SQLite FTS5：FTS5 的 unicode61 分词器不切中文，需要自己塞预切好的
空格串；与其这样，不如直接实现 BM25F，把标题/标签/规格/描述的权重显式写出来，
既能控制召回质量，也方便在面试里讲清楚每个参数在干什么。

打分公式（BM25F 简化版）：
    score(q,d) = Σ_t idf(t) * tf_w(t,d)*(k1+1) / (tf_w(t,d) + k1*(1-b+b*dl/avgdl))
其中 tf_w 是各字段加权后的词频，dl 是加权文档长度。
"""

from __future__ import annotations

import math
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.core.text import tokenize

FIELD_WEIGHTS = {
    "title": 3.0,
    "brand_category": 3.0,
    "tags": 2.0,
    "specs": 1.5,
    "description": 1.0,
}


def doc_term_weights(doc: dict[str, Any]) -> dict[str, float]:
    tf: dict[str, float] = defaultdict(float)

    def add(text: str, weight: float) -> None:
        if not text:
            return
        for token in tokenize(text):
            tf[token] += weight

    add(doc.get("title", ""), FIELD_WEIGHTS["title"])
    add(f"{doc.get('brand','')} {doc.get('category','')}", FIELD_WEIGHTS["brand_category"])
    add(" ".join(doc.get("tags") or []), FIELD_WEIGHTS["tags"])
    for key, value in (doc.get("specs") or {}).items():
        add(f"{key}{value}", FIELD_WEIGHTS["specs"])
    add((doc.get("description") or "")[:120], FIELD_WEIGHTS["description"])
    return dict(tf)


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.sku_ids: list[str] = []
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 1.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}

    # ------------------------------------------------------------------ build
    def build(self, docs: Iterable[dict[str, Any]]) -> None:
        started = time.perf_counter()
        sku_ids: list[str] = []
        doc_len: list[float] = []
        raw: dict[str, list[tuple[int, float]]] = defaultdict(list)

        for idx, doc in enumerate(docs):
            sku_ids.append(doc["sku_id"])
            weights = doc_term_weights(doc)
            doc_len.append(sum(weights.values()))
            for token, weight in weights.items():
                raw[token].append((idx, weight))

        self.sku_ids = sku_ids
        self.doc_len = np.asarray(doc_len, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(doc_len) else 1.0
        n = len(sku_ids)

        self.postings = {}
        self._idf = {}
        for token, pairs in raw.items():
            ids = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
            tfs = np.fromiter((p[1] for p in pairs), dtype=np.float32, count=len(pairs))
            self.postings[token] = (ids, tfs)
            df = len(pairs)
            self._idf[token] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

        self.build_seconds = time.perf_counter() - started

    # ------------------------------------------------------------------ search
    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        tokens = list(dict.fromkeys(tokenize(query)))
        if not tokens or not self.postings:
            return []
        scores = np.zeros(len(self.sku_ids), dtype=np.float32)
        k1, b = self.k1, self.b
        for token in tokens:
            entry = self.postings.get(token)
            if entry is None:
                continue
            ids, tfs = entry
            denom = tfs + k1 * (1.0 - b + b * self.doc_len[ids] / self.avgdl)
            scores[ids] += self._idf[token] * tfs * (k1 + 1.0) / denom
        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "k1": self.k1, "b": self.b, "sku_ids": self.sku_ids,
                    "doc_len": self.doc_len, "avgdl": self.avgdl,
                    "postings": self.postings, "idf": self._idf,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        index = cls(data["k1"], data["b"])
        index.sku_ids = data["sku_ids"]
        index.doc_len = data["doc_len"]
        index.avgdl = data["avgdl"]
        index.postings = data["postings"]
        index._idf = data["idf"]
        return index

    def stats(self) -> dict[str, Any]:
        return {
            "docs": len(self.sku_ids),
            "vocab": len(self.postings),
            "avg_doc_len": round(self.avgdl, 1),
            "k1": self.k1,
            "b": self.b,
        }

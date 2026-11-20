"""消融实验：BM25 / 向量 / RRF / 重排 各自贡献多少。

用法：python scripts/eval_ablation.py
输出：终端表格 + data/eval_ablation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.eval.dataset import build_dataset  # noqa: E402
from app.eval.runner import evaluate  # noqa: E402
from app.llm.embeddings import HashingEmbedder  # noqa: E402
from app.retrieval.engine import SearchEngine  # noqa: E402
from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from app.retrieval.rerank import FeatureReranker  # noqa: E402


class _EmptyBM25:
    def search(self, query: str, top_k: int = 50):  # noqa: ANN001
        return []


class _EmptyVector:
    def __init__(self, dim: int = 512):
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self.embedder = HashingEmbedder(dim)

    def search(self, query: str, top_k: int = 50):  # noqa: ANN001
        return []


class _RRFOnlyReranker:
    name = "rrf_only"

    def rerank(self, query, candidates, slots, top_k):  # noqa: ANN001
        for cand in candidates:
            cand.rerank_score = min(1.0, cand.rrf_score * 45)
            cand.reasons = ["仅按 RRF 排序"]
        return candidates[:top_k]


def build_variants(engine: SearchEngine):
    settings = engine.settings
    return {
        "bm25_only": lambda bm25, vec, rr: RetrievalPipeline(bm25, _EmptyVector(), rr, settings),
        "vector_only": lambda bm25, vec, rr: RetrievalPipeline(_EmptyBM25(), vec, rr, settings),
        "hybrid_rrf": lambda bm25, vec, rr: RetrievalPipeline(bm25, vec, _RRFOnlyReranker(), settings),
        "hybrid_rerank": lambda bm25, vec, rr: RetrievalPipeline(bm25, vec, FeatureReranker(), settings),
    }


def main() -> None:
    engine = SearchEngine().ensure_ready()
    dataset = build_dataset(engine.docs, seed=7)
    results = []
    for label, factory in build_variants(engine).items():
        pipeline = factory(engine.bm25, engine.vector, engine.reranker)
        pipeline.bind_docs(engine.docs)
        report = evaluate(engine, dataset, k=10, searcher=lambda q, s, k, p=pipeline: p.search(q, s, k), label=label)
        results.append(report["summary"])
        kinds = report["summary"].get("by_kind", {})
        key = kinds.get("keyword", {})
        sem = kinds.get("semantic", {})
        print(
            f"{label:<16} 全部 recall={report['summary']['recall@k']:.3f} | "
            f"关键词 recall={key.get('recall@k', 0):.3f} ndcg={key.get('ndcg@k', 0):.3f} | "
            f"语义 recall={sem.get('recall@k', 0):.3f} ndcg={sem.get('ndcg@k', 0):.3f} | "
            f"{report['summary']['latency_ms']:.1f}ms"
        )

    out = engine.settings.data_dir / "eval_ablation.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()

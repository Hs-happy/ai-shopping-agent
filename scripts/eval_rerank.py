"""重排对比：特征重排 vs BGE CrossEncoder。

CrossEncoder 需要 BAAI/bge-reranker-base（约 1GB），首次会自动下载；
设置 HF_ENDPOINT=https://hf-mirror.com 可走镜像。

用法：python scripts/eval_rerank.py [--queries 16]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.eval.dataset import build_dataset  # noqa: E402
from app.eval.runner import evaluate  # noqa: E402
from app.retrieval.engine import SearchEngine  # noqa: E402
from app.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from app.retrieval.rerank import CrossEncoderReranker, FeatureReranker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=int, default=16)
    args = parser.parse_args()

    engine = SearchEngine().ensure_ready()
    dataset = build_dataset(engine.docs, seed=7)[: args.queries]

    reports = []
    for label, reranker in (("feature", FeatureReranker()), ("bge-reranker", CrossEncoderReranker())):
        pipeline = RetrievalPipeline(engine.bm25, engine.vector, reranker, engine.settings)
        pipeline.bind_docs(engine.docs)
        report = evaluate(engine, dataset, k=10, searcher=lambda q, s, k, p=pipeline: p.search(q, s, k), label=label)
        summary = report["summary"]
        reports.append(summary)
        print(
            f"{label:<13} recall@10={summary['recall@k']:.3f} ndcg@10={summary['ndcg@k']:.3f} "
            f"mrr@10={summary['mrr@k']:.3f} 延迟={summary['latency_ms']:.0f}ms"
        )

    out = engine.settings.data_dir / "eval_rerank.json"
    out.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()

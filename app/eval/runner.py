"""检索评测：Recall@K / Precision@K / MRR@K / nDCG@K / 约束满足率 / 需求点命中率。

评测集分两类：
- keyword：查询里直接出现品类/品牌/卖点词，检验精确匹配；
- semantic：查询里不出现卖点词（"地铁上想把噪音隔掉"），检验语义召回。
两类分开统计，才能看出混合检索到底在哪一类查询上有收益。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from app.eval.dataset import build_dataset, relevant_ids, save_dataset
from app.retrieval.engine import SearchEngine

Searcher = Callable[[str, dict[str, Any], int], Any]


def _dcg(rel_flags: list[int]) -> float:
    return sum(flag / math.log2(i + 2) for i, flag in enumerate(rel_flags))


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    return {
        "queries": len(rows),
        "hit@k": round(sum(r["hit@k"] for r in rows) / n, 4),
        "recall@k": round(sum(r["recall@k"] for r in rows) / n, 4),
        "precision@k": round(sum(r["precision@k"] for r in rows) / n, 4),
        "mrr@k": round(sum(r["mrr@k"] for r in rows) / n, 4),
        "ndcg@k": round(sum(r["ndcg@k"] for r in rows) / n, 4),
        "constraint_rate": round(sum(r["constraint"] for r in rows) / n, 4),
        "attr_rate": round(sum(r["attr_hit"] for r in rows) / n, 4),
        "latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 2),
    }


def evaluate(engine: SearchEngine, dataset: list[dict[str, Any]], k: int = 10,
             searcher: Searcher | None = None, label: str = "hybrid+rerank") -> dict[str, Any]:
    search = searcher or (lambda q, s, kk: engine.search(q, s, top_k=kk))
    rows: list[dict[str, Any]] = []

    for item in dataset:
        slots = item["slots"]
        result = search(item["query"], slots, k)
        rel = relevant_ids(engine.docs, slots)
        if not rel:
            continue
        got = [hit.doc["sku_id"] for hit in result.hits]
        docs = [hit.doc for hit in result.hits]
        flags = [1 if sku in rel else 0 for sku in got]
        first = next((i + 1 for i, flag in enumerate(flags) if flag), None)
        ideal = _dcg([1] * min(len(rel), k))

        def strict(doc: dict[str, Any]) -> bool:
            return (
                (not slots.get("category") or doc["category"] == slots["category"])
                and (not slots.get("brand") or doc["brand"] == slots["brand"])
                and (not slots.get("price_max") or doc["price"] <= slots["price_max"])
                and (not slots.get("price_min") or doc["price"] >= slots["price_min"])
            )

        attrs = slots.get("attributes") or []
        rows.append(
            {
                "id": item["id"], "query": item["query"], "slots": slots,
                "kind": item.get("kind", "keyword"),
                "relevant": len(rel), "returned": len(got),
                "hit@k": 1.0 if first else 0.0,
                "recall@k": round(sum(flags) / min(len(rel), k), 4),
                "precision@k": round(sum(flags) / max(len(got), 1), 4),
                "mrr@k": round(1.0 / first, 4) if first else 0.0,
                "ndcg@k": round(_dcg(flags) / ideal, 4) if ideal else 0.0,
                "constraint": round(sum(1 for doc in docs if strict(doc)) / max(len(docs), 1), 4),
                "attr_hit": round(
                    sum(1 for doc in docs if all(a in doc["tags"] for a in attrs)) / max(len(docs), 1), 4
                ) if attrs else 1.0,
                "confidence": result.confidence,
                "latency_ms": result.stats["latency"]["total_ms"],
                "top_titles": [hit_.doc["title"][:36] for hit_ in result.hits[:3]],
            }
        )

    summary = {"config": label, "k": k, **_agg(rows)}
    summary["by_kind"] = {
        kind: _agg([r for r in rows if r["kind"] == kind])
        for kind in ("keyword", "semantic")
        if any(r["kind"] == kind for r in rows)
    }
    return {"summary": summary, "rows": rows}


def main(k: int = 10, seed: int = 7, out: Path | None = None) -> dict[str, Any]:
    engine = SearchEngine().ensure_ready()
    dataset = build_dataset(engine.docs, seed=seed)
    save_dataset(dataset, engine.settings.eval_path)
    report = evaluate(engine, dataset, k=k)
    out = out or engine.settings.data_dir / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = main(args.k, args.seed)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    worst = sorted(report["rows"], key=lambda r: r["recall@k"])[:5]
    print("\n召回最差的 5 条：")
    for row in worst:
        print(f"  [{row['kind']}] {row['query']}  recall={row['recall@k']} attr={row['attr_hit']} rel={row['relevant']}")

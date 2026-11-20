"""RRF（Reciprocal Rank Fusion）融合。

BM25 分数和余弦相似度量纲不同，直接加权没有意义；RRF 只看名次，
score(d) = Σ_r 1/(k + rank_r(d))，对两路召回的头部都友好，
k 取 60 是原论文的经验值：k 越大，分数越平滑，长尾越容易被抬起来。
"""

from __future__ import annotations

from typing import Iterable, Sequence


def rrf(
    rankings: Sequence[Sequence[tuple[int, float]]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """rankings: 每路召回是 [(doc_idx, score), ...]，已按分数降序。"""
    weights = list(weights or [1.0] * len(rankings))
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_idx, _score) in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda item: -item[1])


def intersect_top(rankings: Iterable[Sequence[tuple[int, float]]]) -> set[int]:
    sets = [set(doc for doc, _ in ranking) for ranking in rankings]
    return set.intersection(*sets) if sets else set()

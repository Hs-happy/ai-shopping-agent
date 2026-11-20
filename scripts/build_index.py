"""构建/校验检索索引，并跑几条样例查询。

用法：
    python scripts/build_index.py            # 有缓存就复用
    python scripts/build_index.py --rebuild  # 强制重建
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.engine import SearchEngine  # noqa: E402

SAMPLES = [
    ("5000以内拍照好的手机", {"category": "手机", "price_max": 5000, "attributes": ["拍照"]}),
    ("通勤用的降噪耳机，1000左右", {"category": "无线耳机", "price_max": 1200, "attributes": ["降噪"]}),
    ("学生党轻薄本 预算6000", {"category": "笔记本电脑", "price_max": 6000, "attributes": ["轻薄", "学生"]}),
    ("想买个相机拍vlog", {"category": "相机", "attributes": ["视频"]}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    engine = SearchEngine().ensure_ready(force=args.rebuild)
    print(f"索引就绪，耗时 {time.perf_counter() - started:.1f}s")
    print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))

    for query, slots in SAMPLES:
        result = engine.search(query, slots, top_k=3)
        print(f"\n=== {query}  (confidence={result.confidence})")
        print(f"    stats={json.dumps(result.stats['latency'], ensure_ascii=False)} relaxed={result.relaxed}")
        for hit in result.hits:
            doc = hit.doc
            print(f"    - {doc['title'][:42]:<44} ¥{doc['price']:<7.0f} {doc['brand']:<6} "
                  f"score={hit.rerank_score:.3f} bm25#{hit.bm25_rank} vec#{hit.vec_rank} reasons={hit.reasons}")


if __name__ == "__main__":
    main()

"""数据清洗与入库。

原始数据 → 清洗 → products_clean.jsonl + SQLite。
清洗规则：
1. 品牌中英文别名归一（apple/苹果 → Apple）；
2. 价格统一成 float（去掉 ￥/元/千分位）；
3. 规格单位统一（寸 → 英寸、G → GB）；
4. 标题去营销前缀与重复铺货后缀；
5. 按 (品牌, 清洗后标题) 去重，只保留价格更低/库存更高的一条；
6. 用领域规则重算卖点标签，并拼出检索用 search_text。
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.text import normalize, parse_price
from app.data.taxonomy import CATEGORIES
from app.store.db import Database

BRAND_MAP: dict[str, str] = {}
for _cat in CATEGORIES.values():
    for _brand in _cat.brands:
        BRAND_MAP[_brand.name.lower()] = _brand.name
        for _alias in _brand.aliases:
            BRAND_MAP[_alias.lower()] = _brand.name

_HTML_RE = re.compile(r"<[^>]+>")
_BRACKET_RE = re.compile(r"【[^】]*】")
_SPACES_RE = re.compile(r"\s+")
_SIZE_RE = re.compile(r"(\d(?:\.\d)?)\s*寸")
_GB_RE = re.compile(r"(\d+)\s*G(?![a-zA-Z])")


def clean_title(title: str) -> str:
    title = _BRACKET_RE.sub("", title or "")
    title = title.replace("全新未拆封", " ").replace("官方正品", " ")
    title = _HTML_RE.sub(" ", title)
    return _SPACES_RE.sub(" ", title).strip()


def clean_brand(brand: str) -> str:
    key = normalize(brand or "")
    return BRAND_MAP.get(key, brand.strip() if brand else "")


def clean_specs(specs: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (specs or {}).items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        text = _SIZE_RE.sub(r"\1英寸", text)
        text = _GB_RE.sub(r"\1GB", text)
        out[str(key).strip()] = text
    return out


def clean_description(text: str) -> str:
    text = _HTML_RE.sub(" ", text or "")
    return _SPACES_RE.sub(" ", text).strip()


def build_search_text(row: dict[str, Any]) -> str:
    spec_text = " ".join(f"{k}{v}" for k, v in row["specs"].items())
    parts = [
        row["title"], row["brand"], row["category"],
        " ".join(row["tags"]), spec_text, row["description"][:240],
    ]
    return _SPACES_RE.sub(" ", " ".join(p for p in parts if p)).strip()


def clean_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    category = (raw.get("category") or "").strip()
    if category not in CATEGORIES:
        return None
    title = clean_title(str(raw.get("title") or ""))
    if len(title) < 4:
        return None
    price = parse_price(str(raw.get("price", "")))
    if not price or price < 10:
        # -1/0 这类脏价格经正则解析会变成 1，10 元以下的 3C 商品一律视为脏数据
        return None
    brand = clean_brand(str(raw.get("brand") or ""))
    if not brand:
        return None

    specs = clean_specs(raw.get("specs") or {})
    rating = float(raw.get("rating") or 0.0)
    stock = int(raw.get("stock") or 0)
    tags = list(raw.get("tags") or [])
    try:
        rule_tags = CATEGORIES[category].tag_rule(specs, float(price), rating)
        tags = sorted(set(rule_tags) | set(tags))
    except Exception:  # 规格脏数据兜底：保留原始标签
        tags = sorted(set(tags))
    tags = [t for t in tags if t][:8]

    row = {
        "sku_id": str(raw["sku_id"]),
        "title": title,
        "brand": brand,
        "category": category,
        "price": round(float(price), 2),
        "stock": max(stock, 0),
        "rating": rating,
        "sales": int(raw.get("sales") or 0),
        "specs": specs,
        "tags": tags,
        "description": clean_description(str(raw.get("description") or "")),
        "shop": str(raw.get("shop") or ""),
    }
    row["search_text"] = build_search_text(row)
    return row


def run(raw_path: Path | None = None, out_path: Path | None = None, db_path: Path | None = None) -> dict[str, Any]:
    raw_path = Path(raw_path or settings.raw_jsonl)
    out_path = Path(out_path or settings.products_jsonl)
    db_path = Path(db_path or settings.db_path)

    stats: Counter[str] = Counter()
    seen: dict[str, dict[str, Any]] = {}
    if not raw_path.exists():
        raise FileNotFoundError(f"原始数据不存在: {raw_path}，请先运行 python -m app.data.generator")

    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            stats["raw"] += 1
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue
            row = clean_record(raw)
            if not row:
                stats["invalid"] += 1
                continue
            key = f"{row['brand']}|{row['title']}"
            if key in seen:
                stats["duplicated"] += 1
                kept = seen[key]
                # 同款保留库存更高、价格更低的那条
                if (row["stock"], -row["price"]) > (kept["stock"], -kept["price"]):
                    seen[key] = row
                continue
            seen[key] = row

    rows = list(seen.values())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    db = Database(db_path)
    db.replace_products(rows)

    stats["clean"] = len(rows)
    stats["db"] = db.count_products()
    stats["categories"] = len({r["category"] for r in rows})
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并清洗 SKU 数据")
    parser.add_argument("--generate", type=int, default=0, help="先生成 N 条原始数据（0 表示跳过）")
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()
    if args.generate:
        from app.data.generator import generate

        started = time.perf_counter()
        gen = generate(args.generate, args.seed)
        print(f"[1/2] 生成原始数据 {gen['total']} 条，耗时 {time.perf_counter() - started:.1f}s")
    t0 = time.perf_counter()
    stats = run()
    print(f"[2/2] 清洗入库完成 {stats}，耗时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

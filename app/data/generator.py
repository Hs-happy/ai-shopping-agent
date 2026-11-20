"""合成 3C/数码 SKU 原始数据。

生成的原始数据故意"脏"：价格混着 ¥/元/千分位，品牌有中英文别名，
标题带营销前缀，部分字段缺失，还有重复铺货的行——这些都由 ETL 统一清洗。
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from app.config import settings
from app.data.taxonomy import CATEGORIES, COLORS, SELLING_POINTS, VARIANTS, Brand, Category

GEN_RANGES = {
    "手机": (12, 17),
    "笔记本电脑": (2022, 2026),
    "平板电脑": (3, 8),
    "无线耳机": (1, 5),
    "智能手表": (2, 9),
    "相机": (1, 6),
    "蓝牙音箱": (1, 5),
    "机械键盘": (1, 5),
    "游戏鼠标": (1, 5),
    "显示器": (1, 6),
}

SHOPS = ("品牌官方旗舰店", "京东自营", "天猫旗舰店", "授权专卖店", "品牌直营店")
# 同一型号不同"版本"在电商里很常见，用来保证标题唯一性、避免被 ETL 当成重复铺货
VERSION_TAGS = ("官方标配", "礼盒装", "套装版", "2025款", "国行版", "企业采购版", "直播间专享", "以旧换新专享")


def _pick_brand(cat: Category, rng: random.Random) -> Brand:
    # 越便宜的品牌铺货越多，接近真实货架分布
    weights = [1.0 / (b.base_price**0.35) for b in cat.brands]
    return rng.choices(cat.brands, weights=weights, k=1)[0]


def _round_price(value: float, cat: Category) -> int:
    low, high = cat.price_range
    value = min(max(value, low), high)
    price = int(round(value / 100.0)) * 100 - 1
    return int(min(max(price, low), high))


def _title(cat: Category, brand: Brand, gen: int, variant: str, color: str, specs: dict[str, str], rng) -> str:
    series = rng.choice(brand.series)
    name = series if series != brand.name else f"{brand.name}{gen}"
    if series == brand.name:
        head = f"{name}{variant}"
    else:
        head = f"{brand.name} {name}{gen}{variant}"
    suffix = cat.title_suffix
    if cat.name == "手机":
        return f"{head} {color} {specs['运行内存']}+{specs['机身存储']} {suffix}"
    if cat.name == "笔记本电脑":
        return f"{head} {color} {specs['内存']}+{specs['硬盘'].split()[0]} {suffix}"
    if cat.name == "平板电脑":
        return f"{head} {color} {specs['存储']} {suffix}"
    if cat.name == "显示器":
        return f"{head} {specs['尺寸']} {specs['分辨率']} {specs['刷新率']} {suffix}"
    if cat.name in ("机械键盘", "相机"):
        return f"{head} {color} {suffix}"
    return f"{head} {color} {suffix}"


def _description(title: str, specs: dict[str, str], tags: list[str], rng: random.Random) -> str:
    points = [SELLING_POINTS[t] for t in tags if t in SELLING_POINTS]
    rng.shuffle(points)
    parts = [f"{title}。"]
    if points:
        parts.append(points[0] + "。")
    if len(points) > 1:
        parts.append(points[1] + "。")
    spec_line = "，".join(f"{k}{v}" for k, v in list(specs.items())[:6])
    parts.append(f"核心参数：{spec_line}。")
    parts.append(rng.choice(["现货速发，支持 7 天无理由退货。", "官方质保一年，全国联保。", "下单赠原装保护壳。", "顺丰包邮，次日达。"]))
    return "".join(parts)


def _dirty(rec: dict, rng: random.Random, brand: Brand) -> dict:
    out = dict(rec)
    price = int(out["price"])
    q = rng.random()
    if q < 0.18:
        out["price"] = f"￥{price:,.2f}"
    elif q < 0.32:
        out["price"] = f"{price}元"
    elif q < 0.38:
        out["price"] = f"{price}"

    if brand.aliases and rng.random() < 0.08:
        out["brand"] = rng.choice(brand.aliases)
    elif rng.random() < 0.04:
        out["brand"] = brand.name.lower()

    if rng.random() < 0.02:
        out["title"] = "【官方正品】" + out["title"]
    if rng.random() < 0.03:
        out["title"] = out["title"] + " 全新未拆封"
    if rng.random() < 0.02:
        out["title"] = out["title"].replace(" ", "  ")

    specs = out["specs"]
    for key, value in list(specs.items()):
        specs[key] = value.replace("英寸", "寸") if rng.random() < 0.15 else value
        specs[key] = value.replace("GB", "G") if rng.random() < 0.1 else specs[key]

    if rng.random() < 0.35:
        out["description"] = out["description"].replace("。", "。<br>", 1)
    if rng.random() < 0.1:
        out["description"] = "<b>" + out["description"] + "</b>"
    if rng.random() < 0.015:
        out["stock"] = None
    if rng.random() < 0.02:
        out["rating"] = None
    if rng.random() < 0.002:
        out["price"] = rng.choice([-1, 0])
    if rng.random() < 0.001:
        out["title"] = ""
    return out


def generate(count: int = 50_000, seed: int = 20260601, out_path: Path | None = None) -> dict:
    rng = random.Random(seed)
    out_path = Path(out_path or settings.raw_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = list(CATEGORIES)
    per = count // len(names)
    counts = {name: per for name in names}
    for i in range(count - per * len(names)):
        counts[names[i]] += 1

    stats = {"total": 0, "duplicates": 0}
    used_titles: set[tuple[str, str]] = set()
    with out_path.open("w", encoding="utf-8") as fh:
        for name in names:
            cat = CATEGORIES[name]
            for _ in range(counts[name]):
                brand = _pick_brand(cat, rng)
                price = _round_price(brand.base_price * rng.uniform(*brand.price_span), cat)
                tier = 0 if price < cat.tiers[0] else 1 if price < cat.tiers[1] else 2 if price < cat.tiers[2] else 3
                specs = cat.spec_builder(rng, float(price), tier)
                rating = round(rng.uniform(3.9, 4.95), 1)
                tags = sorted(set(cat.tag_rule(specs, float(price), rating)))
                color = rng.choice(COLORS[name])
                variant = rng.choice(VARIANTS)
                gen = rng.randint(*GEN_RANGES[name])
                title = _title(cat, brand, gen, variant, color, specs, rng)
                base_title = title
                for attempt in range(len(VERSION_TAGS) + 1):
                    if (brand.name, title) not in used_titles:
                        break
                    title = f"{base_title} {VERSION_TAGS[attempt % len(VERSION_TAGS)]}{'' if attempt < len(VERSION_TAGS) else attempt}"
                used_titles.add((brand.name, title))
                rec = {
                    "sku_id": "SKU" + uuid.uuid4().hex[:10].upper(),
                    "title": title,
                    "brand": brand.name,
                    "category": name,
                    "price": price,
                    "specs": specs,
                    "stock": int(rng.choices([0, rng.randint(1, 40), rng.randint(40, 600)], weights=[7, 2, 4])[0]),
                    "rating": rating,
                    "sales": int(min(99_999, rng.lognormvariate(6.1, 1.35))),
                    "tags": tags,
                    "description": _description(title, specs, tags, rng),
                    "shop": rng.choice(SHOPS),
                }
                rec = _dirty(rec, rng, brand)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["total"] += 1

                # 约 3% 重复铺货：同款不同链接
                if rng.random() < 0.03:
                    dup = json.loads(json.dumps(rec))
                    dup["sku_id"] = "SKU" + uuid.uuid4().hex[:10].upper()
                    dup["title"] = str(dup["title"]) + " 全新未拆封"
                    dup["shop"] = rng.choice(SHOPS)
                    fh.write(json.dumps(dup, ensure_ascii=False) + "\n")
                    stats["total"] += 1
                    stats["duplicates"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="生成合成 3C SKU 数据")
    parser.add_argument("--count", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--out", type=Path, default=settings.raw_jsonl)
    args = parser.parse_args()
    stats = generate(args.count, args.seed, args.out)
    print(f"生成完成: {stats} -> {args.out}")


if __name__ == "__main__":
    main()

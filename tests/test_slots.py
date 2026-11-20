from app.agent.slots import Slots, extract_by_rules, merge_slots


def test_extract_category_price_attr():
    slots = extract_by_rules("5000以内拍照好的手机")
    assert slots.category == "手机"
    assert slots.price_max == 5000
    assert "拍照" in slots.attributes


def test_extract_brand_alias():
    slots = extract_by_rules("苹果的笔记本推荐一款")
    assert slots.brand == "Apple"
    assert slots.category == "笔记本电脑"


def test_extract_price_range():
    slots = extract_by_rules("3000到4000之间的耳机")
    assert slots.price_min == 3000
    assert slots.price_max == 4000


def test_exclude_brand():
    slots = extract_by_rules("不要小米的手机")
    assert "小米" in slots.exclude_brands
    assert slots.brand is None


def test_merge_keeps_old_attributes_and_overrides_brand():
    old = Slots(category="手机", price_max=5000, attributes=["拍照"])
    new = Slots(brand="华为")
    merged = merge_slots(old, new)
    assert merged.category == "手机"
    assert merged.brand == "华为"
    assert merged.price_max == 5000
    assert "拍照" in merged.attributes


def test_merge_drops_excluded_brand():
    old = Slots(category="手机", brand="小米")
    new = Slots(exclude_brands=["小米"])
    merged = merge_slots(old, new)
    assert merged.brand is None


def test_missing_requires_category():
    assert "category" in Slots().missing()
    assert Slots(category="手机").missing() == ["preference"]

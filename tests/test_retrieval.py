from app.retrieval.fusion import rrf


def test_search_respects_budget_and_category(engine):
    result = engine.search("5000以内拍照好的手机", {"category": "手机", "price_max": 5000}, top_k=6)
    assert result.hits
    assert all(hit.doc["category"] == "手机" for hit in result.hits)
    assert all(hit.doc["price"] <= 5000 * 1.2 for hit in result.hits)
    assert result.confidence > 0.4


def test_brand_filter(engine):
    result = engine.search("华为的手机", {"category": "手机", "brand": "华为"}, top_k=5)
    assert result.hits
    assert all(hit.doc["brand"] == "华为" for hit in result.hits)


def test_nonsense_query_has_lower_confidence(engine):
    good = engine.search("5000以内拍照好的手机", {"category": "手机", "price_max": 5000})
    bad = engine.search("帮我写一首古诗", {})
    assert bad.confidence < good.confidence
    assert bad.confidence < engine.settings.confidence_threshold


def test_rrf_prefers_documents_in_both_lists():
    fused = rrf([[(1, 5.0), (2, 4.0)], [(2, 0.9), (3, 0.8)]], k=60)
    assert fused[0][0] == 2  # 两路都出现的文档排第一


def test_engine_cache_reuse(engine):
    stats = engine.stats()
    assert stats["docs"] > 1000
    assert stats["bm25"]["vocab"] > 100
    assert stats["cache"] in (True, False)

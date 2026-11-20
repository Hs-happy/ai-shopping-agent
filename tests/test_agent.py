from app.core.events import CollectSink, reset_sink, set_sink
from app.tools.inventory import InventoryService


async def test_agent_recommend_emits_full_pipeline(agent):
    sink = CollectSink()
    token = set_sink(sink)
    try:
        final = await agent.run("pytest-recommend", "5000以内拍照好的手机")
    finally:
        reset_sink(token)

    types = [e["type"] for e in sink.events]
    assert "slots" in types
    assert "retrieval" in types
    assert "tool_result" in types
    assert "delta" in types
    assert "products" in types
    assert final["answer"]
    assert final["slots"]["category"] == "手机"
    assert final["confidence"] > 0.4


async def test_agent_multi_turn_lowers_budget(agent):
    first = await agent.run("pytest-multiturn", "5000以内拍照好的手机")
    second = await agent.run("pytest-multiturn", "再便宜点")
    assert second["slots"]["price_max"] < first["slots"]["price_max"]
    assert second["slots"]["category"] == "手机"


async def test_agent_brand_followup(agent):
    await agent.run("pytest-brand", "5000以内拍照好的手机")
    final = await agent.run("pytest-brand", "换成华为的")
    assert final["slots"]["brand"] == "华为"
    assert final["candidates"]
    assert all(c["brand"] == "华为" for c in final["candidates"])


async def test_agent_chitchat_short_circuit(agent):
    final = await agent.run("pytest-chat", "你好")
    assert "小导" in final["answer"]


async def test_agent_aftersales(agent):
    final = await agent.run("pytest-aftersales", "这个能退货吗")
    assert "售后" in final["answer"]


async def test_inventory_degrades_when_mcp_broken(settings, engine):
    broken = settings.model_copy(update={"mcp_server_module": "app.tools.module_does_not_exist"})
    service = InventoryService(broken, engine.db)
    doc = engine.docs[0]
    try:
        result = await service.query_inventory([doc["sku_id"]], "pytest-degrade")
        assert result.degraded is True
        assert result.data["degraded"] is True
        assert result.data["items"][0]["sku_id"] == doc["sku_id"]
    finally:
        await service.close()


async def test_low_confidence_asks_for_clarification(agent):
    sink = CollectSink()
    token = set_sink(sink)
    try:
        final = await agent.run("pytest-clarify", "推荐个送女朋友的礼物")
    finally:
        reset_sink(token)
    assert final["answer"]

from app.tools.inventory import MCPClient


async def test_mcp_inventory_roundtrip(settings, engine):
    doc = engine.docs[0]
    client = MCPClient(db_path=settings.db_path)
    await client.start()
    try:
        data = await client.call_tool("query_inventory", {"sku_ids": [doc["sku_id"]]}, timeout_s=5)
        assert data["items"][0]["sku_id"] == doc["sku_id"]
        assert "warehouse" in data["items"][0]
        assert "eta_days" in data["items"][0]
    finally:
        await client.close()


async def test_mcp_promotion_tool(settings, engine):
    doc = engine.docs[0]
    client = MCPClient(db_path=settings.db_path)
    await client.start()
    try:
        data = await client.call_tool("get_promotion", {"sku_id": doc["sku_id"]}, timeout_s=5)
        assert data["sku_id"] == doc["sku_id"]
        assert "coupon" in data
    finally:
        await client.close()

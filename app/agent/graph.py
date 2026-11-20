"""LangGraph 状态机装配。

                ┌────────────┐
   user query → │ understand │
                └─────┬──────┘
          chitchat/售后 │ 选购
                ┌─────▼──────┐        ┌─────────┐
                │  retrieve  │◄───────┤ rewrite │
                └─────┬──────┘        └────┬────┘
          conf<阈值 且能改写 │                ▲
                      └───┴────────────────┘
                      │ conf 达标
                ┌─────▼──────┐
                │    plan    │  决定要不要调库存工具
                └─────┬──────┘
                ┌─────▼──────┐
                │    tool    │  MCP 调用 + 超时降级
                └─────┬──────┘
                ┌─────▼──────┐
                │  respond   │  流式生成 / 模板兜底 / 澄清追问
                └────────────┘
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.nodes import AgentDeps, make_nodes
from app.agent.state import AgentState


def build_graph(deps: AgentDeps):
    nodes = make_nodes(deps)
    settings = deps.settings

    def route_after_understand(state: AgentState) -> str:
        if state.get("intent") in ("chitchat", "aftersales"):
            return "respond"
        return "retrieve"

    def route_after_retrieve(state: AgentState) -> str:
        confidence = float(state.get("confidence") or 0.0)
        can_rewrite = state.get("rewrite_count", 0) < settings.max_rewrite
        if confidence < settings.confidence_threshold and can_rewrite:
            return "rewrite"
        return "plan"

    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("understand")
    graph.add_conditional_edges(
        "understand", route_after_understand, {"retrieve": "retrieve", "respond": "respond"}
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"rewrite": "rewrite", "plan": "plan"}
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("plan", "tool")
    graph.add_edge("tool", "respond")
    graph.add_edge("respond", END)
    return graph.compile()

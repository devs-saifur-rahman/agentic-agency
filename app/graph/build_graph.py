from __future__ import annotations
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.llm.base import LLMProvider
from app.graph import nodes as N

def build_graph(llm: LLMProvider):
    g = StateGraph(AgentState)

    # Wrap LLM-dependent nodes so StateGraph can call them with state/config
    def route_guard_node(state: AgentState, config=None):
        return N.route_guard(state, llm=llm, config=config)

    def write_answer_node(state: AgentState, config=None):
        return N.write_answer(state, llm=llm, config=config)

    g.add_node("ingest_user", N.ingest_user)
    g.add_node("route_guard", route_guard_node)
    g.add_node("handle_command", N.handle_command)
    g.add_node("refuse", N.refuse)
    g.add_node("write_answer", write_answer_node)

    g.set_entry_point("ingest_user")
    g.add_edge("ingest_user", "route_guard")

    g.add_conditional_edges(
        "route_guard",
        N.next_after_route,
        {
            "handle_command": "handle_command",
            "refuse": "refuse",
            "write_answer": "write_answer",
        },
    )

    g.add_edge("handle_command", END)
    g.add_edge("refuse", END)
    g.add_edge("write_answer", END)

    return g.compile()
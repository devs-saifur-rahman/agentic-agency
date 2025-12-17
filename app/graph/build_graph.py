from __future__ import annotations
from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.llm.base import LLMProvider
from app.graph import nodes as N

from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.llm.base import LLMProvider
from app.graph import nodes as N


def build_graph(llm: LLMProvider):
    g = StateGraph(AgentState)

    # ---- LLM-wrapped nodes ----
    def route_guard_node(state: AgentState, config=None):
        return N.route_guard(state, llm=llm, config=config)

    def plan_search_node(state: AgentState, config=None):
        return N.plan_search(state, llm=llm, config=config)

    def reformulate_search_node(state: AgentState, config=None):
        return N.reformulate_search(state, llm=llm, config=config)

    def write_answer_node(state: AgentState, config=None):
        return N.write_answer(state, llm=llm, config=config)

    # ---- Nodes ----
    g.add_node("ingest_user", N.ingest_user)
    g.add_node("route_guard", route_guard_node)

    g.add_node("handle_command", N.handle_command)
    g.add_node("refuse", N.refuse)

    g.add_node("plan_search", plan_search_node)
    g.add_node("search_web", N.search_web)
    g.add_node("reformulate_search", reformulate_search_node)
    g.add_node("write_answer", write_answer_node)

    # ---- Entry ----
    g.set_entry_point("ingest_user")
    g.add_edge("ingest_user", "route_guard")

    # ---- Route decision ----
    g.add_conditional_edges(
        "route_guard",
        N.next_after_route,
        {
            "handle_command": "handle_command",
            "refuse": "refuse",
            "plan_search": "plan_search",
        },
    )

    # ---- Football path ----
    g.add_edge("plan_search", "search_web")

    g.add_conditional_edges(
        "search_web",
        N.next_after_search,
        {
            "search_web": "search_web",                  # retry
            "reformulate_search": "reformulate_search",  # one-time query rewrite
            "write_answer": "write_answer",              # proceed to answer
        },
    )

    g.add_edge("reformulate_search", "search_web")
    g.add_edge("write_answer", END)

    # ---- Terminal paths ----
    g.add_edge("handle_command", END)
    g.add_edge("refuse", END)

    return g.compile()
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
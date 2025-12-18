from __future__ import annotations
from langgraph.graph import StateGraph, END
from typing import Any

from langgraph.prebuilt import ToolNode, tools_condition

from app.tools.web_search_tool import web_search_tool
from app.state import AgentState
from app.llm.base import LLMProvider
from app.graph import nodes as N




def build_graph(llm: LLMProvider, checkpointer: Any = None):
    g = StateGraph(AgentState)

    # ---- LLM with tools ----
    chat = llm.get_chat_model()
    llm_with_tools = chat.bind_tools([web_search_tool])


    # ---- LLM-wrapped nodes ----
    def route_guard_node(state: AgentState, config=None):
        return N.route_guard(state, llm=llm, config=config)

    def plan_search_node(state: AgentState, config=None):
        return N.plan_search(state, llm=llm, config=config)

    def reformulate_search_node(state: AgentState, config=None):
        return N.reformulate_search(state, llm=llm, config=config)

    def extract_facts_node(state: AgentState, config=None):
        return N.extract_facts(state, llm=llm, config=config)

    def write_answer_node(state: AgentState, config=None):
        return N.write_answer(state, llm=llm, config=config)
    
    def resolve_selection_node(state: AgentState, config=None):
        return N.resolve_selection(state, llm=llm, config=config)
    
    def agent_search_node(state: AgentState, config=None):
        return N.agent_search(state, llm=llm_with_tools, config=config)


    tool_node = ToolNode([web_search_tool])

    # ---- Nodes ----
    g.add_node("ingest_user", N.ingest_user)
    g.add_node("route_guard", route_guard_node)

    g.add_node("handle_command", N.handle_command)
    g.add_node("refuse", N.refuse)
 
    g.add_node("agent_search", agent_search_node)
    g.add_node("tools", tool_node)


    g.add_node("plan_search", plan_search_node)
    g.add_node("search_web", N.search_web)
    g.add_node("reformulate_search", reformulate_search_node)

    g.add_node("resolve_selection", resolve_selection_node)

    g.add_node("extract_facts", extract_facts_node)
    g.add_node("clarify_ambiguity", N.clarify_ambiguity)

    g.add_node("write_answer", write_answer_node)

    # ---- Entry ----
    g.set_entry_point("ingest_user")
    g.add_edge("ingest_user", "route_guard")

    # ✅ Built-in tools routing
    g.add_conditional_edges("agent_search", 
                            tools_condition, 
                            {
                                "tools": "tools", 
                                "__end__": "extract_facts",
                                "end": "extract_facts"
                                }
                            )
    g.add_edge("tools", "agent_search")

    # ---- Route decision ----
    g.add_conditional_edges(
        "route_guard",
        N.next_after_route,
        {
            "handle_command": "handle_command",
            "resolve_selection": "resolve_selection",
            "refuse": "refuse",
            "agent_search": "agent_search",
        },
    )

    # ---- Football path ----
    g.add_edge("plan_search", "search_web")

    # After search, either retry/reformulate OR proceed to extraction (not write_answer)
    g.add_conditional_edges(
        "search_web",
        N.next_after_search,
        {
            "search_web": "search_web",                  # retry
            "reformulate_search": "reformulate_search",  # one-time rewrite
            "extract_facts": "extract_facts",            # proceed to extraction
        },
    )

    g.add_edge("reformulate_search", "search_web")

    # After extraction, either clarify or answer
    g.add_conditional_edges(
        "extract_facts",
        N.next_after_extract,
        {
            "clarify_ambiguity": "clarify_ambiguity",
            "write_answer": "write_answer",
        },
    )

    g.add_edge("clarify_ambiguity", END)
    g.add_edge("write_answer", END)

    g.add_edge("resolve_selection", "write_answer")

    # ---- Terminal paths ----
    g.add_node("done", lambda s, config=None: s)
    g.add_edge("done", END)
    g.add_conditional_edges(
        "handle_command",
        N.next_after_command,
        {"write_answer": "write_answer", "end": "done"},
    )

    g.add_edge("refuse", END)

    return g.compile(checkpointer=checkpointer)
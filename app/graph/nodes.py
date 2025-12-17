from __future__ import annotations
import json
from typing import Any, Literal
import time
from app.tools.web_search import web_search

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage

from app.llm.base import LLMProvider
from app.state import AgentState, Route

# ---------- Event emission (progress + token streaming) ----------
def emit_progress(config: dict | None, msg: str) -> None:
    emitter = (config or {}).get("configurable", {}).get("emit")
    if callable(emitter):
        emitter({"type": "progress", "message": msg})

def emit_token(config: dict | None, token: str) -> None:
    emitter = (config or {}).get("configurable", {}).get("emit")
    if callable(emitter):
        emitter({"type": "token", "text": token})


def plan_search(state: AgentState, llm: LLMProvider, config: dict | None = None) -> AgentState:
    emit_progress(config, "Planning web search query...")
    state["needs_search"] = True
    state.setdefault("retry_count", 0)
    state.setdefault("max_retries", 2)
    state.setdefault("reformulate_count", 0)

    messages = [
        SystemMessage(content=PLAN_SEARCH_SYSTEM),
        HumanMessage(content=state["user_query"].strip()),
    ]
    raw = llm.invoke(messages)

    try:
        data = json.loads(raw)
        plan = SearchPlan.model_validate(data)
        state["search_query"] = plan.search_query
        return state
    except (json.JSONDecodeError, ValidationError) as e:
        state.setdefault("errors", []).append({"node": "plan_search", "message": str(e), "recoverable": True})
        # fallback: simple direct query
        state["search_query"] = f'{state["user_query"].strip()} football result'
        return state
# ---------- Router schema ----------
class RouteDecision(BaseModel):
    route: Route
    command: str | None = None
    command_args: dict[str, Any] = Field(default_factory=dict)


ROUTER_SYSTEM = """You are a strict router for a football SCORE/RESULT Q&A agent.

Allowed topic (FOOTBALL_SCORES):
- match results, scores, opponents, scorers, last game, previous game, head-to-head results (scores/results only)

Commands (COMMAND):
- /undo N
- /rewind CHECKPOINT_ID
- /restart

Everything else is OUT_OF_SCOPE.

Return ONLY valid JSON matching this schema:
{
  "route": "COMMAND" | "FOOTBALL_SCORES" | "OUT_OF_SCOPE",
  "command": null | "UNDO" | "REWIND" | "RESTART",
  "command_args": { ... }
}

Rules:
- If the user starts with "/" and matches a command, route=COMMAND with parsed args.
- Do not add any extra keys.
- No commentary. JSON only.
"""
REFORMULATE_SYSTEM = """You improve a failing football match-result web search query.
Return ONLY JSON: { "search_query": "..." }
Rules:
- Make it more specific with team names, competition, and keywords: result, score, scorers, match report.
- No tool names. JSON only.
"""

def reformulate_search(state: AgentState, llm: LLMProvider, config: dict | None = None) -> AgentState:
    emit_progress(config, "Reformulating search query (one last try)...")
    state["reformulate_count"] = int(state.get("reformulate_count", 0)) + 1

    messages = [
        SystemMessage(content=REFORMULATE_SYSTEM),
        HumanMessage(content=f"User question: {state['user_query']}\nPrevious query: {state.get('search_query','')}"),
    ]
    raw = llm.invoke(messages)
    try:
        data = json.loads(raw)
        plan = SearchPlan.model_validate(data)
        state["search_query"] = plan.search_query
    except Exception:
        # fallback slight tweak
        state["search_query"] = f'{state["user_query"].strip()} match report score scorers'
    return state
def search_web(state: AgentState, config: dict | None = None) -> AgentState:
    q = state.get("search_query", "").strip()
    emit_progress(config, f"Searching web: {q!r}")

    try:
        results = web_search(q, max_results=6)
        state["search_results"] = results
        return state
    except Exception as e:
        state.setdefault("errors", []).append({"node": "search_web", "message": str(e), "recoverable": True})
        state["retry_count"] = int(state.get("retry_count", 0)) + 1
        # small backoff
        time.sleep(0.5)
        state["search_results"] = None
        return state

def ingest_user(state: AgentState, config: dict | None = None) -> AgentState:
    emit_progress(config, "Ingesting user message...")
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": state["user_query"]})
    state.setdefault("errors", [])
    return state


def route_guard(state: AgentState, llm: LLMProvider, config: dict | None = None) -> AgentState:
    emit_progress(config, "Routing request (scope/command detection)...")
    user_text = state["user_query"].strip()

    # Fast-path deterministic parsing for commands (no LLM needed)
    if user_text.startswith("/"):
        parts = user_text.split()
        cmd = parts[0].lower()
        if cmd == "/undo":
            n = 1
            if len(parts) > 1 and parts[1].isdigit():
                n = int(parts[1])
            state["route"] = "COMMAND"
            state["command"] = "UNDO"
            state["command_args"] = {"n": n}
            return state
        if cmd == "/rewind" and len(parts) > 1:
            state["route"] = "COMMAND"
            state["command"] = "REWIND"
            state["command_args"] = {"checkpoint_id": parts[1]}
            return state
        if cmd == "/restart":
            state["route"] = "COMMAND"
            state["command"] = "RESTART"
            state["command_args"] = {}
            return state

    # Otherwise use LLM router (structured JSON)
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=user_text),
    ]
    raw = llm.invoke(messages)

    try:
        data = json.loads(raw)
        decision = RouteDecision.model_validate(data)
        state["route"] = decision.route
        state["command"] = decision.command
        state["command_args"] = decision.command_args
        return state
    except (json.JSONDecodeError, ValidationError) as e:
        # Safe fallback: if router fails, refuse (prevents scope leakage)
        state.setdefault("errors", []).append({"node": "route_guard", "message": str(e), "recoverable": True})
        state["route"] = "OUT_OF_SCOPE"
        state["command"] = None
        state["command_args"] = {}
        return state


def handle_command(state: AgentState, config: dict | None = None) -> AgentState:
    emit_progress(config, "Handling command...")
    cmd = state.get("command")
    args = state.get("command_args", {})

    # Slice 1: only confirm (real undo/rewind comes with Postgres checkpointing in a later slice)
    if cmd == "UNDO":
        state["final_answer"] = f"✅ Command received: UNDO last {args.get('n', 1)} message(s). (Will be implemented with Postgres checkpoints in Slice 4.)"
        return state
    if cmd == "REWIND":
        state["final_answer"] = f"✅ Command received: REWIND to checkpoint '{args.get('checkpoint_id')}'. (Will be implemented with Postgres checkpoints in Slice 4.)"
        return state
    if cmd == "RESTART":
        state["final_answer"] = "✅ Command received: RESTART. (Will be implemented with new thread_id + Postgres checkpoints in Slice 4.)"
        return state

    state["final_answer"] = "⚠️ Unknown command."
    return state


def refuse(state: AgentState, config: dict | None = None) -> AgentState:
    emit_progress(config, "Refusing (out of scope)...")
    state["final_answer"] = (
        "I can only answer questions about **football match scores/results** (opponent, scoreline, scorers, last/previous match, head-to-head results). "
        "Ask me a score/result question."
    )
    return state


ANSWER_SYSTEM = """You answer ONLY football score/result questions using the provided WEB SEARCH RESULTS.
Rules:
- Use ONLY the provided results (snippets + URLs). Do not use prior knowledge.
- If evidence is insufficient or conflicting, ask a clarifying question.
- Always include 1–3 citations as plain URLs at the end under 'Sources:'.
- If out of scope, refuse briefly.
"""


def write_answer(state: AgentState, llm: LLMProvider, config: dict | None = None) -> AgentState:
    emit_progress(config, "Generating final answer from web evidence (streaming)...")
    user_text = state["user_query"].strip()
    results = state.get("search_results") or []

    evidence_block = "\n".join(
        [f"- Title: {r['title']}\n  URL: {r['url']}\n  Snippet: {r['snippet']}" for r in results[:6]]
    ) or "NO_RESULTS"

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Question: {user_text}\n\nWEB SEARCH RESULTS:\n{evidence_block}"),
    ]

    out = []
    for tok in llm.stream(messages):
        emit_token(config, tok)
        out.append(tok)

    state["final_answer"] = "".join(out).strip()
    # store citations as the URLs we provided (not perfect, but ok for Slice 2)
    state["citations"] = [r["url"] for r in results[:3]]
    return state


# --------- Conditional routing helpers (pure functions) ----------
def next_after_route(state: AgentState) -> Literal["handle_command", "refuse", "plan_search"]:
    r = state.get("route")
    if r == "COMMAND":
        return "handle_command"
    if r == "OUT_OF_SCOPE":
        return "refuse"
    return "plan_search"


def need_retry_search(state: AgentState) -> bool:
    # tool error occurred if search_results missing AND retry_count incremented
    
    return (state.get("search_results") is None) and (int(state.get("retry_count", 0)) < int(state.get("max_retries", 2)))

def search_results_empty(state: AgentState) -> bool:
    return not (state.get("search_results") or [])

def can_reformulate(state: AgentState) -> bool:
    return int(state.get("reformulate_count", 0)) < 1

class SearchPlan(BaseModel):
    search_query: str = Field(min_length=3)

PLAN_SEARCH_SYSTEM = """You create a web search query to answer football SCORE/RESULT questions.
Return ONLY JSON:
{ "search_query": "..." }

Rules:
- The query must be specific (team, opponent, match name like El Clasico, and 'result'/'score'/'scorers' if needed).
- If user asks "previous match / game before that", include 'previous match before last' phrasing.
- No tool names. JSON only.
"""

def next_after_search(state: AgentState) -> Literal["search_web", "reformulate_search", "no_results_clarify", "write_answer"]:
    if need_retry_search(state):
        return "search_web"

    if search_results_empty(state):
        if can_reformulate(state):
            return "reformulate_search"
        return "no_results_clarify"

    return "write_answer"



def no_results_clarify(state: AgentState, config: dict | None = None) -> AgentState:
    emit_progress(config, "No search results; asking user to clarify...")
    state["final_answer"] = (
        "I couldn't find reliable web results for that query.\n"
        "Please clarify one of these so I can search again:\n"
        "- competition (La Liga / UCL / Copa del Rey)\n"
        "- approximate date (e.g., Nov 2025)\n"
        "- opponent name\n"
    )
    state["citations"] = []
    return state
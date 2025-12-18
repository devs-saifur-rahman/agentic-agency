from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, Optional
import concurrent.futures
import os

from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.llm.base import LLMProvider
from app.state import AgentState, Route
from app.tools.web_search import web_search

def _as_text(x) -> str:
    """Normalize LLM outputs (AIMessage/AIMessageChunk/str/etc.) into a plain string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # LangChain messages usually have .content
    if isinstance(x, BaseMessage):
        return x.content or ""
    # fallback
    return str(x)

def _extract_tool_call_ids(ai: AIMessage) -> set[str]:
    ids: set[str] = set()

    # Preferred: ai.tool_calls (LC)
    tc = getattr(ai, "tool_calls", None)
    if tc:
        for c in tc:
            if isinstance(c, dict):
                if c.get("id"):
                    ids.add(c["id"])
            else:
                cid = getattr(c, "id", None)
                if cid:
                    ids.add(cid)

    # Fallback: additional_kwargs["tool_calls"]
    ak = getattr(ai, "additional_kwargs", None) or {}
    tc2 = ak.get("tool_calls") or []
    for c in tc2:
        if isinstance(c, dict) and c.get("id"):
            ids.add(c["id"])

    return ids

def _sanitize_messages_for_tool_models(msgs):
    cleaned = []
    for i, m in enumerate(msgs):
        if isinstance(m, ToolMessage):
            # Only keep tool messages if previous message had tool_calls
            if i > 0 and hasattr(msgs[i-1], "tool_calls"):
                cleaned.append(m)
            continue
        cleaned.append(m)
    return cleaned


# ---------- JSON parsing helpers (robust against minor model formatting) ----------
def _extract_json_object(text: str) -> str | None:
    """Return substring from first '{' to last '}' inclusive, else None."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end+1]

def _as_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, BaseMessage):
        return x.content or ""
    return str(x)

AGENT_SYSTEM = """You are a football scores assistant.
You may use tools when needed to answer.
Only answer football match score/result questions (opponent, scoreline, scorers, last/previous match, head-to-head).
If the user asks something else, say it's out of scope.
"""


def agent_search(state: AgentState, llm, config=None) -> AgentState:
    emit_progress(config, "Agent deciding whether to call tools...")
    
    if state.get("has_searched"):
        return state 
    
    msgs = state.get("messages") or []

    # 🔍 DEBUG — ADD THESE LINES
    print("DEBUG message types (before sanitize):",
          [type(m).__name__ for m in msgs[:12]])

    for m in msgs:
        if type(m).__name__ in ("AIMessage", "ToolMessage"):
            print(
                "DEBUG first tool-ish:",
                type(m).__name__,
                "tool_calls=", getattr(m, "tool_calls", None),
                "tool_call_id=", getattr(m, "tool_call_id", None),
            )
            break
    # 🔍 END DEBUG

    msgs = _sanitize_messages_for_tool_models(state.get("messages", []))

    # ensure system at top (once)
    if not msgs or not isinstance(msgs[0], SystemMessage):
        msgs = [SystemMessage(content=AGENT_SYSTEM)] + msgs

    ai_msg = llm.invoke(msgs)
    state["messages"] = msgs + [ai_msg]
    state["has_searched"] = True   # ✅ SET FLAG
    return state





def _try_parse_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _repair_json_with_llm(raw_text: str, llm: LLMProvider) -> str:
    """Ask model to output ONLY valid JSON, no extra text."""
    fixer_system = (
        "You are a strict JSON repair assistant. "
        "Given malformed/extra-text output, return ONLY a valid JSON object "
        "with no markdown, no commentary, no trailing text."
    )
    messages = [
        SystemMessage(content=fixer_system),
        HumanMessage(content=raw_text),
    ]
    return _invoke_with_timeout(llm, messages) 

DEFAULT_LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "45"))

def _invoke_with_timeout(llm: LLMProvider, messages, timeout_s: int = DEFAULT_LLM_TIMEOUT_S) -> str:
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(llm.invoke, messages)
    try:
        out = fut.result(timeout=timeout_s)
        return _as_text(out)   # ✅ normalize AIMessage → str
    finally:
        fut.cancel()
        ex.shutdown(wait=False, cancel_futures=True)



def _parse_selection_index(text: str) -> int | None:
    """0-based index for inputs like '2', '2.', 'option 2'. None if not a selection."""
    if not text:
        return None
    t = text.strip().lower()

    if t.isdigit():
        n = int(t)
        return n - 1 if n >= 1 else None

    m = re.match(r"^\s*(\d+)\s*[\.\)]\s*$", t)
    if m:
        n = int(m.group(1))
        return n - 1 if n >= 1 else None

    m = re.search(r"\b(option|pick|choose|select)\s+(\d+)\b", t)
    if m:
        n = int(m.group(2))
        return n - 1 if n >= 1 else None

    return None


# ---------- Event emission (progress + token streaming) ----------
def emit_progress(config: RunnableConfig | None, msg: str) -> None:
    emitter = (config or {}).get("configurable", {}).get("emit")
    if callable(emitter):
        emitter({"type": "progress", "message": msg})

def emit_token(config: RunnableConfig | None, token: str) -> None:
    emitter = (config or {}).get("configurable", {}).get("emit")
    if callable(emitter):
        emitter({"type": "token", "text": token})


def plan_search(state: AgentState, llm: LLMProvider, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Planning web search query...")
    state["needs_search"] = True
    state.setdefault("retry_count", 0)
    state.setdefault("max_retries", 2)
    state.setdefault("reformulate_count", 0)

    messages = [
        SystemMessage(content=PLAN_SEARCH_SYSTEM),
        HumanMessage(content=state["user_query"].strip()),
    ]
    # Use timeout to prevent hanging (common with local models)
    try:
        msg = _invoke_with_timeout(llm, messages)
        raw = msg.content if hasattr(msg, "content") else str(msg)
    except concurrent.futures.TimeoutError:
        state.setdefault("errors", []).append({
            "node": "plan_search",
            "message": f"LLM invoke timed out after {DEFAULT_LLM_TIMEOUT_S}s",
            "recoverable": True,
        })
        emit_progress(config, f"PlanSearch timed out after {DEFAULT_LLM_TIMEOUT_S}s; using fallback query")
        # Fallback: build a simple direct query
        state["search_query"] = f"{state['user_query'].strip()} football result"
        return state
    
    # 1) direct parse
    data = _try_parse_json(raw)

    # 2) try extracting the first {...} block
    if data is None:
        candidate = _extract_json_object(raw)
        if candidate:
            data = _try_parse_json(candidate)

    # 3) one repair attempt via model
    if data is None:
        repaired = _repair_json_with_llm(raw, llm)
        data = _try_parse_json(repaired)
        if data is None:
            candidate = _extract_json_object(repaired) if repaired else None
            if candidate:
                data = _try_parse_json(candidate)

    try:
        if data is None:
            raise json.JSONDecodeError("Could not parse JSON", raw, 0)
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

def reformulate_search(state: AgentState, llm: LLMProvider, config: RunnableConfig | None = None) -> AgentState:
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
def search_web(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    q = state.get("search_query", "").strip()
    emit_progress(config, f"Searching web: {q!r}")

    try:
        results = web_search(q, max_results=8)        
        state["search_results"] = results
        return state
    except Exception as e:
        state.setdefault("errors", []).append({"node": "search_web", "message": str(e), "recoverable": True})
        state["retry_count"] = int(state.get("retry_count", 0)) + 1
        # small backoff
        time.sleep(0.5)
        state["search_results"] = None
        return state

def ingest_user(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Ingesting user message...")
    state.setdefault("messages", [])
    state.setdefault("errors", [])
    # store as LangChain message
    state["messages"].append(HumanMessage(content=state["user_query"]))
    return {"messages": [HumanMessage(content=state["user_query"])]}


def route_guard(state: AgentState, llm: LLMProvider, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Routing request (scope/command detection)...")
    user_text = state["user_query"].strip()

    # If we are awaiting a selection, route to resolve_selection (numeric OR text)
    if state.get("awaiting_selection") is True and (state.get("extracted_facts") or []):
        state["route"] = "COMMAND"
        state["command"] = "resolve_selection"
        state["command_args"] = {"selection_text": user_text}
        return state

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
    # Use timeout to prevent hanging (common with local models)
    try:
        raw_msg = _invoke_with_timeout(llm, messages)
        raw = _as_text(raw_msg)
        data = _try_parse_json(raw)
    except concurrent.futures.TimeoutError:
        state.setdefault("errors", []).append({
            "node": "route_guard",
            "message": f"LLM invoke timed out after {DEFAULT_LLM_TIMEOUT_S}s",
            "recoverable": True,
        })
        emit_progress(config, "RouteGuard timed out; falling back to OUT_OF_SCOPE")
        state["route"] = "OUT_OF_SCOPE"
        state["command"] = None
        state["command_args"] = {}
        return state


    if data is None:
        candidate = _extract_json_object(raw)
        if candidate:
            data = _try_parse_json(candidate)

    if data is None:
        repaired = _repair_json_with_llm(raw, llm)
        data = _try_parse_json(repaired)
        if data is None:
            candidate = _extract_json_object(repaired) if repaired else None
            if candidate:
                data = _try_parse_json(candidate)

    try:
        if data is None:
            raise json.JSONDecodeError("Could not parse JSON", raw, 0)
        decision = RouteDecision.model_validate(data)
        # Debug: show router decision
        emit_progress(config, f"Router raw decision: route={decision.route} command={decision.command} args={decision.command_args}")
        state["route"] = decision.route
        state["command"] = decision.command
        state["command_args"] = decision.command_args
        # Normalize (defensive) and emit final routing choice
        if state.get("route") not in ("COMMAND", "FOOTBALL_SCORES", "OUT_OF_SCOPE"):
            state["route"] = "OUT_OF_SCOPE"
            state["command"] = None
            state["command_args"] = {}
        emit_progress(config, f"RouteGuard decided: {state.get('route')} (command={state.get('command')})")
        return state
    except (json.JSONDecodeError, ValidationError) as e:
        # Safe fallback: if router fails, refuse (prevents scope leakage)
        preview = (raw or "").replace("\n", " ")[:300]
        state.setdefault("errors", []).append({
            "node": "route_guard",
            "message": f"{str(e)} | raw_preview={preview}",
            "recoverable": True,
        })
        emit_progress(config, "RouteGuard parse failed; falling back to OUT_OF_SCOPE")
        state["route"] = "OUT_OF_SCOPE"
        state["command"] = None
        state["command_args"] = {}
        return state


def handle_command(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Handling command...")
    cmd = state.get("command")
    args = state.get("command_args", {})

    # Slice 1: only confirm (real undo/rewind comes with Postgres checkpointing in a later slice)
    if cmd == "UNDO":
        state["final_answer"] = f"✅ Command received: UNDO last {args.get('n', 1)} message(s). )"
        return state
    if cmd == "REWIND":
        state["final_answer"] = f"✅ Command received: REWIND to checkpoint '{args.get('checkpoint_id')}'. "
        return state
    if cmd == "RESTART":
        state["final_answer"] = "✅ Command received: RESTART. "
        return state

    state["final_answer"] = "⚠️ Unknown command."
    return state


def refuse(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Refusing (out of scope)...")
    state["final_answer"] = (
        "I can only answer questions about **football match scores/results** (opponent, scoreline, scorers, last/previous match, head-to-head results). "
        "Ask me a score/result question."
    )
    return state


ANSWER_SYSTEM = """You answer ONLY football score/result questions using the PROVIDED MATCH FACTS.
Rules:
- Use ONLY the provided facts and their source_url.
- If facts are insufficient, ask a clarifying question (do not guess).
- Keep it concise.
- End with 'Sources:' and include 1–3 URLs.
"""

def write_answer(state: AgentState, llm: LLMProvider, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Writing answer from extracted facts (streaming)...")

    facts = state.get("extracted_facts") or []
    selected = state.get("selected_fact")
    facts_to_use = [selected] if selected else facts[:2]

    fact_block = "\n".join([json.dumps(f, ensure_ascii=False) for f in facts_to_use if f]) or "NO_FACTS"

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Question: {state['user_query']}\n\nMATCH FACTS:\n{fact_block}"),
    ]

    out_text: list[str] = []
    for chunk in llm.stream(messages):
        # ✅ ChatOpenAI returns AIMessageChunk; Ollama wrappers may return str
        piece = chunk if isinstance(chunk, str) else (getattr(chunk, "content", "") or "")
        if not piece:
            continue
        emit_token(config, piece)   # emit plain string
        out_text.append(piece)

    state["final_answer"] = "".join(out_text).strip()

    # citations from fact source_url
    cites = []
    for f in facts_to_use:
        if not f:
            continue
        u = f.get("source_url")
        if u:
            cites.append(u)
    state["citations"] = list(dict.fromkeys(cites))  # de-dupe, keep order

    return state



def next_after_route(state: AgentState) -> Literal["handle_command", "refuse", "agent_search", "resolve_selection"]:
    r = state.get("route")
    if r == "COMMAND":
        if state.get("command") == "resolve_selection":
            return "resolve_selection"
        return "handle_command"
    if r == "OUT_OF_SCOPE":
        return "refuse"
    return "agent_search"


def next_after_extract_Debug(state: AgentState) -> Literal["clarify_ambiguity", "write_answer"]:
    nxt = "write_answer"
    facts = state.get("extracted_facts") or []
    if not facts or state.get("ambiguous") is True or (state.get("confidence") in ("low", "medium") and state.get("selected_fact") is None):
        nxt = "clarify_ambiguity"
    print("DEBUG next_after_extract:", nxt, "| facts:", len(facts), "| amb:", state.get("ambiguous"), "| conf:", state.get("confidence"))
    return nxt


def next_after_extract(state: AgentState) -> Literal["clarify_ambiguity", "write_answer"]:
    facts = state.get("extracted_facts") or []
    if not facts:
        return "clarify_ambiguity"

    if state.get("ambiguous") is True:
        return "clarify_ambiguity"

    f = state.get("selected_fact") or facts[0]

    has_teams = bool(f.get("home_team")) and bool(f.get("away_team"))
    has_score = bool(f.get("score"))
    has_source = bool(f.get("source_url"))

    if has_teams and has_score and has_source:
        return "write_answer"

    return "clarify_ambiguity"

def need_retry_search(state: AgentState) -> bool:
    # tool error occurred if search_results missing AND retry_count incremented
    
    return (state.get("search_results") is None) and (int(state.get("retry_count", 0)) < int(state.get("max_retries", 2)))

def search_results_empty(state: AgentState) -> bool:
    return not (state.get("search_results") or [])

def can_reformulate(state: AgentState) -> bool:
    return int(state.get("reformulate_count", 0)) < 1

class SearchPlan(BaseModel):
    search_query: str = Field(min_length=3)

class MatchFact(BaseModel):
    date: Optional[str] = None              # ISO if possible, else null
    competition: Optional[str] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    score: Optional[str] = None             # e.g., "2-1"
    scorers: list[str] = Field(default_factory=list)
    source_url: str

class ExtractOutput(BaseModel):
    candidates: list[MatchFact] = Field(default_factory=list)
    ambiguous: bool
    confidence: Literal["low", "medium", "high"]
    missing_fields: list[str] = Field(default_factory=list)
    selected_index: Optional[int] = None

EXTRACT_SYSTEM = """You extract football match RESULT facts from WEB SEARCH RESULTS.

You MUST:
- Use ONLY the provided results (title/snippet/url).
- Output JSON matching this schema:
{
  "candidates": [
    {
      "date": "YYYY-MM-DD" or null,
      "competition": "..." or null,
      "home_team": "..." or null,
      "away_team": "..." or null,
      "score": "X-Y" or null,
      "scorers": ["..."],
      "source_url": "https://..."
    }
  ],
  "ambiguous": true/false,
  "confidence": "low"|"medium"|"high",
  "missing_fields": ["..."],
  "selected_index": 0 or null
}

Rules:
- Create 1 candidate per distinct match you can identify.
- If multiple different matches appear, set ambiguous=true unless the user clearly asked for multiple.
- Choose selected_index only when confidence is high and the user's question clearly points to one match.
- No extra keys. JSON only.
"""

PLAN_SEARCH_SYSTEM = """You create a web search query to answer football SCORE/RESULT questions.
Return ONLY JSON:
{ "search_query": "..." }

Rules:
- The query must be specific (team, opponent, match name like El Clasico, and 'result'/'score'/'scorers' if needed).
- If user asks "previous match / game before that", include 'previous match before last' phrasing.
- No tool names. JSON only.
"""


def next_after_search(state: AgentState) -> Literal["search_web", "reformulate_search", "extract_facts"]:
    if need_retry_search(state):
        return "search_web"
    if search_results_empty(state) and can_reformulate(state):
        return "reformulate_search"
    return "extract_facts"



# def no_results_clarify(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
#     emit_progress(config, "No search results; asking user to clarify...")
#     state["final_answer"] = (
#         "I couldn't find reliable web results for that query.\n"
#         "Please clarify one of these so I can search again:\n"
#         "- competition (La Liga / UCL / Copa del Rey)\n"
#         "- approximate date (e.g., Nov 2025)\n"
#         "- opponent name\n"
#     )
#     state["citations"] = []
#     return state

def clarify_ambiguity(state: AgentState, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Ambiguous results; asking user to disambiguate...")

    cands = state.get("extracted_facts") or []
    lines = []
    for i, c in enumerate(cands[:4], start=1):
        ht = c.get("home_team") or "?"
        at = c.get("away_team") or "?"
        sc = c.get("score") or "?"
        dt = c.get("date") or "?"
        comp = c.get("competition") or "?"
        lines.append(f"{i}) {ht} vs {at} — {sc} — {comp} — {dt}")

    prompt = (
        "I found multiple possible matches for your question.\n"
        "Which one do you mean? Reply with 1/2/3… or tell me the date/competition.\n\n"
        + ("\n".join(lines) if lines else "No clear candidates were extracted; please specify date/competition/opponent.")
    )

    state["pending_user_query"] = state.get("user_query", "")
    state["awaiting_selection"] = True
    state["final_answer"] = prompt
    state["citations"] = []
    return state

def _latest_tool_results_as_search_results(messages: list[Any]) -> list[dict[str, Any]]:
    """
    ToolNode returns ToolMessage(content=...) where content may be:
    - a JSON string
    - python-like repr
    - already a list/dict depending on provider
    We normalize to list[dict(title,url,snippet)].
    """
    if not messages:
        return []

    # scan from the end for ToolMessages (latest tool output)
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, list):
                return content
            if isinstance(content, dict):
                return [content]
            if isinstance(content, str):
                # try JSON parse
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    if isinstance(parsed, dict):
                        return [parsed]
                except Exception:
                    return []
    return []


def extract_facts(state: AgentState, llm: LLMProvider, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Extracting match facts (structured)...")

    # Prefer explicit search_results; else pull from latest tool output
    results = state.get("search_results") or []
    if not results:
        results = _latest_tool_results_as_search_results(state.get("messages") or [])
        state["search_results"] = results
    
    # Keep evidence smaller to reduce local-model JSON failures/hangs
    evidence_block = "\n".join(
        [f"- Title: {r['title']}\n  URL: {r['url']}\n  Snippet: {r['snippet']}" for r in results[:5]]
    ) or "NO_RESULTS"

    messages = [
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=f"User question: {state['user_query']}\n\nWEB SEARCH RESULTS:\n{evidence_block}"),
    ]

    # 0) Call model with timeout (prevents hangs on local models)
    try:
        msg = _invoke_with_timeout(llm, messages)
        raw = msg.content if hasattr(msg, "content") else str(msg)
    except concurrent.futures.TimeoutError:
        state.setdefault("errors", []).append({
            "node": "extract_facts",
            "message": f"LLM invoke timed out after {DEFAULT_LLM_TIMEOUT_S}s",
            "recoverable": True,
        })
        state["extracted_facts"] = []
        state["ambiguous"] = True
        state["confidence"] = "low"
        state["missing_fields"] = ["score", "teams", "date"]
        state["selected_fact"] = None
        emit_progress(config, f"ExtractFacts timed out after {DEFAULT_LLM_TIMEOUT_S}s; falling back")
        return state

    # 1) direct parse
    data = _try_parse_json(raw)

    # 2) try extracting the first {...} block
    if data is None:
        candidate = _extract_json_object(raw)
        if candidate:
            data = _try_parse_json(candidate)

    # 3) one repair attempt via model
    if data is None:
        repaired = _repair_json_with_llm(raw, llm)
        data = _try_parse_json(repaired)
        if data is None:
            candidate = _extract_json_object(repaired) if repaired else None
            if candidate:
                data = _try_parse_json(candidate)

    try:
        if data is None:
            raise json.JSONDecodeError("Could not parse JSON", raw, 0)

        out = ExtractOutput.model_validate(data)

        candidates = [c.model_dump() for c in out.candidates]
        state["extracted_facts"] = candidates
        state["ambiguous"] = bool(out.ambiguous)
        state["confidence"] = out.confidence
        state["missing_fields"] = list(out.missing_fields)

        sel = None
        if out.selected_index is not None and 0 <= out.selected_index < len(candidates):
            sel = candidates[out.selected_index]
        state["selected_fact"] = sel

        return state

    except (json.JSONDecodeError, ValidationError) as e:
        preview = (raw or "").replace("\n", " ")[:300]
        state.setdefault("errors", []).append({
            "node": "extract_facts",
            "message": f"{str(e)} | raw_preview={preview}",
            "recoverable": True,
        })
        state["extracted_facts"] = []
        state["ambiguous"] = True
        state["confidence"] = "low"
        state["missing_fields"] = ["score", "teams", "date"]
        state["selected_fact"] = None
        return state

SELECTION_SYSTEM = """
You are selecting which match candidate the user means.
Return ONLY valid JSON: {"selected_index": <int or null>}.

Rules:
- Use the candidates ONLY (no external knowledge).
- If the user clearly refers to one candidate by opponent/competition/date/score -> select it.
- If unclear -> selected_index must be null.
"""

class SelectionDecision(BaseModel):
    selected_index: int | None = None


def resolve_selection(state: AgentState, llm: LLMProvider, config: RunnableConfig | None = None) -> AgentState:
    emit_progress(config, "Resolving selection (numeric/text)...")

    text = (state.get("command_args") or {}).get("selection_text", "") or ""
    facts = state.get("extracted_facts") or []

    # 1) Deterministic numeric selection
    idx = _parse_selection_index(text)
    if idx is not None:
        if 0 <= idx < len(facts):
            state["selected_fact"] = facts[idx]
            state["ambiguous"] = False
            state["confidence"] = "high"
            state["awaiting_selection"] = False
            # restore original question
            oq = (state.get("pending_user_query") or "").strip()
            if oq:
                state["user_query"] = oq
            # next: write_answer
            state["command"] = "select_fact"
            state["final_answer"] = None
            return state

        state["final_answer"] = "That selection is out of range. Reply with 1/2/3…"
        state["awaiting_selection"] = True
        return state

    # 2) Text-based selection using LLM over candidates (no web)
    options = []
    for i, f in enumerate(facts):
        options.append(
            f"{i+1}) {f.get('home_team')} vs {f.get('away_team')} — {f.get('score')} — {f.get('competition')} — {f.get('date')}"
        )
    options_block = "\n".join(options)

    messages = [
        SystemMessage(content=SELECTION_SYSTEM),
        HumanMessage(content=f"User reply: {text}\n\nCandidates:\n{options_block}\n\nReturn JSON only."),
    ]

    try:
        msg = _invoke_with_timeout(llm, messages)
        raw = msg.content if hasattr(msg, "content") else str(msg)
    except concurrent.futures.TimeoutError:
        state["final_answer"] = "I couldn’t resolve your selection in time. Reply with 1/2/3…"
        state["awaiting_selection"] = True
        return state

    data = _try_parse_json(raw)
    if data is None:
        cand = _extract_json_object(raw)
        if cand:
            data = _try_parse_json(cand)
    if data is None:
        # one repair attempt
        try:
            repaired = _repair_json_with_llm(raw, llm)
            data = _try_parse_json(repaired) or _try_parse_json(_extract_json_object(repaired) or "")
        except Exception:
            data = None

    try:
        if data is None:
            raise json.JSONDecodeError("Could not parse JSON", raw, 0)
        decision = SelectionDecision.model_validate(data)
    except Exception:
        state["final_answer"] = "I couldn't understand your choice. Reply with 1/2/3… or mention opponent/date/competition."
        state["awaiting_selection"] = True
        return state

    if decision.selected_index is None:
        state["final_answer"] = "I’m not sure which one you mean. Reply with 1/2/3… or mention opponent/date/competition."
        state["awaiting_selection"] = True
        return state

    if not (0 <= decision.selected_index < len(facts)):
        state["final_answer"] = "That selection is out of range. Reply with 1/2/3…"
        state["awaiting_selection"] = True
        return state

    state["selected_fact"] = facts[decision.selected_index]
    state["ambiguous"] = False
    state["confidence"] = "high"
    state["awaiting_selection"] = False

    oq = (state.get("pending_user_query") or "").strip()
    if oq:
        state["user_query"] = oq

    state["command"] = "select_fact"  # for graph routing
    state["final_answer"] = None
    return state


def next_after_command(state: AgentState) -> Literal["write_answer", "end"]:
    # After resolve_selection, we set command=select_fact when ready to answer
    if state.get("command") == "select_fact" and state.get("selected_fact") is not None:
        return "write_answer"
    return "end"





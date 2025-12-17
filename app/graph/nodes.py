from __future__ import annotations
import json
from typing import Any, Literal

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


ANSWER_SYSTEM = """You answer ONLY football score/result questions.
- If the question is not about match results/scores/scorers, refuse briefly.
- If you don't have evidence yet, say you need to search (we'll add search in Slice 2).
Keep it concise.
"""


def write_answer(state: AgentState, llm: LLMProvider, config: dict | None = None) -> AgentState:
    emit_progress(config, "Generating final answer (streaming)...")

    # Slice 1: no web search yet. We will be honest.
    user_text = state["user_query"].strip()

    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=user_text),
    ]

    # Stream tokens to UI/console while building final_answer
    out = []
    for tok in llm.stream(messages):
        emit_token(config, tok)
        out.append(tok)

    state["final_answer"] = "".join(out).strip()
    return state


# --------- Conditional routing helpers (pure functions) ----------
def next_after_route(state: AgentState) -> Literal["handle_command", "refuse", "write_answer"]:
    r = state.get("route")
    if r == "COMMAND":
        return "handle_command"
    if r == "OUT_OF_SCOPE":
        return "refuse"
    return "write_answer"
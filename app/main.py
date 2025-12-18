from __future__ import annotations
import uuid
from rich.console import Console
from rich.text import Text

from app.llm.factory import get_llm
from app.graph.build_graph import build_graph
from app.state import AgentState
import traceback


console = Console()
THREAD_STATE: dict[str, AgentState] = {}

def make_emitter():
    def emit(evt: dict):
        t = evt.get("type")
        if t == "progress":
            console.print(f"[bold cyan][progress][/bold cyan] {evt.get('message')}")
        elif t == "token":
            console.print(evt.get("text", ""), end="")
    return emit

def run_once(app, thread_id: str, user_query: str, prev_state: AgentState | None):
    saw_token = False

    def emit(event: dict):
        nonlocal saw_token
        t = event.get("type")
        if t == "progress":
            console.print(f"{event.get('message')}")
        elif t == "token":
            saw_token = True
            console.print(event.get("text", ""), end="")

    # Start from previous state if available
    state: AgentState = dict(prev_state or {})
    state["thread_id"] = thread_id
    state["user_query"] = user_query
    state.setdefault("messages", [])
    state.setdefault("errors", [])

    result: AgentState = app.invoke(state, config={"configurable": {"emit": emit}})

    # If nothing was streamed, print final_answer explicitly
    final_answer = (result or {}).get("final_answer")
    if (not saw_token) and final_answer:
        console.print(final_answer)

    cites = (result or {}).get("citations") or []
    if (not saw_token) and cites:
        console.print("\nSources:\n")
        for u in cites[:3]:
            console.print(u)
            console.print("\n")

    return result


def main():
    llm = get_llm()
    app = build_graph(llm)

    thread_id = str(uuid.uuid4())
    THREAD_STATE[thread_id] = {}
    console.print("[bold green]Football Agent (Slice 3)[/bold green]")
    console.print("Type a question. Commands: /undo N, /rewind ID, /restart. Type 'exit' to quit.\n")

    while True:
        user_query = console.input("[bold]You:[/bold] ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break

        if user_query.strip() == "/restart":
            thread_id = str(uuid.uuid4())
            THREAD_STATE[thread_id] = {} 
            console.print("[bold yellow]Started a new thread.[/bold yellow]")
            continue

        prev = THREAD_STATE.get(thread_id)
        result = run_once(app, thread_id, user_query, prev)
        THREAD_STATE[thread_id] = result


        # Show final answer (already streamed), but keep for debugging/state visibility
        if result.get("errors"):
            console.print(f"[dim]errors: {result['errors']}[/dim]")

if __name__ == "__main__":
    main()
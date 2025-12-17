from __future__ import annotations
import uuid
from rich.console import Console
from rich.text import Text

from app.llm.factory import get_llm
from app.graph.build_graph import build_graph
from app.state import AgentState

console = Console()

def make_emitter():
    def emit(evt: dict):
        t = evt.get("type")
        if t == "progress":
            console.print(f"[bold cyan][progress][/bold cyan] {evt.get('message')}")
        elif t == "token":
            console.print(evt.get("text", ""), end="")
    return emit

def run_once(app, thread_id: str, user_query: str):
    emit = make_emitter()

    state: AgentState = {
        "thread_id": thread_id,
        "user_query": user_query,
        "messages": [],
        "errors": [],
    }

    # Pass emitter through config (LangGraph-friendly)
    result = app.invoke(
        state,
        config={"configurable": {"emit": emit}},
    )
    console.print("")  # newline after token stream
    return result

def main():
    llm = get_llm()
    app = build_graph(llm)

    thread_id = str(uuid.uuid4())
    console.print("[bold green]Football Agent (Slice 1)[/bold green]")
    console.print("Type a question. Commands: /undo N, /rewind ID, /restart. Type 'exit' to quit.\n")

    while True:
        user_query = console.input("[bold]You:[/bold] ").strip()
        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit"):
            break

        if user_query.strip() == "/restart":
            thread_id = str(uuid.uuid4())
            console.print("[bold yellow]Started a new thread.[/bold yellow]")

        result = run_once(app, thread_id, user_query)

        # Show final answer (already streamed), but keep for debugging/state visibility
        if result.get("errors"):
            console.print(f"[dim]errors: {result['errors']}[/dim]")

if __name__ == "__main__":
    main()
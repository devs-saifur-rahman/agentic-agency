from __future__ import annotations
import os
import uuid
from rich.console import Console

from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.postgres import PostgresSaver

from app.llm.factory import get_llm
from app.graph.build_graph import build_graph
from app.state import AgentState


console = Console()

VALID_COMMANDS = {"/undo", "/rewind", "/restart", "/history", "/threads", "/switch", "/new"}
THREAD_CHECKPOINT: dict[str, str | None] = {}
KNOWN_THREADS: set[str] = set()



def get_latest_checkpoint_id(app, thread_id: str) -> str | None:
    cfg = {"configurable": {"thread_id": thread_id}}
    snap = app.get_state(cfg)
    # snap.config is usually something like {"configurable": {"thread_id":..., "checkpoint_id":...}}
    return (snap.config.get("configurable") or {}).get("checkpoint_id")

def get_nth_prev_checkpoint_id(app, thread_id: str, n: int) -> str | None:
    cfg = {"configurable": {"thread_id": thread_id}}
    history = list(app.get_state_history(cfg))
    # history[0] is usually latest. n=1 means previous.
    idx = n
    if idx < 0 or idx >= len(history):
        return None
    return (history[idx].config.get("configurable") or {}).get("checkpoint_id")

def show_history(app, thread_id: str, limit: int = 10):
    cfg = {"configurable": {"thread_id": thread_id}}
    items = list(app.get_state_history(cfg))

    console.print(f"[bold]Last {min(limit, len(items))} checkpoints (latest first):[/bold]")
    for i, snap in enumerate(items[:limit], start=0):
        ck = (snap.config.get("configurable") or {}).get("checkpoint_id")
        # Best-effort summary from state
        vals = snap.values or {}
        last_q = (vals.get("user_query") or "")[:60]
        last_ans = (vals.get("final_answer") or "")[:60]
        console.print(f"{i}) {ck}  | Q: {last_q}  | A: {last_ans}")


def make_emitter():
    def emit(evt: dict):
        t = evt.get("type")
        if t == "progress":
            console.print(f"[bold cyan][progress][/bold cyan] {evt.get('message')}")
        elif t == "token":
            console.print(evt.get("text", ""), end="")
    return emit


def run_once(app, thread_id: str, user_query: str, checkpoint_id: str | None = None):
    cfg = {"configurable": {"thread_id": thread_id}}
    if checkpoint_id:
        cfg["configurable"]["checkpoint_id"] = checkpoint_id

    input_state: AgentState = {"user_query": user_query}

    # stream_mode="messages" gives AIMessageChunk (and tool messages depending on model)
    for msg, meta in app.stream(input_state, config=cfg, stream_mode="messages"):
        if isinstance(msg, AIMessageChunk):
            ak = msg.additional_kwargs or {}
            reasoning = ak.get("reasoning") or ak.get("thinking")
            if reasoning:
                console.print(f"[dim]{reasoning}[/dim]")

            if msg.content:
                console.print(msg.content, end="")

    # newline after stream
    console.print("")

    # fetch final state snapshot (for citations/errors/checkpoint)
    snap = app.get_state({"configurable": {"thread_id": thread_id}})
    final_state = snap.values or {}

    cites = final_state.get("citations") or []
    if cites:
        console.print("\nSources:")
        for u in cites[:3]:
            console.print(u)

    return final_state



def main():
    
    dsn = os.getenv("POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("POSTGRES_DSN is missing. Set it in your environment/.env")
        
    llm = get_llm()    
    with PostgresSaver.from_conn_string(dsn) as checkpointer:
        checkpointer.setup() 
        app = build_graph(llm, checkpointer=checkpointer)

        thread_id = str(uuid.uuid4())
        KNOWN_THREADS.add(thread_id)
        THREAD_CHECKPOINT[thread_id] = None
        console.print(f"[dim]New thread_id: {thread_id}[/dim]")
        console.print("[bold green]Football Agent (Slice 3)[/bold green]")
        console.print("Type a question. Commands: /history, /undo N, /rewind ID, /restart, /threads, /switch <id>, /new. Type 'exit' to quit.\n")

        while True:
            user_query = console.input("\n[bold]You:[/bold] ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit"):
                break

            if user_query.startswith("/") and user_query.split()[0] not in VALID_COMMANDS:
                console.print("[yellow]Unknown command. Try /history, /undo N, /rewind ID, /restart, /threads, /switch <id>, /new[/yellow]")
                continue


            if user_query.strip() == "/restart":
                thread_id = str(uuid.uuid4())
                KNOWN_THREADS.add(thread_id)
                THREAD_CHECKPOINT[thread_id] = None
                console.print(f"[bold yellow]New thread_id: {thread_id}[/bold yellow]")
                continue

            if user_query.startswith("/undo"):
                parts = user_query.split()
                n = 1
                if len(parts) > 1 and parts[1].isdigit():
                    n = int(parts[1])

                ck = get_nth_prev_checkpoint_id(app, thread_id, n)
                if not ck:
                    console.print("[yellow]No such checkpoint to undo to.[/yellow]")
                    continue

                THREAD_CHECKPOINT[thread_id] = ck
                console.print(f"[dim]Active pointer checkpoint: {THREAD_CHECKPOINT[thread_id]}[/dim]")
                show_history(app, thread_id, limit=3)
                continue

            if user_query.startswith("/rewind"):
                parts = user_query.split()
                if len(parts) < 2:
                    console.print("[yellow]Usage: /rewind <checkpoint_id>[/yellow]")
                    continue
                THREAD_CHECKPOINT[thread_id] = parts[1].strip()
                console.print(f"[dim]Active pointer checkpoint: {THREAD_CHECKPOINT[thread_id]}[/dim]")
                show_history(app, thread_id, limit=3)
                continue

            if user_query.startswith("/history"):
                parts = user_query.split()
                n = 10
                if len(parts) > 1 and parts[1].isdigit():
                    n = int(parts[1])
                show_history(app, thread_id, limit=n)
                continue
            if user_query.strip() == "/threads":
                console.print("[bold]Known threads:[/bold]")
                for t in sorted(KNOWN_THREADS):
                    mark = "*" if t == thread_id else " "
                    console.print(f"{mark} {t}")
                continue

            if user_query.startswith("/switch"):
                parts = user_query.split()
                if len(parts) < 2:
                    console.print("[yellow]Usage: /switch <thread_id>[/yellow]")
                    continue
                target = parts[1].strip()
                if target not in KNOWN_THREADS:
                    console.print("[yellow]Unknown thread_id. Use /threads to list.[/yellow]")
                    continue
                thread_id = target
                console.print(f"[bold yellow]Switched to thread: {thread_id}[/bold yellow]")
                continue

            if user_query.strip() == "/new":
                thread_id = str(uuid.uuid4())
                KNOWN_THREADS.add(thread_id)
                THREAD_CHECKPOINT[thread_id] = None
                console.print(f"[bold yellow]New thread_id: {thread_id}[/bold yellow]")
                continue


            ck = THREAD_CHECKPOINT.get(thread_id)
            result = run_once(app, thread_id, user_query, checkpoint_id=ck)

             # After a successful run, advance pointer to latest
            latest_ck = get_latest_checkpoint_id(app, thread_id)
            THREAD_CHECKPOINT[thread_id] = latest_ck
            console.print(f"\n[dim]thread={thread_id} | checkpoint={latest_ck}[/dim]")

            # Show final answer (already streamed), but keep for debugging/state visibility
            if result.get("errors"):
                console.print(f"[dim]errors: {result['errors']}[/dim]")
            

if __name__ == "__main__":
    main()
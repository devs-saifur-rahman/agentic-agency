from typing import TypedDict, Literal, Any, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

Route = Literal["COMMAND", "FOOTBALL_SCORES", "OUT_OF_SCOPE"]

class AgentState(TypedDict, total=False):
    thread_id: str
    user_query: str

    messages: Annotated[list[BaseMessage], add_messages]

    route: Route
    command: str | None
    command_args: dict[str, Any]

    needs_search: bool
    search_query: str
    search_results: list[dict[str, Any]] | None
    citations: list[str]

    extracted_facts: list[dict[str, Any]]
    selected_fact: dict[str, Any] | None
    ambiguous: bool
    confidence: str
    missing_fields: list[str]

    awaiting_selection: bool
    pending_user_query: str

    retry_count: int
    max_retries: int
    reformulate_count: int

    final_answer: str | None
    errors: list[dict[str, Any]]

    has_searched: bool


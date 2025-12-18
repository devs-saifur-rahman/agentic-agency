from typing import TypedDict, Literal, Any

Route = Literal["COMMAND", "FOOTBALL_SCORES", "OUT_OF_SCOPE"]

class AgentState(TypedDict, total=False):
    # ---- Conversation ----
    thread_id: str
    user_query: str
    messages: list[dict[str, Any]]

    # ---- Routing ----
    route: Route
    command: str | None
    command_args: dict[str, Any]

    # ---- Search / Evidence (Slice 2) ----
    needs_search: bool
    search_query: str
    search_results: list[dict[str, Any]] | None
    citations: list[str]

    retry_count: int
    max_retries: int
    reformulate_count: int

    # ---- Output ----
    final_answer: str | None

    # ---- Reliability ----
    errors: list[dict[str, Any]]
    
    # ---- Extracted facts (Slice 3) ----
    extracted_facts: list[dict[str, Any]]
    ambiguous: bool
    confidence: Literal["low", "medium", "high"]
    missing_fields: list[str]
    selected_fact: dict[str, Any] | None
    
    # ---- Clarification / Selection ----
    awaiting_selection: bool
    pending_user_query: str

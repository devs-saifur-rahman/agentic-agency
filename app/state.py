from typing import TypedDict, Literal, NotRequired, Any

Route = Literal["COMMAND", "FOOTBALL_SCORES", "OUT_OF_SCOPE"]

class AgentState(TypedDict):
    # Conversation
    thread_id: str
    user_query: str
    messages: list[dict[str, Any]]  # minimal message dicts; later we can use LangChain Message objects

    # Routing
    route: NotRequired[Route]
    command: NotRequired[str | None]
    command_args: NotRequired[dict[str, Any]]

    # Output
    final_answer: NotRequired[str | None]

    # Reliability
    errors: NotRequired[list[dict[str, Any]]]
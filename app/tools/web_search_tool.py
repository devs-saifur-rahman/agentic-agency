from __future__ import annotations
from typing import Any
from langchain_core.tools import tool
from app.tools.web_search import web_search  # your existing function

@tool("web_search", return_direct=False)
def web_search_tool(query: str) -> list[dict[str, Any]]:
    """Search the web for football match results. Returns a list of {title,url,snippet}."""
    return web_search(query)

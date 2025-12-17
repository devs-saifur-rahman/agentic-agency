from __future__ import annotations
from typing import List, Dict
from ddgs import DDGS

def web_search(query: str, max_results: int = 8) -> List[Dict]:
    results: List[Dict] = []
    with DDGS() as ddgs:
        for r in ddgs.text(
            query,
            max_results=max_results,
            safesearch="off",
            region="wt-wt",     # global
            timelimit="m"       # recent (month); use "w" or "d" if needed
        ):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return [x for x in results if x["url"]]
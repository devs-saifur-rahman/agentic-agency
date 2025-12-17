from __future__ import annotations
from typing import Iterable
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage

class OllamaProvider:
    def __init__(self, model: str, base_url: str):
        self._llm = ChatOllama(model=model, base_url=base_url)

    def invoke(self, messages: list[BaseMessage]) -> str:
        resp = self._llm.invoke(messages)
        return getattr(resp, "content", "") or ""

    def stream(self, messages: list[BaseMessage]) -> Iterable[str]:
        for chunk in self._llm.stream(messages):
            txt = getattr(chunk, "content", "") or ""
            if txt:
                yield txt
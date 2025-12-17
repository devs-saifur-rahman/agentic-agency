from __future__ import annotations
from typing import Iterable
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

class OpenAIProvider:
    def __init__(self, model: str, api_key: str):
        self._llm = ChatOpenAI(model=model, api_key=api_key)

    def invoke(self, messages: list[BaseMessage]) -> str:
        resp = self._llm.invoke(messages)
        return getattr(resp, "content", "") or ""

    def stream(self, messages: list[BaseMessage]) -> Iterable[str]:
        for chunk in self._llm.stream(messages):
            txt = getattr(chunk, "content", "") or ""
            if txt:
                yield txt
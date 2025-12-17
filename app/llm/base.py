from __future__ import annotations
from typing import Protocol, Iterable
from langchain_core.messages import BaseMessage

class LLMProvider(Protocol):
    def invoke(self, messages: list[BaseMessage]) -> str: ...
    def stream(self, messages: list[BaseMessage]) -> Iterable[str]: ...
from __future__ import annotations
from typing import Protocol, Any
from langchain_core.messages import BaseMessage

class LLMProvider(Protocol):
    def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...
    def get_chat_model(self) -> Any: ...

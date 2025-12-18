from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage

class OllamaProvider:
    def __init__(self, model: str, base_url: str):
        self._llm = ChatOllama(model=model, base_url=base_url)

    def get_chat_model(self):
        return self._llm

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        return self._llm.invoke(messages)

    def stream(self, messages: list[BaseMessage]):
        for chunk in self._llm.stream(messages):
            yield chunk

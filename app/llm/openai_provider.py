from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

class OpenAIProvider:
    def __init__(self, model: str, api_key: str):
        self._llm = ChatOpenAI(model=model, api_key=api_key)

    def get_chat_model(self):
        return self._llm

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage:
        # IMPORTANT: return the full AIMessage
        return self._llm.invoke(messages)

    def stream(self, messages: list[BaseMessage]):
        # stream AIMessageChunk (not strings)
        for chunk in self._llm.stream(messages):
            yield chunk

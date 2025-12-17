from __future__ import annotations
from app.config import settings
from app.llm.openai_provider import OpenAIProvider
from app.llm.ollama_provider import OllamaProvider
from app.llm.base import LLMProvider

def get_llm() -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing but LLM_PROVIDER=openai")
        return OpenAIProvider(model=settings.openai_model, api_key=settings.openai_api_key)

    if settings.llm_provider == "ollama":
        return OllamaProvider(model=settings.ollama_model, base_url=settings.ollama_base_url)

    raise RuntimeError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
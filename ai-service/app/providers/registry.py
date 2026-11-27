from __future__ import annotations

from app.config import Settings
from app.providers.base import ModelProvider, ProviderAdapter
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider


def build_provider(settings: Settings) -> ModelProvider:
    if settings.provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.provider_timeout_seconds)
    if settings.provider == "mock":
        return MockProvider(chunk_delay=0.02)
    if settings.provider == "openai":
        if not settings.openai_api_key or not settings.openai_api_key.get_secret_value().strip():
            raise ValueError("AI_OPENAI_API_KEY is required for OpenAI")
        if any(
            not m or m.startswith("mock-")
            for m in (settings.chat_model, settings.extract_model, settings.embedding_model)
        ):
            raise ValueError("set AI_CHAT_MODEL, AI_EXTRACT_MODEL and AI_EMBEDDING_MODEL for OpenAI")
        return OpenAIProvider(settings.openai_api_key.get_secret_value(), settings.provider_timeout_seconds)
    raise ValueError(f"unknown provider: {settings.provider}")


def build_adapter(settings: Settings, provider: ModelProvider | None = None) -> ProviderAdapter:
    return ProviderAdapter(
        provider or build_provider(settings),
        timeout_seconds=settings.provider_timeout_seconds,
        max_concurrency=settings.provider_max_concurrency,
    )

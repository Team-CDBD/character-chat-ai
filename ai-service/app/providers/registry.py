from __future__ import annotations

from app.config import Settings
from app.providers.base import ModelProvider, ProviderAdapter
from app.providers.mock import MockProvider


def build_provider(settings: Settings) -> ModelProvider:
    if settings.provider == "mock":
        return MockProvider(chunk_delay=0.02)
    # Real providers are added after the W2 Korean evaluation picks a model (plan §12).
    raise ValueError(f"unknown provider: {settings.provider}")


def build_adapter(settings: Settings, provider: ModelProvider | None = None) -> ProviderAdapter:
    return ProviderAdapter(
        provider or build_provider(settings),
        timeout_seconds=settings.provider_timeout_seconds,
        max_concurrency=settings.provider_max_concurrency,
    )

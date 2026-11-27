from __future__ import annotations

import logging

from app.config import Settings
from app.providers.base import ModelProvider, ProviderAdapter
from app.providers.mock import MockProvider
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider
from app.providers.tokens import TokenCounter, load_hf_tokenizer, load_tiktoken

log = logging.getLogger(__name__)


def build_provider(settings: Settings) -> ModelProvider:
    if settings.provider == "ollama":
        return OllamaProvider(settings.ollama_base_url, settings.provider_timeout_seconds,
                              num_ctx=settings.ollama_num_ctx)
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


def build_counter(settings: Settings, provider: ModelProvider) -> TokenCounter:
    if settings.tokenizer_file:
        return TokenCounter(exact=load_hf_tokenizer(settings.tokenizer_file), exact_source="exact:hf")
    if provider.name == "openai":
        exact = load_tiktoken(settings.chat_model)
        if exact is not None:
            return TokenCounter(exact=exact, exact_source="exact:tiktoken")
    if provider.name == "mock":
        return TokenCounter(calibrate=False, fallback=provider.count_tokens)
    return TokenCounter(calibrate=settings.token_calibration, fallback=provider.count_tokens)


def check_context_fits(settings: Settings) -> None:
    ctx = settings.tuning.context
    if settings.provider == "ollama" and ctx.input_budget + ctx.output_tokens > settings.ollama_num_ctx:
        raise ValueError(
            f"context budget {ctx.input_budget} + output {ctx.output_tokens} tokens exceeds "
            f"AI_OLLAMA_NUM_CTX={settings.ollama_num_ctx}"
        )


def build_adapter(settings: Settings, provider: ModelProvider | None = None) -> ProviderAdapter:
    check_context_fits(settings)
    provider = provider or build_provider(settings)
    adapter = ProviderAdapter(
        provider,
        timeout_seconds=settings.provider_timeout_seconds,
        max_concurrency=settings.provider_max_concurrency,
        counter=build_counter(settings, provider),
        count_model=settings.chat_model,
    )
    log.info("token counting: %s", adapter.token_count_source)
    return adapter

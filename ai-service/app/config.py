"""Service settings. Every value can be overridden by an environment variable prefixed with ``AI_``."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.tuning import Tuning, deep_merge


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore", env_nested_delimiter="__",
                                      env_ignore_empty=True)

    env: str = "local"
    database_url: str = "postgresql+asyncpg://ai:ai@localhost:5432/ai"

    # Service-to-service auth: BE sends ``Authorization: Bearer <token>``.
    service_tokens: list[str] = Field(default_factory=lambda: ["local-dev-token"])

    provider: Literal["ollama", "openai", "mock"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: SecretStr | None = None
    provider_timeout_seconds: float = 120.0
    provider_max_concurrency: int = 1
    chat_model: str = "gemma4:12b"
    extract_model: str = "gemma4:12b"
    embedding_model: str = "embeddinggemma"
    ollama_num_ctx: int = 8192

    # Token counting (providers/tokens.py). Exact when a tokenizer is available, otherwise an estimate
    # calibrated from the token counts the provider reports.
    tokenizer_file: str | None = None  # Hugging Face tokenizer.json for the chat model (e.g. Gemma)
    token_calibration: bool = True

    @model_validator(mode="before")
    @classmethod
    def provider_defaults(cls, values):
        profiles = {
            "openai": ("gpt-4o-mini", "gpt-4o-mini", "text-embedding-3-small"),
            "mock": ("mock-chat", "mock-extract", "mock-embed"),
        }
        values = dict(values)
        file = values.get("tuning_file")
        if file:
            from_file = json.loads(Path(file).read_text(encoding="utf-8"))
            env_tuning = values.get("tuning") or {}
            if isinstance(env_tuning, Tuning):
                env_tuning = env_tuning.model_dump(exclude_unset=True)
            if isinstance(env_tuning, str):
                env_tuning = json.loads(env_tuning)
            if isinstance(env_tuning, dict):
                values["tuning"] = deep_merge(from_file, env_tuning)
        if values.get("provider") in profiles:
            for key, value in zip(
                ("chat_model", "extract_model", "embedding_model"), profiles[values["provider"]], strict=True
            ):
                values.setdefault(key, value)
            values.setdefault("provider_max_concurrency", 8)
            values.setdefault("provider_timeout_seconds", 30.0)
        else:
            values.setdefault("generation_slots", 1)
            values.setdefault("memory_slots", 1)
            values.setdefault("lease_seconds", 600.0)
        return values

    # Worker slots (plan §12: generation 8, memory 4)
    generation_slots: int = 8
    memory_slots: int = 4
    lifecycle_batch: int = 50
    poll_interval_seconds: float = 0.2
    lease_seconds: float = 30.0

    # Dialogue / context (plan §04, §06)
    recent_window_turns: int = 6
    min_recent_window_turns: int = 3
    memory_wait_max_seconds: float = 3.0
    context_wait_default_seconds: float = 10.0
    snapshot_ttl_hours: int = 24
    summary_every_turns: int = 6
    retrieval_top_k: int = 8
    retrieval_gate: float = 0.25
    unfinished_gate: float = 0.30
    regenerate_temperature_bump: float = 0.1

    # Embedding re-index sweep (worker): re-embeds memories whose embedding is missing, failed or was
    # made by a different AI_EMBEDDING_MODEL. 0 disables the sweep (memory jobs still retry their own).
    reindex_interval_seconds: float = 60.0
    reindex_sweep_batch: int = 32
    # Failed revision checks (model down) are re-queued after this long; until then the conversation's
    # unchecked character facts stay hidden.
    revision_retry_after_seconds: float = 300.0

    # Data-decided tuning values with defaults (app/tuning.py, docs/tuning.md)
    tuning_file: str | None = None
    tuning: Tuning = Field(default_factory=Tuning)

    # SSE
    sse_poll_interval_seconds: float = 0.1
    sse_keepalive_seconds: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

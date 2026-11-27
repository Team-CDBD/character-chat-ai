"""Service settings. Every value can be overridden by an environment variable prefixed with ``AI_``."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")

    env: str = "local"
    database_url: str = "postgresql+asyncpg://ai:ai@localhost:5432/ai"

    # Service-to-service auth: BE sends ``Authorization: Bearer <token>``.
    service_tokens: list[str] = Field(default_factory=lambda: ["local-dev-token"])

    # Provider selection. "mock" until the model is chosen (plan §12, W2).
    provider: str = "mock"
    openai_api_key: SecretStr | None = None
    provider_timeout_seconds: float = 30.0
    provider_max_concurrency: int = 8
    chat_model: str = "mock-chat"
    extract_model: str = "mock-extract"
    embedding_model: str = "mock-embed"

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

    # SSE
    sse_poll_interval_seconds: float = 0.1
    sse_keepalive_seconds: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings, get_settings
from app.persistence.db import Database
from app.providers.base import ModelProvider, ProviderAdapter
from app.providers.registry import build_adapter


@dataclass
class Services:
    settings: Settings
    db: Database
    provider: ProviderAdapter

    @classmethod
    def create(cls, settings: Settings | None = None, provider: ModelProvider | None = None) -> Services:
        settings = settings or get_settings()
        return cls(settings=settings, db=Database(settings.database_url), provider=build_adapter(settings, provider))

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings


class Database:
    def __init__(self, url: str | None = None) -> None:
        self.engine: AsyncEngine = create_async_engine(url or get_settings().database_url, pool_pre_ping=True)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as s:
            yield s

    @asynccontextmanager
    async def tx(self) -> AsyncIterator[AsyncSession]:
        async with self.sessionmaker() as s, s.begin():
            yield s

    async def dispose(self) -> None:
        await self.engine.dispose()

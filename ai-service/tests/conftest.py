from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

from app.config import Settings
from app.dialogue.manager import DialogueManager
from app.lifecycle.handler import process_pending
from app.main import create_app
from app.memory.pipeline import MemoryPipeline
from app.persistence import jobs
from app.providers.mock import MockProvider
from app.services import Services

ROOT = Path(__file__).resolve().parents[1]
TEST_DB = os.environ.get("AI_TEST_DATABASE_URL", "postgresql+asyncpg://ai:ai@localhost:5432/ai_test")
TOKEN = "test-token"
TABLES = [
    "ai_conversation", "lifecycle_cursor", "event_inbox", "settings_snapshot", "committed_turn", "revocation",
    "memory", "unfinished_event", "relationship_event", "relationship_state", "summary", "context_snapshot",
    "ai_job", "stream_event", "usage_ledger",
]


def _db_available() -> bool:
    import asyncio

    import asyncpg

    async def ping() -> None:
        conn = await asyncpg.connect(TEST_DB.replace("+asyncpg", ""), timeout=3)
        await conn.close()

    try:
        asyncio.run(ping())
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def migrated_db() -> str:
    if not _db_available():
        pytest.skip(f"PostgreSQL not reachable at {TEST_DB}")
    env = {**os.environ, "AI_DATABASE_URL": TEST_DB}
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True,
                   capture_output=True)
    return TEST_DB


@pytest_asyncio.fixture
async def services(migrated_db: str):
    settings = Settings(
        provider="mock",
        database_url=migrated_db,
        service_tokens=[TOKEN],
        poll_interval_seconds=0.01,
        memory_wait_max_seconds=0.2,
        context_wait_default_seconds=1.0,
        sse_poll_interval_seconds=0.01,
        lease_seconds=30,
    )
    sv = Services.create(settings, MockProvider())
    async with sv.db.tx() as s:
        await s.execute(text(f"TRUNCATE {', '.join(TABLES)}"))
    yield sv
    await sv.db.dispose()


@pytest.fixture
def mock(services: Services) -> MockProvider:
    return services.provider.provider  # type: ignore[return-value]


@pytest_asyncio.fixture
async def client(services: Services):
    app = create_app(services)
    app.state.services = services
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://ai", headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        yield c


async def drain(sv: Services, *, generation: bool = True, memory: bool = True) -> None:
    """Run workers deterministically until no work is left."""
    dialogue, pipeline = DialogueManager(sv), MemoryPipeline(sv)
    for _ in range(200):
        progressed = await process_pending(sv.db, tuning=sv.settings.tuning) > 0
        if generation:
            job = await jobs.claim(sv.db, "GENERATION", "test", sv.settings.lease_seconds)
            if job:
                await dialogue.run_attempt(job)
                progressed = True
        if memory:
            for kind in ("BASELINE", "REVISION", "MEMORY"):
                job = await jobs.claim(sv.db, kind, "test", sv.settings.lease_seconds)
                if job:
                    await pipeline.run(job)
                    progressed = True
        if not progressed:
            return
    raise AssertionError("drain did not converge")

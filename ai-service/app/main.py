"""ai-api entrypoint: ``uvicorn app.main:app``."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from app.api.routes import error_handler, router
from app.services import Services


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or Services.create()
        try:
            yield
        finally:
            if services is None:
                await app.state.services.db.dispose()

    app = FastAPI(
        title="Character Chat AI Service",
        version="0.1.0",
        summary="BE-AI contract /internal/ai/v1",
        lifespan=lifespan,
    )
    app.include_router(router)
    app.add_exception_handler(HTTPException, error_handler)  # type: ignore[arg-type]

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        async with app.state.services.db.session() as s:
            await s.execute(text("SELECT 1"))
        return {"status": "ok"}

    return app


app = create_app()

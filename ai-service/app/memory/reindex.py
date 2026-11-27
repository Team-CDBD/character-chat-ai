"""Embedding re-index path (plan §07): after ``AI_EMBEDDING_MODEL`` changes, or after embedding outages.

A memory is *stale* when it is still valid and its vector is missing, failed, or was made by a
different embedding model than the configured one. Stale memories are never lost: retrieval scores
them by keywords (``KEYWORD_FALLBACK``) until they are re-embedded. Three things re-embed them:

* every memory job retries a few stale rows of its own conversation (``tuning.memory.reindex_batch``);
* the worker's sweep re-embeds stale rows of all active conversations in the background
  (``AI_REINDEX_INTERVAL_SECONDS``, ``AI_REINDEX_SWEEP_BATCH``);
* ``python -m tools.reindex_embeddings`` does the same on demand and shows progress.

Writes are conditional: a row deleted by reset/delete in the meantime is simply not updated, and a row
already re-embedded by someone else is left as is. Only one sweep runs at a time across all workers and
CLI runs (Postgres advisory lock), and the sweep steps aside while user-facing generations are queued,
since a local Ollama usually has a single model slot. If a batch fails, its rows are retried one by one
and the ones that still fail are marked ``FAILED`` and moved to the back of the queue, so one bad row
cannot block the rest.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from app.persistence.db import Database
from app.persistence.models import new_id
from app.providers.base import ProviderAdapter, ProviderError

log = logging.getLogger(__name__)

STALE = "(m.index_status <> 'INDEXED' OR m.embedding_version IS DISTINCT FROM :ev)"
# Stale rows whose backoff has passed (or that never failed). Never-failed rows come first, then failed
# rows by their retry time, so a row that keeps failing moves further back each time.
DUE = "(m.index_retry_at IS NULL OR m.index_retry_at <= now())"
DUE_ORDER = "m.index_retry_at NULLS FIRST, m.created_at"
BACKOFF_BASE_SECONDS = 60
BACKOFF_MAX_SECONDS = 86400

# Marks a failed embedding attempt: FAILED plus exponential backoff (1 min, 2, 4 ... up to 1 day).
# Conditional on the row still being stale for this model, so a late failure cannot overwrite a row
# that another path (memory job / sweep) has meanwhile indexed.
MARK_FAILED_SQL = f"""
    UPDATE memory m SET index_status = 'FAILED', index_attempts = m.index_attempts + 1,
        index_retry_at = now() + make_interval(secs => LEAST({BACKOFF_MAX_SECONDS},
            {BACKOFF_BASE_SECONDS} * power(2, LEAST(m.index_attempts, 20))))
    WHERE m.id = ANY(:ids) AND {STALE}
"""
OUTAGE_AFTER = 3  # consecutive single-row failures that mean "provider down"
SWEEP_LOCK_KEY = 0x6169_7265_696E  # "aireindex"; pg advisory lock shared by worker sweeps and the CLI
_ACTIVE_GENERATION = (
    "SELECT 1 FROM ai_job WHERE kind = 'GENERATION' "
    "AND status IN ('ACCEPTED','WAITING_CONTEXT') LIMIT 1"  # queued, not in flight: steady traffic can't starve it
)
EMBED_TEXT = "concat_ws(' ', m.predicate, m.value, m.evidence)"


@dataclass
class ReindexResult:
    embedded: int = 0
    failed: int = 0
    remaining: int = 0
    locked: bool = False  # another sweep was running
    by_version: dict[str, int] = field(default_factory=dict)


async def stale_counts(db: Database, model: str) -> ReindexResult:
    async with db.session() as s:
        rows = (await s.execute(text(
            f"""
            SELECT COALESCE(m.embedding_version, '(none)') || ' / ' || m.index_status, count(*)
            FROM memory m JOIN ai_conversation c ON c.tenant_id = m.tenant_id
                 AND c.conversation_id = m.conversation_id AND c.epoch = m.epoch AND c.status = 'ACTIVE'
            WHERE m.valid_to IS NULL AND {STALE}
            GROUP BY 1 ORDER BY 1
            """
        ), {"ev": model})).all()
    return ReindexResult(remaining=sum(n for _, n in rows), by_version={k: n for k, n in rows})


# Session-level advisory lock: needs a direct PostgreSQL connection (not pgbouncer in transaction mode).
@asynccontextmanager
async def sweep_lock(db: Database) -> AsyncIterator[bool]:
    """Session-level advisory lock held on one connection for the whole sweep. Yields False if taken."""
    async with db.engine.connect() as conn:
        got = bool(await conn.scalar(text("SELECT pg_try_advisory_lock(:k)"), {"k": SWEEP_LOCK_KEY}))
        await conn.commit()
        try:
            yield got
        finally:
            if got:
                await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": SWEEP_LOCK_KEY})
                await conn.commit()


async def generation_waiting(db: Database) -> bool:
    async with db.session() as s:
        return (await s.execute(text(_ACTIVE_GENERATION))).first() is not None


async def reindex_batch(db: Database, provider: ProviderAdapter, model: str, batch: int) -> ReindexResult:
    """Re-embed up to ``batch`` stale memories across active conversations.

    One provider call; if it fails, each row is tried alone and rows that still fail are marked FAILED.
    Rows already FAILED are picked last.
    """
    async with db.session() as s:
        rows = (await s.execute(text(
            f"""
            SELECT m.id, m.tenant_id, m.owner_id, m.conversation_id, {EMBED_TEXT} AS body
            FROM memory m JOIN ai_conversation c ON c.tenant_id = m.tenant_id
                 AND c.conversation_id = m.conversation_id AND c.epoch = m.epoch AND c.status = 'ACTIVE'
            WHERE m.valid_to IS NULL AND {STALE} AND {DUE}
            ORDER BY {DUE_ORDER}
            LIMIT :n
            """
        ), {"ev": model, "n": batch})).mappings().all()
    if not rows:
        return ReindexResult()
    try:
        vectors, usage = await provider.embed([r["body"] for r in rows], model)
    except ProviderError as exc:
        log.warning("embedding re-index batch failed (%s); retrying rows one by one", type(exc).__name__)
        if len(rows) == 1:
            await _mark_failed(db, rows, model)
            return ReindexResult(failed=1)
        total = ReindexResult()
        for r in rows:
            one = await _embed_rows(db, provider, model, [r])
            total.embedded += one.embedded
            total.failed += one.failed
            if total.failed >= OUTAGE_AFTER and not total.embedded:
                break  # every row fails: the provider is down, not one bad row; try again next sweep
        return total
    return await _write(db, model, rows, vectors, usage)


async def _embed_rows(db: Database, provider: ProviderAdapter, model: str, rows: list[Any]) -> ReindexResult:
    try:
        vectors, usage = await provider.embed([r["body"] for r in rows], model)
    except ProviderError:
        await _mark_failed(db, rows, model)
        return ReindexResult(failed=len(rows))
    return await _write(db, model, rows, vectors, usage)


async def _mark_failed(db: Database, rows: list[Any], model: str) -> None:
    async with db.tx() as s:
        await s.execute(text(MARK_FAILED_SQL), {"ids": [r["id"] for r in rows], "ev": model})


async def _write(db: Database, model: str, rows: list[Any], vectors: list[list[float]], usage: Any) -> ReindexResult:
    written: list[Any] = []
    async with db.tx() as s:
        for r, vec in zip(rows, vectors, strict=True):
            res = await s.execute(text(
                f"""
                UPDATE memory m SET embedding = CAST(:v AS vector), embedding_version = :ev, index_status = 'INDEXED',
                    index_attempts = 0, index_retry_at = NULL
                WHERE m.id = :id AND m.valid_to IS NULL AND {STALE}
                  AND EXISTS (SELECT 1 FROM ai_conversation c WHERE c.tenant_id = m.tenant_id
                              AND c.conversation_id = m.conversation_id AND c.epoch = m.epoch AND c.status = 'ACTIVE')
                """
            ), {"v": json.dumps(vec), "ev": model, "id": r["id"]})
            if res.rowcount:  # type: ignore[attr-defined]
                written.append(r)
        if written:
            await _ledger(s, rows, written, usage)
    return ReindexResult(embedded=len(written))


async def _ledger(s: Any, rows: list[Any], written: list[Any], usage: Any) -> None:
    """One ledger row per conversation, tokens split by text length (one batched call serves many).
    Only rows this call actually wrote are charged."""
    total_chars = sum(len(r["body"]) for r in rows) or 1
    groups: dict[tuple[str, str, str], int] = {}
    for r in written:
        key = (r["tenant_id"], r["owner_id"], r["conversation_id"])
        groups[key] = groups.get(key, 0) + len(r["body"])
    for (tenant, owner, conversation), chars in groups.items():
        await s.execute(text(
            "INSERT INTO usage_ledger (id, tenant_id, owner_id, conversation_id, attempt_id, purpose, model, "
            "input_tokens, output_tokens, est_cost, created_at) "
            "VALUES (:id, :t, :o, :c, 'reindex', 'embed', :m, :i, 0, 0, now())"
        ), {"id": new_id(), "t": tenant, "o": owner, "c": conversation, "m": usage.model,
            "i": round(usage.input_tokens * chars / total_chars)})


async def reindex_all(
    db: Database, provider: ProviderAdapter, model: str, batch: int, *, max_batches: int | None = None,
    yield_to_generation: bool = False, pause_seconds: float = 0.0,
) -> ReindexResult:
    """Run batches until nothing is stale, a batch makes no progress, or ``max_batches`` is reached.

    Returns immediately (``locked=True``) if another sweep holds the lock. With ``yield_to_generation``
    the sweep stops as soon as a user-facing generation is queued; the next sweep continues.
    """
    total = ReindexResult()
    async with sweep_lock(db) as got:
        if not got:
            total.locked = True
            return total
        done = 0
        while max_batches is None or done < max_batches:
            if yield_to_generation and await generation_waiting(db):
                break
            result = await reindex_batch(db, provider, model, batch)
            done += 1
            total.embedded += result.embedded
            total.failed += result.failed
            if result.embedded == 0 or result.failed:
                break  # failed rows now sort last; the next sweep starts with the fresh ones
            if pause_seconds:
                await asyncio.sleep(pause_seconds)
    total.remaining = (await stale_counts(db, model)).remaining
    return total

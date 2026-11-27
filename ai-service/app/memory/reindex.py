"""Embedding re-index path (plan §07): after ``AI_EMBEDDING_MODEL`` changes, or after embedding outages.

A memory is *stale* when it is still valid and its vector is missing, failed, or was made by a
different embedding model than the configured one. Stale memories are never lost: retrieval scores
them by keywords (``KEYWORD_FALLBACK``) until they are re-embedded. Three things re-embed them:

* every memory job retries a few stale rows of its own conversation (``tuning.memory.reindex_batch``);
* the worker's sweep re-embeds stale rows of all active conversations in the background
  (``AI_REINDEX_INTERVAL_SECONDS``, ``AI_REINDEX_SWEEP_BATCH``);
* ``python -m tools.reindex_embeddings`` does the same on demand and shows progress.

Writes are conditional: a row deleted by reset/delete in the meantime is simply not updated, and a row
already re-embedded by someone else is left as is.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from app.persistence.db import Database
from app.persistence.models import new_id
from app.providers.base import ProviderAdapter, ProviderError

log = logging.getLogger(__name__)

STALE = "(m.index_status <> 'INDEXED' OR m.embedding_version IS DISTINCT FROM :ev)"
EMBED_TEXT = "concat_ws(' ', m.predicate, m.value, m.evidence)"


@dataclass
class ReindexResult:
    embedded: int = 0
    failed: int = 0
    remaining: int = 0
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


async def reindex_batch(db: Database, provider: ProviderAdapter, model: str, batch: int) -> ReindexResult:
    """Re-embed up to ``batch`` stale memories across active conversations. One provider call."""
    async with db.session() as s:
        rows = (await s.execute(text(
            f"""
            SELECT m.id, m.tenant_id, m.owner_id, m.conversation_id, {EMBED_TEXT} AS body
            FROM memory m JOIN ai_conversation c ON c.tenant_id = m.tenant_id
                 AND c.conversation_id = m.conversation_id AND c.epoch = m.epoch AND c.status = 'ACTIVE'
            WHERE m.valid_to IS NULL AND {STALE}
            ORDER BY m.created_at
            LIMIT :n
            """
        ), {"ev": model, "n": batch})).mappings().all()
    if not rows:
        return ReindexResult()
    try:
        vectors, usage = await provider.embed([r["body"] for r in rows], model)
    except ProviderError as exc:
        log.warning("embedding re-index failed: %s", type(exc).__name__)
        return ReindexResult(failed=len(rows))
    embedded = 0
    async with db.tx() as s:
        for r, vec in zip(rows, vectors, strict=True):
            res = await s.execute(text(
                f"""
                UPDATE memory m SET embedding = CAST(:v AS vector), embedding_version = :ev, index_status = 'INDEXED'
                WHERE m.id = :id AND m.valid_to IS NULL AND {STALE}
                  AND EXISTS (SELECT 1 FROM ai_conversation c WHERE c.tenant_id = m.tenant_id
                              AND c.conversation_id = m.conversation_id AND c.epoch = m.epoch AND c.status = 'ACTIVE')
                """
            ), {"v": json.dumps(vec), "ev": model, "id": r["id"]})
            embedded += res.rowcount  # type: ignore[attr-defined]
        await _ledger(s, rows, usage)
    return ReindexResult(embedded=embedded)


async def _ledger(s: Any, rows: list[Any], usage: Any) -> None:
    """One ledger row per conversation, tokens split by text length (one batched call serves many)."""
    total_chars = sum(len(r["body"]) for r in rows) or 1
    groups: dict[tuple[str, str, str], int] = {}
    for r in rows:
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
    db: Database, provider: ProviderAdapter, model: str, batch: int, *, max_batches: int | None = None
) -> ReindexResult:
    """Run batches until nothing is stale, a batch fails, or ``max_batches`` is reached."""
    total = ReindexResult()
    done = 0
    while max_batches is None or done < max_batches:
        result = await reindex_batch(db, provider, model, batch)
        done += 1
        total.embedded += result.embedded
        total.failed += result.failed
        if result.failed or result.embedded == 0:
            break
    total.remaining = (await stale_counts(db, model)).remaining
    return total

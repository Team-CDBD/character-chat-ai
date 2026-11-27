"""Retriever (plan §07): scope filter -> hybrid score -> relevance gate.

The scope filter is injected by the server from the job row; callers cannot pass their own
filters. At MVP scale (hundreds of memories per conversation) exact pgvector distance after
the scope filter is enough; an ANN index comes after latency/recall measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge.context_builder import MemoryLine
from app.understanding.normalize import keywords

W_SEMANTIC, W_KEYWORD, W_IMPORTANCE, W_TIME = 0.55, 0.15, 0.15, 0.15


@dataclass(frozen=True)
class Scope:
    tenant_id: str
    owner_id: str
    conversation_id: str
    epoch: int


@dataclass
class Retrieved:
    lines: list[MemoryLine]
    degraded: list[str] = field(default_factory=list)


def keyword_score(query_kw: set[str], doc: str) -> float:
    if not query_kw:
        return 0.0
    doc_kw = keywords(doc)
    hits = sum(1 for q in query_kw if q in doc_kw or any(q in d or d in q for d in doc_kw if len(d) >= 2))
    return min(1.0, hits / len(query_kw))


def time_score(mem_type: str, resolved_at: datetime | None, created_at: datetime, now: datetime) -> float:
    if mem_type in ("SCHEDULE", "UNFINISHED") and resolved_at is not None:
        days = abs((resolved_at - now).total_seconds()) / 86400
        return 1.0 if days <= 2 else max(0.0, 1.0 - (days - 2) / 12)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400)
    return math.exp(-age_days / 30)


def score(semantic: float, kw: float, importance: float, t: float) -> float:
    return W_SEMANTIC * semantic + W_KEYWORD * kw + W_IMPORTANCE * importance + W_TIME * t


def render(row: Any, tz) -> str:
    when = row["created_at"].astimezone(tz)
    tag = " (극중)" if row["is_fiction"] else ""
    detail = f" [{row['time_expr']}]" if row["time_expr"] else ""
    return f'- ({when.month:02d}/{when.day:02d}){tag} {row["predicate"]}: {row["value"]}{detail} ← "{row["evidence"]}"'


async def retrieve(
    s: AsyncSession,
    scope: Scope,
    query: str,
    query_embedding: list[float] | None,
    embedding_version: str,
    now: datetime,
    tz,
    *,
    top_k: int,
    gate: float,
    exclude_ids: set[str] | None = None,
) -> Retrieved:
    degraded: list[str] = []
    params: dict[str, Any] = {
        "t": scope.tenant_id, "o": scope.owner_id, "c": scope.conversation_id, "e": scope.epoch,
        "ev": embedding_version,
    }
    sim_sql = "NULL::float"
    if query_embedding is not None:
        sim_sql = (
            "CASE WHEN embedding IS NOT NULL AND embedding_version = :ev "
            "THEN 1 - (embedding <=> CAST(:q AS vector)) END"
        )
        params["q"] = json.dumps(query_embedding)
    else:
        degraded.append("QUERY_EMBEDDING_UNAVAILABLE")
    rows = (
        await s.execute(
            text(
                f"""
                SELECT id, type, predicate, value, evidence, time_expr, resolved_at, is_fiction, importance,
                       created_at, index_status, {sim_sql} AS sim
                FROM memory
                WHERE tenant_id = :t AND owner_id = :o AND conversation_id = :c AND epoch = :e
                  AND valid_to IS NULL AND conflicts_with_revision IS NULL
                ORDER BY created_at DESC
                LIMIT 500
                """
            ),
            params,
        )
    ).mappings().all()

    q_kw = keywords(query)
    scored: list[tuple[float, Any]] = []
    fallback = False
    for r in rows:
        if exclude_ids and r["id"] in exclude_ids:
            continue
        kw = keyword_score(q_kw, f"{r['predicate']} {r['value']} {r['evidence']}")
        if r["sim"] is None:
            fallback = True
            semantic = kw  # unindexed memories are never silently dropped: keyword stands in
        else:
            semantic = max(0.0, float(r["sim"]))
        total = score(semantic, kw, float(r["importance"]), time_score(r["type"], r["resolved_at"], r["created_at"], now))
        if total >= gate:
            scored.append((total, r))
    if fallback and rows:
        degraded.append("KEYWORD_FALLBACK")
    scored.sort(key=lambda x: x[0], reverse=True)
    lines = [MemoryLine(id=r["id"], text=render(r, tz), score=round(sc, 4)) for sc, r in scored[:top_k]]
    return Retrieved(lines=lines, degraded=degraded)

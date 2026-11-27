"""DB-backed job queue: FOR UPDATE SKIP LOCKED claim, lease, fencing token (plan §12).

Every write a worker makes on behalf of a job goes through :class:`FencedJob`, which checks
the fencing token in the same statement. A worker whose lease expired (or whose job was
revoked by a reset/delete, which bumps the token) gets :class:`LeaseLost` and must stop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ACTIVE_STATUSES
from app.persistence.db import Database

_JSON_COLUMNS = {"usage", "degraded", "payload"}


class LeaseLost(Exception):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ClaimedJob:
    id: str
    kind: str
    tenant_id: str
    owner_id: str
    conversation_id: str
    epoch: int
    turn_seq: int
    attempt_id: str
    response_id: str | None
    payload: dict[str, Any]
    mode: str | None
    previous_status: str
    fencing_token: int
    run_count: int
    deadline_at: datetime | None
    snapshot_id: str | None


_CLAIM_GENERATION = text(
    """
    WITH candidate AS (
        SELECT id, status FROM ai_job
        WHERE kind = 'GENERATION'
          AND (status = 'ACCEPTED'
               OR (status = ANY(:active) AND lease_until < now()))
        ORDER BY created_at
        LIMIT 1
        FOR UPDATE SKIP LOCKED
    )
    UPDATE ai_job j SET
        status = CASE WHEN c.status = 'STREAMING' THEN 'STREAMING' ELSE 'RUNNING' END,
        lease_until = now() + make_interval(secs => :lease),
        lease_owner = :owner,
        fencing_token = j.fencing_token + 1,
        run_count = j.run_count + 1,
        updated_at = now()
    FROM candidate c
    WHERE j.id = c.id
    RETURNING j.*, c.status AS previous_status
    """
)

# Memory jobs run strictly in turn order per conversation, so memoryThroughTurnSeq is a watermark.
_CLAIM_ORDERED = text(
    """
    WITH candidate AS (
        SELECT j.id, j.status FROM ai_job j
        JOIN ai_conversation c
          ON c.tenant_id = j.tenant_id AND c.conversation_id = j.conversation_id AND c.epoch = j.epoch
        WHERE j.kind = :kind
          AND c.status = 'ACTIVE'
          AND (j.status = 'ACCEPTED' OR (j.status = 'RUNNING' AND j.lease_until < now()))
          AND (:kind <> 'MEMORY' OR (
              j.turn_seq <= c.memory_through_turn_seq + 1
              AND NOT EXISTS (
                  SELECT 1 FROM ai_job earlier
                  WHERE earlier.tenant_id = j.tenant_id AND earlier.conversation_id = j.conversation_id
                    AND earlier.epoch = j.epoch AND earlier.kind = 'MEMORY' AND earlier.turn_seq < j.turn_seq
                    AND earlier.status IN ('ACCEPTED', 'RUNNING')
              )
          ))
          -- One revision check per conversation at a time: an older and a newer check must not interleave.
          AND (:kind <> 'REVISION' OR NOT EXISTS (
              SELECT 1 FROM ai_job other
              WHERE other.tenant_id = j.tenant_id AND other.conversation_id = j.conversation_id
                AND other.epoch = j.epoch AND other.kind = 'REVISION' AND other.id <> j.id
                AND other.status = 'RUNNING' AND other.lease_until >= now()
          ))
        ORDER BY j.created_at
        LIMIT 1
        FOR UPDATE OF j SKIP LOCKED
    )
    UPDATE ai_job j SET
        status = 'RUNNING',
        lease_until = now() + make_interval(secs => :lease),
        lease_owner = :owner,
        fencing_token = j.fencing_token + 1,
        run_count = j.run_count + 1,
        updated_at = now()
    FROM candidate c
    WHERE j.id = c.id
    RETURNING j.*, c.status AS previous_status
    """
)


def _to_claimed(row: Any) -> ClaimedJob:
    m = row._mapping
    return ClaimedJob(
        id=m["id"],
        kind=m["kind"],
        tenant_id=m["tenant_id"],
        owner_id=m["owner_id"],
        conversation_id=m["conversation_id"],
        epoch=m["epoch"],
        turn_seq=m["turn_seq"],
        attempt_id=m["attempt_id"],
        response_id=m["response_id"],
        payload=m["payload"],
        mode=m["mode"],
        previous_status=m["previous_status"],
        fencing_token=m["fencing_token"],
        run_count=m["run_count"],
        deadline_at=m["deadline_at"],
        snapshot_id=m["snapshot_id"],
    )


async def claim(db: Database, kind: str, owner: str, lease_seconds: float) -> ClaimedJob | None:
    async with db.tx() as s:
        if kind == "GENERATION":
            row = (
                await s.execute(
                    _CLAIM_GENERATION, {"active": list(ACTIVE_STATUSES), "lease": lease_seconds, "owner": owner}
                )
            ).first()
        else:
            row = (await s.execute(_CLAIM_ORDERED, {"kind": kind, "lease": lease_seconds, "owner": owner})).first()
        return _to_claimed(row) if row else None


class FencedJob:
    def __init__(self, db: Database, job: ClaimedJob, lease_seconds: float) -> None:
        self.db = db
        self.job = job
        self.lease_seconds = lease_seconds

    @property
    def token(self) -> int:
        return self.job.fencing_token

    async def guard(self, s: AsyncSession, **updates: Any) -> int:
        """Extend the lease (and apply updates) iff we still hold the job. Returns next_event_seq."""
        sets = ", ".join(
            f"{k} = CAST(:{k} AS jsonb)" if k in _JSON_COLUMNS else f"{k} = :{k}" for k in updates
        )
        sql = text(
            f"""
            UPDATE ai_job SET lease_until = now() + make_interval(secs => :lease), updated_at = now()
            {", " + sets if sets else ""}
            WHERE id = :id AND fencing_token = :token AND status = ANY(:active)
            RETURNING next_event_seq
            """
        )
        params = {"id": self.job.id, "token": self.token, "lease": self.lease_seconds, "active": list(ACTIVE_STATUSES)}
        params.update({k: _json(v) if k in _JSON_COLUMNS else v for k, v in updates.items()})
        row = (await s.execute(sql, params)).first()
        if row is None:
            raise LeaseLost(self.job.id)
        return row[0]

    async def set_status(self, status: str, **updates: Any) -> None:
        async with self.db.tx() as s:
            await self.guard(s, status=status, **updates)

    async def append_event(self, s: AsyncSession, event: str, data: dict[str, Any]) -> int:
        seq = await self.guard(s)
        await s.execute(
            text(
                "UPDATE ai_job SET next_event_seq = next_event_seq + 1 WHERE id = :id AND fencing_token = :token"
            ),
            {"id": self.job.id, "token": self.token},
        )
        await s.execute(
            text(
                "INSERT INTO stream_event (job_id, event_seq, event, data, created_at) "
                "VALUES (:id, :seq, :event, CAST(:data AS jsonb), now())"
            ),
            {"id": self.job.id, "seq": seq, "event": event, "data": _json(data)},
        )
        return seq

    async def finish(self, s: AsyncSession, status: str, **updates: Any) -> None:
        """Terminal transition. Must be the last guarded write in the transaction."""
        await self.guard(s, **updates)
        res = await s.execute(
            text(
                "UPDATE ai_job SET status = :status, completed_at = now(), lease_until = NULL "
                "WHERE id = :id AND fencing_token = :token"
            ),
            {"status": status, "id": self.job.id, "token": self.token},
        )
        if res.rowcount != 1:  # type: ignore[attr-defined]
            raise LeaseLost(self.job.id)


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def lease_deadline(seconds: float) -> datetime:
    return utcnow() + timedelta(seconds=seconds)


async def requeue_failed_revisions(db: Database, after_seconds: float) -> int:
    """Give failed revision checks another chance once the model may be back.

    A failed check leaves that conversation's character facts hidden (fail closed); without this they
    would stay hidden until the next character edit.
    """
    async with db.tx() as s:
        res = await s.execute(
            text(
                """
                UPDATE ai_job j SET status = 'ACCEPTED', run_count = 0, error_code = NULL, error_detail = NULL,
                       completed_at = NULL, updated_at = now()
                FROM ai_conversation c
                WHERE j.kind = 'REVISION' AND j.status = 'FAILED'
                  AND j.completed_at < now() - make_interval(secs => :after)
                  AND c.tenant_id = j.tenant_id AND c.conversation_id = j.conversation_id AND c.epoch = j.epoch
                  AND c.status = 'ACTIVE'
                """
            ),
            {"after": after_seconds},
        )
        return res.rowcount  # type: ignore[attr-defined]

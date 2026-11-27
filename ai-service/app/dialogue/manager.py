"""Dialogue Manager (plan §04, §05, §09): one generation attempt, fixed order, no agent loop.

    validate boundary -> wait context -> (reuse | build + freeze) snapshot -> compose -> result
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ErrorCode, GenerationMode, GenerationRequest, JobStatus, error_body
from app.knowledge import composer, prompts
from app.knowledge.context_builder import (
    BudgetExceeded,
    ContextInputs,
    RecentTurn,
    build,
)
from app.knowledge.output_policy import ChunkPolicy, PolicyBlocked, expects_banmal
from app.knowledge.retriever import Scope, retrieve
from app.persistence.jobs import ClaimedJob, FencedJob, LeaseLost, utcnow
from app.persistence.models import new_id
from app.providers.base import ChatParams, PromptMessage, ProviderError, ProviderTimeout
from app.relationship.reducer import Stage, guideline
from app.relationship.unfinished import OpenEventView, describe, select_for_context
from app.services import Services
from app.understanding.normalize import derive_query_hint, normalize

log = logging.getLogger(__name__)
WEEKDAYS = "월화수목금토일"
MAX_PROVIDER_TRIES = 2


class AttemptFailed(Exception):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Snapshot:
    id: str
    messages: list[PromptMessage]
    character: dict[str, Any]
    degraded: list[str] = field(default_factory=list)


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


def humanize_gap(last: datetime | None, now: datetime) -> str:
    if last is None:
        return "처음 대화"
    delta = now - last
    if delta < timedelta(minutes=10):
        return "방금 전"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() // 60)}분 전"
    if delta < timedelta(days=1):
        return f"{int(delta.total_seconds() // 3600)}시간 전"
    return f"{delta.days}일 전"


def time_line(now: datetime, tz: ZoneInfo, last: datetime | None) -> str:
    local = now.astimezone(tz)
    return (
        f"지금: {local:%Y-%m-%d}({WEEKDAYS[local.weekday()]}) {local:%H:%M} · 마지막 대화: {humanize_gap(last, now)}"
    )


class DialogueManager:
    def __init__(self, services: Services) -> None:
        self.sv = services
        self.cfg = services.settings

    # ------------------------------------------------------------ entry
    async def run_attempt(self, job: ClaimedJob) -> None:
        fenced = FencedJob(self.sv.db, job, self.cfg.lease_seconds)
        try:
            req = GenerationRequest.model_validate(job.payload)
            if job.previous_status == JobStatus.STREAMING:
                # A previous worker died mid-stream. Partial output cannot be resumed (plan §09).
                raise AttemptFailed(ErrorCode.PROVIDER_ERROR, "worker lost during streaming")
            if req.deadline_at <= utcnow():
                raise AttemptFailed(ErrorCode.CONTEXT_NOT_READY, "deadline passed before start")
            conv = await self.wait_context(fenced, req)
            snapshot = await self.snapshot_for(fenced, req, conv)
            await self.compose(fenced, req, snapshot)
        except LeaseLost:
            log.info("lease lost or job revoked", extra={"job": job.id})
        except AttemptFailed as exc:
            await self._fail(fenced, exc.code, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("generation crashed", extra={"job": job.id})
            await self._fail(fenced, ErrorCode.INTERNAL, type(exc).__name__)

    # ------------------------------------------------------------ boundary & waiting
    async def _boundary(self, s: AsyncSession, req: GenerationRequest) -> dict[str, Any] | None:
        row = (
            await s.execute(
                text(
                    """
                    SELECT c.*, COALESCE(cur.applied_seq, 0) AS applied_seq
                    FROM ai_conversation c
                    LEFT JOIN lifecycle_cursor cur ON cur.tenant_id = c.tenant_id
                         AND cur.aggregate_type = 'CONVERSATION' AND cur.aggregate_id = c.conversation_id
                    WHERE c.tenant_id = :t AND c.owner_id = :o AND c.conversation_id = :c AND c.epoch = :e
                    """
                ),
                {"t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "e": req.epoch},
            )
        ).mappings().first()
        return dict(row) if row else None

    async def wait_context(self, fenced: FencedJob, req: GenerationRequest) -> dict[str, Any]:
        """Stage 1: lifecycle seq applied (DB-only, fast). Stage 2: memory watermark (short wait)."""
        limit = min(req.deadline_at, utcnow() + timedelta(seconds=self.cfg.context_wait_default_seconds))
        waiting_marked = False
        while True:
            async with self.sv.db.session() as s:
                conv = await self._boundary(s, req)
            if conv is not None and conv["status"] != "ACTIVE":
                raise AttemptFailed(ErrorCode.REVOKED, "conversation epoch revoked")
            ready = (
                conv is not None
                and conv["applied_seq"] >= req.required_lifecycle_seq
                and conv["character_rev"] >= req.character_revision
            )
            if ready:
                break
            if utcnow() >= limit:
                raise AttemptFailed(ErrorCode.CONTEXT_NOT_READY, "lifecycle events not applied before deadline")
            if not waiting_marked:
                await fenced.set_status(JobStatus.WAITING_CONTEXT.value)
                waiting_marked = True
            await asyncio.sleep(0.05)

        # Stage 2: turns leaving the recent window must already be memorized.
        need = req.turn_seq - self.cfg.recent_window_turns
        memory_limit = min(limit, utcnow() + timedelta(seconds=self.cfg.memory_wait_max_seconds))
        while conv["memory_through_turn_seq"] < need and utcnow() < memory_limit:
            await asyncio.sleep(0.1)
            async with self.sv.db.session() as s:
                conv = await self._boundary(s, req) or conv
        await fenced.set_status(JobStatus.RUNNING.value)
        return conv

    # ------------------------------------------------------------ snapshot
    async def snapshot_for(self, fenced: FencedJob, req: GenerationRequest, conv: dict[str, Any]) -> Snapshot:
        if req.mode == GenerationMode.REGENERATE or req.context_snapshot_id:
            snap = await self._load_snapshot(req, req.context_snapshot_id)
            if snap is None:
                if req.mode == GenerationMode.REGENERATE:
                    # Never silently rebuild: a different context is not a "regenerate" (plan §09).
                    raise AttemptFailed(ErrorCode.SNAPSHOT_EXPIRED, "context snapshot missing or expired")
            else:
                await fenced.set_status(JobStatus.RUNNING.value, snapshot_id=snap.id)
                return snap
        if req.mode == GenerationMode.RETRY:
            snap = await self._load_snapshot(req, None)
            if snap is not None:
                await fenced.set_status(JobStatus.RUNNING.value, snapshot_id=snap.id)
                return snap
        return await self._build_and_freeze(fenced, req, conv)

    async def _load_snapshot(self, req: GenerationRequest, snapshot_id: str | None) -> Snapshot | None:
        id_filter = "AND id = :id" if snapshot_id is not None else ""
        async with self.sv.db.session() as s:
            row = (
                await s.execute(
                    text(
                        f"""
                        SELECT * FROM context_snapshot
                        WHERE turn_seq = :ts {id_filter} AND tenant_id = :t AND owner_id = :o AND conversation_id = :c AND epoch = :e
                          AND expires_at > now()
                        ORDER BY created_at DESC LIMIT 1
                        """
                    ),
                    {"id": snapshot_id, "ts": req.turn_seq, "t": req.tenant_id, "o": req.owner_id,
                     "c": req.conversation_id, "e": req.epoch},
                )
            ).mappings().first()
        if row is None:
            return None
        content = row["content"]
        return Snapshot(
            id=row["id"],
            messages=[PromptMessage(m["role"], m["content"]) for m in content["messages"]],
            character=content.get("character", {}),
            degraded=list(row["degraded"] or []),
        )

    async def _build_and_freeze(self, fenced: FencedJob, req: GenerationRequest, conv: dict[str, Any]) -> Snapshot:
        tz = ZoneInfo(req.timezone)
        now = req.user_message.sent_at.astimezone(UTC)
        scope = Scope(req.tenant_id, req.owner_id, req.conversation_id, req.epoch)
        degraded: list[str] = []
        window = self.cfg.recent_window_turns
        lag = (req.turn_seq - 1) - conv["memory_through_turn_seq"]
        if lag > window:
            window = lag  # widen with raw text instead of silently missing memories
            degraded.append("MEMORY_LAG_WINDOW_WIDENED")

        async with self.sv.db.session() as s:
            settings = (
                await s.execute(
                    text(
                        "SELECT * FROM settings_snapshot WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e "
                        "AND character_rev=:cr AND persona_rev=:pr"
                    ),
                    {"t": req.tenant_id, "c": req.conversation_id, "e": req.epoch, "cr": req.character_revision,
                     "pr": req.persona_revision},
                )
            ).mappings().first()
            if settings is None:
                raise AttemptFailed(ErrorCode.CONTEXT_NOT_READY, "settings revision not received")

            turns = (
                await s.execute(
                    text(
                        """
                        SELECT turn_seq, user_text, assistant_text FROM committed_turn
                        WHERE tenant_id=:t AND owner_id=:o AND conversation_id=:c AND epoch=:e AND turn_seq <= :base
                        ORDER BY turn_seq DESC LIMIT :n
                        """
                    ),
                    {"t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "e": req.epoch,
                     "base": req.base_committed_turn_seq, "n": window + 1},
                )
            ).all()
            recent = [RecentTurn(r[0], r[1], r[2], is_opening_scene=r[0] == 0) for r in reversed(turns)]
            if req.base_committed_turn_seq > 0 and (not turns or turns[0][0] < req.base_committed_turn_seq):
                raise AttemptFailed(ErrorCode.CONTEXT_NOT_READY, "base committed turn not projected yet")

            rel = (
                await s.execute(
                    text("SELECT stage FROM relationship_state WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e"),
                    {"t": req.tenant_id, "c": req.conversation_id, "e": req.epoch},
                )
            ).scalar()

            summary = (
                await s.execute(
                    text(
                        "SELECT text FROM summary WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e "
                        "ORDER BY to_turn_seq DESC LIMIT 1"
                    ),
                    {"t": req.tenant_id, "c": req.conversation_id, "e": req.epoch},
                )
            ).scalar()

            user_text = normalize(req.user_message.text)
            query = derive_query_hint(user_text, [t.user_text for t in recent if t.user_text])
            query_vec: list[float] | None = None
            try:
                vecs, _ = await self.sv.provider.embed([query], self.cfg.embedding_model)
                query_vec = vecs[0]
            except ProviderError:
                degraded.append("QUERY_EMBEDDING_FAILED")
            retrieved = await retrieve(
                s, scope, query, query_vec, self.cfg.embedding_model, now, tz,
                character_revision=req.character_revision, top_k=self.cfg.retrieval_top_k, gate=self.cfg.retrieval_gate,
            )
            degraded.extend(retrieved.degraded)

            sim_sql = (
                "CASE WHEN m.embedding IS NOT NULL AND m.embedding_version = :ev "
                "THEN 1 - (m.embedding <=> CAST(:q AS vector)) ELSE 0 END"
                if query_vec
                else "0"
            )
            # Open promises/plans survive a character edit (unlike character facts in retrieval):
            # an edit does not cancel what the two already agreed to do.
            open_params: dict[str, Any] = {
                "t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "e": req.epoch,
            }
            if query_vec:
                open_params |= {"ev": self.cfg.embedding_model, "q": json.dumps(query_vec)}
            open_rows = (
                await s.execute(
                    text(
                        f"""
                        SELECT u.id, u.title, u.status, u.due_at, u.created_at, {sim_sql} AS sim
                        FROM unfinished_event u JOIN memory m ON m.id = u.memory_id
                        WHERE u.tenant_id=:t AND u.owner_id=:o AND u.conversation_id=:c AND u.epoch=:e
                          AND u.status = 'OPEN'
                        """
                    ),
                    open_params,
                )
            ).mappings().all()

        recent_assistant = " ".join(t.assistant_text for t in recent[-3:])
        events = [OpenEventView(r["id"], r["title"], r["status"], r["due_at"], r["created_at"]) for r in open_rows]
        mentioned = {e.id for e in events if e.title and e.title[:6] in recent_assistant}
        picked = select_for_context(
            events, now, {r["id"]: float(r["sim"] or 0) for r in open_rows}, mentioned, self.cfg.unfinished_gate
        )

        character = settings["character_card"]
        inputs = ContextInputs(
            policy_text=prompts.chat_policy(req.generation_policy_version),
            character=character,
            character_rev=req.character_revision,
            persona=settings["persona"],
            relationship_guideline=guideline(Stage(rel)) if rel else None,
            open_events=[describe(ev, st, now, tz) for ev, st in picked],
            memories=retrieved.lines,
            summary=summary,
            time_line=time_line(now, tz, conv["last_turn_at"]),
            recent_turns=recent,
            current_user_text=user_text,
            recent_window=window + (1 if recent and recent[0].is_opening_scene else 0),
            min_recent_window=self.cfg.min_recent_window_turns,
        )
        try:
            built = build(inputs, self.sv.provider.count_tokens)
        except BudgetExceeded as exc:
            raise AttemptFailed(ErrorCode.BUDGET_EXCEEDED, str(exc)) from exc
        degraded.extend(built.degraded)

        snapshot_id = new_id()
        content = built.content() | {"character": character}
        async with self.sv.db.tx() as s:
            await fenced.guard(s)
            await s.execute(
                text(
                    """
                    INSERT INTO context_snapshot (id, tenant_id, owner_id, conversation_id, epoch, turn_seq, content,
                        input_hash, model_version, prompt_version, memory_ids, degraded, expires_at, created_at)
                    VALUES (:id, :t, :o, :c, :e, :ts, CAST(:content AS jsonb), :hash, :mv, :pv,
                            CAST(:mids AS jsonb), CAST(:deg AS jsonb), :exp, now())
                    """
                ),
                {"id": snapshot_id, "t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "e": req.epoch,
                 "ts": req.turn_seq, "content": _j(content), "hash": built.input_hash(), "mv": self.cfg.chat_model,
                 "pv": req.generation_policy_version, "mids": _j(built.memory_ids), "deg": _j(degraded),
                 "exp": utcnow() + timedelta(hours=self.cfg.snapshot_ttl_hours)},
            )
            await fenced.guard(s, snapshot_id=snapshot_id, degraded=degraded)
        return Snapshot(snapshot_id, built.messages, character, degraded)

    # ------------------------------------------------------------ compose
    async def compose(self, fenced: FencedJob, req: GenerationRequest, snap: Snapshot) -> None:
        temperature = 0.8 + (self.cfg.regenerate_temperature_bump if req.mode == GenerationMode.REGENERATE else 0)
        params = ChatParams(model=self.cfg.chat_model, temperature=temperature, max_output_tokens=600)
        started = False

        async def emit(chunk: str) -> None:
            nonlocal started
            async with self.sv.db.tx() as s:
                if not started:
                    await fenced.guard(s, status=JobStatus.STREAMING.value)
                    started = True
                await fenced.append_event(s, "delta", {"text": chunk})

        last_error: Exception | None = None
        result = None
        for _ in range(MAX_PROVIDER_TRIES):
            policy = ChunkPolicy(expects_banmal=expects_banmal(snap.character.get("speechStyle", "")))
            try:
                result = await composer.generate(self.sv.provider, snap.messages, params, policy, emit)
                break
            except composer.StreamStartedError as exc:
                code = ErrorCode.PROVIDER_TIMEOUT if isinstance(exc.cause, ProviderTimeout) else ErrorCode.PROVIDER_ERROR
                raise AttemptFailed(code, "provider failed after streaming started") from exc
            except PolicyBlocked as exc:
                raise AttemptFailed(ErrorCode.POLICY_BLOCKED, "output policy blocked the answer") from exc
            except ProviderError as exc:
                last_error = exc
                if req.deadline_at <= utcnow():
                    break
        if result is None:
            code = ErrorCode.PROVIDER_TIMEOUT if isinstance(last_error, ProviderTimeout) else ErrorCode.PROVIDER_ERROR
            raise AttemptFailed(code, "provider failed")

        usage = result.usage.as_dict()
        degraded = snap.degraded + sorted(result.flags)
        async with self.sv.db.tx() as s:
            await fenced.append_event(
                s, "completed",
                {"responseId": req.response_id, "text": result.text, "usage": usage, "contextSnapshotId": snap.id},
            )
            await s.execute(
                text(
                    """
                    INSERT INTO usage_ledger (id, tenant_id, owner_id, conversation_id, attempt_id, purpose, model,
                        input_tokens, output_tokens, est_cost, created_at)
                    VALUES (:id, :t, :o, :c, :a, 'gen', :m, :i, :out, 0, now())
                    """
                ),
                {"id": new_id(), "t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "a": req.attempt_id,
                 "m": result.usage.model, "i": result.usage.input_tokens, "out": result.usage.output_tokens},
            )
            await fenced.finish(s, JobStatus.COMPLETED.value, result_text=result.text, usage=usage, degraded=degraded)

    async def _fail(self, fenced: FencedJob, code: ErrorCode, message: str) -> None:
        try:
            async with self.sv.db.tx() as s:
                await fenced.append_event(s, "failed", {"error": error_body(code, message)})
                await fenced.finish(s, JobStatus.FAILED.value, error_code=code.value, error_detail=message)
        except LeaseLost:
            pass

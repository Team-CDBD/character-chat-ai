"""AI API /internal/ai/v1 (plan §10).

Accepting and executing are separate: no handler here ever calls a model. Requests are
persisted and answered with 202; workers do the rest.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    ConversationState,
    ErrorBody,
    ErrorCode,
    GenerationAccepted,
    GenerationRequest,
    GenerationResult,
    JobStatus,
    LifecycleAccepted,
    LifecycleEvent,
    LifecycleReceipt,
    Usage,
)
from app.lifecycle.handler import is_revoked, lock_conversation_scope, receive_event
from app.persistence.models import new_id
from app.services import Services

router = APIRouter(prefix="/internal/ai/v1")


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def require_service_auth(request: Request, authorization: Annotated[str | None, Header()] = None) -> None:
    tokens: list[str] = request.app.state.services.settings.service_tokens
    supplied = (authorization or "").removeprefix("Bearer ").strip()
    if not supplied or not any(hmac.compare_digest(supplied, t) for t in tokens):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, {"code": "UNAUTHORIZED", "message": "service token required"})


def api_error(code: int, err: str, message: str) -> HTTPException:
    return HTTPException(code, {"code": err, "message": message})


def payload_hash(req: GenerationRequest) -> str:
    # traceId is transport metadata, not part of the request's meaning.
    body = req.model_dump(by_alias=True, mode="json", exclude={"trace_id"})
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------- generations


@router.post(
    "/generations",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=GenerationAccepted,
    response_model_by_alias=True,
    dependencies=[Depends(require_service_auth)],
    responses={409: {"description": "IDEMPOTENCY_CONFLICT | ACTIVE_GENERATION_EXISTS | EPOCH_REVOKED"},
               404: {"description": "CONVERSATION_NOT_REGISTERED"}},
)
async def accept_generation(req: GenerationRequest, sv: ServicesDep) -> GenerationAccepted:
    h = payload_hash(req)
    try:
        return await _accept(sv, req, h)
    except IntegrityError as exc:
        if "uq_active_generation" in str(exc.orig):
            raise api_error(409, "ACTIVE_GENERATION_EXISTS", "a generation is already running for this conversation") from exc
        # Lost a race with an identical request: re-read and answer idempotently.
        async with sv.db.session() as s:
            row = (
                await s.execute(
                    text("SELECT id, payload_hash, status FROM ai_job WHERE kind='GENERATION' AND attempt_id=:a"),
                    {"a": req.attempt_id},
                )
            ).first()
        if row is None or row[1] != h:
            raise api_error(409, "IDEMPOTENCY_CONFLICT", "same attemptId with a different payload") from exc
        return GenerationAccepted(ai_job_id=row[0], status=JobStatus(row[2]))


async def _accept(sv: Services, req: GenerationRequest, h: str) -> GenerationAccepted:
    """Check the conversation and store the job in ONE transaction under the conversation lock.

    Reset / delete take the same lock before revoking (lifecycle/handler.py ``_revoke_epoch``), so the two
    are strictly ordered per conversation: either this job exists first and is revoked + scrubbed by the
    cleanup, or the cleanup committed first and this request is refused. The user's message is never stored
    for a conversation whose data was already deleted.
    """
    async with sv.db.tx() as s:
        existing = (
            await s.execute(
                text("SELECT id, payload_hash, status FROM ai_job WHERE kind='GENERATION' AND attempt_id=:a"),
                {"a": req.attempt_id},
            )
        ).first()
        if existing:
            if existing[1] != h:
                raise api_error(409, "IDEMPOTENCY_CONFLICT", "same attemptId with a different payload")
            return GenerationAccepted(ai_job_id=existing[0], status=JobStatus(existing[2]))

        await lock_conversation_scope(s, req.tenant_id, req.conversation_id)
        conv = (
            await s.execute(
                text("SELECT status, character_id FROM ai_conversation WHERE tenant_id=:t AND owner_id=:o "
                     "AND conversation_id=:c AND epoch=:e"),
                {"t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "e": req.epoch},
            )
        ).first()
        if (conv is not None and conv[0] == "REVOKED") or await is_revoked(
            s, req.tenant_id, req.conversation_id, req.epoch, conv[1] if conv else None
        ):
            raise api_error(409, "EPOCH_REVOKED", "conversation epoch was reset or deleted")
        if conv is None:
            pending = (
                await s.execute(
                    text(
                        "SELECT 1 FROM event_inbox WHERE tenant_id=:t AND aggregate_type='CONVERSATION' "
                        "AND aggregate_id=:c AND status IN ('RECEIVED', 'FAILED') LIMIT 1"
                    ),
                    {"t": req.tenant_id, "c": req.conversation_id},
                )
            ).first()
            if pending is None:
                raise api_error(404, "CONVERSATION_NOT_REGISTERED", "ConversationOpened not received for this epoch")

        job_id = new_id()
        await _insert_generation(s, job_id, req, h)
    return GenerationAccepted(ai_job_id=job_id, status=JobStatus.ACCEPTED)


async def _insert_generation(s: AsyncSession, job_id: str, req: GenerationRequest, h: str) -> None:
    await s.execute(
        text(
            """
            INSERT INTO ai_job (id, kind, tenant_id, owner_id, conversation_id, epoch, turn_seq, attempt_id,
                response_id, payload_hash, payload, mode, status, fencing_token, run_count, deadline_at,
                next_event_seq, degraded, created_at, updated_at)
            VALUES (:id, 'GENERATION', :t, :o, :c, :e, :ts, :a, :r, :h, CAST(:p AS jsonb), :m, 'ACCEPTED', 0, 0,
                :d, 2, '[]'::jsonb, now(), now())
            """
        ),
        {"id": job_id, "t": req.tenant_id, "o": req.owner_id, "c": req.conversation_id, "e": req.epoch,
         "ts": req.turn_seq, "a": req.attempt_id, "r": req.response_id, "h": h,
         "p": req.model_dump_json(by_alias=True), "m": req.mode.value, "d": req.deadline_at},
    )
    await s.execute(
        text(
            "INSERT INTO stream_event (job_id, event_seq, event, data, created_at) "
            "VALUES (:id, 1, 'accepted', CAST(:d AS jsonb), now())"
        ),
        {"id": job_id, "d": json.dumps({"aiJobId": job_id, "attemptId": req.attempt_id})},
    )


def _result(row: Any) -> GenerationResult:
    error = None
    if row["error_code"]:
        code = ErrorCode(row["error_code"])
        error = ErrorBody(code=code, message=row["error_detail"] or code.value, retryable=code.retryable)
    usage = None
    if row["usage"]:
        u = row["usage"]
        usage = Usage(input_tokens=u.get("inputTokens", 0), output_tokens=u.get("outputTokens", 0), model=u.get("model"))
    return GenerationResult(
        ai_job_id=row["id"], attempt_id=row["attempt_id"], response_id=row["response_id"],
        status=JobStatus(row["status"]), text=row["result_text"], usage=usage, error=error,
        context_snapshot_id=row["snapshot_id"], degraded=list(row["degraded"] or []),
    )


async def _load_job(sv: Services, job_id: str):
    async with sv.db.session() as s:
        row = (
            await s.execute(text("SELECT * FROM ai_job WHERE id=:id AND kind='GENERATION'"), {"id": job_id})
        ).mappings().first()
    if row is None:
        raise api_error(404, "NOT_FOUND", "generation not found")
    return row


@router.get(
    "/generations/{ai_job_id}",
    response_model=GenerationResult,
    response_model_by_alias=True,
    dependencies=[Depends(require_service_auth)],
)
async def get_result(ai_job_id: str, sv: ServicesDep) -> GenerationResult:
    """Stored result only; never calls the model again (plan §09 화면 이탈)."""
    return _result(await _load_job(sv, ai_job_id))


def _sse(seq: int, event: str, data: dict[str, Any]) -> bytes:
    return f"id: {seq}\nevent: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


@router.get(
    "/generations/{ai_job_id}/events",
    dependencies=[Depends(require_service_auth)],
    response_class=StreamingResponse,
    responses={200: {"content": {"text/event-stream": {}}, "description": "accepted / delta / completed / failed"}},
)
async def stream_events(
    ai_job_id: str,
    sv: ServicesDep,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after: Annotated[int | None, Query(description="Alternative to the Last-Event-ID header")] = None,
) -> StreamingResponse:
    await _load_job(sv, ai_job_id)
    start = after if after is not None else int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
    cfg = sv.settings

    async def gen() -> AsyncIterator[bytes]:
        cursor = start
        idle = 0.0
        while True:
            async with sv.db.session() as s:
                events = (
                    await s.execute(
                        text("SELECT event_seq, event, data FROM stream_event WHERE job_id=:id AND event_seq > :c "
                             "ORDER BY event_seq LIMIT 200"),
                        {"id": ai_job_id, "c": cursor},
                    )
                ).all()
                job = None if events else (
                    await s.execute(text("SELECT * FROM ai_job WHERE id=:id"), {"id": ai_job_id})
                ).mappings().first()
            for seq, event, data in events:
                cursor = seq
                yield _sse(seq, event, data)
                if event in ("completed", "failed"):
                    return
            if events:
                idle = 0.0
                continue
            if job is not None and JobStatus(job["status"]).terminal:
                # Events outside retention: replace with the final result (plan §10).
                res = _result(job).model_dump(by_alias=True, mode="json")
                yield _sse(job["next_event_seq"], "completed" if job["status"] == "COMPLETED" else "failed", res)
                return
            await asyncio.sleep(cfg.sse_poll_interval_seconds)
            idle += cfg.sse_poll_interval_seconds
            if idle >= cfg.sse_keepalive_seconds:
                idle = 0.0
                yield b": keepalive\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- lifecycle


@router.post(
    "/lifecycle-events",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=LifecycleAccepted,
    response_model_by_alias=True,
    dependencies=[Depends(require_service_auth)],
)
async def post_lifecycle_event(event: LifecycleEvent, sv: ServicesDep) -> LifecycleAccepted:
    """202 means "stored in the Inbox", not "applied"."""
    return LifecycleAccepted(receipt_id=await receive_event(sv.db, event))


@router.get(
    "/lifecycle-events/{receipt_id}",
    response_model=LifecycleReceipt,
    response_model_by_alias=True,
    dependencies=[Depends(require_service_auth)],
)
async def get_lifecycle_receipt(receipt_id: str, sv: ServicesDep) -> LifecycleReceipt:
    async with sv.db.session() as s:
        row = (
            await s.execute(text("SELECT id, event_id, status, error, applied_at, attempts, next_attempt_at "
                                 "FROM event_inbox WHERE id=:id"), {"id": receipt_id})
        ).first()
    if row is None:
        raise api_error(404, "NOT_FOUND", "receipt not found")
    return LifecycleReceipt(receipt_id=row[0], event_id=row[1], status=row[2], error=row[3], applied_at=row[4],
                            attempts=row[5], next_retry_at=row[6] if row[2] == "FAILED" else None)


@router.get(
    "/conversations/{conversation_id}/state",
    response_model=ConversationState,
    response_model_by_alias=True,
    dependencies=[Depends(require_service_auth)],
)
async def get_conversation_state(
    conversation_id: str,
    sv: ServicesDep,
    tenant_id: Annotated[str, Query(alias="tenantId")],
    epoch: int,
) -> ConversationState:
    """Diagnostics only: watermarks, never memory contents (MVP has no Memory UI)."""
    async with sv.db.session() as s:
        row = (
            await s.execute(
                text(
                    """
                    SELECT c.*, COALESCE(cur.applied_seq, 0) AS applied_seq FROM ai_conversation c
                    LEFT JOIN lifecycle_cursor cur ON cur.tenant_id = c.tenant_id
                         AND cur.aggregate_type = 'CONVERSATION' AND cur.aggregate_id = c.conversation_id
                    WHERE c.tenant_id=:t AND c.conversation_id=:c AND c.epoch=:e
                    """
                ),
                {"t": tenant_id, "c": conversation_id, "e": epoch},
            )
        ).mappings().first()
    if row is None:
        raise api_error(404, "NOT_FOUND", "conversation epoch not found")
    return ConversationState(
        conversation_id=conversation_id, epoch=epoch, status=row["status"], applied_lifecycle_seq=row["applied_seq"],
        memory_through_turn_seq=row["memory_through_turn_seq"],
        embedding_through_turn_seq=row["embedding_through_turn_seq"],
        character_revision=row["character_rev"], persona_revision=row["persona_rev"],
    )


def error_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, dict) else {"code": "ERROR", "message": str(exc.detail)}
    return JSONResponse({"error": detail}, status_code=exc.status_code)

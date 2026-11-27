"""Lifecycle Handler (plan §05, §09, §10).

* ``receive_event`` only persists to the Inbox (202 != applied).
* ``process_pending`` applies events strictly in ``aggregateSeq`` order per aggregate.
  A gap holds the aggregate; nothing is skipped.
* Reset / delete revoke the epoch first (tombstone + fencing bump on running jobs), then clean up.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import TypeAdapter
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    ACTIVE_STATUSES,
    AnswerCommittedPayload,
    CharacterDeletedPayload,
    CharacterRevisionChangedPayload,
    ConversationOpenedPayload,
    ConversationResetPayload,
    ErrorCode,
    LifecycleEvent,
    error_body,
)
from app.persistence.db import Database
from app.persistence.models import new_id
from app.relationship.reducer import baseline_state

log = logging.getLogger(__name__)
_event_adapter: TypeAdapter[Any] = TypeAdapter(LifecycleEvent)


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


# ---------------------------------------------------------------- inbox


async def receive_event(db: Database, event: Any) -> str:
    payload = event.payload.model_dump(by_alias=True, mode="json")
    async with db.tx() as s:
        if "epoch" in payload:
            conversation = payload["conversationId"]
            await _lock_inbox_scope(s, event.tenant_id, conversation)
            if await is_revoked(s, event.tenant_id, conversation, payload["epoch"], payload.get("characterId")):
                payload = {}
        row = (
            await s.execute(
                text(
                    """
                    INSERT INTO event_inbox (id, event_id, tenant_id, owner_id, aggregate_type, aggregate_id,
                                             aggregate_seq, type, payload, status, received_at)
                    VALUES (:id, :event_id, :tenant, :owner, :atype, :aid, :seq, :type, CAST(:payload AS jsonb),
                            'RECEIVED', now())
                    ON CONFLICT (event_id) DO NOTHING
                    RETURNING id
                    """
                ),
                {
                    "id": new_id(),
                    "event_id": event.event_id,
                    "tenant": event.tenant_id,
                    "owner": event.owner_id,
                    "atype": event.aggregate_type.value,
                    "aid": event.aggregate_id,
                    "seq": event.aggregate_seq,
                    "type": event.type,
                    "payload": _j(payload),
                },
            )
        ).first()
        if row:
            return row[0]
        existing = (
            await s.execute(text("SELECT id FROM event_inbox WHERE event_id = :e"), {"e": event.event_id})
        ).scalar_one()
        return existing


async def process_pending(db: Database, batch: int = 50) -> int:
    """Apply ready events. Returns how many were applied (or failed)."""
    async with db.session() as s:
        rows = (
            await s.execute(
                text(
                    """
                    SELECT i.id FROM event_inbox i
                    LEFT JOIN lifecycle_cursor c
                      ON c.tenant_id = i.tenant_id AND c.aggregate_type = i.aggregate_type
                     AND c.aggregate_id = i.aggregate_id
                    WHERE i.status = 'RECEIVED' AND i.aggregate_seq <= COALESCE(c.applied_seq, 0) + 1
                    ORDER BY i.aggregate_seq, i.received_at
                    LIMIT :n
                    """
                ),
                {"n": batch},
            )
        ).all()
    done = 0
    for (inbox_id,) in rows:
        if await _apply_one(db, inbox_id):
            done += 1
    return done


async def _apply_one(db: Database, inbox_id: str) -> bool:
    try:
        async with db.tx() as s:
            ev = (
                await s.execute(
                    text("SELECT * FROM event_inbox WHERE id = :id AND status = 'RECEIVED' FOR UPDATE SKIP LOCKED"),
                    {"id": inbox_id},
                )
            ).mappings().first()
            if ev is None:
                return False
            await s.execute(
                text(
                    "INSERT INTO lifecycle_cursor (tenant_id, aggregate_type, aggregate_id, applied_seq) "
                    "VALUES (:t, :at, :a, 0) ON CONFLICT DO NOTHING"
                ),
                {"t": ev["tenant_id"], "at": ev["aggregate_type"], "a": ev["aggregate_id"]},
            )
            cursor = (
                await s.execute(
                    text(
                        "SELECT applied_seq FROM lifecycle_cursor WHERE tenant_id=:t AND aggregate_type=:at "
                        "AND aggregate_id=:a FOR UPDATE"
                    ),
                    {"t": ev["tenant_id"], "at": ev["aggregate_type"], "a": ev["aggregate_id"]},
                )
            ).scalar_one()
            if ev["aggregate_seq"] <= cursor:
                # Same seq re-sent under a new event id: already applied.
                await _mark(s, inbox_id, "APPLIED", "duplicate_seq")
                return True
            if ev["aggregate_seq"] != cursor + 1:
                return False  # gap: hold
            if ev["payload"]:
                await APPLIERS[ev["type"]](s, ev["tenant_id"], ev["owner_id"], ev["payload"])
            await s.execute(
                text(
                    "UPDATE lifecycle_cursor SET applied_seq = :seq WHERE tenant_id=:t AND aggregate_type=:at "
                    "AND aggregate_id=:a"
                ),
                {"seq": ev["aggregate_seq"], "t": ev["tenant_id"], "at": ev["aggregate_type"], "a": ev["aggregate_id"]},
            )
            await _mark(s, inbox_id, "APPLIED", None)
        return True
    except Exception as exc:  # the aggregate stays blocked until the event is fixed and re-sent
        log.exception("lifecycle event failed", extra={"inbox_id": inbox_id})
        async with db.tx() as s:
            await _mark(s, inbox_id, "FAILED", type(exc).__name__)
        return True


async def _mark(s: AsyncSession, inbox_id: str, status: str, error: str | None) -> None:
    await s.execute(
        text("UPDATE event_inbox SET status = :st, error = :err, applied_at = now(), payload = '{}'::jsonb WHERE id = :id"),
        {"st": status, "err": error, "id": inbox_id},
    )


# ---------------------------------------------------------------- revocation helpers


async def is_revoked(s: AsyncSession, tenant: str, conversation_id: str, epoch: int, character_id: str | None) -> bool:
    row = (
        await s.execute(
            text(
                """
                SELECT 1 FROM revocation WHERE tenant_id = :t AND (
                    (scope = 'CHARACTER' AND target_id = :ch)
                 OR (scope = 'CONVERSATION' AND target_id = :c)
                 OR (scope = 'EPOCH' AND target_id = :c AND epoch = :e))
                LIMIT 1
                """
            ),
            {"t": tenant, "ch": character_id or "", "c": conversation_id, "e": epoch},
        )
    ).first()
    return row is not None


async def _revoke_epoch(s: AsyncSession, tenant: str, conversation_id: str, epoch: int, reason: str) -> None:
    """Block first, then clean. Running jobs lose their fencing token immediately."""
    await s.execute(
        text(
            "INSERT INTO revocation (id, tenant_id, scope, target_id, epoch, reason, created_at) "
            "VALUES (:id, :t, 'EPOCH', :c, :e, :r, now())"
        ),
        {"id": new_id(), "t": tenant, "c": conversation_id, "e": epoch, "r": reason},
    )
    await s.execute(
        text("UPDATE ai_conversation SET status = 'REVOKED' WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e"),
        {"t": tenant, "c": conversation_id, "e": epoch},
    )
    jobs = (
        await s.execute(
            text(
                """
                UPDATE ai_job SET status = 'REVOKED', fencing_token = fencing_token + 1, completed_at = now(),
                       error_code = 'REVOKED', lease_until = NULL, next_event_seq = next_event_seq + 1
                WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e AND status = ANY(:active)
                RETURNING id, kind, next_event_seq - 1 AS seq
                """
            ),
            {"t": tenant, "c": conversation_id, "e": epoch, "active": list(ACTIVE_STATUSES)},
        )
    ).all()
    for job_id, kind, seq in jobs:
        if kind == "GENERATION":
            await s.execute(
                text(
                    "INSERT INTO stream_event (job_id, event_seq, event, data, created_at) "
                    "VALUES (:id, :seq, 'failed', CAST(:d AS jsonb), now())"
                ),
                {"id": job_id, "seq": seq, "d": _j({"error": error_body(ErrorCode.REVOKED, reason)})},
            )
    await _purge_epoch(s, tenant, conversation_id, epoch, include_settings=False)


_DERIVED_TABLES = (
    "committed_turn",
    "memory",
    "unfinished_event",
    "relationship_event",
    "relationship_state",
    "summary",
    "context_snapshot",
)


async def _lock_inbox_scope(s: AsyncSession, tenant: str, conversation_id: str) -> None:
    await s.execute(text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
                    {"scope": _j([tenant, conversation_id])})


async def _purge_epoch(s: AsyncSession, tenant: str, conversation_id: str, epoch: int | None, *, include_settings: bool) -> None:
    tables = _DERIVED_TABLES + (("settings_snapshot",) if include_settings else ())
    epoch_filter = "" if epoch is None else " AND epoch = :e"
    params: dict[str, Any] = {"t": tenant, "c": conversation_id}
    if epoch is not None:
        params["e"] = epoch
    await _lock_inbox_scope(s, tenant, conversation_id)
    inbox_epoch = "AND payload->>'epoch' = CAST(:e AS text)" if epoch is not None else ""
    await s.execute(text(
        "UPDATE event_inbox SET payload = '{}'::jsonb WHERE tenant_id=:t AND payload->>'conversationId'=:c "
        f"AND type IN ('ConversationOpened','CharacterRevisionChanged','AnswerCommitted') {inbox_epoch}"
    ), {**params, "e": str(epoch)} if epoch is not None else params)
    for table in tables:
        await s.execute(text(f"DELETE FROM {table} WHERE tenant_id=:t AND conversation_id=:c{epoch_filter}"), params)
    # Keep job rows (status/result metadata for BE queries) but drop generated text and stream bodies.
    await s.execute(
        text(
            f"DELETE FROM stream_event WHERE job_id IN (SELECT id FROM ai_job WHERE tenant_id=:t "
            f"AND conversation_id=:c{epoch_filter})"
        ),
        params,
    )
    await s.execute(
        text(f"UPDATE ai_job SET result_text = NULL, payload = '{{}}'::jsonb WHERE tenant_id=:t AND conversation_id=:c{epoch_filter}"),
        params,
    )


# ---------------------------------------------------------------- appliers


async def _open_epoch(
    s: AsyncSession,
    tenant: str,
    owner: str,
    conversation_id: str,
    character_id: str,
    epoch: int,
    character_rev: int,
    persona_rev: int,
    character: dict[str, Any],
    persona: dict[str, Any] | None,
    opening_scene: dict[str, Any] | None,
) -> None:
    inserted = (
        await s.execute(
            text(
                """
                INSERT INTO ai_conversation (id, tenant_id, owner_id, conversation_id, epoch, character_id, status,
                    character_rev, persona_rev, memory_through_turn_seq, embedding_through_turn_seq,
                    summary_through_turn_seq, created_at)
                VALUES (:id, :t, :o, :c, :e, :ch, 'ACTIVE', :cr, :pr, 0, 0, 0, now())
                ON CONFLICT (tenant_id, conversation_id, epoch) DO NOTHING
                RETURNING id
                """
            ),
            {"id": new_id(), "t": tenant, "o": owner, "c": conversation_id, "e": epoch, "ch": character_id,
             "cr": character_rev, "pr": persona_rev},
        )
    ).first()
    if inserted is None:
        return
    scope = {"t": tenant, "o": owner, "c": conversation_id, "e": epoch}
    await s.execute(
        text(
            """
            INSERT INTO settings_snapshot (id, tenant_id, owner_id, conversation_id, epoch, character_rev, persona_rev,
                character_card, persona, opening_scene, created_at)
            VALUES (:id, :t, :o, :c, :e, :cr, :pr, CAST(:card AS jsonb), CAST(:persona AS jsonb),
                    CAST(:scene AS jsonb), now())
            ON CONFLICT DO NOTHING
            """
        ),
        {**scope, "id": new_id(), "cr": character_rev, "pr": persona_rev, "card": _j(character),
         "persona": _j(persona), "scene": _j(opening_scene)},
    )
    if opening_scene:
        await s.execute(
            text(
                """
                INSERT INTO committed_turn (id, tenant_id, owner_id, conversation_id, epoch, turn_seq, user_msg_id,
                    user_text, assistant_msg_id, assistant_text, character_rev, sent_at, created_at)
                VALUES (:id, :t, :o, :c, :e, 0, NULL, NULL, :mid, :txt, :cr, now(), now())
                ON CONFLICT DO NOTHING
                """
            ),
            {**scope, "id": new_id(), "mid": opening_scene.get("messageId"),
             "txt": ((f"*{opening_scene.get('narration')}*\n" if opening_scene.get("narration") else "")
                     + (opening_scene.get("characterLine") or "")).strip(),
             "cr": character_rev},
        )
    base = baseline_state(50, 40, 0)
    await s.execute(
        text(
            """
            INSERT INTO relationship_state (id, tenant_id, owner_id, conversation_id, epoch, trust, closeness, tension,
                stage, baseline_source, daily_gain, through_turn_seq, created_at)
            VALUES (:id, :t, :o, :c, :e, :tr, :cl, :tn, :st, 'DEFAULT', '{}'::jsonb, 0, now())
            ON CONFLICT DO NOTHING
            """
        ),
        {**scope, "id": new_id(), "tr": base.trust, "cl": base.closeness, "tn": base.tension, "st": base.stage.value},
    )
    await enqueue_job(s, "BASELINE", tenant, owner, conversation_id, epoch, 0, f"baseline:{conversation_id}:{epoch}", {})


async def enqueue_job(
    s: AsyncSession, kind: str, tenant: str, owner: str, conversation_id: str, epoch: int, turn_seq: int,
    attempt_id: str, payload: dict[str, Any],
) -> None:
    await s.execute(
        text(
            """
            INSERT INTO ai_job (id, kind, tenant_id, owner_id, conversation_id, epoch, turn_seq, attempt_id,
                payload_hash, payload, status, fencing_token, run_count, next_event_seq, degraded, created_at, updated_at)
            VALUES (:id, :k, :t, :o, :c, :e, :ts, :a, '', CAST(:p AS jsonb), 'ACCEPTED', 0, 0, 1, '[]'::jsonb,
                    now(), now())
            ON CONFLICT (kind, attempt_id) DO NOTHING
            """
        ),
        {"id": new_id(), "k": kind, "t": tenant, "o": owner, "c": conversation_id, "e": epoch, "ts": turn_seq,
         "a": attempt_id, "p": _j(payload)},
    )


async def apply_conversation_opened(s: AsyncSession, tenant: str, owner: str, raw: dict[str, Any]) -> None:
    p = ConversationOpenedPayload.model_validate(raw)
    if await is_revoked(s, tenant, p.conversation_id, p.epoch, p.character_id):
        return
    await _open_epoch(
        s, tenant, owner, p.conversation_id, p.character_id, p.epoch, p.character_revision, p.persona_revision,
        p.character.model_dump(by_alias=True), p.persona.model_dump(by_alias=True) if p.persona else None,
        p.opening_scene.model_dump(by_alias=True) if p.opening_scene else None,
    )


async def _active_conversation(s: AsyncSession, tenant: str, owner: str, conversation_id: str, epoch: int):
    return (
        await s.execute(
            text(
                "SELECT * FROM ai_conversation WHERE tenant_id=:t AND owner_id=:o AND conversation_id=:c "
                "AND epoch=:e AND status='ACTIVE' FOR UPDATE"
            ),
            {"t": tenant, "o": owner, "c": conversation_id, "e": epoch},
        )
    ).mappings().first()


async def apply_answer_committed(s: AsyncSession, tenant: str, owner: str, raw: dict[str, Any]) -> None:
    p = AnswerCommittedPayload.model_validate(raw)
    conv = await _active_conversation(s, tenant, owner, p.conversation_id, p.epoch)
    if conv is None:
        return  # revoked epoch or unknown conversation: late event is ignored
    await s.execute(
        text(
            """
            INSERT INTO committed_turn (id, tenant_id, owner_id, conversation_id, epoch, turn_seq, user_msg_id,
                user_text, assistant_msg_id, assistant_text, character_rev, sent_at, created_at)
            VALUES (:id, :t, :o, :c, :e, :ts, :um, :ut, :am, :at, :cr, :sent, now())
            ON CONFLICT DO NOTHING
            """
        ),
        {"id": new_id(), "t": tenant, "o": owner, "c": p.conversation_id, "e": p.epoch, "ts": p.turn_seq,
         "um": p.user_message.id, "ut": p.user_message.text, "am": p.assistant_message.id,
         "at": p.assistant_message.text, "cr": p.character_revision, "sent": p.user_message.sent_at},
    )
    await s.execute(
        text(
            "UPDATE ai_conversation SET last_turn_at = GREATEST(COALESCE(last_turn_at, :sent), :sent) "
            "WHERE id = :id"
        ),
        {"sent": p.user_message.sent_at, "id": conv["id"]},
    )
    await enqueue_job(
        s, "MEMORY", tenant, owner, p.conversation_id, p.epoch, p.turn_seq,
        f"memory:{p.conversation_id}:{p.epoch}:{p.turn_seq}", {},
    )


async def apply_revision_changed(s: AsyncSession, tenant: str, owner: str, raw: dict[str, Any]) -> None:
    p = CharacterRevisionChangedPayload.model_validate(raw)
    conv = await _active_conversation(s, tenant, owner, p.conversation_id, p.epoch)
    if conv is None:
        return
    prev = (
        await s.execute(
            text(
                "SELECT opening_scene FROM settings_snapshot WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"t": tenant, "c": p.conversation_id, "e": p.epoch},
        )
    ).scalar()
    await s.execute(
        text(
            """
            INSERT INTO settings_snapshot (id, tenant_id, owner_id, conversation_id, epoch, character_rev, persona_rev,
                character_card, persona, opening_scene, created_at)
            VALUES (:id, :t, :o, :c, :e, :cr, :pr, CAST(:card AS jsonb), CAST(:persona AS jsonb),
                    CAST(:scene AS jsonb), now())
            ON CONFLICT DO NOTHING
            """
        ),
        {"id": new_id(), "t": tenant, "o": owner, "c": p.conversation_id, "e": p.epoch, "cr": p.character_revision,
         "pr": p.persona_revision, "card": _j(p.character.model_dump(by_alias=True)),
         "persona": _j(p.persona.model_dump(by_alias=True) if p.persona else None), "scene": _j(prev)},
    )
    await s.execute(
        text("UPDATE ai_conversation SET character_rev = :cr, persona_rev = :pr WHERE id = :id"),
        {"cr": p.character_revision, "pr": p.persona_revision, "id": conv["id"]},
    )
    # Retrieval checks the source turn revision even when extraction completes after this edit.


async def apply_conversation_reset(s: AsyncSession, tenant: str, owner: str, raw: dict[str, Any]) -> None:
    p = ConversationResetPayload.model_validate(raw)
    old = (
        await s.execute(
            text("SELECT * FROM ai_conversation WHERE tenant_id=:t AND owner_id=:o AND conversation_id=:c AND epoch=:e"),
            {"t": tenant, "o": owner, "c": p.conversation_id, "e": p.old_epoch},
        )
    ).mappings().first()
    if old is None:
        return
    settings = (
        await s.execute(
            text(
                "SELECT * FROM settings_snapshot WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e "
                "AND character_rev=:cr AND persona_rev=:pr"
            ),
            {"t": tenant, "c": p.conversation_id, "e": p.old_epoch, "cr": old["character_rev"], "pr": old["persona_rev"]},
        )
    ).mappings().first()
    if old["status"] == "ACTIVE":
        await _revoke_epoch(s, tenant, p.conversation_id, p.old_epoch, "ConversationReset")
    if settings is None or await is_revoked(s, tenant, p.conversation_id, p.new_epoch, old["character_id"]):
        return
    # Settings, persona and opening scene are kept (plan §09); everything derived starts over.
    await _open_epoch(
        s, tenant, owner, p.conversation_id, old["character_id"], p.new_epoch, old["character_rev"],
        old["persona_rev"], settings["character_card"], settings["persona"], settings["opening_scene"],
    )


async def apply_character_deleted(s: AsyncSession, tenant: str, owner: str, raw: dict[str, Any]) -> None:
    p = CharacterDeletedPayload.model_validate(raw)
    await s.execute(
        text(
            "INSERT INTO revocation (id, tenant_id, scope, target_id, epoch, reason, created_at) "
            "VALUES (:id, :t, 'CHARACTER', :ch, NULL, 'CharacterDeleted', now())"
        ),
        {"id": new_id(), "t": tenant, "ch": p.character_id},
    )
    convs = (
        await s.execute(
            text("SELECT conversation_id, epoch, status FROM ai_conversation WHERE tenant_id=:t AND owner_id=:o "
                 "AND (character_id=:ch OR conversation_id = ANY(:ids))"),
            {"t": tenant, "o": owner, "ch": p.character_id, "ids": p.conversation_ids},
        )
    ).all()
    for conversation_id, epoch, status in convs:
        if status == "ACTIVE":
            await _revoke_epoch(s, tenant, conversation_id, epoch, "CharacterDeleted")
    for conversation_id in {c[0] for c in convs}:
        await _purge_epoch(s, tenant, conversation_id, None, include_settings=True)
        await s.execute(
            text("DELETE FROM ai_conversation WHERE tenant_id=:t AND conversation_id=:c"),
            {"t": tenant, "c": conversation_id},
        )
        await s.execute(
            text("DELETE FROM ai_job WHERE tenant_id=:t AND conversation_id=:c AND kind <> 'GENERATION'"),
            {"t": tenant, "c": conversation_id},
        )
        await s.execute(
            text(
                "INSERT INTO revocation (id, tenant_id, scope, target_id, epoch, reason, created_at) "
                "VALUES (:id, :t, 'CONVERSATION', :c, NULL, 'CharacterDeleted', now())"
            ),
            {"id": new_id(), "t": tenant, "c": conversation_id},
        )


APPLIERS = {
    "ConversationOpened": apply_conversation_opened,
    "AnswerCommitted": apply_answer_committed,
    "CharacterRevisionChanged": apply_revision_changed,
    "ConversationReset": apply_conversation_reset,
    "CharacterDeleted": apply_character_deleted,
}


def parse_event(raw: dict[str, Any]) -> Any:
    return _event_adapter.validate_python(raw)

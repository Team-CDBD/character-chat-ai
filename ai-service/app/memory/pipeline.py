"""Memory Pipeline (cold path, plan §07-§08).

    committed turn -> prefilter -> extract (light model, JSON) -> validate evidence
                   -> reduce memories + relationship + unfinished events (one transaction)
                   -> watermark++ -> embed new memories -> rolling summary

Runs strictly in turn order per conversation (the claim query enforces it).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.knowledge import prompts
from app.knowledge.retriever import CHARACTER_REVISION_FILTER
from app.memory import prefilter
from app.memory.reducer import ExistingMemory, reduce
from app.memory.validator import EXTRACTOR_VERSION, MemoryType, TurnText, parse_extraction, validate
from app.persistence.jobs import ClaimedJob, FencedJob, LeaseLost
from app.persistence.models import new_id
from app.providers.base import ChatParams, PromptMessage, ProviderError
from app.relationship.reducer import (
    REDUCER_VERSION,
    ProposedRelationEvent,
    ReduceContext,
    RelationState,
    Stage,
    apply_relation_events,
    baseline_state,
)
from app.services import Services

log = logging.getLogger(__name__)
MAX_RUNS = 3


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


class MemoryPipeline:
    def __init__(self, services: Services) -> None:
        self.sv = services
        self.cfg = services.settings

    async def run(self, job: ClaimedJob) -> None:
        fenced = FencedJob(self.sv.db, job, self.cfg.lease_seconds)
        try:
            if job.kind == "BASELINE":
                await self._baseline(fenced)
            else:
                await self._memory(fenced)
        except LeaseLost:
            log.info("memory job lost lease / revoked", extra={"job": job.id})
        except ProviderError as exc:
            await self._retry_or_skip(fenced, exc)

    async def _retry_or_skip(self, fenced: FencedJob, exc: Exception) -> None:
        job = fenced.job
        try:
            async with self.sv.db.tx() as s:
                if job.run_count < MAX_RUNS:
                    await fenced.guard(s, status="ACCEPTED", error_code="PROVIDER_ERROR")
                    return
                # Give up on extraction for this turn but keep the watermark moving: the raw turn
                # is still in the projection and can be reprocessed later.
                if job.kind == "MEMORY":
                    await self._advance_watermark(s, job, embedding=False)
                await fenced.finish(s, "FAILED", error_code="PROVIDER_ERROR", error_detail=type(exc).__name__)
        except LeaseLost:
            pass

    # ------------------------------------------------------------ baseline
    async def _baseline(self, fenced: FencedJob) -> None:
        job = fenced.job
        async with self.sv.db.session() as s:
            card = (
                await s.execute(
                    text(
                        "SELECT character_card FROM settings_snapshot WHERE tenant_id=:t AND conversation_id=:c "
                        "AND epoch=:e ORDER BY created_at LIMIT 1"
                    ),
                    {"t": job.tenant_id, "c": job.conversation_id, "e": job.epoch},
                )
            ).scalar()
        relation_text = (card or {}).get("relationshipToUser", "")
        result, usage = await self.sv.provider.complete_json(
            [PromptMessage("system", prompts.load("baseline-v1")), PromptMessage("user", relation_text or "(없음)")],
            ChatParams(model=self.cfg.extract_model, temperature=0),
            "baseline-v1",
        )
        try:
            base = baseline_state(float(result["trust"]), float(result["closeness"]), float(result.get("tension", 0)))
        except (KeyError, TypeError, ValueError):
            base = baseline_state(50, 40, 0)
        async with self.sv.db.tx() as s:
            await fenced.guard(s)
            await s.execute(
                text(
                    """
                    UPDATE relationship_state SET trust=:tr, closeness=:cl, tension=:tn, stage=:st, baseline_source='MODEL'
                    WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e AND through_turn_seq = 0
                      AND baseline_source = 'DEFAULT'
                    """
                ),
                {"tr": base.trust, "cl": base.closeness, "tn": base.tension, "st": base.stage.value,
                 "t": job.tenant_id, "c": job.conversation_id, "e": job.epoch},
            )
            await self._ledger(s, job, "baseline", usage)
            await fenced.finish(s, "COMPLETED")

    # ------------------------------------------------------------ memory
    async def _memory(self, fenced: FencedJob) -> None:
        job = fenced.job
        scope = {"t": job.tenant_id, "o": job.owner_id, "c": job.conversation_id, "e": job.epoch}
        async with self.sv.db.session() as s:
            projected = await s.scalar(text(
                "SELECT memory_through_turn_seq FROM ai_conversation "
                "WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e"
            ), scope)
            resume = projected is not None and projected >= job.turn_seq
            if resume:
                pending = (await s.execute(text(
                    "SELECT id, concat_ws(' ', predicate, value, evidence) FROM memory "
                    "WHERE tenant_id=:t AND owner_id=:o AND conversation_id=:c AND epoch=:e "
                    "AND source_turn_seq=:ts AND index_status <> 'INDEXED' ORDER BY id"
                ), {**scope, "ts": job.turn_seq})).all()
        if resume:
            await self._finish_memory(fenced, [(mid, body) for mid, body in pending])
            return

        async with self.sv.db.session() as s:
            turn_row = (
                await s.execute(
                    text(
                        "SELECT * FROM committed_turn WHERE tenant_id=:t AND owner_id=:o AND conversation_id=:c "
                        "AND epoch=:e AND turn_seq=:ts"
                    ),
                    {**scope, "ts": job.turn_seq},
                )
            ).mappings().first()
            open_events = (
                await s.execute(
                    text(
                        "SELECT id, title, due_at FROM unfinished_event WHERE tenant_id=:t AND owner_id=:o "
                        "AND conversation_id=:c AND epoch=:e AND status='OPEN'"
                    ),
                    scope,
                )
            ).mappings().all()
        if turn_row is None:
            async with self.sv.db.tx() as s:
                await self._advance_watermark(s, job, embedding=True)
                await fenced.finish(s, "COMPLETED", error_detail="turn missing")
            return

        turn = TurnText(
            job.turn_seq, turn_row["user_msg_id"], turn_row["user_text"], turn_row["assistant_msg_id"],
            turn_row["assistant_text"],
        )
        report = None
        usage = None
        if prefilter.worth_extracting(turn.user_text, turn.assistant_text):
            raw, usage = await self.sv.provider.complete_json(
                self._extract_messages(turn, turn_row["sent_at"], open_events),
                ChatParams(model=self.cfg.extract_model, temperature=0),
                "extract-v1",
            )
            report = validate(parse_extraction(raw), turn, {e["id"] for e in open_events})

        new_memory_ids: list[tuple[str, str]] = []
        async with self.sv.db.tx() as s:
            await fenced.guard(s)
            if report is not None:
                new_memory_ids = await self._apply_report(s, job, scope, turn, turn_row["sent_at"], turn_row["character_rev"], report)
            if usage is not None:
                await self._ledger(s, job, "extract", usage)
            await self._advance_watermark(s, job, embedding=not new_memory_ids)
            await fenced.guard(s)

        await self._finish_memory(fenced, new_memory_ids)

    async def _finish_memory(self, fenced: FencedJob, pending: list[tuple[str, str]]) -> None:
        if pending:
            await self._embed(fenced, pending)
        await self._maybe_summarize(fenced)
        async with self.sv.db.tx() as s:
            await fenced.finish(s, "COMPLETED")

    def _extract_messages(self, turn: TurnText, sent_at: datetime, open_events: list[Any]) -> list[PromptMessage]:
        tz = ZoneInfo("Asia/Seoul")
        events = "\n".join(f"- id={e['id']}: {e['title']}" for e in open_events) or "(없음)"
        body = (
            f"발화 시각: {sent_at.astimezone(tz).isoformat()}\n"
            f"<open_events>\n{events}\n</open_events>\n"
            f"[user id={turn.user_msg_id}] {turn.user_text}\n"
            f"[character id={turn.assistant_msg_id}] {turn.assistant_text}"
        )
        return [PromptMessage("system", prompts.load("extract-v1")), PromptMessage("user", body)]

    async def _apply_report(self, s: AsyncSession, job: ClaimedJob, scope: dict[str, Any], turn: TurnText,
                            sent_at: datetime, character_rev: int, report) -> list[tuple[str, str]]:
        existing_rows = (
            await s.execute(
                text(
                    "SELECT id, type, subject, predicate, value FROM memory m WHERE tenant_id=:t AND owner_id=:o "
                    f"AND conversation_id=:c AND epoch=:e AND valid_to IS NULL AND {CHARACTER_REVISION_FILTER}"
                ),
                {**scope, "cr": character_rev},
            )
        ).all()
        existing = [ExistingMemory(*r) for r in existing_rows]
        inserted: list[tuple[str, str]] = []
        for d in reduce(report.memories, existing):
            if d.action != "INSERT":
                continue
            m = d.memory
            mid = new_id()
            if d.supersedes_id:
                await s.execute(
                    text("UPDATE memory SET valid_to = now() WHERE id = :id AND tenant_id = :t AND conversation_id = :c"),
                    {"id": d.supersedes_id, "t": job.tenant_id, "c": job.conversation_id},
                )
            await s.execute(
                text(
                    """
                    INSERT INTO memory (id, tenant_id, owner_id, conversation_id, epoch, type, subject, predicate, value,
                        time_expr, resolved_at, time_uncertainty, is_fiction, importance, source_msg_ids, source_turn_seq,
                        evidence, supersedes_id, extractor_version, index_status, created_at)
                    VALUES (:id, :t, :o, :c, :e, :type, :subj, :pred, :val, :texpr, :rat, :tunc, :fic, :imp,
                        CAST(:src AS jsonb), :ts, :ev, :sup, :xv, 'PENDING', :created)
                    """
                ),
                {**scope, "id": mid, "type": m.type.value, "subj": m.subject, "pred": m.predicate, "val": m.value,
                 "texpr": m.time_expression, "rat": m.resolved_at, "tunc": m.time_uncertainty, "fic": m.is_fiction,
                 "imp": m.importance, "src": _j(m.source_message_ids), "ts": job.turn_seq, "ev": m.evidence,
                 "sup": d.supersedes_id, "xv": EXTRACTOR_VERSION, "created": sent_at},
            )
            inserted.append((mid, f"{m.predicate} {m.value} {m.evidence}"))
            if m.opens_unfinished_event and not m.is_fiction and m.type in (
                MemoryType.SCHEDULE, MemoryType.UNFINISHED, MemoryType.SHARED_EVENT
            ):
                await s.execute(
                    text(
                        """
                        INSERT INTO unfinished_event (id, tenant_id, owner_id, conversation_id, epoch, memory_id, status,
                            title, due_at, created_at)
                        VALUES (:id, :t, :o, :c, :e, :mid, 'OPEN', :title, :due, :created)
                        """
                    ),
                    {**scope, "id": new_id(), "mid": mid, "title": m.value, "due": m.resolved_at, "created": sent_at},
                )

        resolved_shared = False
        for r in report.resolves:
            res = await s.execute(
                text(
                    "UPDATE unfinished_event SET status = :st, resolved_turn_seq = :ts WHERE id = :id AND tenant_id=:t "
                    "AND owner_id=:o AND conversation_id=:c AND epoch=:e AND status='OPEN'"
                ),
                {**scope, "st": r.outcome, "ts": job.turn_seq, "id": r.unfinished_event_id},
            )
            resolved_shared = resolved_shared or (res.rowcount > 0 and r.outcome == "RESOLVED")  # type: ignore[attr-defined]

        await self._apply_relationship(s, job, scope, sent_at, report, resolved_shared)
        return inserted

    async def _apply_relationship(self, s: AsyncSession, job: ClaimedJob, scope: dict[str, Any], sent_at: datetime,
                                  report, resolved_shared: bool) -> None:
        state_row = (
            await s.execute(
                text(
                    "SELECT * FROM relationship_state WHERE tenant_id=:t AND owner_id=:o AND conversation_id=:c "
                    "AND epoch=:e FOR UPDATE"
                ),
                scope,
            )
        ).mappings().first()
        if state_row is None or not report.relation_events:
            if state_row is not None:
                await s.execute(
                    text("UPDATE relationship_state SET through_turn_seq = :ts WHERE id = :id"),
                    {"ts": job.turn_seq, "id": state_row["id"]},
                )
            return
        keys = {
            r[0]
            for r in (
                await s.execute(
                    text("SELECT event_key FROM relationship_event WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e"),
                    scope,
                )
            ).all()
        }
        last_kind = (
            await s.execute(
                text(
                    "SELECT kind FROM relationship_event WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e "
                    "AND kind IN ('CONFLICT','RECONCILIATION') ORDER BY source_turn_seq DESC, created_at DESC LIMIT 1"
                ),
                scope,
            )
        ).scalar()
        state = RelationState(
            trust=state_row["trust"], closeness=state_row["closeness"], tension=state_row["tension"],
            stage=Stage(state_row["stage"]), daily_gain=dict(state_row["daily_gain"] or {}),
        )
        proposals = [
            ProposedRelationEvent(e.kind, tuple(e.source_message_ids), e.evidence, e.topic) for e in report.relation_events
        ]
        day = sent_at.astimezone(ZoneInfo("Asia/Seoul")).date().isoformat()
        new_state, applied = apply_relation_events(
            state, proposals,
            ReduceContext(day=day, open_conflict=last_kind == "CONFLICT", resolved_shared_event=resolved_shared,
                          existing_keys=frozenset(keys)),
        )
        for ev in applied:
            await s.execute(
                text(
                    """
                    INSERT INTO relationship_event (id, tenant_id, owner_id, conversation_id, epoch, event_key, kind,
                        delta, source_turn_seq, source_msg_ids, evidence, reducer_version, created_at)
                    VALUES (:id, :t, :o, :c, :e, :k, :kind, CAST(:d AS jsonb), :ts, CAST(:src AS jsonb), :ev, :rv, now())
                    ON CONFLICT DO NOTHING
                    """
                ),
                {**scope, "id": new_id(), "k": ev.event_key, "kind": ev.kind.value, "d": _j(ev.delta),
                 "ts": job.turn_seq, "src": _j(list(ev.proposal.source_msg_ids)), "ev": ev.proposal.evidence,
                 "rv": REDUCER_VERSION},
            )
        await s.execute(
            text(
                """
                UPDATE relationship_state SET trust=:tr, closeness=:cl, tension=:tn, stage=:st,
                    daily_gain=CAST(:dg AS jsonb), through_turn_seq=:ts
                WHERE id = :id
                """
            ),
            {"tr": new_state.trust, "cl": new_state.closeness, "tn": new_state.tension, "st": new_state.stage.value,
             "dg": _j(new_state.daily_gain), "ts": job.turn_seq, "id": state_row["id"]},
        )

    async def _advance_watermark(self, s: AsyncSession, job: ClaimedJob, *, embedding: bool) -> None:
        await s.execute(
            text(
                f"""
                UPDATE ai_conversation SET memory_through_turn_seq = GREATEST(memory_through_turn_seq, :ts)
                {", embedding_through_turn_seq = GREATEST(embedding_through_turn_seq, :ts)" if embedding else ""}
                WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e AND status='ACTIVE'
                """
            ),
            {"ts": job.turn_seq, "t": job.tenant_id, "c": job.conversation_id, "e": job.epoch},
        )

    async def _embed(self, fenced: FencedJob, items: list[tuple[str, str]]) -> None:
        """Embedding failure never loses the structured memory: it stays PENDING and keyword search covers it."""
        job = fenced.job
        try:
            vectors, usage = await self.sv.provider.embed([t for _, t in items], self.cfg.embedding_model)
        except ProviderError:
            async with self.sv.db.tx() as s:
                await fenced.guard(s)
                await s.execute(
                    text("UPDATE memory SET index_status='FAILED' WHERE id = ANY(:ids)"), {"ids": [i for i, _ in items]}
                )
            raise
        async with self.sv.db.tx() as s:
            await fenced.guard(s)
            for (mid, _), vec in zip(items, vectors, strict=True):
                await s.execute(
                    text(
                        "UPDATE memory SET embedding = CAST(:v AS vector), embedding_version = :ver, index_status='INDEXED' "
                        "WHERE id = :id"
                    ),
                    {"v": json.dumps(vec), "ver": self.cfg.embedding_model, "id": mid},
                )
            await self._ledger(s, job, "embed", usage)
            await self._advance_watermark(s, job, embedding=True)

    async def _maybe_summarize(self, fenced: FencedJob) -> None:
        """Roll turns that left the recent window into a summary every ``summary_every_turns`` turns."""
        job = fenced.job
        window, every = self.cfg.recent_window_turns, self.cfg.summary_every_turns
        async with self.sv.db.session() as s:
            through = (
                await s.execute(
                    text("SELECT summary_through_turn_seq FROM ai_conversation WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e"),
                    {"t": job.tenant_id, "c": job.conversation_id, "e": job.epoch},
                )
            ).scalar() or 0
            upto = job.turn_seq - window
            if upto - through < every:
                return
            rows = (
                await s.execute(
                    text(
                        "SELECT turn_seq, user_text, assistant_text FROM committed_turn WHERE tenant_id=:t AND "
                        "conversation_id=:c AND epoch=:e AND turn_seq > :a AND turn_seq <= :b ORDER BY turn_seq"
                    ),
                    {"t": job.tenant_id, "c": job.conversation_id, "e": job.epoch, "a": through, "b": upto},
                )
            ).all()
            previous = (
                await s.execute(
                    text("SELECT text FROM summary WHERE tenant_id=:t AND conversation_id=:c AND epoch=:e "
                         "ORDER BY to_turn_seq DESC LIMIT 1"),
                    {"t": job.tenant_id, "c": job.conversation_id, "e": job.epoch},
                )
            ).scalar()
        dialogue = "\n".join(
            (f"user: {u}\n" if u else "") + f"character: {a}" for _, u, a in rows
        )
        result, usage = await self.sv.provider.complete_json(
            [PromptMessage("system", prompts.load("summary-v1")),
             PromptMessage("user", f"<previous_summary>{previous or ''}</previous_summary>\n{dialogue}")],
            ChatParams(model=self.cfg.extract_model, temperature=0),
            "summary-v1",
        )
        summary_text = str(result.get("summary", "")).strip()
        async with self.sv.db.tx() as s:
            await fenced.guard(s)
            if summary_text:
                await s.execute(
                    text(
                        "INSERT INTO summary (id, tenant_id, owner_id, conversation_id, epoch, from_turn_seq, to_turn_seq, "
                        "text, version, created_at) VALUES (:id, :t, :o, :c, :e, :a, :b, :txt, 'summary-v1', now())"
                    ),
                    {"id": new_id(), "t": job.tenant_id, "o": job.owner_id, "c": job.conversation_id, "e": job.epoch,
                     "a": 1 if not previous else through + 1, "b": upto, "txt": summary_text},
                )
            await s.execute(
                text("UPDATE ai_conversation SET summary_through_turn_seq = :b WHERE tenant_id=:t AND conversation_id=:c "
                     "AND epoch=:e"),
                {"b": upto, "t": job.tenant_id, "c": job.conversation_id, "e": job.epoch},
            )
            await self._ledger(s, job, "summary", usage)

    async def _ledger(self, s: AsyncSession, job: ClaimedJob, purpose: str, usage) -> None:
        await s.execute(
            text(
                "INSERT INTO usage_ledger (id, tenant_id, owner_id, conversation_id, attempt_id, purpose, model, "
                "input_tokens, output_tokens, est_cost, created_at) VALUES (:id, :t, :o, :c, :a, :p, :m, :i, :out, 0, now())"
            ),
            {"id": new_id(), "t": job.tenant_id, "o": job.owner_id, "c": job.conversation_id, "a": job.attempt_id,
             "p": purpose, "m": usage.model, "i": usage.input_tokens, "out": usage.output_tokens},
        )


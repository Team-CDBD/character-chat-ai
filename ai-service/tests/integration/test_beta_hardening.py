"""Pre-beta hardening (2026-11-28 review):

1. delete / reset vs generation accept are ordered per conversation (no old message body survives),
2. one deadline from model-slot wait to the final save (fail once, slot released),
3. a FAILED lifecycle event is retried and same-eventId re-sends stay idempotent,
plus 422 on bad time input, the 24h retention sweep, and no DB connection held while embedding.
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from app.api import routes
from app.dialogue.manager import DialogueManager
from app.lifecycle import handler
from app.lifecycle.handler import RetryPolicy, process_pending
from app.persistence import jobs
from app.providers.base import ChatParams, PromptMessage
from app.providers.mock import MockProvider
from tests.conftest import drain
from tests.integration.test_generation_api import parse_sse
from tools.mock_be import MockConversation

API = "/internal/ai/v1"
T0 = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
SECRET = "비밀번호는 고양이123"


async def post_event(client, ev) -> str:
    r = await client.post(f"{API}/lifecycle-events", json=ev)
    assert r.status_code == 202, r.text
    return r.json()["receiptId"]


async def open_conv(client, services) -> MockConversation:
    conv = MockConversation()
    await post_event(client, conv.opened())
    await drain(services)
    return conv


async def scalar(services, sql, **params):
    async with services.db.session() as s:
        return (await s.execute(text(sql), params)).scalar()


async def secret_copies(services, needle: str) -> int:
    """Rows anywhere in the AI DB that still hold the user's message text."""
    like = f"%{needle}%"
    return await scalar(
        services,
        """
        SELECT (SELECT count(*) FROM ai_job WHERE payload::text LIKE :p OR COALESCE(result_text, '') LIKE :p)
             + (SELECT count(*) FROM stream_event WHERE data::text LIKE :p)
             + (SELECT count(*) FROM event_inbox WHERE payload::text LIKE :p)
             + (SELECT count(*) FROM committed_turn WHERE COALESCE(user_text, '') LIKE :p)
             + (SELECT count(*) FROM context_snapshot WHERE content::text LIKE :p)
        """,
        p=like,
    )


async def blocked(task: asyncio.Task, seconds: float = 0.3) -> bool:
    await asyncio.sleep(seconds)
    return not task.done()


# ---------------------------------------------------------------- 1. delete/reset vs accept


def _cleanup_event(conv: MockConversation, kind: str):
    if kind == "reset":
        epoch = conv.epoch
        ev = conv.reset()
        conv.epoch = epoch  # the generation below still targets the old epoch
        return ev
    return conv.character_deleted()


@pytest.mark.parametrize("kind", ["reset", "delete"])
async def test_accept_first_then_cleanup_scrubs_the_new_job(client, services, monkeypatch, kind):
    """Accept holds the conversation lock (job inserted, not committed); the cleanup must wait, then revoke
    and scrub that job. Before the fix the cleanup ran past it and the message body survived."""
    conv = await open_conv(client, services)
    await post_event(client, _cleanup_event(conv, kind))  # in the Inbox, not applied yet

    inserted, release = asyncio.Event(), asyncio.Event()
    original = routes._insert_generation

    async def paused_insert(*args, **kwargs):
        await original(*args, **kwargs)
        inserted.set()
        await release.wait()  # transaction still open: lock held

    monkeypatch.setattr(routes, "_insert_generation", paused_insert)
    accept = asyncio.create_task(client.post(f"{API}/generations", json=conv.generation(1, SECRET, T0)))
    await asyncio.wait_for(inserted.wait(), 5)
    lifecycle = asyncio.create_task(process_pending(services.db))
    try:
        assert await blocked(lifecycle), "cleanup must wait for the in-flight accept"
    finally:
        release.set()
    r = await asyncio.wait_for(accept, 10)
    assert r.status_code == 202, r.text
    await asyncio.wait_for(lifecycle, 10)
    await drain(services)

    job = r.json()["aiJobId"]
    # Revoked by the cleanup (delete also keeps generation rows, for BE's status queries).
    assert await scalar(services, "SELECT status FROM ai_job WHERE id=:id", id=job) == "REVOKED"
    assert await secret_copies(services, SECRET) == 0
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "REVOKED"


@pytest.mark.parametrize("kind", ["reset", "delete"])
async def test_cleanup_first_then_accept_is_refused(client, services, monkeypatch, kind):
    """The cleanup holds the lock mid-transaction; accept must wait, then see the revocation and refuse.
    Before the fix accept read ACTIVE before the cleanup committed and stored the message afterwards."""
    conv = await open_conv(client, services)
    await post_event(client, _cleanup_event(conv, kind))

    purged, release = asyncio.Event(), asyncio.Event()
    original = handler._purge_epoch

    async def paused_purge(*args, **kwargs):
        await original(*args, **kwargs)
        purged.set()
        await release.wait()  # cleanup transaction still open

    monkeypatch.setattr(handler, "_purge_epoch", paused_purge)
    lifecycle = asyncio.create_task(process_pending(services.db))
    await asyncio.wait_for(purged.wait(), 5)
    accept = asyncio.create_task(client.post(f"{API}/generations", json=conv.generation(1, SECRET, T0)))
    try:
        assert await blocked(accept), "accept must wait for the in-flight cleanup"
    finally:
        release.set()
    await asyncio.wait_for(lifecycle, 10)
    r = await asyncio.wait_for(accept, 10)
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "EPOCH_REVOKED"
    assert await scalar(services, "SELECT count(*) FROM ai_job WHERE kind='GENERATION'") == 0
    assert await secret_copies(services, SECRET) == 0


async def test_concurrent_accept_and_reset_never_leave_the_body(client, services):
    """No pauses: fire both at once many times; whichever wins, nothing of the old epoch survives."""
    for i in range(15):
        conv = await open_conv(client, services)
        await post_event(client, _cleanup_event(conv, "reset"))
        needle = f"{SECRET}-{i}"
        r, _ = await asyncio.gather(
            client.post(f"{API}/generations", json=conv.generation(1, needle, T0)),
            process_pending(services.db),
        )
        assert r.status_code in (202, 409), r.text
        await drain(services)
        assert await secret_copies(services, needle) == 0


# ---------------------------------------------------------------- 2. one deadline


class SlowStartMock(MockProvider):
    """Waits ``first_token_delay`` before the first chunk when the user line contains '느리게'."""

    def __init__(self, first_token_delay: float = 5.0, **kw) -> None:
        super().__init__(**kw)
        self.first_token_delay = first_token_delay

    async def stream_chat(self, messages: list[PromptMessage], params: ChatParams):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "느리게" in last_user:
            await asyncio.sleep(self.first_token_delay)
        async for chunk in super().stream_chat(messages, params):
            yield chunk


async def stream_rows(services, job_id):
    async with services.db.session() as s:
        return (await s.execute(text("SELECT event FROM stream_event WHERE job_id=:id ORDER BY event_seq"),
                                {"id": job_id})).scalars().all()


async def run_one(services, dialogue=None):
    job = await jobs.claim(services.db, "GENERATION", "test", services.settings.lease_seconds)
    assert job is not None
    await (dialogue or DialogueManager(services)).run_attempt(job)
    return job.id


async def test_deadline_before_first_token_fails_once_and_frees_the_slot(client, services, monkeypatch):
    slow = SlowStartMock(first_token_delay=10)
    monkeypatch.setattr(services.provider, "provider", slow)
    conv = await open_conv(client, services)
    job = (await client.post(f"{API}/generations",
                             json=conv.generation(1, "느리게 답해줘", T0, deadline=timedelta(seconds=0.8)))).json()["aiJobId"]
    started = time.monotonic()
    await run_one(services)
    assert time.monotonic() - started < 3, "the attempt must stop at the deadline, not after the model"

    result = (await client.get(f"{API}/generations/{job}")).json()
    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "PROVIDER_TIMEOUT"
    assert result["error"]["message"] == "deadline passed before first token"
    assert (await stream_rows(services, job)).count("failed") == 1
    assert not services.provider._slots.locked()


async def test_deadline_mid_stream_fails_once(client, services, monkeypatch):
    mock = MockProvider(chunk_delay=0.05, chunk_size=2)
    mock.reply_script = lambda _m: "응. " * 200  # ~5 s of streaming
    monkeypatch.setattr(services.provider, "provider", mock)
    conv = await open_conv(client, services)
    job = (await client.post(f"{API}/generations",
                             json=conv.generation(1, "길게", T0, deadline=timedelta(seconds=1.2)))).json()["aiJobId"]
    started = time.monotonic()
    await run_one(services)
    assert time.monotonic() - started < 3

    result = (await client.get(f"{API}/generations/{job}")).json()
    assert result["status"] == "FAILED" and result["text"] is None
    assert result["error"]["message"] == "deadline passed during streaming"
    events = await stream_rows(services, job)
    assert "delta" in events and "completed" not in events
    assert events.count("failed") == 1 and events[-1] == "failed"
    assert not services.provider._slots.locked()

    sse = parse_sse((await client.get(f"{API}/generations/{job}/events")).text)
    assert sse[-1][1] == "failed"


async def test_next_request_gets_the_single_slot_after_a_deadline(client, services, monkeypatch):
    """Ollama default: one call at a time. A's deadline passes while it holds the slot; B, queued behind it,
    must get the slot right away and finish (before the fix A kept the model busy for its whole answer)."""
    monkeypatch.setattr(services.provider, "provider", SlowStartMock(first_token_delay=30))
    monkeypatch.setattr(services.provider, "_slots", asyncio.Semaphore(1))
    a, b = await open_conv(client, services), await open_conv(client, services)
    job_a = (await client.post(f"{API}/generations",
                               json=a.generation(1, "느리게", T0, deadline=timedelta(seconds=0.8)))).json()["aiJobId"]
    claim_a = await jobs.claim(services.db, "GENERATION", "w1", 30)
    job_b = (await client.post(f"{API}/generations",
                               json=b.generation(1, "빨리", T0, deadline=timedelta(seconds=10)))).json()["aiJobId"]
    claim_b = await jobs.claim(services.db, "GENERATION", "w2", 30)
    started = time.monotonic()
    dm = DialogueManager(services)
    await asyncio.gather(dm.run_attempt(claim_a), dm.run_attempt(claim_b))
    assert time.monotonic() - started < 5

    ra = (await client.get(f"{API}/generations/{job_a}")).json()
    rb = (await client.get(f"{API}/generations/{job_b}")).json()
    assert ra["status"] == "FAILED" and ra["error"]["code"] == "PROVIDER_TIMEOUT"
    assert rb["status"] == "COMPLETED" and rb["text"]


async def test_slot_wait_counts_against_the_deadline(client, services, monkeypatch):
    """B's deadline passes while it waits for the slot: B fails at its deadline, once, without a model call."""
    slow = SlowStartMock(first_token_delay=3)
    monkeypatch.setattr(services.provider, "provider", slow)
    monkeypatch.setattr(services.provider, "_slots", asyncio.Semaphore(1))
    a, b = await open_conv(client, services), await open_conv(client, services)
    await client.post(f"{API}/generations", json=a.generation(1, "느리게", T0, deadline=timedelta(seconds=20)))
    claim_a = await jobs.claim(services.db, "GENERATION", "w1", 30)
    job_b = (await client.post(f"{API}/generations",
                               json=b.generation(1, "빨리", T0, deadline=timedelta(seconds=0.8)))).json()["aiJobId"]
    claim_b = await jobs.claim(services.db, "GENERATION", "w2", 30)
    dm = DialogueManager(services)
    task_a = asyncio.create_task(dm.run_attempt(claim_a))
    await asyncio.sleep(0.2)  # A holds the slot
    started = time.monotonic()
    await dm.run_attempt(claim_b)
    assert time.monotonic() - started < 2
    rb = (await client.get(f"{API}/generations/{job_b}")).json()
    assert rb["status"] == "FAILED" and rb["error"]["code"] in ("PROVIDER_TIMEOUT", "CONTEXT_NOT_READY")
    assert (await stream_rows(services, job_b)).count("failed") == 1
    await task_a


async def test_late_answer_is_not_saved_after_the_deadline(client, services, monkeypatch):
    from app.dialogue import manager

    conv = await open_conv(client, services)
    job = (await client.post(f"{API}/generations", json=conv.generation(1, "안녕", T0))).json()["aiJobId"]
    original = manager.composer.generate

    async def generate_then_late(*args, **kwargs):
        out = await original(*args, **kwargs)
        monkeypatch.setattr(manager, "utcnow", lambda: datetime.now(UTC) + timedelta(days=1))
        return out

    monkeypatch.setattr(manager.composer, "generate", generate_then_late)
    await run_one(services)
    result = (await client.get(f"{API}/generations/{job}")).json()
    assert result["status"] == "FAILED" and result["text"] is None
    assert result["error"]["message"] == "deadline passed before the answer was saved"
    events = await stream_rows(services, job)
    assert "completed" not in events and events.count("failed") == 1


# ---------------------------------------------------------------- 3. failed lifecycle events


def flaky(monkeypatch, event_type: str, failures: int):
    original = handler.APPLIERS[event_type]
    state = {"left": failures, "calls": 0}

    async def maybe_fail(*args, **kwargs):
        state["calls"] += 1
        if state["left"] > 0:
            state["left"] -= 1
            raise RuntimeError("transient")
        return await original(*args, **kwargs)

    monkeypatch.setitem(handler.APPLIERS, event_type, maybe_fail)
    return state


async def receipt(client, rid):
    return (await client.get(f"{API}/lifecycle-events/{rid}")).json()


async def test_failed_event_resent_with_same_id_recovers_and_unblocks_later_events(client, services, monkeypatch):
    conv = await open_conv(client, services)
    state = flaky(monkeypatch, "AnswerCommitted", failures=1)
    first = conv.committed(1, "안녕", "어.", T0)
    second = conv.committed(2, "뭐해", "그냥.", T0 + timedelta(minutes=1))
    r1 = await post_event(client, first)
    r2 = await post_event(client, second)
    await drain(services, generation=False, memory=False)

    failed = await receipt(client, r1)
    assert failed["status"] == "FAILED" and failed["attempts"] == 1 and failed["nextRetryAt"]
    assert (await receipt(client, r2))["status"] == "RECEIVED"  # held behind seq 2, order kept

    assert await post_event(client, first) == r1  # same eventId -> same receipt, re-armed now
    await drain(services, generation=False, memory=False)
    assert (await receipt(client, r1))["status"] == "APPLIED"
    assert (await receipt(client, r2))["status"] == "APPLIED"

    # Re-sending an applied event again changes nothing.
    assert await post_event(client, first) == r1
    await drain(services, generation=False, memory=False)
    assert state["calls"] == 3  # fail, succeed (turn 1), turn 2; never a second apply of turn 1
    turns = await scalar(services, "SELECT count(*) FROM committed_turn WHERE conversation_id=:c AND turn_seq=1",
                         c=conv.conversation_id)
    assert turns == 1
    assert await scalar(services, "SELECT count(*) FROM ai_job WHERE kind='MEMORY' AND conversation_id=:c",
                        c=conv.conversation_id) == 2


async def test_failed_event_is_retried_after_backoff_without_resend(client, services, monkeypatch):
    conv = await open_conv(client, services)
    flaky(monkeypatch, "AnswerCommitted", failures=1)
    rid = await post_event(client, conv.committed(1, "안녕", "어.", T0))
    await process_pending(services.db, retry=RetryPolicy(base_seconds=60))
    assert (await receipt(client, rid))["status"] == "FAILED"
    assert await process_pending(services.db) == 0  # backoff not over yet

    async with services.db.tx() as s:  # time passes
        await s.execute(text("UPDATE event_inbox SET next_attempt_at = now() - interval '1 second' WHERE id=:id"),
                        {"id": rid})
    await process_pending(services.db)
    assert (await receipt(client, rid))["status"] == "APPLIED"


async def test_exhausted_event_waits_for_a_resend(client, services, monkeypatch):
    conv = await open_conv(client, services)
    flaky(monkeypatch, "AnswerCommitted", failures=2)
    ev = conv.committed(1, "안녕", "어.", T0)
    rid = await post_event(client, ev)
    policy = RetryPolicy(base_seconds=0, max_attempts=2)
    for _ in range(3):
        await process_pending(services.db, retry=policy)
    gave_up = await receipt(client, rid)
    assert gave_up["status"] == "FAILED" and gave_up["attempts"] == 2 and gave_up["nextRetryAt"] is None
    assert await process_pending(services.db, retry=policy) == 0

    assert await post_event(client, ev) == rid
    assert (await receipt(client, rid))["attempts"] == 0
    await process_pending(services.db, retry=policy)
    assert (await receipt(client, rid))["status"] == "APPLIED"


# ---------------------------------------------------------------- 4. time input at the door


@pytest.mark.parametrize(
    "patch",
    [
        {"timezone": "Mars/Olympus"},
        {"timezone": "asia/seoul"},
        {"timezone": ""},
        {"deadlineAt": "2026-11-28T10:00:00"},  # no UTC offset
        {"userMessage": {"id": "u1", "text": "안녕", "sentAt": "2026-11-28T10:00:00"}},
    ],
)
async def test_bad_time_input_is_422_at_accept(client, services, patch):
    conv = await open_conv(client, services)
    body = conv.generation(1, "안녕", T0) | patch
    r = await client.post(f"{API}/generations", json=body)
    assert r.status_code == 422, r.text
    assert await scalar(services, "SELECT count(*) FROM ai_job WHERE kind='GENERATION'") == 0


async def test_valid_offsets_are_accepted(client, services):
    conv = await open_conv(client, services)
    body = conv.generation(1, "안녕", T0) | {"timezone": "America/New_York"}
    body["deadlineAt"] = (datetime.now(UTC) + timedelta(minutes=1)).astimezone().isoformat()
    assert (await client.post(f"{API}/generations", json=body)).status_code == 202


# ---------------------------------------------------------------- 5. 24h retention


async def test_retention_sweep_deletes_old_bodies_but_keeps_the_result(client, services):
    conv = await open_conv(client, services)
    old = (await client.post(f"{API}/generations", json=conv.generation(1, SECRET, T0))).json()["aiJobId"]
    await drain(services)
    new = (await client.post(f"{API}/generations",
                             json=conv.generation(2, "최근 질문", T0, attempt="b"))).json()["aiJobId"]
    await drain(services)
    before = (await client.get(f"{API}/generations/{old}")).json()
    async with services.db.tx() as s:
        await s.execute(text("UPDATE ai_job SET completed_at = now() - interval '25 hours' WHERE id=:id"), {"id": old})
        await s.execute(text("UPDATE context_snapshot SET expires_at = now() - interval '1 hour' WHERE id=:id"),
                        {"id": before["contextSnapshotId"]})

    purged = await jobs.purge_expired(services.db, 24 * 3600, batch=1)  # batch 1: loops until done
    assert purged.snapshots == 1 and purged.jobs == 1 and purged.stream_events > 0
    assert await scalar(services, "SELECT count(*) FROM stream_event WHERE job_id=:id", id=old) == 0
    assert await scalar(services, "SELECT count(*) FROM context_snapshot WHERE id=:id",
                        id=before["contextSnapshotId"]) == 0
    assert await scalar(services, "SELECT payload::text FROM ai_job WHERE id=:id", id=old) == "{}"

    # BE can still read the final answer, by GET and by SSE recovery.
    after = (await client.get(f"{API}/generations/{old}")).json()
    assert after["status"] == "COMPLETED" and after["text"] == before["text"]
    sse = parse_sse((await client.get(f"{API}/generations/{old}/events", headers={"Last-Event-ID": "1"})).text)
    assert len(sse) == 1 and sse[0][1] == "completed" and sse[0][2]["text"] == before["text"]

    # Recent work untouched; a second sweep has nothing to do.
    assert await scalar(services, "SELECT count(*) FROM stream_event WHERE job_id=:id", id=new) > 0
    again = await jobs.purge_expired(services.db, 24 * 3600, batch=500)
    assert (again.snapshots, again.jobs) == (0, 0)

    # Regenerating from the purged snapshot is refused, never silently rebuilt.
    regen = conv.generation(1, SECRET, T0, mode="REGENERATE", attempt="r", snapshot_id=before["contextSnapshotId"])
    job = (await client.post(f"{API}/generations", json=regen)).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["error"]["code"] == "SNAPSHOT_EXPIRED"


# ---------------------------------------------------------------- 6. no DB connection while waiting on the model


async def test_no_db_connection_is_held_while_embedding(client, services, mock, monkeypatch):
    seen: list[int] = []
    original = mock.embed

    async def watching_embed(texts, model):
        seen.append(services.db.engine.pool.checkedout())
        return await original(texts, model)

    monkeypatch.setattr(mock, "embed", watching_embed)
    conv = await open_conv(client, services)
    await post_event(client, conv.committed(1, "나 민트초코 좋아해", "그래.", T0))
    await drain(services)
    await client.post(f"{API}/generations", json=conv.generation(2, "내가 뭐 좋아한다고 했지", T0))
    await drain(services)
    assert seen, "query embedding should have run"
    assert max(seen) == 0, f"connections checked out during embed: {seen}"

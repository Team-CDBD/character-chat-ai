from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from app.persistence import jobs
from app.persistence.jobs import FencedJob, LeaseLost
from app.providers.base import ProviderError
from tests.conftest import drain
from tools.mock_be import MockConversation

API = "/internal/ai/v1"
T0 = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)


def parse_sse(body: str) -> list[tuple[int, str, dict]]:
    out = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        out.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return out


async def open_conv(client, services) -> MockConversation:
    conv = MockConversation()
    r = await client.post(f"{API}/lifecycle-events", json=conv.opened())
    assert r.status_code == 202
    await drain(services)
    return conv


async def test_requires_service_token(client):
    r = await client.get(f"{API}/generations/x", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


async def test_generation_streams_and_result_is_restorable(client, services, mock):
    conv = await open_conv(client, services)
    r = await client.post(f"{API}/generations", json=conv.generation(1, "좋은 아침", T0))
    assert r.status_code == 202
    job_id = r.json()["aiJobId"]
    await drain(services)

    result = (await client.get(f"{API}/generations/{job_id}")).json()
    assert result["status"] == "COMPLETED"
    assert result["contextSnapshotId"]
    assert result["usage"]["outputTokens"] > 0

    events = parse_sse((await client.get(f"{API}/generations/{job_id}/events")).text)
    assert events[0][1] == "accepted" and events[-1][1] == "completed"
    assert [e[0] for e in events] == list(range(1, len(events) + 1))
    assert "".join(e[2]["text"] for e in events if e[1] == "delta") == result["text"]

    # reconnect after event 2: only the rest is replayed
    replay = parse_sse((await client.get(f"{API}/generations/{job_id}/events", headers={"Last-Event-ID": "2"})).text)
    assert replay[0][0] == 3

    # Opening scene is Turn 0, and the character card is in the first (stable) system layer.
    messages = mock.chat_calls[0]
    assert "<character>" in messages[0].content and "강현우" in messages[0].content
    assert any(m.role == "assistant" and "우산 없지" in m.content for m in messages)
    assert len(mock.chat_calls) == 1  # GET / SSE never call the model again


async def test_idempotent_accept_and_conflicts(client, services):
    conv = await open_conv(client, services)
    req = conv.generation(1, "안녕", T0)
    a = (await client.post(f"{API}/generations", json=req)).json()
    b = (await client.post(f"{API}/generations", json={**req, "traceId": "other"})).json()
    assert a["aiJobId"] == b["aiJobId"]

    r = await client.post(f"{API}/generations", json={**req, "userMessage": {**req["userMessage"], "text": "다른"}})
    assert r.status_code == 409 and r.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"

    other = conv.generation(1, "안녕", T0, attempt="b")
    r = await client.post(f"{API}/generations", json=other)
    assert r.status_code == 409 and r.json()["error"]["code"] == "ACTIVE_GENERATION_EXISTS"


async def test_unregistered_conversation_is_rejected(client, services):
    conv = MockConversation()
    r = await client.post(f"{API}/generations", json=conv.generation(1, "안녕", T0))
    assert r.status_code == 404 and r.json()["error"]["code"] == "CONVERSATION_NOT_REGISTERED"


async def test_regenerate_reuses_snapshot_without_previous_candidate(client, services, mock):
    conv = await open_conv(client, services)
    first = (await client.post(f"{API}/generations", json=conv.generation(1, "오늘 뭐해", T0))).json()["aiJobId"]
    await drain(services)
    snap = (await client.get(f"{API}/generations/{first}")).json()["contextSnapshotId"]

    regen = conv.generation(1, "오늘 뭐해", T0, mode="REGENERATE", attempt="b", snapshot_id=snap)
    second = (await client.post(f"{API}/generations", json=regen)).json()["aiJobId"]
    await drain(services)
    res = (await client.get(f"{API}/generations/{second}")).json()
    assert res["status"] == "COMPLETED" and res["contextSnapshotId"] == snap
    assert mock.chat_calls[0] == mock.chat_calls[1]  # identical context, previous answer not included


async def test_regenerate_with_missing_snapshot_fails_explicitly(client, services):
    conv = await open_conv(client, services)
    regen = conv.generation(1, "오늘 뭐해", T0, mode="REGENERATE", snapshot_id="00000000-0000-0000-0000-000000000000")
    job = (await client.post(f"{API}/generations", json=regen)).json()["aiJobId"]
    await drain(services)
    res = (await client.get(f"{API}/generations/{job}")).json()
    assert res["status"] == "FAILED" and res["error"]["code"] == "SNAPSHOT_EXPIRED"
    assert res["error"]["retryable"] is False


async def test_provider_error_before_stream_is_retryable_failure(client, services, mock):
    conv = await open_conv(client, services)
    mock.fail_next = ProviderError("boom")
    job = (await client.post(f"{API}/generations", json=conv.generation(1, "안녕", T0))).json()["aiJobId"]
    await drain(services)
    # first try fails, the in-attempt retry succeeds
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"


async def test_context_not_ready_when_lifecycle_seq_missing(client, services):
    conv = await open_conv(client, services)
    req = conv.generation(1, "안녕", T0)
    req["requiredLifecycleSeq"] = 99
    job = (await client.post(f"{API}/generations", json=req)).json()["aiJobId"]
    await drain(services)
    res = (await client.get(f"{API}/generations/{job}")).json()
    assert res["error"]["code"] == "CONTEXT_NOT_READY" and res["error"]["retryable"] is True
    events = parse_sse((await client.get(f"{API}/generations/{job}/events")).text)
    assert events[-1][1] == "failed"


async def test_fencing_rejects_stale_worker(client, services):
    conv = await open_conv(client, services)
    await client.post(f"{API}/generations", json=conv.generation(1, "안녕", T0))
    stale = await jobs.claim(services.db, "GENERATION", "w1", 30)
    async with services.db.tx() as s:  # lease expires, another worker takes over
        await s.execute(text("UPDATE ai_job SET lease_until = now() - interval '1 second' WHERE id = :id"),
                        {"id": stale.id})
    fresh = await jobs.claim(services.db, "GENERATION", "w2", 30)
    assert fresh.id == stale.id and fresh.fencing_token == stale.fencing_token + 1
    try:
        await FencedJob(services.db, stale, 30).set_status("STREAMING")
        raise AssertionError("stale worker write accepted")
    except LeaseLost:
        pass


async def test_conversation_state_exposes_watermarks_only(client, services):
    conv = await open_conv(client, services)
    await client.post(f"{API}/lifecycle-events", json=conv.committed(1, "안녕", "어.", T0))
    await drain(services)
    state = (await client.get(f"{API}/conversations/{conv.conversation_id}/state",
                              params={"tenantId": conv.tenant_id, "epoch": 1})).json()
    assert state["appliedLifecycleSeq"] == 2 and state["memoryThroughTurnSeq"] == 1
    assert set(state) == {"conversationId", "epoch", "status", "appliedLifecycleSeq", "memoryThroughTurnSeq",
                          "embeddingThroughTurnSeq", "characterRevision", "personaRevision"}


async def test_deadline_in_past_fails(client, services):
    conv = await open_conv(client, services)
    req = conv.generation(1, "안녕", T0, deadline=timedelta(seconds=-1))
    job = (await client.post(f"{API}/generations", json=req)).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "FAILED"

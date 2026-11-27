from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from tests.conftest import drain
from tools.mock_be import HYUNWOO, MockConversation

API = "/internal/ai/v1"
DAY1 = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)  # 11/7 21:00 KST; "다음 주 금요일" = 11/13


def scripted_extractor(messages):
    """Stands in for the light extraction model: proposes memories from the user line."""
    body = messages[-1].content
    user_line = next(line for line in body.splitlines() if line.startswith("[user id="))
    uid = user_line.split("id=")[1].split("]")[0]
    out = {"memories": [], "relationEvents": [], "resolves": []}
    if "면접" in user_line and "있어" in user_line:
        out["memories"].append({
            "type": "SCHEDULE", "subject": "user", "predicate": "has_interview", "value": "최종 면접",
            "timeExpression": "다음 주 금요일", "resolvedAt": "2026-11-13T01:00:00Z", "timeUncertainty": "DAY",
            "importance": 0.9, "sourceMessageIds": [uid], "evidence": "다음 주 금요일에 면접 있어",
            "opensUnfinishedEvent": True,
        })
    if "민트초코 싫어해" in user_line:
        out["memories"].append({"type": "USER_FACT", "subject": "user", "predicate": "mint_choco", "value": "싫어함",
                                "importance": 0.6, "sourceMessageIds": [uid], "evidence": "민트초코 싫어해"})
    if "민트초코 좋아해" in user_line:
        out["memories"].append({"type": "USER_FACT", "subject": "user", "predicate": "mint_choco", "value": "좋아함",
                                "importance": 0.6, "sourceMessageIds": [uid], "evidence": "민트초코 좋아해"})
    if "힘들어" in user_line:
        out["relationEvents"].append({"kind": "PERSONAL_DISCLOSURE", "sourceMessageIds": [uid],
                                      "evidence": "취업 때문에 좀 힘들어", "topic": "취업"})
    if "현우는 대학생" in user_line:  # model trying to invent a user fact from the character's words
        out["memories"].append({"type": "USER_FACT", "subject": "user", "predicate": "job", "value": "대학생",
                                "sourceMessageIds": ["a1"], "evidence": "나 대학생이야"})
    return out


async def post(client, ev):
    r = await client.post(f"{API}/lifecycle-events", json=ev)
    assert r.status_code == 202, r.text
    return r.json()["receiptId"]


async def rows(services, sql, **params):
    async with services.db.session() as s:
        return (await s.execute(text(sql), params)).all()


async def test_events_apply_in_aggregate_order_and_dedupe(client, services):
    conv = MockConversation()
    opened = conv.opened()
    commit = conv.committed(1, "안녕", "어.", DAY1)
    r2 = await post(client, commit)  # arrives before seq 1
    await drain(services)
    assert (await client.get(f"{API}/lifecycle-events/{r2}")).json()["status"] == "RECEIVED"  # held at gap

    r1 = await post(client, opened)
    assert await post(client, opened) == r1  # same eventId -> same receipt
    await drain(services)
    assert (await client.get(f"{API}/lifecycle-events/{r1}")).json()["status"] == "APPLIED"
    assert (await client.get(f"{API}/lifecycle-events/{r2}")).json()["status"] == "APPLIED"
    turns = await rows(services, "SELECT turn_seq FROM committed_turn ORDER BY turn_seq")
    assert [t[0] for t in turns] == [0, 1]  # Turn 0 = opening scene


async def test_golden_scenario_interview_recalled_on_day7(client, services, mock):
    """기획서 §30: Day1 면접 언급 -> Day3 다른 주제 -> Day7 재방문 시 자연스럽게 재언급 가능한 Context."""
    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나 다음 주 금요일에 면접 있어", "...면접? 잘 봐.", DAY1))
    await post(client, conv.committed(2, "오늘 점심 뭐 먹지", "아무거나.", DAY1 + timedelta(days=2)))
    await drain(services)

    mem = await rows(services, "SELECT type, value, index_status FROM memory")
    assert mem == [("SCHEDULE", "최종 면접", "INDEXED")]
    assert len(await rows(services, "SELECT 1 FROM unfinished_event WHERE status='OPEN'")) == 1

    day7 = datetime(2026, 11, 12, 23, 10, tzinfo=UTC)  # 11/13 08:10 KST, the interview day
    job = (await client.post(f"{API}/generations", json=conv.generation(3, "좋은 아침", day7))).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"

    system_variable = mock.chat_calls[-1][1].content
    assert "<open_events>" in system_variable and "최종 면접: 오늘" in system_variable
    assert "<memories>" in system_variable and "최종 면접" in system_variable
    assert "마지막 대화: 3일 전" in system_variable


async def test_correction_supersedes_previous_fact(client, services, mock):
    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나 민트초코 싫어해", "그래.", DAY1))
    await post(client, conv.committed(2, "근데 이제 민트초코 좋아해", "갑자기?", DAY1 + timedelta(days=1)))
    await drain(services)
    valid = await rows(services, "SELECT value FROM memory WHERE valid_to IS NULL")
    assert valid == [("좋아함",)]
    assert len(await rows(services, "SELECT 1 FROM memory")) == 2


async def test_false_memory_from_character_words_is_rejected(client, services, mock):
    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "현우는 대학생 때 뭐 했어?", "나 대학생이야. 아직.", DAY1))
    await drain(services)
    assert await rows(services, "SELECT 1 FROM memory") == []


async def test_relationship_event_applies_once(client, services, mock):
    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "요즘 취업 때문에 좀 힘들어", "...밥은 먹었어?", DAY1))
    await post(client, conv.committed(2, "진짜 취업 때문에 좀 힘들어", "알아.", DAY1 + timedelta(hours=1)))
    await drain(services)
    state = await rows(services, "SELECT trust, closeness, through_turn_seq, baseline_source FROM relationship_state")
    assert state == [(54.0, 42.0, 2, "MODEL")]  # mock baseline 50/40 + one disclosure, topic dedupe
    assert len(await rows(services, "SELECT 1 FROM relationship_event")) == 1


async def test_trivial_turns_skip_extraction(client, services, mock):
    calls = []
    mock.json_scripts["extract-v1"] = lambda m: calls.append(1) or {"memories": []}
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "ㅋㅋㅋㅋ", "뭐야.", DAY1))
    await post(client, conv.committed(2, "응", "어.", DAY1))
    await drain(services)
    assert calls == []
    state = await rows(services, "SELECT memory_through_turn_seq FROM ai_conversation")
    assert state == [(2,)]


async def test_revision_change_applies_to_next_turn(client, services, mock):
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.revision_changed(HYUNWOO | {"background": "현재 대학원생이다."}))
    await drain(services)
    job = (await client.post(f"{API}/generations", json=conv.generation(1, "안녕", DAY1))).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"
    assert "revision=2" in mock.chat_calls[-1][0].content and "대학원생" in mock.chat_calls[-1][0].content


async def test_reset_revokes_old_epoch_and_keeps_settings(client, services, mock):
    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나 다음 주 금요일에 면접 있어", "잘 봐.", DAY1))
    await drain(services)
    running = (await client.post(f"{API}/generations", json=conv.generation(2, "안녕", DAY1))).json()["aiJobId"]

    await post(client, conv.reset())
    await drain(services, generation=False)

    res = (await client.get(f"{API}/generations/{running}")).json()
    assert res["status"] == "REVOKED"
    events = (await client.get(f"{API}/generations/{running}/events")).text
    assert "REVOKED" in events

    assert await rows(services, "SELECT 1 FROM memory") == []
    assert await rows(services, "SELECT epoch FROM committed_turn") == [(2,)]  # new epoch's Turn 0 only
    old = MockConversation(conversation_id=conv.conversation_id, character_id=conv.character_id, epoch=1)
    r = await client.post(f"{API}/generations", json=old.generation(3, "안녕", DAY1))
    assert r.status_code == 409 and r.json()["error"]["code"] == "EPOCH_REVOKED"

    # Late AnswerCommitted for the old epoch cannot resurrect anything.
    late = old.committed(5, "늦은 이벤트", "...", DAY1)
    late["aggregateSeq"] = conv.last_seq + 1
    await post(client, late)
    await drain(services)
    assert await rows(services, "SELECT 1 FROM committed_turn WHERE epoch = 1") == []

    job = (await client.post(f"{API}/generations", json=conv.generation(1, "다시 안녕", DAY1))).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"
    assert "강현우" in mock.chat_calls[-1][0].content


async def test_character_deleted_purges_and_blocks(client, services, mock):
    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나 민트초코 싫어해", "그래.", DAY1))
    await drain(services)
    receipt = await post(client, conv.character_deleted())
    await drain(services)
    assert (await client.get(f"{API}/lifecycle-events/{receipt}")).json()["status"] == "APPLIED"
    for table in ("ai_conversation", "memory", "committed_turn", "settings_snapshot", "relationship_state"):
        assert await rows(services, f"SELECT 1 FROM {table}") == [], table
    assert len(await rows(services, "SELECT 1 FROM revocation WHERE scope='CHARACTER'")) == 1

    # a re-delivered / restored ConversationOpened does not bring the character back
    again = MockConversation(conversation_id="conv-new", character_id=conv.character_id)
    await post(client, again.opened())
    await drain(services)
    assert await rows(services, "SELECT 1 FROM ai_conversation") == []


async def test_crash_after_projection_resumes_before_next_turn(client, services, mock, monkeypatch):
    import asyncio

    import pytest

    from app.memory.pipeline import MemoryPipeline
    from app.persistence.jobs import claim

    conv = MockConversation()
    await post(client, conv.opened())
    await drain(services)
    mock.json_scripts["extract-v1"] = scripted_extractor
    await post(client, conv.committed(1, "나 다음 주 금요일에 면접 있어", "응", DAY1))
    await post(client, conv.committed(2, "응", "응", DAY1))
    await drain(services, memory=False)
    pipeline = MemoryPipeline(services)
    job = await claim(services.db, "MEMORY", "old-worker", 30)

    async def interrupted(*args):
        raise asyncio.CancelledError

    with monkeypatch.context() as m:
        m.setattr(pipeline, "_embed", interrupted)
        with pytest.raises(asyncio.CancelledError):
            await pipeline.run(job)
    assert await claim(services.db, "MEMORY", "other-worker", 30) is None
    async with services.db.tx() as s:
        await s.execute(text("UPDATE ai_job SET lease_until=now() - interval '1 second' WHERE id=:id"), {"id": job.id})
    await drain(services)
    assert await rows(services, "SELECT index_status FROM memory") == [("INDEXED",)]
    assert await rows(services, "SELECT run_count, status FROM ai_job WHERE kind='MEMORY' ORDER BY turn_seq") == [
        (2, "COMPLETED"), (1, "COMPLETED")
    ]
    assert await rows(services, "SELECT count(*) FROM usage_ledger WHERE purpose='extract'") == [(1,)]


async def test_delete_scrubs_queued_and_late_inbox_bodies(client, services):
    conv = MockConversation()
    await post(client, conv.opened())
    await drain(services)
    ev = conv.committed(1, "비공개 대화", "답변", DAY1)
    ev["aggregateSeq"] = 10
    await post(client, ev)
    await post(client, conv.character_deleted())
    await drain(services)
    await post(client, conv.committed(2, "늦은 원문", "응", DAY1))
    assert await rows(services, "SELECT count(*) FROM event_inbox WHERE payload <> '{}'::jsonb") == [(0,)]


async def test_revision_filters_character_memory_even_when_extraction_lags(client, services, mock):
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나는 고양이를 키워", "나는 대학생이야", DAY1))
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    mock.json_scripts["extract-v1"] = lambda _: {"memories": [
        {"type": "SHARED_EVENT", "subject": "character", "predicate": "직업", "value": "대학생",
         "sourceMessageIds": ["a1"], "evidence": "대학생이야"},
        {"type": "USER_FACT", "subject": "user", "predicate": "반려동물", "value": "고양이",
         "sourceMessageIds": ["u1"], "evidence": "고양이를 키워"},
    ]}
    mock.json_scripts["revision-check-v1"] = lambda msgs: {"conflicts": [
        {"id": line.split("id=")[1].split(":")[0], "reason": "직업 변경"}
        for line in msgs[-1].content.splitlines() if line.startswith("- id=") and "대학생" in line
    ]}
    await drain(services)
    assert await rows(services, "SELECT count(*) FROM memory") == [(2,)]
    assert await rows(services, "SELECT predicate, conflicts_with_revision FROM memory WHERE subject='character'") == [
        ("직업", 2)]
    services.settings.retrieval_gate = 0
    job = (await client.post(f"{API}/generations", json=conv.generation(2, "기억나?", DAY1))).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"
    memories = mock.chat_calls[-1][1].content
    assert "반려동물: 고양이" in memories and "직업: 대학생" not in memories


async def test_failed_event_keeps_body_until_conversation_is_deleted(client, services, monkeypatch):
    from app.lifecycle import handler

    conv = MockConversation()
    await post(client, conv.opened())
    await drain(services)

    async def broken(*args):
        raise RuntimeError("transient")

    monkeypatch.setitem(handler.APPLIERS, "AnswerCommitted", broken)
    await post(client, conv.committed(1, "원문", "답변", DAY1))
    await drain(services)
    assert await rows(services, "SELECT status, payload->>'conversationId' FROM event_inbox "
                                "WHERE type='AnswerCommitted'") == [("FAILED", conv.conversation_id)]
    monkeypatch.undo()
    await post(client, conv.character_deleted())
    await drain(services)
    assert await rows(services, "SELECT count(*) FROM event_inbox WHERE payload <> '{}'::jsonb") == [(0,)]


async def test_embedding_outage_completes_job_and_reindexes_later(client, services, mock):
    from app.providers.base import ProviderError

    conv = MockConversation()
    await post(client, conv.opened())
    await drain(services)
    mock.json_scripts["extract-v1"] = scripted_extractor
    real_embed = mock.embed

    async def down(texts, model):
        raise ProviderError("embedding down")

    mock.embed = down
    await post(client, conv.committed(1, "나 다음 주 금요일에 면접 있어", "응", DAY1))
    await drain(services)
    assert await rows(services, "SELECT run_count, status FROM ai_job WHERE kind='MEMORY'") == [(1, "COMPLETED")]
    assert await rows(services, "SELECT index_status, index_attempts, index_retry_at > now() FROM memory") == [
        ("FAILED", 1, True)]

    mock.embed = real_embed
    await post(client, conv.committed(2, "응", "응", DAY1))
    await drain(services)
    assert await rows(services, "SELECT index_status FROM memory") == [("FAILED",)]  # still backing off
    async with services.db.tx() as s:
        await s.execute(text("UPDATE memory SET index_retry_at = now() - interval '1 second'"))
    await post(client, conv.committed(3, "응", "응", DAY1))
    await drain(services)
    assert await rows(services, "SELECT index_status, index_attempts, index_retry_at FROM memory") == [
        ("INDEXED", 0, None)]


async def test_character_promise_survives_character_edit(client, services, mock):
    mock.json_scripts["extract-v1"] = lambda _: {"memories": [
        {"type": "UNFINISHED", "subject": "character", "predicate": "promise", "value": "노래 들려주기",
         "timeExpression": "다음 주 금요일", "resolvedAt": "2026-11-13", "timeUncertainty": "DAY",
         "importance": 0.9, "sourceMessageIds": ["a1"], "evidence": "다음 주 금요일에 노래 들려줄게",
         "opensUnfinishedEvent": True},
    ]}
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "다음 주 금요일에 노래 불러줘", "다음 주 금요일에 노래 들려줄게", DAY1))
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert len(await rows(services, "SELECT 1 FROM unfinished_event WHERE status='OPEN'")) == 1
    day7 = datetime(2026, 11, 12, 23, 10, tzinfo=UTC)  # due day in KST
    job = (await client.post(f"{API}/generations", json=conv.generation(2, "좋은 아침", day7))).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"
    system_variable = mock.chat_calls[-1][1].content
    assert "<open_events>" in system_variable and "노래 들려주기" in system_variable.split("<open_events>")[1]


def _character_memories_script(_):
    return {"memories": [
        {"type": "SHARED_EVENT", "subject": "character", "predicate": "직업", "value": "대학생",
         "sourceMessageIds": ["a1"], "evidence": "나는 대학생이야"},
        {"type": "SHARED_EVENT", "subject": "character", "predicate": "좋아하는 음식", "value": "라면",
         "sourceMessageIds": ["a1"], "evidence": "라면을 제일 좋아해"},
        {"type": "SHARED_EVENT", "subject": "shared", "predicate": "함께한 일", "value": "한강 산책",
         "sourceMessageIds": ["a1"], "evidence": "한강 산책 또 가자"},
    ]}


async def _memories_in_prompt(client, services, mock, conv, turn):
    services.settings.retrieval_gate = 0
    job = (await client.post(f"{API}/generations", json=conv.generation(turn, "기억나?", DAY1))).json()["aiJobId"]
    await drain(services)
    assert (await client.get(f"{API}/generations/{job}")).json()["status"] == "COMPLETED"
    return mock.chat_calls[-1][1].content


async def test_character_edit_hides_only_conflicting_memories(client, services, mock):
    conv = MockConversation()
    await post(client, conv.opened())
    mock.json_scripts["extract-v1"] = _character_memories_script
    await post(client, conv.committed(1, "너 요즘 학교 다니면서 뭐 해?", "나는 대학생이야. 라면을 제일 좋아해. 한강 산책 또 가자", DAY1))
    await drain(services)
    seen: list[str] = []

    def judge(msgs):
        seen.append(msgs[-1].content)
        return {"conflicts": [{"id": line.split("id=")[1].split(":")[0], "reason": "직업 변경"}
                              for line in msgs[-1].content.splitlines() if "대학생" in line and line.startswith("- id=")]
                + [{"id": "not-a-memory-id", "reason": "invented"}]}

    mock.json_scripts["revision-check-v1"] = judge
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert len(seen) == 1 and "배경:" in seen[0] and "-> 직장인" in seen[0]
    assert "id=" in seen[0] and "한강 산책" in seen[0]  # shared memories are checked too
    memories = await _memories_in_prompt(client, services, mock, conv, 2)
    assert "좋아하는 음식: 라면" in memories and "함께한 일: 한강 산책" in memories
    assert "직업: 대학생" not in memories
    assert sorted(await rows(services, "SELECT predicate, checked_character_rev, conflicts_with_revision FROM memory "
                                "")) == sorted([("좋아하는 음식", 2, None), ("직업", 1, 2), ("함께한 일", 2, None)])
    assert await rows(services, "SELECT count(*) FROM usage_ledger WHERE purpose='revision'") == [(1,)]


async def test_failed_revision_check_keeps_character_facts_hidden(client, services, mock):
    from app.providers.base import ProviderError

    conv = MockConversation()
    await post(client, conv.opened())
    mock.json_scripts["extract-v1"] = _character_memories_script
    await post(client, conv.committed(1, "너 요즘 학교 다니면서 뭐 해?", "나는 대학생이야. 라면을 제일 좋아해. 한강 산책 또 가자", DAY1))
    await drain(services)

    def down(_):
        raise ProviderError("model down")

    mock.json_scripts["revision-check-v1"] = down
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert await rows(services, "SELECT status, run_count FROM ai_job WHERE kind='REVISION'") == [("FAILED", 3)]
    memories = await _memories_in_prompt(client, services, mock, conv, 2)
    assert "함께한 일: 한강 산책" in memories  # shared stays visible
    assert "라면" not in memories and "대학생" not in memories  # unchecked character facts: fail closed


async def test_embedding_model_change_reindexes_old_vectors(client, services, mock):
    from app.memory.reindex import reindex_all, stale_counts

    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나 다음 주 금요일에 면접 있어", "응", DAY1))
    await post(client, conv.committed(2, "나 민트초코 싫어해", "응", DAY1))
    await drain(services)
    assert await rows(services, "SELECT DISTINCT embedding_version FROM memory") == [("mock-embed",)]

    services.settings.embedding_model = "mock-embed-v2"
    status = await stale_counts(services.db, "mock-embed-v2")
    assert status.remaining == 2 and status.by_version == {"mock-embed / INDEXED": 2}
    job = (await client.post(f"{API}/generations", json=conv.generation(3, "면접 얘기", DAY1))).json()["aiJobId"]
    await drain(services)
    assert "KEYWORD_FALLBACK" in (await client.get(f"{API}/generations/{job}")).json()["degraded"]

    # A memory job of the conversation re-embeds a few stale rows on its own ...
    services.settings.tuning = services.settings.tuning.model_copy(
        update={"memory": services.settings.tuning.memory.model_copy(update={"reindex_batch": 1})})
    await post(client, conv.committed(3, "응", "응", DAY1))
    await drain(services)
    assert (await stale_counts(services.db, "mock-embed-v2")).remaining == 1
    # ... and the sweep / CLI finishes the rest.
    result = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10)
    assert (result.embedded, result.remaining) == (1, 0)
    assert await rows(services, "SELECT DISTINCT embedding_version, index_status FROM memory") == [
        ("mock-embed-v2", "INDEXED")]
    assert await rows(services, "SELECT count(*) FROM usage_ledger WHERE attempt_id='reindex'") == [(1,)]
    job = (await client.post(f"{API}/generations", json=conv.generation(4, "면접 얘기", DAY1))).json()["aiJobId"]
    await drain(services)
    assert "KEYWORD_FALLBACK" not in (await client.get(f"{API}/generations/{job}")).json()["degraded"]


async def test_reindex_skips_revoked_conversations_and_survives_outage(client, services, mock):
    from app.memory.reindex import reindex_all, stale_counts
    from app.providers.base import ProviderError

    mock.json_scripts["extract-v1"] = scripted_extractor
    conv = MockConversation()
    await post(client, conv.opened())
    await post(client, conv.committed(1, "나 다음 주 금요일에 면접 있어", "응", DAY1))
    await drain(services)
    real_embed = mock.embed

    async def down(texts, model):
        raise ProviderError("down")

    mock.embed = down
    result = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10)
    assert (result.embedded, result.failed, result.remaining) == (0, 1, 1)
    assert await rows(services, "SELECT index_status, embedding_version FROM memory") == [("FAILED", "mock-embed")]
    mock.embed = real_embed
    await post(client, conv.reset())
    await drain(services)
    assert (await stale_counts(services.db, "mock-embed-v2")).remaining == 0


def _judge_job_conflicts(msgs):
    """Conflict when the new card says 직장인 and the memory says 대학생."""
    body = msgs[-1].content
    new_card = body.split("<character>")[1]
    if "직장인" not in new_card:
        return {"conflicts": []}
    return {"conflicts": [{"id": line.split("id=")[1].split(":")[0], "reason": "직업"}
                          for line in body.splitlines() if line.startswith("- id=") and "대학생" in line]}


async def _conversation_with_character_facts(client, services, mock):
    conv = MockConversation()
    await post(client, conv.opened())
    mock.json_scripts["extract-v1"] = _character_memories_script
    await post(client, conv.committed(1, "너 요즘 학교 다니면서 뭐 해?", "나는 대학생이야. 라면을 제일 좋아해. 한강 산책 또 가자", DAY1))
    await drain(services)
    return conv


async def test_reverted_edit_brings_conflicting_memory_back(client, services, mock):
    conv = await _conversation_with_character_facts(client, services, mock)
    mock.json_scripts["revision-check-v1"] = _judge_job_conflicts
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert await rows(services, "SELECT conflicts_with_revision FROM memory WHERE predicate='직업'") == [(2,)]
    await post(client, conv.revision_changed(HYUNWOO))  # edit undone
    await drain(services)
    assert await rows(services, "SELECT checked_character_rev, conflicts_with_revision FROM memory "
                                "WHERE predicate='직업'") == [(3, None)]
    assert "직업: 대학생" in await _memories_in_prompt(client, services, mock, conv, 2)


async def test_slow_check_for_older_edit_cannot_undo_newer_edit(client, services, mock):
    from app.lifecycle.handler import process_pending

    conv = await _conversation_with_character_facts(client, services, mock)
    real = mock.complete_json
    calls: list[str] = []

    async def slow_first_call(messages, params, schema_name):
        if schema_name == "revision-check-v1":
            calls.append(messages[-1].content)
            if len(calls) == 1:  # while judging rev2, the user edits again (rev3 reverts the job)
                await post(client, conv.revision_changed(HYUNWOO | {"background": "대학원생"}))
                await process_pending(services.db, tuning=services.settings.tuning)
                return {"conflicts": [{"id": line.split("id=")[1].split(":")[0]}
                                      for line in messages[-1].content.splitlines() if line.startswith("- id=")]}
        return await real(messages, params, schema_name)

    mock.complete_json = slow_first_call
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert len(calls) == 2 and "-> 대학원생" in calls[1]  # judged again against the newer card
    assert await rows(services, "SELECT DISTINCT checked_character_rev, conflicts_with_revision FROM memory "
                                "WHERE subject <> 'user'") == [(3, None)]
    assert await rows(services, "SELECT status, count(*) FROM ai_job WHERE kind='REVISION' GROUP BY 1") == [
        ("COMPLETED", 2)]


async def test_only_one_revision_check_runs_per_conversation(client, services, mock):
    from app.persistence import jobs

    conv = await _conversation_with_character_facts(client, services, mock)
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await post(client, conv.revision_changed(HYUNWOO | {"background": "대학원생"}))
    await drain(services, memory=False)
    first = await jobs.claim(services.db, "REVISION", "w1", 30)
    assert first is not None
    assert await jobs.claim(services.db, "REVISION", "w2", 30) is None


async def test_failed_revision_check_is_requeued_later(client, services, mock):
    from app.persistence import jobs
    from app.providers.base import ProviderError

    conv = await _conversation_with_character_facts(client, services, mock)

    def down(_):
        raise ProviderError("model down")

    mock.json_scripts["revision-check-v1"] = down
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert await jobs.requeue_failed_revisions(services.db, after_seconds=3600) == 0  # not yet
    mock.json_scripts["revision-check-v1"] = _judge_job_conflicts
    assert await jobs.requeue_failed_revisions(services.db, after_seconds=0) == 1
    await drain(services)
    assert sorted(await rows(services, "SELECT predicate, checked_character_rev, conflicts_with_revision FROM memory "
                                "WHERE subject <> 'user'")) == sorted([
        ("좋아하는 음식", 2, None), ("직업", 1, 2), ("함께한 일", 2, None)])


async def test_one_bad_memory_does_not_block_the_reindex_sweep(client, services, mock):
    from app.memory.reindex import reindex_all
    from app.providers.base import ProviderError

    await _conversation_with_character_facts(client, services, mock)
    real_embed = mock.embed

    async def rejects_ramen(texts, model):
        if any("라면" in t for t in texts):
            raise ProviderError("input rejected")
        return await real_embed(texts, model)

    mock.embed = rejects_ramen
    result = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10)
    assert (result.embedded, result.failed, result.remaining) == (2, 1, 1)
    assert sorted(await rows(services, "SELECT predicate, index_status, embedding_version FROM memory")) == sorted([
        ("좋아하는 음식", "FAILED", "mock-embed"), ("직업", "INDEXED", "mock-embed-v2"),
        ("함께한 일", "INDEXED", "mock-embed-v2")])


async def test_reindex_sweep_runs_once_across_workers_and_yields_to_users(client, services, mock):
    from app.memory.reindex import reindex_all, sweep_lock

    conv = await _conversation_with_character_facts(client, services, mock)
    async with sweep_lock(services.db) as got:
        assert got
        busy = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10)
        assert busy.locked and busy.embedded == 0
    await client.post(f"{API}/generations", json=conv.generation(2, "안녕", DAY1))  # queued, not run
    waiting = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10, yield_to_generation=True)
    assert waiting.embedded == 0 and waiting.remaining == 3
    await drain(services)
    done = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10, yield_to_generation=True)
    assert (done.embedded, done.remaining) == (3, 0)


async def test_mixed_vector_dimensions_during_model_switch(client, services, mock):
    from app.providers.mock import hash_embed

    conv = await _conversation_with_character_facts(client, services, mock)

    async def small(texts, model):
        return [hash_embed(t, dim=16) for t in texts], (await mock.__class__.embed(mock, texts, model))[1]

    mock.embed = small
    services.settings.embedding_model = "mock-embed-16"
    job = (await client.post(f"{API}/generations", json=conv.generation(2, "라면 좋아해?", DAY1))).json()["aiJobId"]
    await drain(services)
    body = (await client.get(f"{API}/generations/{job}")).json()
    assert body["status"] == "COMPLETED" and "KEYWORD_FALLBACK" in body["degraded"]


def _rule_following_judge(msgs):
    """Follows the prompt: 'check' looks only at changed fields, 'recheck' at the whole new card."""
    body = msgs[-1].content
    changed = body.split("<changed_fields>")[1].split("</changed_fields>")[0]
    card = body.split("<character>")[1].split("</character>")[0]
    says_worker = "직장인" in (card if "<mode>recheck</mode>" in body else changed)
    return {"conflicts": [{"id": line.split("id=")[1].split(":")[0]} for line in body.splitlines()
                          if says_worker and line.startswith("- id=") and "대학생" in line]}


async def test_unrelated_edit_keeps_conflicting_memory_hidden(client, services, mock):
    conv = await _conversation_with_character_facts(client, services, mock)
    seen: list[str] = []
    mock.json_scripts["revision-check-v1"] = lambda m: (seen.append(m[-1].content), _rule_following_judge(m))[1]
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인", "speechStyle": "존댓말"}))
    await drain(services)
    assert any("<mode>recheck</mode>" in c and "대학생" in c for c in seen)
    assert await rows(services, "SELECT predicate, conflicts_with_revision FROM memory "
                                "WHERE conflicts_with_revision IS NOT NULL") == [("직업", 3)]
    assert "직업: 대학생" not in await _memories_in_prompt(client, services, mock, conv, 2)


def test_revision_attempt_ids_fit_the_column():
    from app.lifecycle.handler import revision_attempt_id

    assert revision_attempt_id("conv-1", 1, 2) == "revision:conv-1:1:2"
    long = revision_attempt_id("c" * 64, 12, 345, "t999")
    assert len(long) <= 64 and long == revision_attempt_id("c" * 64, 12, 345, "t999")


async def test_rows_that_keep_failing_back_off_and_do_not_block_others(client, services, mock):
    from app.memory.reindex import reindex_all
    from app.providers.base import ProviderError

    await _conversation_with_character_facts(client, services, mock)  # 3 memories, oldest first
    order = [r[0] for r in await rows(services, "SELECT predicate FROM memory ORDER BY created_at, id")]
    real_embed = mock.embed
    down = {"on": True}

    async def flaky(texts, model):
        if down["on"]:
            raise ProviderError("down")
        return await real_embed(texts, model)

    mock.embed = flaky
    first = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10)
    assert first.embedded == 0 and first.failed == 3
    assert await rows(services, "SELECT count(*) FROM memory WHERE index_retry_at > now()") == [(3,)]
    # Nothing is due yet: the sweep does not hammer the same rows again.
    again = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=10)
    assert (again.embedded, again.failed) == (0, 0)
    # Make the newest row due first: retry order follows the schedule, not creation time.
    async with services.db.tx() as s:
        await s.execute(text("UPDATE memory SET index_retry_at = now() - interval '1 minute' WHERE predicate = :p"),
                        {"p": order[-1]})
    down["on"] = False
    third = await reindex_all(services.db, services.provider, "mock-embed-v2", batch=1, max_batches=1)
    assert third.embedded == 1
    assert await rows(services, "SELECT predicate FROM memory WHERE embedding_version = 'mock-embed-v2'") == [
        (order[-1],)]
    # A second failure doubles the wait.
    down["on"] = True
    async with services.db.tx() as s:
        await s.execute(text("UPDATE memory SET index_retry_at = now() - interval '1 second' WHERE predicate = :p"),
                        {"p": order[0]})
    await reindex_all(services.db, services.provider, "mock-embed-v2", batch=1, max_batches=1)
    assert await rows(services, "SELECT index_attempts, index_retry_at - now() > interval '100 seconds' FROM memory "
                                "WHERE predicate = :p", p=order[0]) == [(2, True)]


async def test_late_failure_does_not_overwrite_an_indexed_row(client, services, mock):
    from app.memory.reindex import MARK_FAILED_SQL

    await _conversation_with_character_facts(client, services, mock)
    ids = [r[0] for r in await rows(services, "SELECT id FROM memory")]
    async with services.db.tx() as s:  # e.g. the sweep failed on rows a memory job already re-embedded
        await s.execute(text(MARK_FAILED_SQL), {"ids": ids, "ev": "mock-embed"})
    assert await rows(services, "SELECT DISTINCT index_status, index_attempts FROM memory") == [("INDEXED", 0)]
    async with services.db.tx() as s:  # still stale for the new model: this failure is recorded
        await s.execute(text(MARK_FAILED_SQL), {"ids": ids, "ev": "mock-embed-v2"})
    assert await rows(services, "SELECT DISTINCT index_status, index_attempts FROM memory") == [("FAILED", 1)]


async def test_malformed_revision_answer_retries_instead_of_unhiding(client, services, mock):
    conv = await _conversation_with_character_facts(client, services, mock)
    mock.json_scripts["revision-check-v1"] = _judge_job_conflicts
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인"}))
    await drain(services)
    assert await rows(services, "SELECT conflicts_with_revision FROM memory WHERE predicate='직업'") == [(2,)]
    answers = iter([{}, {"conflicts": "oops"}])
    mock.json_scripts["revision-check-v1"] = lambda m: (lambda a: _judge_job_conflicts(m) if a == "ok" else a)(
        next(answers, "ok"))
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인", "speechStyle": "존댓말"}))
    await drain(services)
    # Two malformed answers were retried; the job only succeeded on the third (well-formed) answer.
    assert await rows(services, "SELECT run_count, status FROM ai_job WHERE kind='REVISION' "
                                "AND payload->>'characterRevision' = '3'") == [(3, "COMPLETED")]
    mock.json_scripts["revision-check-v1"] = lambda _: {}
    await post(client, conv.revision_changed(HYUNWOO | {"background": "직장인", "speechStyle": "반말"}))
    await drain(services)
    assert await rows(services, "SELECT status FROM ai_job WHERE kind='REVISION' "
                                "AND payload->>'characterRevision' = '4'") == [("FAILED",)]
    assert await rows(services, "SELECT checked_character_rev FROM memory WHERE predicate='라면' "
                                "OR predicate='좋아하는 음식'") == [(3,)]  # not marked compatible with rev 4
    assert "라면" not in await _memories_in_prompt(client, services, mock, conv, 2)

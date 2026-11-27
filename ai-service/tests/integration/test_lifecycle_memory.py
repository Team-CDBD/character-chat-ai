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

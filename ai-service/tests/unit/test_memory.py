from app.memory.prefilter import worth_extracting
from app.memory.reducer import ExistingMemory, reduce
from app.memory.validator import TurnText, parse_extraction, validate

TURN = TurnText(
    turn_seq=11,
    user_msg_id="u11",
    user_text="나 다음 주 금요일에 면접 있어. 그리고 민트초코 싫어해",
    assistant_msg_id="a11",
    assistant_text="면접? 잘 봐. 끝나면 우리 영화 보러 가자.",
)


def mem(**kw):
    base = {
        "type": "USER_FACT", "subject": "user", "predicate": "dislikes", "value": "민트초코",
        "sourceMessageIds": ["u11"], "evidence": "민트초코 싫어해", "importance": 0.7,
    }
    base.update(kw)
    return base


def test_prefilter():
    assert not worth_extracting("ㅋㅋㅋㅋ", "ㅋㅋ")
    assert not worth_extracting("응", "그래")
    assert not worth_extracting("좋은 아침!", "응")
    assert worth_extracting("나 다음 주 금요일에 면접 있어", "...")
    assert worth_extracting("민트초코 싫어해", "...")


def test_valid_memory_passes():
    report = validate(parse_extraction({"memories": [mem()]}), TURN, set())
    assert len(report.memories) == 1 and not report.rejected


def test_evidence_must_be_in_source_message():
    report = validate(parse_extraction({"memories": [mem(evidence="딸기를 좋아해")]}), TURN, set())
    assert report.memories == [] and report.rejected == [("memory", "evidence_not_found")]


def test_unknown_message_id_rejected():
    report = validate(parse_extraction({"memories": [mem(sourceMessageIds=["u99"])]}), TURN, set())
    assert report.memories == []


def test_user_fact_cannot_come_from_character():
    m = mem(evidence="영화 보러 가자", sourceMessageIds=["a11"], predicate="plans", value="영화")
    report = validate(parse_extraction({"memories": [m]}), TURN, set())
    assert report.memories == [] and report.rejected[0][1] == "user_fact_from_character"


def test_character_suggestion_does_not_open_unfinished_event():
    m = mem(type="SHARED_EVENT", subject="both", predicate="plan", value="영화", evidence="영화 보러 가자",
            sourceMessageIds=["a11"], opensUnfinishedEvent=True)
    report = validate(parse_extraction({"memories": [m]}), TURN, set())
    assert len(report.memories) == 1 and report.memories[0].opens_unfinished_event is False


def test_partial_salvage_of_malformed_output():
    out = parse_extraction({"memories": [mem(), {"type": "NOPE"}], "relationEvents": "bad"})
    assert len(out.memories) == 1


def test_resolve_needs_known_open_event():
    raw = {"resolves": [{"unfinishedEventId": "ue-x", "outcome": "RESOLVED", "sourceMessageIds": ["u11"]}]}
    assert validate(parse_extraction(raw), TURN, set()).resolves == []
    assert len(validate(parse_extraction(raw), TURN, {"ue-x"}).resolves) == 1


def test_reducer_supersedes_and_dedupes():
    out = parse_extraction({"memories": [mem(predicate="likes", value="민트초코", evidence="민트초코")]})
    existing = [ExistingMemory("m1", "USER_FACT", "user", "likes", "딸기")]
    decisions = reduce(out.memories, existing)
    assert decisions[0].action == "INSERT" and decisions[0].supersedes_id == "m1"

    existing = [ExistingMemory("m1", "USER_FACT", "user", "likes", "민트 초코")]
    assert reduce(out.memories, existing)[0].action == "SKIP_DUPLICATE"

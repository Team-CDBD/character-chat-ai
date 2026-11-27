import pytest

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


def test_short_self_disclosures_reach_extraction():
    assert worth_extracting("난 대전에 살아", "")
    assert worth_extracting("난 고양이를 키워", "")


def test_preference_cannot_reverse_quoted_dislike():
    result = validate(parse_extraction({"memories": [mem(predicate="likes")]}), TURN, set())
    assert result.rejected == [("memory", "unsupported_preference")]


def test_prefilter_covers_tense_and_spacing_without_keyword_false_positive():
    for text in ("부산에 살았어", "서울 살아", "강아지 키운다"):
        assert worth_extracting(text, ""), text
    assert not worth_extracting("키워드 뭐였지", "")


def test_preference_contradiction_checks_the_words_after_the_target():
    cases = [
        ("likes", "민트초코 진짜 싫어해", False),
        ("likes", "민트초코 싫어해요 ㅋㅋ", False),
        ("likes", "나 민트초코 싫어", False),
        ("likes", "민트초코 싫어하는데", False),
        ("likes", "민트초코 좋아하는 사람 싫어", True),
        ("dislikes", "민트초코 안 좋아해", True),
        ("dislikes", "민트초코 좋아해", False),
    ]
    for predicate, evidence, accepted in cases:
        turn = TurnText(11, "u11", evidence, "a11", "응")
        result = validate(parse_extraction({"memories": [mem(predicate=predicate, evidence=evidence)]}), turn, set())
        assert (result.memories != []) is accepted, (predicate, evidence)
    turn = TurnText(11, "u11", "민트초코 싫어해 근데 딸기는 좋아", "a11", "응")
    out = parse_extraction({"memories": [mem(predicate="likes", value="딸기", evidence="민트초코 싫어해 근데 딸기는 좋아")]})
    assert len(validate(out, turn, set()).memories) == 1


def test_subject_is_normalized_so_revision_filter_cannot_be_bypassed():
    turn = TurnText(11, "u11", "나 민트초코 싫어해", "a11", "나는 대학생이야")
    out = parse_extraction({"memories": [
        mem(type="SHARED_EVENT", subject="assistant", predicate="job", value="대학생", evidence="대학생이야",
            sourceMessageIds=["a11"]),
        mem(type="SHARED_EVENT", subject="hyunwoo", predicate="job", value="대학생", evidence="대학생이야",
            sourceMessageIds=["a11"]),
        mem(type="SHARED_EVENT", subject="both", predicate="plan", value="영화", evidence="대학생이야",
            sourceMessageIds=["a11"]),
    ]})
    assert [m.subject for m in validate(out, turn, set()).memories] == ["character", "character", "shared"]


def test_revision_check_helpers_only_trust_sent_ids_and_real_changes():
    from app.memory.pipeline import changed_fields, parse_conflicts

    old = {"name": "현우", "background": "대학생", "speechStyle": "반말"}
    new = {"name": "현우", "background": "직장인", "speechStyle": "반말", "extra": "고양이를 키운다"}
    assert changed_fields(old, new) == {"배경": ("대학생", "직장인"), "추가 설정": ("", "고양이를 키운다")}
    assert changed_fields(new, dict(new)) == {}
    assert set(changed_fields(None, new)) == {"이름", "배경", "말투", "추가 설정"}
    result = {"conflicts": [{"id": "m1"}, {"id": "zzz"}, "m2"]}
    assert parse_conflicts(result, {"m1", "m2"}) == {"m1", "m2"}  # ids we did not send are ignored
    assert parse_conflicts({"conflicts": []}, {"m1"}) == set()


@pytest.mark.parametrize("bad", [{}, {"conflicts": "m1"}, {"conflicts": None}, {"conflicts": [3]},
                                 {"conflicts": [{"reason": "no id"}]}, {"conflicts": [{"id": 7}]}, ["m1"]])
def test_malformed_revision_check_answer_is_an_error_not_no_conflicts(bad):
    from app.memory.pipeline import parse_conflicts
    from app.providers.base import ProviderError

    with pytest.raises(ProviderError):
        parse_conflicts(bad, {"m1"})

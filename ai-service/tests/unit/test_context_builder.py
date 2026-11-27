from app.knowledge.context_builder import (
    INPUT_BUDGET,
    ContextInputs,
    MemoryLine,
    RecentTurn,
    build,
    opening_scene_turn,
)
from app.providers.base import estimate_tokens

CARD = {"name": "강현우", "personality": "무뚝뚝하다", "speechStyle": "반말. 문장이 짧다."}


def make(**kw):
    base = dict(
        policy_text="정책", character=CARD, character_rev=4, persona={"name": "예진"},
        relationship_guideline="편한 사이다.", open_events=[], memories=[], summary=None,
        time_line="지금: 2026-11-14(금) 08:10", recent_turns=[], current_user_text="좋은 아침",
        recent_window=6,
    )
    base.update(kw)
    return ContextInputs(**base)


def test_layer_order_stable_first():
    built = build(make(), estimate_tokens)
    assert built.messages[0].role == "system" and "<character>" in built.messages[0].content
    assert "<time>" in built.messages[1].content
    assert built.messages[-1].content == "좋은 아침"


def test_opening_scene_is_turn_zero():
    t0 = opening_scene_turn({"narration": "비 오는 퇴근길", "characterLine": "우산 없어?", "messageId": "s0"})
    built = build(make(recent_turns=[t0]), estimate_tokens)
    assert built.included_turns == [0]
    assert built.messages[2].role == "assistant" and "우산 없어?" in built.messages[2].content


def test_recent_window_shrinks_but_keeps_minimum_and_character():
    long = "가" * 1000
    turns = [RecentTurn(i, long, long) for i in range(1, 7)]
    built = build(make(recent_turns=turns), estimate_tokens)
    assert len(built.included_turns) == 3
    assert "RECENT_WINDOW_SHRUNK" in built.degraded
    assert "강현우" in built.messages[0].content


def test_memories_ranked_and_budgeted():
    mems = [MemoryLine(f"m{i}", "- 기억 " + "나" * 300, score=i / 10) for i in range(10)]
    built = build(make(memories=mems), estimate_tokens)
    assert built.memory_ids[0] == "m9"
    assert built.layer_tokens["memories"] <= 900


def test_total_never_exceeds_budget():
    mems = [MemoryLine(f"m{i}", "기억" * 100, score=0.5) for i in range(20)]
    turns = [RecentTurn(i, "가" * 700, "나" * 700) for i in range(1, 7)]
    built = build(make(memories=mems, summary="요약" * 2000, recent_turns=turns), estimate_tokens)
    assert sum(built.layer_tokens.values()) <= INPUT_BUDGET


def test_huge_turns_go_below_minimum_instead_of_failing():
    huge = "가" * 3000
    turns = [RecentTurn(i, huge, huge) for i in range(1, 7)]
    built = build(make(recent_turns=turns), estimate_tokens)
    assert built.included_turns == [6]
    assert "RECENT_BELOW_MINIMUM" in built.degraded


def test_input_hash_is_deterministic():
    assert build(make(), estimate_tokens).input_hash() == build(make(), estimate_tokens).input_hash()

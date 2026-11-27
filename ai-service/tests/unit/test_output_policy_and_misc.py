from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from app.dialogue.manager import humanize_gap, time_line
from app.knowledge.output_policy import ChunkPolicy, expects_banmal
from app.relationship.unfinished import OpenEventView, effective_status, select_for_context
from app.understanding.normalize import derive_query_hint, normalize

NOW = datetime(2026, 11, 13, 23, 10, tzinfo=UTC)  # 2026-11-14 08:10 KST


def test_chunks_are_released_at_sentence_boundaries():
    p = ChunkPolicy(expects_banmal=True)
    assert p.feed("왔어") == []
    assert p.feed("? 늦었네. 밥") == ["왔어? ", "늦었네. "]
    assert p.flush() == ["밥"]


def test_role_break_and_honorific_are_flagged_not_blocked():
    p = ChunkPolicy(expects_banmal=True)
    p.feed("저는 AI라서 잘 모르겠습니다. ")
    assert {"ROLE_BREAK", "SPEECH_STYLE_HONORIFIC"} <= p.flags
    assert not p.check_final("저는 AI라서요").blocked


def test_expects_banmal():
    assert expects_banmal("반말. 문장이 짧다.")
    assert not expects_banmal("존댓말을 쓴다")


def test_unfinished_due_and_expired():
    ev = OpenEventView("e1", "최종 면접", "OPEN", NOW + timedelta(hours=3), NOW - timedelta(days=7))
    assert effective_status(ev, NOW) == "DUE"
    assert effective_status(ev, NOW + timedelta(days=20)) == "EXPIRED"


def test_unfinished_gate_and_limit():
    due = OpenEventView("due", "면접", "OPEN", NOW, NOW)
    off = OpenEventView("off", "이사", "OPEN", NOW + timedelta(days=10), NOW)
    rel = OpenEventView("rel", "시험", "OPEN", NOW + timedelta(days=10), NOW)
    picked = select_for_context([off, rel, due], NOW, {"rel": 0.5, "off": 0.1}, set(), gate=0.3)
    assert [p[0].id for p in picked] == ["due", "rel"]
    assert select_for_context([due], NOW, {}, {"due"}, gate=0.3) == []


def test_query_hint_resolves_deictic():
    assert "면접" in derive_query_hint("그때 거기 어땠어?", ["다음 주 면접 장소가 강남이야"])
    assert normalize("ㅋㅋㅋㅋㅋㅋ  진짜") == "ㅋㅋㅋ 진짜"


def test_time_line():
    line = time_line(NOW, ZoneInfo("Asia/Seoul"), NOW - timedelta(days=2))
    assert line == "지금: 2026-11-14(토) 08:10 · 마지막 대화: 2일 전"
    assert humanize_gap(None, NOW) == "처음 대화"

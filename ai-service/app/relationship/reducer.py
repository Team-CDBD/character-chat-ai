"""Deterministic Relationship Reducer (plan §08).

The model only *proposes* relation events; this module decides how much they move the
state. Message counts never change the relationship.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from app.tuning import DEFAULT_TUNING, RelationshipTuning

_D = DEFAULT_TUNING.relationship
REDUCER_VERSION = "rel-reducer-v1"
SCALE_MIN, SCALE_MAX = 0.0, 100.0
HYSTERESIS = _D.hysteresis
DAILY_POSITIVE_CAP = _D.daily_positive_cap  # SHARED_EXPERIENCE / SUPPORT closeness per day


class RelationKind(StrEnum):
    PERSONAL_DISCLOSURE = "PERSONAL_DISCLOSURE"
    PROMISE_KEPT = "PROMISE_KEPT"
    PROMISE_BROKEN = "PROMISE_BROKEN"
    CONFLICT = "CONFLICT"
    RECONCILIATION = "RECONCILIATION"
    SHARED_EXPERIENCE = "SHARED_EXPERIENCE"
    SUPPORT = "SUPPORT"


# kind -> (trust, closeness, tension); defaults live in app/tuning.py
DELTAS: dict[RelationKind, tuple[float, float, float]] = {RelationKind(k): v for k, v in _D.deltas.items()}
DAILY_CAPPED = {RelationKind.SHARED_EXPERIENCE, RelationKind.SUPPORT}


class Stage(StrEnum):
    DISTANT = "DISTANT"  # 거리감
    FAMILIAR = "FAMILIAR"  # 익숙함
    CLOSE = "CLOSE"  # 가까움
    TENSE = "TENSE"  # 긴장 상태 (overrides)


@dataclass
class RelationState:
    trust: float
    closeness: float
    tension: float
    stage: Stage
    daily_gain: dict[str, float] = field(default_factory=dict)  # "YYYY-MM-DD" -> closeness gained


@dataclass(frozen=True)
class ProposedRelationEvent:
    kind: str
    source_msg_ids: tuple[str, ...]
    evidence: str
    topic: str | None = None  # e.g. disclosure topic; same topic counts once


@dataclass(frozen=True)
class AppliedRelationEvent:
    event_key: str
    kind: RelationKind
    delta: dict[str, float]
    proposal: ProposedRelationEvent


@dataclass(frozen=True)
class ReduceContext:
    day: str  # local date of the turn, for the daily cap
    open_conflict: bool  # RECONCILIATION only valid when a conflict is open
    resolved_shared_event: bool  # PROMISE_KEPT only when a linked SHARED_EVENT was RESOLVED
    existing_keys: frozenset[str]


def _clamp(v: float) -> float:
    return max(SCALE_MIN, min(SCALE_MAX, v))


def event_key(kind: RelationKind, proposal: ProposedRelationEvent) -> str:
    """Stable key so the same event is never applied twice (re-delivery, re-extraction)."""
    if kind == RelationKind.PERSONAL_DISCLOSURE and proposal.topic:
        basis = f"{kind}:topic:{proposal.topic.strip().lower()}"
    else:
        basis = f"{kind}:" + ",".join(sorted(proposal.source_msg_ids))
    return hashlib.sha256(basis.encode()).hexdigest()[:40]


def baseline_state(
    trust: float, closeness: float, tension: float = 0.0, tuning: RelationshipTuning = _D
) -> RelationState:
    t, c, x = _clamp(trust), _clamp(closeness), _clamp(tension)
    return RelationState(trust=t, closeness=c, tension=x, stage=stage_of(t, c, x, previous=None, tuning=tuning))


def default_baseline(tuning: RelationshipTuning = _D) -> RelationState:
    return baseline_state(tuning.default_trust, tuning.default_closeness, tuning.default_tension, tuning)


def apply_relation_events(
    state: RelationState, proposals: list[ProposedRelationEvent], ctx: ReduceContext,
    tuning: RelationshipTuning = _D,
) -> tuple[RelationState, list[AppliedRelationEvent]]:
    trust, closeness, tension = state.trust, state.closeness, state.tension
    daily = dict(state.daily_gain)
    applied: list[AppliedRelationEvent] = []
    seen = set(ctx.existing_keys)
    open_conflict = ctx.open_conflict

    for p in proposals:
        try:
            kind = RelationKind(p.kind)
        except ValueError:
            continue  # unknown kinds from the model are ignored
        if not p.source_msg_ids or not p.evidence.strip():
            continue  # evidence is mandatory
        if kind == RelationKind.RECONCILIATION and not open_conflict:
            continue
        if kind == RelationKind.PROMISE_KEPT and not ctx.resolved_shared_event:
            continue
        key = event_key(kind, p)
        if key in seen:
            continue
        dt, dc, dx = tuning.deltas[kind.value]
        if kind in DAILY_CAPPED:
            used = daily.get(ctx.day, 0.0)
            dc = max(0.0, min(dc, tuning.daily_positive_cap - used))
            daily[ctx.day] = used + dc
            if dc == 0:
                continue
        trust, closeness, tension = _clamp(trust + dt), _clamp(closeness + dc), _clamp(tension + dx)
        if kind == RelationKind.CONFLICT:
            open_conflict = True
        if kind == RelationKind.RECONCILIATION:
            open_conflict = False
        seen.add(key)
        applied.append(AppliedRelationEvent(key, kind, {"trust": dt, "closeness": dc, "tension": dx}, p))

    # keep only recent days in the cap ledger
    daily = {d: v for d, v in sorted(daily.items())[-7:]}
    new_stage = stage_of(trust, closeness, tension, previous=state.stage, tuning=tuning)
    return RelationState(trust, closeness, tension, new_stage, daily), applied


def stage_of(
    trust: float, closeness: float, tension: float, previous: Stage | None, tuning: RelationshipTuning = _D
) -> Stage:
    """Stage with hysteresis (default ±3) so the tone does not flip every turn."""
    h = tuning.hysteresis

    def above(value: float, threshold: float, currently_above: bool) -> bool:
        return value >= threshold - h if currently_above else value >= threshold + (h if previous is not None else 0)

    if above(tension, tuning.tense_tension, previous == Stage.TENSE):
        return Stage.TENSE
    was_close = previous == Stage.CLOSE
    if above(closeness, tuning.close_closeness, was_close) and above(trust, tuning.close_trust, was_close):
        return Stage.CLOSE
    was_familiar_or_more = previous in (Stage.FAMILIAR, Stage.CLOSE)
    if above(closeness, tuning.familiar_closeness, was_familiar_or_more):
        return Stage.FAMILIAR
    return Stage.DISTANT


GUIDELINES: dict[Stage, str] = {
    Stage.DISTANT: "아직 거리를 두고 있다. 사적인 관심을 드러내지 않는다.",
    Stage.FAMILIAR: "편한 사이다. 걱정을 툭 던지듯 표현해도 된다.",
    Stage.CLOSE: "상대를 기다렸다는 티가 조금씩 난다. 그래도 직접 말하진 않는다.",
    Stage.TENSE: "최근 다툼 이후라 말이 더 짧아진다. 먼저 사과하지 않는다(설정에 따름).",
}


def guideline(stage: Stage) -> str:
    """Numbers never reach the prompt; only behaviour guidance does (plan 핵심 결정 ⑥)."""
    return f"{GUIDELINES[stage]} 어떤 관계 단계에서도 캐릭터의 성격과 말투를 우선한다."

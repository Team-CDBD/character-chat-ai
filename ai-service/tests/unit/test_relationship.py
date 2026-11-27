from app.relationship.reducer import (
    ProposedRelationEvent,
    ReduceContext,
    Stage,
    apply_relation_events,
    baseline_state,
    guideline,
    stage_of,
)


def ctx(**kw):
    base = dict(day="2026-11-14", open_conflict=False, resolved_shared_event=False, existing_keys=frozenset())
    base.update(kw)
    return ReduceContext(**base)


def ev(kind, ids=("u1",), evidence="근거", topic=None):
    return ProposedRelationEvent(kind, tuple(ids), evidence, topic)


def test_disclosure_applies_fixed_delta_not_model_value():
    state = baseline_state(50, 40)
    new, applied = apply_relation_events(state, [ev("PERSONAL_DISCLOSURE", topic="취업")], ctx())
    assert (new.trust, new.closeness) == (54, 42)
    assert len(applied) == 1


def test_same_event_key_is_never_applied_twice():
    state = baseline_state(50, 40)
    new, applied = apply_relation_events(state, [ev("PERSONAL_DISCLOSURE", topic="취업")], ctx())
    again, applied2 = apply_relation_events(
        new, [ev("PERSONAL_DISCLOSURE", ids=("u9",), topic="취업")], ctx(existing_keys=frozenset({applied[0].event_key}))
    )
    assert applied2 == [] and again.trust == new.trust


def test_reconciliation_requires_open_conflict():
    state = baseline_state(50, 40, 20)
    _, applied = apply_relation_events(state, [ev("RECONCILIATION")], ctx(open_conflict=False))
    assert applied == []
    new, applied = apply_relation_events(state, [ev("CONFLICT", ids=("u1",)), ev("RECONCILIATION", ids=("u2",))], ctx())
    assert [a.kind for a in applied] == ["CONFLICT", "RECONCILIATION"]
    assert new.tension == 18


def test_promise_kept_requires_resolved_shared_event():
    state = baseline_state(50, 40)
    _, applied = apply_relation_events(state, [ev("PROMISE_KEPT")], ctx())
    assert applied == []
    new, _ = apply_relation_events(state, [ev("PROMISE_KEPT")], ctx(resolved_shared_event=True))
    assert new.trust == 55


def test_daily_cap_on_positive_events():
    state = baseline_state(50, 40)
    proposals = [ev("SHARED_EXPERIENCE", ids=(f"u{i}",)) for i in range(5)]
    new, applied = apply_relation_events(state, proposals, ctx())
    assert new.closeness == 46  # +6 cap
    assert len(applied) == 2


def test_events_without_evidence_or_unknown_kind_are_ignored():
    state = baseline_state(50, 40)
    _, applied = apply_relation_events(state, [ev("PERSONAL_DISCLOSURE", evidence=" "), ev("LOVE_BOMB")], ctx())
    assert applied == []


def test_stage_hysteresis():
    assert stage_of(50, 36, 0, previous=None) == Stage.FAMILIAR
    # crossing upward from DISTANT needs +3
    assert stage_of(50, 36, 0, previous=Stage.DISTANT) == Stage.DISTANT
    assert stage_of(50, 38, 0, previous=Stage.DISTANT) == Stage.FAMILIAR
    # staying needs only -3
    assert stage_of(50, 33, 0, previous=Stage.FAMILIAR) == Stage.FAMILIAR
    assert stage_of(50, 31, 0, previous=Stage.FAMILIAR) == Stage.DISTANT
    assert stage_of(90, 90, 55, previous=Stage.CLOSE) == Stage.TENSE


def test_guideline_has_no_numbers():
    for stage in Stage:
        assert not any(ch.isdigit() for ch in guideline(stage))

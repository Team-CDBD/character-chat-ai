"""Unfinished Event lifecycle and exposure rules (plan §08)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.tuning import DEFAULT_TUNING, UnfinishedTuning

_D = DEFAULT_TUNING.unfinished
DUE_WINDOW = timedelta(hours=_D.due_window_hours)
EXPIRE_AFTER = timedelta(days=_D.expire_after_days)
MAX_EXPOSED = _D.max_exposed


@dataclass(frozen=True)
class OpenEventView:
    id: str
    title: str
    status: str  # stored: OPEN | RESOLVED | CANCELLED
    due_at: datetime | None
    created_at: datetime


def effective_status(ev: OpenEventView, now: datetime, tuning: UnfinishedTuning = _D) -> str:
    """DUE / EXPIRED are computed at read time, never stored."""
    if ev.status != "OPEN":
        return ev.status
    if ev.due_at is not None:
        if now > ev.due_at + timedelta(days=tuning.expire_after_days):
            return "EXPIRED"
        if abs(now - ev.due_at) <= timedelta(hours=tuning.due_window_hours):
            return "DUE"
    elif now > ev.created_at + timedelta(days=tuning.undated_expire_after_days):
        return "EXPIRED"
    return "OPEN"


def select_for_context(
    events: list[OpenEventView],
    now: datetime,
    similarity: dict[str, float],
    recently_mentioned: set[str],
    gate: float,
    tuning: UnfinishedTuning = _D,
) -> list[tuple[OpenEventView, str]]:
    """DUE first; non-DUE only when related to the current topic; skip what the character just mentioned."""
    picked: list[tuple[OpenEventView, str, float]] = []
    for ev in events:
        status = effective_status(ev, now, tuning)
        if status not in ("OPEN", "DUE") or ev.id in recently_mentioned:
            continue
        sim = similarity.get(ev.id, 0.0)
        if status == "DUE" or sim >= gate:
            picked.append((ev, status, (1.0 if status == "DUE" else 0.0) + sim))
    picked.sort(key=lambda x: x[2], reverse=True)
    return [(ev, status) for ev, status, _ in picked[: tuning.max_exposed]]


def describe(ev: OpenEventView, status: str, now: datetime, tz) -> str:
    if ev.due_at is None:
        return f"- {ev.title}: 결과는 아직 모름."
    local_due = ev.due_at.astimezone(tz)
    days = (local_due.date() - now.astimezone(tz).date()).days
    when = {0: "오늘", 1: "내일", -1: "어제"}.get(days, f"{local_due.month}/{local_due.day}")
    tail = "예정. 결과는 아직 모름." if days >= 0 else "지났음. 결과는 아직 모름(추정하지 말 것)."
    return f"- {ev.title}: {when}({local_due.month}/{local_due.day}) {tail}"

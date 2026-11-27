"""Evidence validation for extracted memories (plan §07 "False Memory를 막는 규칙").

The extraction model proposes; this module rejects anything without real evidence in the
committed turn. Nothing here trusts the model's output for scope, permissions or settings.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic.alias_generators import to_camel

EXTRACTOR_VERSION = "extract-v1"


class MemoryType(StrEnum):
    USER_FACT = "USER_FACT"
    SCHEDULE = "SCHEDULE"
    SHARED_EVENT = "SHARED_EVENT"
    RELATION_EVENT = "RELATION_EVENT"
    UNFINISHED = "UNFINISHED"


class _M(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class ExtractedMemory(_M):
    type: MemoryType
    subject: str = Field(max_length=32)
    predicate: str = Field(max_length=64)
    value: str = Field(max_length=500)
    time_expression: str | None = Field(default=None, max_length=64)
    resolved_at: datetime | None = None
    time_uncertainty: str | None = None
    is_fiction: bool = False
    importance: float = 0.5
    source_message_ids: list[str] = Field(min_length=1)
    evidence: str = Field(min_length=1, max_length=500)
    opens_unfinished_event: bool = False


class ExtractedRelationEvent(_M):
    kind: str
    source_message_ids: list[str] = Field(min_length=1)
    evidence: str = Field(min_length=1, max_length=500)
    topic: str | None = None


class ExtractedResolve(_M):
    unfinished_event_id: str
    outcome: str  # RESOLVED | CANCELLED
    source_message_ids: list[str] = Field(min_length=1)


class ExtractionOutput(_M):
    memories: list[ExtractedMemory] = Field(default_factory=list)
    relation_events: list[ExtractedRelationEvent] = Field(default_factory=list)
    resolves: list[ExtractedResolve] = Field(default_factory=list)


@dataclass(frozen=True)
class TurnText:
    turn_seq: int
    user_msg_id: str | None
    user_text: str | None
    assistant_msg_id: str | None
    assistant_text: str


@dataclass
class ValidationReport:
    memories: list[ExtractedMemory]
    relation_events: list[ExtractedRelationEvent]
    resolves: list[ExtractedResolve]
    rejected: list[tuple[str, str]]  # (what, reason) - for metrics only, never logged with raw text


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[\s\"'“”‘’.,!?~…·]+", "", text)


def parse_extraction(raw: dict[str, Any]) -> ExtractionOutput:
    try:
        return ExtractionOutput.model_validate(raw)
    except ValidationError:
        # Salvage valid items instead of dropping the whole output.
        out = ExtractionOutput()
        for key, model, target in (
            ("memories", ExtractedMemory, out.memories),
            ("relationEvents", ExtractedRelationEvent, out.relation_events),
            ("resolves", ExtractedResolve, out.resolves),
        ):
            for item in raw.get(key, []) if isinstance(raw, dict) else []:
                try:
                    target.append(model.model_validate(item))
                except ValidationError:
                    continue
        return out


def _evidence_sources(turn: TurnText, ids: list[str], evidence: str) -> set[str] | None:
    """Return which speakers ('user'/'assistant') the evidence is found in, or None if invalid."""
    valid_ids = {i for i in (turn.user_msg_id, turn.assistant_msg_id) if i}
    if not ids or not set(ids) <= valid_ids:
        return None
    ev = normalize(evidence)
    if not ev:
        return None
    found: set[str] = set()
    if turn.user_msg_id in ids and turn.user_text and ev in normalize(turn.user_text):
        found.add("user")
    if turn.assistant_msg_id in ids and ev in normalize(turn.assistant_text):
        found.add("assistant")
    return found or None


_SUBJECTS = {
    "user": "user", "사용자": "user", "유저": "user", "me": "user",
    "character": "character", "assistant": "character", "캐릭터": "character", "bot": "character",
    "shared": "shared", "both": "shared", "we": "shared", "us": "shared", "우리": "shared", "둘": "shared",
}


def _normalize_subject(subject: str, speakers: set[str]) -> str:
    """Map the model's free-form subject onto user / character / shared.

    Retrieval hides character facts from older character revisions, so an unknown subject must not
    slip past that filter: it is the user's only when the evidence is the user's own words.
    """
    known = _SUBJECTS.get(subject.strip().lower())
    if known:
        return known
    return "user" if speakers == {"user"} else "character"


# Sentiment cues searched right after the preference target, earliest match wins.
# Negated liking ("안 좋아", "좋아하지 않") must be tried before plain "좋아".
_SENTIMENT = re.compile(r"(?P<neg>안좋아|좋아하지않|좋아하진않|좋진않)|(?P<dis>싫어|싫다|싫은|싫음|별로)|(?P<like>좋아|좋다|좋음)")
_SENTIMENT_WINDOW = 10  # normalized characters after the target ("진짜 싫어해요" fits)


def _preference_supported(m: ExtractedMemory) -> bool:
    if m.type != MemoryType.USER_FACT or m.predicate not in {"likes", "dislikes"}:
        return True
    # Reject a likes/dislikes whose own quote says the opposite about this target. Other languages and
    # paraphrases are left to the normal source checks and the evaluation set.
    target, ev = normalize(m.value), normalize(m.evidence)
    if not target:
        return True
    for found in re.finditer(re.escape(target), ev):
        cue = _SENTIMENT.search(ev, found.end(), found.end() + _SENTIMENT_WINDOW)
        if cue is None:
            continue
        said = "likes" if cue.lastgroup == "like" else "dislikes"
        return said == m.predicate
    return True


def validate(output: ExtractionOutput, turn: TurnText, open_event_ids: set[str]) -> ValidationReport:
    memories: list[ExtractedMemory] = []
    relations: list[ExtractedRelationEvent] = []
    resolves: list[ExtractedResolve] = []
    rejected: list[tuple[str, str]] = []

    for m in output.memories:
        speakers = _evidence_sources(turn, m.source_message_ids, m.evidence)
        if speakers is None:
            rejected.append(("memory", "evidence_not_found"))
            continue
        m = m.model_copy(update={"subject": _normalize_subject(m.subject, speakers)})
        # Facts about the user come only from the user's own words.
        if m.subject == "user" and "user" not in speakers:
            rejected.append(("memory", "user_fact_from_character"))
            continue
        if m.type in (MemoryType.USER_FACT, MemoryType.SCHEDULE) and "user" not in speakers:
            rejected.append(("memory", "fact_without_user_evidence"))
            continue
        if not _preference_supported(m):
            rejected.append(("memory", "unsupported_preference"))
            continue
        update: dict[str, Any] = {"importance": min(1.0, max(0.0, m.importance))}
        # A character's suggestion alone never becomes a confirmed plan.
        if m.type == MemoryType.SHARED_EVENT and speakers == {"assistant"}:
            update["opens_unfinished_event"] = False
        memories.append(m.model_copy(update=update))

    for r in output.relation_events:
        if _evidence_sources(turn, r.source_message_ids, r.evidence) is None:
            rejected.append(("relation", "evidence_not_found"))
            continue
        relations.append(r)

    for res in output.resolves:
        if res.unfinished_event_id not in open_event_ids or res.outcome not in ("RESOLVED", "CANCELLED"):
            rejected.append(("resolve", "unknown_event"))
            continue
        if turn.user_msg_id not in res.source_message_ids:
            rejected.append(("resolve", "not_from_user"))
            continue
        resolves.append(res)

    return ValidationReport(memories, relations, resolves, rejected)

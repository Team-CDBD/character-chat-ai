"""Memory Reducer: duplicates, corrections (supersedes) and caps. Deterministic."""

from __future__ import annotations

from dataclasses import dataclass

from app.memory.validator import ExtractedMemory, MemoryType, normalize

KEYED_TYPES = {MemoryType.USER_FACT, MemoryType.SCHEDULE}
MAX_NEW_PER_TURN = 5


@dataclass(frozen=True)
class ExistingMemory:
    id: str
    type: str
    subject: str
    predicate: str
    value: str


@dataclass(frozen=True)
class MemoryDecision:
    action: str  # INSERT | SKIP_DUPLICATE
    memory: ExtractedMemory
    supersedes_id: str | None = None


def reduce(new: list[ExtractedMemory], existing: list[ExistingMemory]) -> list[MemoryDecision]:
    """``existing`` must contain only currently valid memories (valid_to IS NULL)."""
    decisions: list[MemoryDecision] = []
    by_key: dict[tuple[str, str, str], ExistingMemory] = {
        (e.type, e.subject, e.predicate): e for e in existing if e.type in KEYED_TYPES
    }
    seen_values = {(e.type, normalize(e.value)) for e in existing}

    ranked = sorted(new, key=lambda m: m.importance, reverse=True)[:MAX_NEW_PER_TURN]
    for m in ranked:
        value_key = (m.type.value, normalize(m.value))
        if m.type in KEYED_TYPES:
            key = (m.type.value, m.subject, m.predicate)
            prev = by_key.get(key)
            if prev is not None and normalize(prev.value) == normalize(m.value):
                decisions.append(MemoryDecision("SKIP_DUPLICATE", m))
                continue
            decisions.append(MemoryDecision("INSERT", m, supersedes_id=prev.id if prev else None))
            by_key[key] = ExistingMemory("pending", m.type.value, m.subject, m.predicate, m.value)
        else:
            if value_key in seen_values:
                decisions.append(MemoryDecision("SKIP_DUPLICATE", m))
                continue
            decisions.append(MemoryDecision("INSERT", m))
        seen_values.add(value_key)
    return decisions

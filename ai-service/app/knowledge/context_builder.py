"""Context Builder (plan §06).

Layers are allocated from highest to lowest priority and trimmed from the lowest. Layers
1-3 (policy, character card, persona) are never trimmed. Stable layers come first so the
provider's prompt caching can reuse them across turns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.providers.base import PromptMessage
from app.tuning import DEFAULT_TUNING, ContextTuning

BUDGETS = dict(DEFAULT_TUNING.context.budgets)  # defaults; effective values come from Settings.tuning
INPUT_BUDGET = DEFAULT_TUNING.context.input_budget  # ~7,260; output reserve (600) is separate


class BudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class RecentTurn:
    turn_seq: int
    user_text: str | None
    assistant_text: str
    is_opening_scene: bool = False


@dataclass(frozen=True)
class MemoryLine:
    id: str
    text: str
    score: float


@dataclass
class ContextInputs:
    policy_text: str
    character: dict[str, Any]
    character_rev: int
    persona: dict[str, Any] | None
    relationship_guideline: str | None
    open_events: list[str]
    memories: list[MemoryLine]
    summary: str | None
    time_line: str
    recent_turns: list[RecentTurn]  # oldest first
    current_user_text: str
    recent_window: int
    min_recent_window: int = 3


@dataclass
class BuiltContext:
    messages: list[PromptMessage]
    memory_ids: list[str]
    layer_tokens: dict[str, int]
    included_turns: list[int]
    degraded: list[str] = field(default_factory=list)

    def content(self) -> dict[str, Any]:
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "memoryIds": self.memory_ids,
            "layerTokens": self.layer_tokens,
            "includedTurns": self.included_turns,
        }

    def input_hash(self) -> str:
        blob = json.dumps(self.content()["messages"], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


CHARACTER_FIELDS = [
    ("name", "이름"),
    ("intro", "소개"),
    ("personality", "성격"),
    ("speechStyle", "말투"),
    ("background", "배경"),
    ("relationshipToUser", "사용자와의 관계"),
    ("extra", "추가 설정"),
]
PERSONA_FIELDS = [("name", "이름"), ("age", "나이"), ("job", "직업"), ("description", "설명")]


def render_character(card: dict[str, Any], revision: int) -> str:
    lines = [f"{label}: {card[key]}" for key, label in CHARACTER_FIELDS if card.get(key)]
    return f"[캐릭터 카드 revision={revision}]\n<character>\n" + "\n".join(lines) + "\n</character>"


def render_persona(persona: dict[str, Any] | None) -> str:
    if not persona:
        return ""
    parts = [f"{label}: {persona[key]}" for key, label in PERSONA_FIELDS if persona.get(key)]
    return "<user_persona>" + " / ".join(parts) + "</user_persona>"


def _turn_messages(t: RecentTurn) -> list[PromptMessage]:
    out: list[PromptMessage] = []
    if t.user_text:
        out.append(PromptMessage("user", t.user_text))
    out.append(PromptMessage("assistant", t.assistant_text))
    return out


def build(
    inputs: ContextInputs, count: Callable[[str], int], tuning: ContextTuning = DEFAULT_TUNING.context
) -> BuiltContext:
    BUDGETS, INPUT_BUDGET = tuning.budgets, tuning.input_budget  # noqa: N806 - same names as the defaults
    degraded: list[str] = []
    tokens: dict[str, int] = {}

    # 1-3: never trimmed. Input length limits on the BE side keep these inside budget.
    policy = inputs.policy_text
    character = render_character(inputs.character, inputs.character_rev)
    persona = render_persona(inputs.persona)
    tokens["policy"], tokens["character"], tokens["persona"] = count(policy), count(character), count(persona)
    for layer in ("policy", "character", "persona"):
        if tokens[layer] > BUDGETS[layer]:
            degraded.append(f"OVER_BUDGET_{layer.upper()}")  # measured, never silently cut

    # 4: relationship guideline + gated open events
    def render_relationship(events: list[str]) -> str:
        parts: list[str] = []
        if inputs.relationship_guideline:
            parts.append(f"<relationship>{inputs.relationship_guideline}</relationship>")
        if events:
            parts.append("<open_events>\n" + "\n".join(events) + "\n</open_events>")
        return "\n".join(parts)

    open_events = list(inputs.open_events)
    relationship = render_relationship(open_events)
    while count(relationship) > BUDGETS["relationship"] and open_events:
        open_events.pop()
        relationship = render_relationship(open_events)
    tokens["relationship"] = count(relationship)

    # 7: time
    time_line = f"<time>{inputs.time_line}</time>"
    tokens["time"] = count(time_line)

    # 8: recent window (whole turns only) + current message
    window = inputs.recent_turns[-inputs.recent_window :] if inputs.recent_window else []
    current_tokens = count(inputs.current_user_text)

    def recent_cost(turns: list[RecentTurn]) -> int:
        return sum(count(t.user_text or "") + count(t.assistant_text) for t in turns) + current_tokens

    while recent_cost(window) > BUDGETS["recent"] and len(window) > inputs.min_recent_window:
        window = window[1:]
        if "RECENT_WINDOW_SHRUNK" not in degraded:
            degraded.append("RECENT_WINDOW_SHRUNK")

    # 5: memories by score
    chosen: list[MemoryLine] = []
    mem_tokens = 0
    for m in sorted(inputs.memories, key=lambda x: x.score, reverse=True):
        c = count(m.text) + 1
        if mem_tokens + c > BUDGETS["memories"]:
            continue
        chosen.append(m)
        mem_tokens += c

    # 6: summary (trim from the front, keep the latest part)
    summary = inputs.summary or ""
    while summary and count(summary) > BUDGETS["summary"]:
        summary = summary[len(summary) // 5 :]

    def total() -> int:
        return (
            sum(tokens[k] for k in ("policy", "character", "persona", "relationship", "time"))
            + sum(count(m.text) + 1 for m in chosen)
            + count(summary)
            + recent_cost(window)
        )

    # Global trim order: low-relevance memories -> summary front -> recent window (min turns)
    while total() > INPUT_BUDGET and chosen:
        chosen.pop()  # lowest score is last
    while total() > INPUT_BUDGET and summary:
        summary = summary[len(summary) // 3 :] if len(summary) > 30 else ""
    while total() > INPUT_BUDGET and len(window) > inputs.min_recent_window:
        window = window[1:]
    # Last resort for very long messages: go below the minimum window rather than fail the turn.
    while total() > INPUT_BUDGET and len(window) > 1:
        window = window[1:]
        if "RECENT_BELOW_MINIMUM" not in degraded:
            degraded.append("RECENT_BELOW_MINIMUM")
    if total() > INPUT_BUDGET:
        raise BudgetExceeded(f"context needs {total()} tokens > {INPUT_BUDGET}")

    tokens["memories"] = sum(count(m.text) + 1 for m in chosen)
    tokens["summary"] = count(summary)
    tokens["recent"] = recent_cost(window)

    stable = "\n\n".join(p for p in (policy, character, persona) if p)
    variable_parts = [relationship]
    if chosen:
        variable_parts.append("<memories>\n" + "\n".join(m.text for m in chosen) + "\n</memories>")
    if summary:
        variable_parts.append(f"<summary>{summary}</summary>")
    variable_parts.append(time_line)
    variable = "\n".join(p for p in variable_parts if p)

    messages = [PromptMessage("system", stable), PromptMessage("system", variable)]
    for t in window:
        messages.extend(_turn_messages(t))
    messages.append(PromptMessage("user", inputs.current_user_text))

    return BuiltContext(
        messages=messages,
        memory_ids=[m.id for m in chosen],
        layer_tokens=tokens,
        included_turns=[t.turn_seq for t in window],
        degraded=degraded,
    )


def opening_scene_turn(scene: dict[str, Any] | None) -> RecentTurn | None:
    if not scene:
        return None
    narration = scene.get("narration") or ""
    line = scene.get("characterLine") or ""
    text = (f"*{narration}*\n" if narration else "") + line
    return RecentTurn(turn_seq=0, user_text=None, assistant_text=text.strip(), is_opening_scene=True)

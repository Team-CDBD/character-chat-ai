"""Output Policy (plan §05).

Text is buffered to a sentence boundary and checked *before* it is sent. In the MVP the
role-break / speech-style checks only produce metrics flags; ``blocked`` is reserved for
hard safety rules (none configured yet).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENTENCE_END = re.compile(r"([.!?。…~]+[\"'”’)]?\s+|\n)")
_ROLE_BREAK = re.compile(r"(저는\s*AI|나는\s*AI|AI\s*(로서|라서|이기\s*때문)|언어\s*모델|인공지능\s*(이라|으로서))")
_HONORIFIC_ENDING = re.compile(r"(습니다|습니까|세요|에요|예요|해요|이요)[.!?]*$")
MAX_BUFFER_CHARS = 80


@dataclass
class PolicyVerdict:
    blocked: bool = False
    flags: set[str] = field(default_factory=set)


class ChunkPolicy:
    def __init__(self, *, expects_banmal: bool) -> None:
        self.expects_banmal = expects_banmal
        self.buffer = ""
        self.flags: set[str] = set()

    def _check(self, text: str) -> PolicyVerdict:
        verdict = PolicyVerdict()
        if _ROLE_BREAK.search(text):
            verdict.flags.add("ROLE_BREAK")
        if self.expects_banmal:
            for sentence in re.split(r"(?<=[.!?\n])\s*", text):
                if _HONORIFIC_ENDING.search(sentence.strip()):
                    verdict.flags.add("SPEECH_STYLE_HONORIFIC")
                    break
        self.flags |= verdict.flags
        return verdict

    def feed(self, delta: str) -> list[str]:
        """Return the chunks that passed the check and may be emitted now."""
        self.buffer += delta
        ready: list[str] = []
        while True:
            m = _SENTENCE_END.search(self.buffer)
            if m is None:
                if len(self.buffer) >= MAX_BUFFER_CHARS:
                    ready.append(self.buffer)
                    self.buffer = ""
                break
            ready.append(self.buffer[: m.end()])
            self.buffer = self.buffer[m.end() :]
        passed: list[str] = []
        for chunk in ready:
            if self._check(chunk).blocked:
                raise PolicyBlocked("chunk blocked")
            passed.append(chunk)
        return passed

    def flush(self) -> list[str]:
        if not self.buffer:
            return []
        chunk, self.buffer = self.buffer, ""
        if self._check(chunk).blocked:
            raise PolicyBlocked("chunk blocked")
        return [chunk]

    def check_final(self, text: str) -> PolicyVerdict:
        verdict = self._check(text)
        if not text.strip():
            verdict.blocked = True
            verdict.flags.add("EMPTY")
        return verdict


class PolicyBlocked(Exception):
    pass


def expects_banmal(speech_style: str) -> bool:
    return "반말" in speech_style and "존댓말" not in speech_style.replace("반말", "")

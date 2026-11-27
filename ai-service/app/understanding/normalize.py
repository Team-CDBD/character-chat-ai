"""Rule-based input normalization and query hints (plan §05 Understanding).

No model calls here by default. A query hint only enriches retrieval; it is never promoted
to an instruction.
"""

from __future__ import annotations

import re
import unicodedata

_DEICTIC = re.compile(r"(그때|거기|그거|그 얘기|그 사람|걔|그날|저번에|아까)")
_REPEAT = re.compile(r"(.)\1{3,}")
_ABBREVIATIONS = {"담주": "다음 주", "오늘밤": "오늘 밤", "낼모레": "내일모레"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).strip()
    text = _REPEAT.sub(lambda m: m.group(1) * 3, text)  # "ㅋㅋㅋㅋㅋㅋ" -> "ㅋㅋㅋ"
    return re.sub(r"\s+", " ", text)


def expand_abbreviations(text: str) -> str:
    for short, full in _ABBREVIATIONS.items():
        text = text.replace(short, full)
    return text


def derive_query_hint(current: str, recent_user_texts: list[str]) -> str:
    """Retrieval query: current message, plus the previous user message when it uses deictic words."""
    query = expand_abbreviations(normalize(current))
    if _DEICTIC.search(query) and recent_user_texts:
        query = f"{query} {expand_abbreviations(normalize(recent_user_texts[-1]))}"
    return query


def keywords(text: str) -> set[str]:
    """Crude Korean keyword set: tokens of length >= 2 with common particles stripped."""
    tokens = re.findall(r"[0-9A-Za-z가-힣]+", text)
    out: set[str] = set()
    for t in tokens:
        t = re.sub(r"(은|는|이|가|을|를|에|에서|으로|로|와|과|도|만|이야|야|이랑|랑|한테|께)$", "", t)
        if len(t) >= 2:
            out.add(t.lower())
    return out

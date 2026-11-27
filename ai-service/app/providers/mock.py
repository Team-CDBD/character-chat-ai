"""Deterministic Mock Provider (plan P0).

Lets BE integrate and lets tests run without a real model. Behaviour can be scripted per
schema so tests can drive the memory / relationship pipelines precisely.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import AsyncIterator, Callable
from typing import Any

from app.providers.base import ChatParams, PromptMessage, ProviderError, Usage, estimate_tokens

EMBED_DIM = 64

JsonScript = Callable[[list[PromptMessage]], dict[str, Any]]

EMPTY_RESULTS: dict[str, dict[str, Any]] = {
    "extract-v1": {"memories": [], "relationEvents": [], "resolves": []},
    "summary-v1": {"summary": ""},
    "baseline-v1": {"trust": 50, "closeness": 40, "tension": 0},
    "revision-check-v1": {"conflicts": []},
}


class MockProvider:
    name = "mock"

    def __init__(self, *, chunk_delay: float = 0.0, chunk_size: int = 4) -> None:
        self.chunk_delay = chunk_delay
        self.chunk_size = chunk_size
        self.json_scripts: dict[str, JsonScript] = {}
        self.reply_script: Callable[[list[PromptMessage]], str] | None = None
        self.fail_next: Exception | None = None
        self.chat_calls: list[list[PromptMessage]] = []

    # -- chat -------------------------------------------------------------
    def _default_reply(self, messages: list[PromptMessage]) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"응. \"{last_user[:40]}\" 말이지. 알겠어."

    async def stream_chat(self, messages: list[PromptMessage], params: ChatParams) -> AsyncIterator[str]:
        self.chat_calls.append(list(messages))
        if self.fail_next is not None:
            exc, self.fail_next = self.fail_next, None
            raise exc
        text = (self.reply_script or self._default_reply)(messages)
        for i in range(0, len(text), self.chunk_size):
            if self.chunk_delay:
                await asyncio.sleep(self.chunk_delay)
            yield text[i : i + self.chunk_size]

    # -- structured -------------------------------------------------------
    async def complete_json(
        self, messages: list[PromptMessage], params: ChatParams, schema_name: str
    ) -> tuple[dict[str, Any], Usage]:
        script = self.json_scripts.get(schema_name)
        if script is not None:
            result = script(messages)
        elif schema_name == "summary-v1":
            body = messages[-1].content if messages else ""
            result = {"summary": body[-600:]}
        elif schema_name in EMPTY_RESULTS:
            result = EMPTY_RESULTS[schema_name]
        else:
            raise ProviderError(f"mock has no script for {schema_name}")
        usage = Usage(
            model=params.model,
            input_tokens=sum(estimate_tokens(m.content) for m in messages),
            output_tokens=estimate_tokens(str(result)),
        )
        return result, usage

    # -- embedding --------------------------------------------------------
    async def embed(self, texts: list[str], model: str) -> tuple[list[list[float]], Usage]:
        return [hash_embed(t) for t in texts], Usage(model=model, input_tokens=sum(map(estimate_tokens, texts)))

    def count_tokens(self, text: str) -> int:
        return estimate_tokens(text)


def hash_embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Character-bigram hashing embedding: deterministic and good enough for overlap-style similarity."""
    vec = [0.0] * dim
    compact = "".join(text.split())
    grams = [compact[i : i + 2] for i in range(max(1, len(compact) - 1))]
    for g in grams:
        h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=4).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

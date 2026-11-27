"""Provider Adapter (plan §05, §12).

All model calls go through :class:`ProviderAdapter`. It normalizes errors, timeouts and
usage so provider differences never leak into the rest of the service, and it owns the
concurrency slot. A concrete provider only implements :class:`ModelProvider`.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class PromptMessage:
    role: Role
    content: str


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "inputTokens": self.input_tokens, "outputTokens": self.output_tokens}


@dataclass
class ChatParams:
    model: str
    temperature: float = 0.8
    max_output_tokens: int = 600
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderError(Exception):
    """Non-timeout provider failure (rate limit, 5xx, malformed output)."""


class ProviderTimeout(ProviderError):
    pass


def estimate_tokens(text: str) -> int:
    """Conservative estimate until the chosen model's official tokenizer is wired in (plan §12)."""
    if not text:
        return 0
    return math.ceil(len(text) / 1.5)


class ModelProvider(Protocol):
    name: str

    def stream_chat(self, messages: list[PromptMessage], params: ChatParams) -> AsyncIterator[str]:
        """Yield text deltas. Must raise ProviderError / ProviderTimeout on failure."""
        ...

    async def complete_json(
        self, messages: list[PromptMessage], params: ChatParams, schema_name: str
    ) -> tuple[dict[str, Any], Usage]: ...

    async def embed(self, texts: list[str], model: str) -> tuple[list[list[float]], Usage]: ...

    def count_tokens(self, text: str) -> int: ...


@dataclass
class StreamOutcome:
    text: str
    usage: Usage


class ProviderAdapter:
    def __init__(self, provider: ModelProvider, *, timeout_seconds: float, max_concurrency: int) -> None:
        self.provider = provider
        self.timeout = timeout_seconds
        self._slots = asyncio.Semaphore(max_concurrency)

    def count_tokens(self, text: str) -> int:
        return self.provider.count_tokens(text)

    async def stream_chat(
        self, messages: list[PromptMessage], params: ChatParams
    ) -> AsyncIterator[str | StreamOutcome]:
        """Yield deltas, then a final :class:`StreamOutcome` with normalized usage."""
        async with self._slots:
            parts: list[str] = []
            iterator = self.provider.stream_chat(messages, params).__aiter__()
            while True:
                try:
                    delta = await asyncio.wait_for(iterator.__anext__(), timeout=self.timeout)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise ProviderTimeout("provider stream timed out") from exc
                except ProviderError:
                    raise
                except Exception as exc:  # normalize unknown SDK errors
                    raise ProviderError(str(exc)) from exc
                parts.append(delta)
                yield delta
            text = "".join(parts)
            usage = Usage(
                model=params.model,
                input_tokens=sum(self.count_tokens(m.content) for m in messages),
                output_tokens=self.count_tokens(text),
            )
            yield StreamOutcome(text=text, usage=usage)

    async def complete_json(
        self, messages: list[PromptMessage], params: ChatParams, schema_name: str
    ) -> tuple[dict[str, Any], Usage]:
        async with self._slots:
            try:
                return await asyncio.wait_for(
                    self.provider.complete_json(messages, params, schema_name), timeout=self.timeout
                )
            except TimeoutError as exc:
                raise ProviderTimeout(f"{schema_name} timed out") from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(str(exc)) from exc

    async def embed(self, texts: list[str], model: str) -> tuple[list[list[float]], Usage]:
        async with self._slots:
            try:
                return await asyncio.wait_for(self.provider.embed(texts, model), timeout=self.timeout)
            except TimeoutError as exc:
                raise ProviderTimeout("embedding timed out") from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(str(exc)) from exc

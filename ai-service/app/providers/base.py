"""Provider Adapter (plan §05, §12).

All model calls go through :class:`ProviderAdapter`. It normalizes errors, timeouts and
usage so provider differences never leak into the rest of the service, and it owns the
concurrency slot. A concrete provider only implements :class:`ModelProvider`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.providers.tokens import TokenCounter, estimate_tokens

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


# ``estimate_tokens`` (re-exported) is the fixed conservative estimate; the adapter prefers an exact or
# calibrated counter (providers/tokens.py).
__all__ = ["estimate_tokens"]


class ModelProvider(Protocol):
    name: str

    def stream_chat(self, messages: list[PromptMessage], params: ChatParams) -> AsyncIterator[str | Usage]:
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
    def __init__(
        self,
        provider: ModelProvider,
        *,
        timeout_seconds: float,
        max_concurrency: int,
        counter: TokenCounter | None = None,
        count_model: str | None = None,
    ) -> None:
        self.provider = provider
        self.timeout = timeout_seconds
        self._slots = asyncio.Semaphore(max_concurrency)
        self.counter = counter or TokenCounter(calibrate=False, fallback=provider.count_tokens)
        self.count_model = count_model  # calibration only learns from calls to this (the chat) model

    def count_tokens(self, text: str) -> int:
        return self.counter.count(text)

    @property
    def token_count_source(self) -> str:
        return self.counter.source

    def _observe(self, messages: list[PromptMessage], params: ChatParams, usage: Usage | None) -> None:
        if usage is not None and usage.input_tokens and (self.count_model is None or params.model == self.count_model):
            self.counter.observe([m.content for m in messages], usage.input_tokens)

    async def stream_chat(
        self, messages: list[PromptMessage], params: ChatParams
    ) -> AsyncIterator[str | StreamOutcome]:
        """Yield deltas, then a final :class:`StreamOutcome` with normalized usage."""
        async with self._slots:
            parts: list[str] = []
            reported_usage = None
            iterator = self.provider.stream_chat(messages, params).__aiter__()
            try:
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
                    if isinstance(delta, Usage):
                        reported_usage = delta
                        continue
                    parts.append(delta)
                    yield delta
            finally:
                # Close the provider stream (its HTTP connection) on every exit: completion, error, the attempt
                # deadline cancelling us, or the caller abandoning the stream. Ollama stops generating when the
                # client goes away, so the slot and the model are free for the next request.
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
            text = "".join(parts)
            self._observe(messages, params, reported_usage)
            usage = reported_usage or Usage(
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
                result, usage = await asyncio.wait_for(
                    self.provider.complete_json(messages, params, schema_name), timeout=self.timeout
                )
            except TimeoutError as exc:
                raise ProviderTimeout(f"{schema_name} timed out") from exc
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError(str(exc)) from exc
        self._observe(messages, params, usage)
        return result, usage

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

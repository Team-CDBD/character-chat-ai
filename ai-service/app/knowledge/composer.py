"""Response Composer (plan §05): one streaming model call, chunks gated by Output Policy.

Once streaming has started, a failure is final for this attempt: we never splice another
model's answer onto a partial one (plan §09).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import aclosing
from dataclasses import dataclass, field

from app.knowledge.output_policy import ChunkPolicy, PolicyBlocked
from app.persistence.jobs import LeaseLost
from app.providers.base import ChatParams, PromptMessage, ProviderAdapter, StreamOutcome, Usage


@dataclass
class Composed:
    text: str
    usage: Usage
    flags: set[str] = field(default_factory=set)


class StreamStartedError(Exception):
    """Provider failed after at least one chunk was sent to the user."""

    def __init__(self, cause: Exception) -> None:
        super().__init__(str(cause))
        self.cause = cause


async def generate(
    provider: ProviderAdapter,
    messages: list[PromptMessage],
    params: ChatParams,
    policy: ChunkPolicy,
    emit: Callable[[str], Awaitable[None]],
) -> Composed:
    emitted_any = False
    outcome: StreamOutcome | None = None
    try:
        # aclosing: if emit() fails (lease lost, policy block) the provider stream is closed right away,
        # which releases the model slot instead of leaving it to garbage collection.
        async with aclosing(provider.stream_chat(messages, params)) as stream:
            async for item in stream:
                if isinstance(item, StreamOutcome):
                    outcome = item
                    break
                for chunk in policy.feed(item):
                    await emit(chunk)
                    emitted_any = True
        for chunk in policy.flush():
            await emit(chunk)
            emitted_any = True
    except (PolicyBlocked, LeaseLost):
        raise
    except Exception as exc:
        if emitted_any:
            raise StreamStartedError(exc) from exc
        raise
    assert outcome is not None
    verdict = policy.check_final(outcome.text)
    if verdict.blocked:
        raise PolicyBlocked("final check failed")
    return Composed(text=outcome.text, usage=outcome.usage, flags=set(policy.flags))

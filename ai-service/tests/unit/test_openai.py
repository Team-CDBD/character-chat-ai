import json

import httpx
import pytest

from app.providers.base import ChatParams, PromptMessage, ProviderAdapter, ProviderError, StreamOutcome
from app.providers.openai import OpenAIProvider

MESSAGES = [PromptMessage("user", "안녕")]
PARAMS = ChatParams("chat-test")


def adapter(handler):
    provider = OpenAIProvider("test-key", 5, transport=httpx.MockTransport(handler))
    return ProviderAdapter(provider, timeout_seconds=5, max_concurrency=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [True, False])
async def test_chat_stream(complete):
    def respond(request):
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["messages"] == [{"role": "user", "content": "안녕"}]
        assert payload["stream_options"]["include_usage"]
        chunks = [
            {"choices": [{"delta": {"content": "반가워"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "model": "chat-test", "usage": {"prompt_tokens": 21, "completion_tokens": 7}},
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return httpx.Response(200, text=body + ("data: [DONE]\n\n" if complete else ""))

    stream = adapter(respond).stream_chat(MESSAGES, PARAMS)
    assert await anext(stream) == "반가워"
    if complete:
        final = await anext(stream)
        assert isinstance(final, StreamOutcome)
        assert (final.text, final.usage.input_tokens, final.usage.output_tokens) == ("반가워", 21, 7)
    else:
        with pytest.raises(ProviderError):
            await anext(stream)
    await stream.aclose()


@pytest.mark.asyncio
async def test_structured_completion_and_embedding_order():
    def respond(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            assert payload["input"] == ["a", "b"]
            return httpx.Response(
                200,
                json={
                    "model": "emb",
                    "usage": {"prompt_tokens": 2},
                    "data": [{"index": 1, "embedding": [0.1]}, {"index": 0, "embedding": [0.2]}],
                },
            )
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "model": "chat-test",
                "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                "choices": [{"finish_reason": "stop", "message": {"content": '{"summary": "요약"}'}}],
            },
        )

    provider = adapter(respond)
    result, usage = await provider.complete_json(MESSAGES, PARAMS, "summary-v1")
    assert result == {"summary": "요약"} and usage.output_tokens == 5
    vectors, usage = await provider.embed(["a", "b"], "emb")
    assert vectors == [[0.2], [0.1]] and usage.input_tokens == 2

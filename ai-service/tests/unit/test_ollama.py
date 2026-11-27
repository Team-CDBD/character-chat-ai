import json

import httpx
import pytest

from app.config import Settings
from app.providers.base import ChatParams, PromptMessage, ProviderAdapter, ProviderError, Usage
from app.providers.ollama import OllamaProvider
from app.providers.registry import build_provider


@pytest.mark.parametrize(
    ("name", "chat", "embed"),
    [
        ("ollama", "gemma4:12b", "embeddinggemma"),
        ("openai", "gpt-4o-mini", "text-embedding-3-small"),
        ("mock", "mock-chat", "mock-embed"),
    ],
)
def test_switch(monkeypatch, name, chat, embed):
    monkeypatch.setenv("AI_PROVIDER", name)
    cfg = Settings(openai_api_key="test", _env_file=None)
    assert cfg.chat_model == chat and cfg.embedding_model == embed
    assert build_provider(cfg).name == name


def adapter(handler):
    provider = OllamaProvider("http://localhost:11434", 5, transport=httpx.MockTransport(handler))
    return ProviderAdapter(provider, timeout_seconds=5, max_concurrency=1)


messages = [PromptMessage("user", "안녕")]
params = ChatParams("gemma4:12b")


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["stop", "length", None])
async def test_native_stream(ending):
    def reply(request):
        payload = json.loads(request.content)
        assert request.url.path == "/api/chat" and payload["stream"] and payload["think"] is False
        assert payload["model"] == "gemma4:12b" and payload["options"]["num_ctx"] == 8192
        rows = [{"done": False, "message": {"content": "반가워", "thinking": "hidden"}}]
        if ending:
            rows.append(
                {
                    "done": True,
                    "done_reason": ending,
                    "message": {"content": ""},
                    "prompt_eval_count": 12,
                    "eval_count": 3,
                }
            )
        return httpx.Response(200, text="\n".join(json.dumps(r) for r in rows))

    stream = adapter(reply).stream_chat(messages, params)
    assert await anext(stream) == "반가워"
    if ending == "stop":
        final = await anext(stream)
        assert final.usage == Usage(params.model, 12, 3)
    else:
        with pytest.raises(ProviderError):
            await anext(stream)
    await stream.aclose()


@pytest.mark.asyncio
async def test_json_and_embeddings():
    def reply(request):
        payload = json.loads(request.content)
        if request.url.path == "/api/embed":
            assert payload["input"] == ["하나", "둘"]
            return httpx.Response(200, json={"embeddings": [[0.1], [0.2]], "prompt_eval_count": 4})
        assert payload["format"] == "json" and payload["stream"] is False
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {"content": '{"summary":"요약"}'},
                "prompt_eval_count": 3,
                "eval_count": 4,
            },
        )

    provider = adapter(reply)
    obj, usage = await provider.complete_json(messages, params, "summary-v1")
    assert obj == {"summary": "요약"} and usage == Usage(params.model, 3, 4)
    vectors, usage = await provider.embed(["하나", "둘"], "embeddinggemma")
    assert vectors == [[0.1], [0.2]] and usage.input_tokens == 4

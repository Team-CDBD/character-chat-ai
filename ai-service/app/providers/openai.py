"""OpenAI Chat Completions and embeddings for the AI worker."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Any

import httpx

from app.providers.base import (
    ChatParams,
    PromptMessage,
    ProviderError,
    ProviderTimeout,
    Usage,
    estimate_tokens,
)


class OpenAIProvider:
    name = "openai"
    count_tokens = staticmethod(estimate_tokens)

    def __init__(self, api_key: str, timeout: float, *, transport: httpx.AsyncBaseTransport | None = None):
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="https://api.openai.com/v1/",
            headers=self._headers,
            transport=self._transport,
            timeout=self._timeout,
        )

    @staticmethod
    def _chat_request(messages: list[PromptMessage], params: ChatParams) -> dict[str, Any]:
        return {
            "model": params.model,
            "messages": [asdict(m) for m in messages],
            "temperature": params.temperature,
            "max_completion_tokens": params.max_output_tokens,
            "store": False,
        }

    async def stream_chat(
        self, messages: list[PromptMessage], params: ChatParams
    ) -> AsyncIterator[str | Usage]:
        payload = self._chat_request(messages, params)
        payload.update(stream=True, stream_options={"include_usage": True})
        stopped = False
        has_text = False
        try:
            async with self._client() as client, client.stream("POST", "chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        if stopped and has_text:
                            return
                        break
                    chunk = json.loads(data)
                    if usage := chunk.get("usage"):
                        yield Usage(chunk["model"], usage["prompt_tokens"], usage["completion_tokens"])
                    for choice in chunk["choices"]:
                        if content := choice["delta"].get("content"):
                            has_text = True
                            yield content
                        reason = choice.get("finish_reason")
                        if reason and reason != "stop":
                            raise ProviderError("OpenAI response did not finish normally")
                        stopped |= reason == "stop"
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("OpenAI stream timed out") from exc
        raise ProviderError("incomplete OpenAI stream")

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("OpenAI request timed out") from exc

    async def complete_json(
        self, messages: list[PromptMessage], params: ChatParams, schema_name: str
    ) -> tuple[dict[str, Any], Usage]:
        payload = self._chat_request(messages, params)
        payload["response_format"] = {"type": "json_object"}
        data = await self._post("chat/completions", payload)
        choice = data["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
            raise ProviderError(f"incomplete {schema_name} response")
        result = json.loads(choice["message"]["content"])
        if not isinstance(result, dict):
            raise ProviderError(f"expected {schema_name} object")
        usage = data["usage"]
        return result, Usage(data["model"], usage["prompt_tokens"], usage["completion_tokens"])

    async def embed(self, texts: list[str], model: str) -> tuple[list[list[float]], Usage]:
        data = await self._post("embeddings", {"model": model, "input": texts, "encoding_format": "float"})
        rows = sorted(data["data"], key=lambda row: row["index"])
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ProviderError("incomplete embedding response")
        return [row["embedding"] for row in rows], Usage(data["model"], data["usage"]["prompt_tokens"])

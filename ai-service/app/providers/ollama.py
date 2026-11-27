"""Native Ollama chat, JSON generation and batch embeddings."""

from __future__ import annotations

import json
import math
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


class OllamaProvider:
    name = "ollama"
    count_tokens = staticmethod(estimate_tokens)

    def __init__(self, base_url: str, timeout: float, *, transport: httpx.AsyncBaseTransport | None = None):
        self.base_url = base_url
        self.timeout = timeout
        self.transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout, transport=self.transport)

    @staticmethod
    def _payload(messages: list[PromptMessage], params: ChatParams) -> dict[str, Any]:
        return {
            "model": params.model,
            "messages": [asdict(m) for m in messages],
            "think": False,
            "options": {
                "num_ctx": 8192,
                "temperature": params.temperature,
                "num_predict": params.max_output_tokens,
            },
        }

    @staticmethod
    def _check_terminal(data: dict[str, Any]) -> None:
        if data.get("error") or data.get("done") is not True or data.get("done_reason") != "stop":
            raise ProviderError("Ollama did not complete the response")

    async def stream_chat(
        self, messages: list[PromptMessage], params: ChatParams
    ) -> AsyncIterator[str | Usage]:
        payload = self._payload(messages, params)
        payload["stream"] = True
        has_text = False
        try:
            async with self._client() as client, client.stream("POST", "api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise ProviderError("Ollama generation failed")
                    content = data["message"]["content"]
                    if content:
                        has_text = True
                        yield content
                    if data["done"]:
                        self._check_terminal(data)
                        if not has_text:
                            raise ProviderError("empty Ollama response")
                        if "prompt_eval_count" in data and "eval_count" in data:
                            yield Usage(params.model, data["prompt_eval_count"], data["eval_count"])
                        return
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama stream timed out") from exc
        raise ProviderError("incomplete Ollama stream")

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with self._client() as client:
                response = await client.post(path, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("Ollama request timed out") from exc

    async def complete_json(
        self, messages: list[PromptMessage], params: ChatParams, schema_name: str
    ) -> tuple[dict[str, Any], Usage]:
        payload = self._payload(messages, params)
        data = await self._post("api/chat", {**payload, "stream": False, "format": "json"})
        self._check_terminal(data)
        result = json.loads(data["message"]["content"])
        if not isinstance(result, dict):
            raise ProviderError(f"expected {schema_name} object")
        return result, Usage(params.model, data["prompt_eval_count"], data["eval_count"])

    async def embed(self, texts: list[str], model: str) -> tuple[list[list[float]], Usage]:
        data = await self._post("api/embed", {"input": texts, "model": model, "truncate": False})
        vectors = data["embeddings"]
        if len(vectors) != len(texts) or any(
            not v or len(v) != len(vectors[0]) or not all(math.isfinite(n) for n in v) for v in vectors
        ):
            raise ProviderError("invalid Ollama embeddings")
        return vectors, Usage(model, data["prompt_eval_count"])

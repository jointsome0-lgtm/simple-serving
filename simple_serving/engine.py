"""A thin client of vLLM's OpenAI-compatible server, which listens on loopback.

It uses GET /health, GET /v1/models, POST /tokenize and POST /v1/chat/completions with streaming. Answers are read
tolerantly, because field names move between vLLM versions; the exact version is pinned on the card. An engine
failure becomes an `EngineError` with a code of section 9 and nothing else: the engine's own error text may quote
the prompt, so it is never read.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .errors import ServiceError
from .validation import ChatRequest

CONNECT_TIMEOUT_S = 5.0
HEALTH_TIMEOUT_S = 10.0
MAX_LINE_BYTES = 1_000_000  # far above any event of a text chunk
# No read timeout for counts and generations: a long prompt takes long to read, and the class's wall time bounds both.
UNBOUNDED = httpx.Timeout(None, connect=CONNECT_TIMEOUT_S)


class EngineError(ServiceError):
    """The engine refused a request (`invalid_request`) or failed (`engine_unavailable`)."""


def refusal(status: int) -> EngineError:
    """The error for an engine's HTTP status. With 400 or 422 the engine found the request itself wrong, such as a
    schema it cannot compile; any other status, 401, 403 and 404 included, means the engine or its setup failed."""
    return EngineError("invalid_request" if status in (400, 422) else "engine_unavailable")


def tokenize_payload(alias: str, request: ChatRequest) -> dict[str, Any]:
    # Counting renders the prompt as generation does (contract section 5). vLLM's defaults for the generation prompt
    # and special tokens are written out in both calls, so a version that changes one cannot make them differ.
    return {"model": alias, "messages": request.messages, "add_generation_prompt": True,
            "add_special_tokens": False, "chat_template_kwargs": {"enable_thinking": request.enable_thinking}}


def generation_payload(alias: str, request: ChatRequest, *, priority: int, cache_salt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": alias, "messages": request.messages, "max_tokens": request.max_tokens,
        "stream": True, "stream_options": {"include_usage": True}, **request.sampling,
        "add_generation_prompt": True, "add_special_tokens": False,
        "chat_template_kwargs": {"enable_thinking": request.enable_thinking},
        "priority": priority, "cache_salt": cache_salt,
    }
    if request.response_format is not None:
        payload["response_format"] = request.response_format
    return payload


class EngineStream:
    """The engine's server-sent events of one generation, one JSON object at a time.

    Closing the stream closes its connection: that is how vLLM behind HTTP is told to abort the request. When `aclose`
    returns, the gateway's engine request has ended locally. That is not a confirmation from the engine, which learns of
    the abort from the closed connection and may compute a little longer (contract section 9).
    """

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._lines = _lines(response)

    async def next_event(self) -> dict[str, Any] | None:
        """The next event, or None at [DONE]. A line that is not part of an event, data that is not UTF-8 or JSON,
        and a body that ends without [DONE] break the stream: EngineError."""
        data: list[bytes] = []
        try:
            async for line in self._lines:
                if line.startswith(b"data:"):
                    data.append(line[5:].removeprefix(b" "))
                elif line.startswith(b":"):
                    continue  # a comment, which server-sent events allow
                elif line:
                    raise EngineError("engine_unavailable")
                elif data:
                    return _event(b"\n".join(data))
        except httpx.HTTPError:
            raise EngineError("engine_unavailable") from None
        raise EngineError("engine_unavailable")

    async def aclose(self) -> None:
        try:
            await self._lines.aclose()
        finally:
            await self._response.aclose()


class EngineClient:
    def __init__(self, base_url: str) -> None:
        # trust_env=False: a proxy from the environment must never stand between the gateway and its engine.
        self._client = httpx.AsyncClient(
            base_url=base_url, trust_env=False,
            timeout=httpx.Timeout(HEALTH_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=32))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def context_length(self, alias: str) -> int:
        """Verify the engine: it answers, serves the alias and reports its context length."""
        await self._call("GET", "/health")
        models = _object(await self._json("GET", "/v1/models"))
        for card in models.get("data") or []:
            if isinstance(card, dict) and card.get("id") == alias:
                length = card.get("max_model_len")
                if type(length) is int and length >= 1:
                    return length
        raise EngineError("engine_unavailable")

    async def count(self, alias: str, request: ChatRequest) -> int:
        answer = _object(await self._json("POST", "/tokenize", tokenize_payload(alias, request), UNBOUNDED))
        count = answer.get("count")
        if type(count) is not int and isinstance(answer.get("tokens"), list):
            count = len(answer["tokens"])
        if type(count) is not int or count < 1:
            raise EngineError("engine_unavailable")
        return count

    async def generate(self, payload: dict[str, Any]) -> EngineStream:
        """Start a generation. It returns once the engine has answered with its status and headers."""
        request = self._client.build_request("POST", "/v1/chat/completions", json=payload, timeout=UNBOUNDED)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError:
            raise EngineError("engine_unavailable") from None
        if response.status_code != 200:
            await response.aclose()
            raise refusal(response.status_code)
        return EngineStream(response)

    async def _call(self, method: str, path: str, payload: Any = None,
                    timeout: httpx.Timeout | None = None) -> httpx.Response:
        try:
            response = await self._client.request(method, path, json=payload,
                                                  timeout=timeout or self._client.timeout)
        except httpx.HTTPError:
            raise EngineError("engine_unavailable") from None
        if response.status_code != 200:
            raise refusal(response.status_code)
        return response

    async def _json(self, method: str, path: str, payload: Any = None, timeout: httpx.Timeout | None = None) -> Any:
        response = await self._call(method, path, payload, timeout)
        try:
            return response.json()
        except ValueError:
            raise EngineError("engine_unavailable") from None


async def _lines(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """The lines of a body as bytes, without their ends. A body is decoded only per event, strictly."""
    pending = b""
    async for chunk in response.aiter_bytes():
        pending += chunk
        *lines, pending = pending.split(b"\n")
        for line in lines:
            yield line.removesuffix(b"\r")
        if len(pending) > MAX_LINE_BYTES:
            raise EngineError("engine_unavailable")
    if pending:
        yield pending


def _event(data: bytes) -> dict[str, Any] | None:
    if data == b"[DONE]":
        return None
    try:
        return _object(json.loads(data.decode("utf-8")))
    except ValueError:  # UnicodeDecodeError included
        raise EngineError("engine_unavailable") from None


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EngineError("engine_unavailable")
    return value

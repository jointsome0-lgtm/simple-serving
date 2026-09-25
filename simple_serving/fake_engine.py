"""A fake of vLLM's OpenAI-compatible server, for the tests and the dev launcher.

It answers GET /health, GET /v1/models, POST /tokenize and POST /v1/chat/completions the way vLLM does: the status and
headers of a generation go out at once, its first event after the prompt is read. A test scripts it per step from the
`engine` block of a case, with gates and delays for the scenarios. Without a script it gives default answers: the count
is ceil(characters of all message contents / 4), and a generation streams a fixed synthetic sentence and stops. As in
vLLM, a request that asks for thinking gets a fixed thought first, in the reasoning field, and `max_tokens`, counted
as characters / 4 too, cuts the answer and finishes it with `length`. A schema that holds a pattern which is not a
regular expression is refused with 400 before the stream, as vLLM refuses a schema it cannot compile. It records every
call with its body and headers, and the moment it noticed that its client left.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from starlette.types import Message, Receive, Scope, Send

SENTENCE = ("The keeper ", "lights the lamp ", "and watches ", "the grey sea.")
THOUGHT = ("The sea is calm, ", "so the lamp comes first.")  # the default reasoning, when a request asks for it
ROUTES = {("GET", "/health"): "health", ("GET", "/v1/models"): "models", ("POST", "/tokenize"): "tokenize",
          ("POST", "/v1/chat/completions"): "generate"}


@dataclass
class Script:
    """What the fake engine does for one request. Fields left unset take the default answers."""

    input_tokens: int | None = None
    # As in a case's `engine` block (contract/README.md): `role`, `content`, `reasoning`, `finish_reason` and `index`
    # fill one chunk; `{"error": ...}` is an error event, `{"raw_hex": ...}` raw data, `{"break": true}` a drop.
    events: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = field(default_factory=dict)  # None: no usage chunk; missing fields are computed
    generate_status: int | None = None
    # Controls for the scenarios.
    hold_tokenize: asyncio.Event | None = None  # hold the count's answer until the gate opens
    hold_headers: asyncio.Event | None = None  # hold a generation before its status line
    hold_first: asyncio.Event | None = None  # hold after the headers, before the first event: a long prompt
    first_delay_s: float = 0.0  # a pause after the headers, before the first event: a long prompt
    delay_s: float = 0.0  # a pause after each event
    endless: bool = False  # after the events, keep generating until the client leaves

    @classmethod
    def from_case(cls, engine: dict[str, Any]) -> Script:
        """The script of a case step's `engine` block."""
        return cls(input_tokens=engine.get("input_tokens"), events=engine.get("events"),
                   usage=engine.get("usage", {}), generate_status=engine.get("generate_status"))


@dataclass
class Call:
    kind: str  # "health", "models", "tokenize", "generate" or "unknown"
    body: Any
    headers: dict[str, str]
    started_at: float
    events_sent: int = 0
    last_event_at: float | None = None
    client_left_at: float | None = None  # when the fake noticed that its client closed the connection
    ended_at: float | None = None
    client_left: asyncio.Event = field(default_factory=asyncio.Event)
    ended: asyncio.Event = field(default_factory=asyncio.Event)


class FakeEngine:
    def __init__(self, served_name: str, max_model_len: int, *, reasoning_field: str = "reasoning") -> None:
        self.served_name = served_name
        self.max_model_len = max_model_len
        self.reasoning_field = reasoning_field  # vLLM has named it `reasoning_content`, later `reasoning`
        self.script: Script | Callable[[Any], Script] | None = None  # a callable picks a script by request body
        self.healthy = True
        self.calls: list[Call] = []

    def calls_of(self, *kinds: str) -> list[Call]:
        return [call for call in self.calls if call.kind in kinds]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        call = Call(kind=ROUTES.get((scope["method"], scope["path"]), "unknown"), body=await _read_json(receive),
                    headers={name.decode("latin-1"): value.decode("latin-1") for name, value in scope["headers"]},
                    started_at=time.monotonic())
        self.calls.append(call)
        watcher = asyncio.create_task(_watch_client(call, receive))
        try:
            await getattr(self, "_" + call.kind)(call, send)
        finally:
            watcher.cancel()
            call.ended_at = time.monotonic()
            call.ended.set()

    async def _health(self, call: Call, send: Send) -> None:
        await _respond(send, 200 if self.healthy else 503, b"")

    async def _models(self, call: Call, send: Send) -> None:
        if not self.healthy:
            await _respond(send, 503, b"")
            return
        await _respond_json(send, 200, {"object": "list", "data": [
            {"id": self.served_name, "object": "model", "created": 0, "owned_by": "vllm", "root": self.served_name,
             "parent": None, "max_model_len": self.max_model_len}]})

    async def _tokenize(self, call: Call, send: Send) -> None:
        script = self._script(call.body)
        if script.hold_tokenize is not None:
            await _hold(script.hold_tokenize, call)
        count = self._count(script, call.body)
        await _respond_json(send, 200, {"count": count, "max_model_len": self.max_model_len,
                                        "tokens": list(range(count)), "token_strs": None})

    async def _generate(self, call: Call, send: Send) -> None:
        script = self._script(call.body)
        if script.hold_headers is not None:
            await _hold(script.hold_headers, call)
        status = script.generate_status
        if status is None and _uncompilable(call.body):
            status = 400
        if status is not None:
            # Engines may quote the prompt in their errors, so this one does; the gateway must never pass it on.
            await _respond_json(send, status, {"error": {
                "message": "cannot serve: " + _prompt_text(call.body), "type": "BadRequestError", "code": status}})
            return
        if call.client_left.is_set():
            return
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
        if script.hold_first is not None:
            await _hold(script.hold_first, call)
        if script.first_delay_s:
            await _hold(asyncio.Event(), call, timeout=script.first_delay_s)
        events = script.events if script.events is not None else self._default_events(call.body)
        for event in events:
            if call.client_left.is_set():
                return
            if event.get("break"):
                return  # an unfinished response: uvicorn drops the connection without the end of the body
            if "raw_hex" in event:
                await self._send_data(call, send, bytes.fromhex(event["raw_hex"]))
            elif "error" in event:
                await self._send_event(call, send, {"error": event["error"]})
            else:
                await self._send_event(call, send, self._chunk(event))
            if script.delay_s:
                await _hold(asyncio.Event(), call, timeout=script.delay_s)
        while script.endless and not call.client_left.is_set():
            await self._send_event(call, send, self._chunk({"content": "and on "}))
            await _hold(asyncio.Event(), call, timeout=script.delay_s or 0.01)
        if call.client_left.is_set():
            return
        if script.usage is not None:
            await self._send_event(call, send, self._usage_chunk(script, call.body, events))
        await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})

    async def _unknown(self, call: Call, send: Send) -> None:
        await _respond_json(send, 404, {"error": {"message": "Not Found", "code": 404}})

    def _script(self, body: Any) -> Script:
        if callable(self.script):
            return self.script(body)
        return self.script or Script()

    def _count(self, script: Script, body: Any) -> int:
        if script.input_tokens is not None:
            return script.input_tokens
        return max(1, math.ceil(len(_prompt_text(body)) / 4))

    def _default_events(self, body: Any) -> list[dict[str, Any]]:
        """The thought when the request asks for thinking, then the sentence, cut where `max_tokens` runs out."""
        kwargs = body.get("chat_template_kwargs") if isinstance(body, dict) else None
        thinking = isinstance(kwargs, dict) and kwargs.get("enable_thinking") is True
        parts = [*(("reasoning", part) for part in THOUGHT if thinking), *(("content", part) for part in SENTENCE)]
        limit = body.get("max_tokens") if isinstance(body, dict) else None
        room = limit * 4 if type(limit) is int else sum(len(part) for _, part in parts)
        events: list[dict[str, Any]] = [{"role": "assistant"}]
        for name, part in parts:
            if room < len(part):
                events += [{name: part[:room]}] if room else []
                return [*events, {"finish_reason": "length"}]
            events.append({name: part})
            room -= len(part)
        return [*events, {"finish_reason": "stop"}]

    def _chunk(self, event: dict[str, Any]) -> dict[str, Any]:
        delta: dict[str, Any] = {}
        if "role" in event:
            delta.update(role=event["role"], content="")  # vLLM's first chunk
        if "reasoning" in event:
            delta[self.reasoning_field] = event["reasoning"]
        if "content" in event:
            delta["content"] = event["content"]
        finish = event.get("finish_reason")
        if finish is not None and not delta:
            delta["content"] = ""  # vLLM's finish chunk
        choice = {"index": event.get("index", 0), "delta": delta, "logprobs": None, "finish_reason": finish}
        return self._head() | {"choices": [choice]}

    def _usage_chunk(self, script: Script, body: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
        given = script.usage or {}
        prompt = given.get("prompt_tokens", self._count(script, body))
        completion = given.get("completion_tokens")
        if completion is None:
            generated = sum(len(event.get("reasoning", "")) + len(event.get("content", "")) for event in events)
            completion = max(1, math.ceil(generated / 4))
        usage: dict[str, Any] = {"prompt_tokens": prompt, "total_tokens": prompt + completion,
                                 "completion_tokens": completion}
        if given.get("cached_tokens") is not None:
            usage["prompt_tokens_details"] = {"cached_tokens": given["cached_tokens"]}
        return self._head() | {"choices": [], "usage": usage}

    def _head(self) -> dict[str, Any]:
        return {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": int(time.time()),
                "model": self.served_name}

    async def _send_event(self, call: Call, send: Send, chunk: dict[str, Any]) -> None:
        await self._send_data(call, send, json.dumps(chunk).encode())

    async def _send_data(self, call: Call, send: Send, data: bytes) -> None:
        await send({"type": "http.response.body", "body": b"data: " + data + b"\n\n", "more_body": True})
        call.events_sent += 1
        call.last_event_at = time.monotonic()


async def _watch_client(call: Call, receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass
    if call.ended_at is None:
        call.client_left_at = time.monotonic()
        call.client_left.set()


async def _hold(gate: asyncio.Event, call: Call, timeout: float | None = None) -> None:
    """Wait until the gate opens, the client leaves or the timeout passes, whichever comes first."""
    waiters = {asyncio.create_task(gate.wait()), asyncio.create_task(call.client_left.wait())}
    try:
        await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()


async def _read_json(receive: Receive) -> Any:
    chunks = []
    while True:
        message: Message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    try:
        return json.loads(b"".join(chunks)) if chunks and any(chunks) else None
    except ValueError:
        return None


def _prompt_text(body: Any) -> str:
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return ""
    return "".join(m["content"] for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str))


def _uncompilable(body: Any) -> bool:
    """Whether the request's schema holds a `pattern` that is not a regular expression, which vLLM cannot compile."""
    fmt = body.get("response_format") if isinstance(body, dict) else None
    spec = fmt.get("json_schema") if isinstance(fmt, dict) else None
    pending = [spec.get("schema") if isinstance(spec, dict) else None]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if isinstance(node.get("pattern"), str):
                try:
                    re.compile(node["pattern"])
                except re.error:
                    return True
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    return False


async def _respond(send: Send, status: int, body: bytes, content_type: bytes = b"text/plain") -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body, "more_body": False})


async def _respond_json(send: Send, status: int, payload: Any) -> None:
    await _respond(send, status, json.dumps(payload).encode(), b"application/json")

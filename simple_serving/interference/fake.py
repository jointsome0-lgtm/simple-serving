"""A synthetic gateway for the prefill-interference dry run, not a GPU model or a throughput prediction.

A cold long request pauses the fake's decoders for a known interval; the exact prompt and scope, once primed, do
not. Empty and metadata chunks precede the first content chunk. Tests can gate that boundary, fragment UTF-8 SSE,
or break either stream. Calls retain counts and timestamps only, never a prompt, key or generated text.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from starlette.types import Receive, Scope, Send

from ..config import Listener
from ..server import Servers, bind

MODEL = "synthetic-prefill"
KEY = "synthetic-interference-key"
MARKER = "synthetic-private-stream-marker-ё"
CONTEXT = 65536
CLASSES = ("reader", "agent", "internal", "external")
LIMITS = {"reader": 4, "agent": 2, "internal": 2, "external": 1}


@dataclass
class Call:
    role: str
    cls: str
    started_at: float
    input_tokens: int
    cached_tokens: int
    first_token_at: float | None = None
    tokens: int = 0
    complete: bool = False
    disconnected: bool = False
    ended_at: float | None = None


class FakeInterference:
    def __init__(self, *, stall_s: float = 0.12, token_delay_s: float = 0.003, fragments: bool = True,
                 post_prefill_tail: bool = True,
                 metadata_gate: asyncio.Event | None = None, gate_decoder: int = 2, early_usage: bool = False,
                 fault: str | None = None, fault_on: str = "decoder", marker: str = MARKER) -> None:
        self.stall_s, self.token_delay_s, self.fragments = stall_s, token_delay_s, fragments
        self.post_prefill_tail = post_prefill_tail
        self.metadata_gate, self.gate_decoder, self.early_usage = metadata_gate, gate_decoder, early_usage
        self.fault, self.fault_on, self.marker = fault, fault_on, marker
        self.metadata_sent, self.first_token_sent, self.long_started = (asyncio.Event() for _ in range(3))
        self.calls: list[Call] = []
        self.pending: set[asyncio.Task[None]] = set()
        self.active = dict.fromkeys(CLASSES, 0)
        self._cache: set[bytes] = set()
        self._blocked_until = 0.0
        self._decoders = 0
        self._started = time.time()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return
        headers = {name.decode().lower(): value.decode() for name, value in scope["headers"]}
        method, path = scope["method"], scope["path"]
        if headers.get("authorization") != f"Bearer {KEY}" and path != "/health":
            await _json(send, 401, {"error": {"code": "unauthorized"}})
            return
        if method == "GET":
            await self._get(path, send)
            return
        body = await _body(receive)
        if path in ("/tokenize", "/v1/chat/completions/input_tokens"):
            name = "count" if path == "/tokenize" else "input_tokens"
            await _json(send, 200, {name: _count(body), "max_model_len": CONTEXT})
        elif path == "/v1/chat/completions":
            await self._generate(body, headers, receive, send)
        else:
            await _json(send, 404, {"error": {"code": "not_found"}})

    async def _get(self, path: str, send: Send) -> None:
        if path == "/health":
            await _json(send, 200, {})
        elif path == "/v1/state":
            await _json(send, 200, {"contract": "2", "boot_id": "synthetic-prefill-boot", "status": "ready",
                                   "model": MODEL, "context_tokens": CONTEXT, "active": self.active,
                                   "waiting": dict.fromkeys(CLASSES, 0), "sleep_requested": False,
                                   "drain_generation": 0, "versions": {"vllm": "synthetic"}})
        elif path == "/v1/models":
            await _json(send, 200, {"object": "list", "data": [{"id": MODEL, "object": "model",
                                                                 "max_model_len": CONTEXT}]})
        elif path in ("/metrics", "/metrics/engine"):
            text = self.metrics(engine=path.endswith("/engine"))
            await _respond(send, 200, text.encode(), b"text/plain; version=0.0.4; charset=utf-8")
        else:
            await _json(send, 404, {"error": {"code": "not_found"}})

    def metrics(self, *, engine: bool) -> str:
        prompt = sum(call.input_tokens for call in self.calls)
        cached = sum(call.cached_tokens for call in self.calls)
        generated = sum(call.tokens for call in self.calls)
        running = sum(self.active.values())
        if engine:
            values = {"num_requests_running": running, "num_requests_waiting": 0, "kv_cache_usage_perc": 0.1,
                      "num_preemptions_total": 0, "prompt_tokens_total": prompt,
                      "generation_tokens_total": generated, "prefix_cache_queries_total": prompt,
                      "prefix_cache_hits_total": cached}
            return "".join(f"vllm:{name}{{engine=\"0\",model_name=\"{MODEL}\"}} {value}\n"
                           for name, value in values.items())
        lines = [f"simple_serving_start_time_seconds {self._started}",
                 "simple_serving_generation_shared_limit 4", f"simple_serving_ours_in_flight {running}",
                 f'simple_serving_accepted_in_flight{{route="/v1/chat/completions"}} {running}']
        for cls in CLASSES:
            lines.extend((f'simple_serving_generation_active_limit{{class="{cls}"}} {LIMITS[cls]}',
                          f'simple_serving_generations_active{{class="{cls}"}} {self.active[cls]}',
                          f'simple_serving_generations_waiting{{class="{cls}"}} 0',
                          f'simple_serving_counts_active{{class="{cls}"}} 0',
                          f'simple_serving_counts_waiting{{class="{cls}"}} 0'))
            ended = sum(call.complete for call in self.calls if call.cls == cls)
            lines.append('simple_serving_requests_total{route="/v1/chat/completions",'
                         f'class="{cls}",outcome="success",code="none"}} {ended}')
        return "\n".join(lines) + "\n"

    async def _generate(self, body: dict[str, Any], headers: dict[str, str], receive: Receive, send: Send) -> None:
        cls = headers.get("x-simple-serving-class", "internal")
        cls = cls if cls in CLASSES else "internal"
        role = "long" if cls == "internal" or body.get("max_tokens", 1) <= 8 or _count(body) > 1024 else "decoder"
        key = hashlib.sha256(json.dumps([body.get("messages"), headers.get("x-simple-serving-scope", "")],
                                       sort_keys=True).encode()).digest()
        count = _count(body)
        call = Call(role, cls, time.monotonic(), count, count if role == "long" and key in self._cache else 0)
        self.calls.append(call)
        if role == "decoder":
            self._decoders += 1
        ordinal = self._decoders
        self.active[cls] += 1
        stream = asyncio.create_task(self._stream(body, call, ordinal, send))
        disconnected = asyncio.create_task(_disconnected(receive))
        self.pending.update((stream, disconnected))
        try:
            await asyncio.wait((stream, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected.done() and not stream.done():
                call.disconnected = True
                stream.cancel()
            elif stream.done():
                await stream
            if call.complete and role == "long":
                self._cache.add(key)
        except asyncio.CancelledError:
            call.disconnected = True
            raise
        finally:
            for task in (stream, disconnected):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stream, disconnected, return_exceptions=True)
            self.pending.difference_update((stream, disconnected))
            self.active[cls] -= 1
            call.ended_at = time.monotonic()

    async def _stream(self, body: dict[str, Any], call: Call, ordinal: int, send: Send) -> None:
        fault = self.fault if call.role == self.fault_on else None
        if fault == "http_error":
            await _json(send, 503, {"error": {"code": "engine_unavailable", "message": self.marker}})
            return
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
        await self._event(send, _chunk({"role": "assistant"}))
        await self._event(send, _chunk({"content": ""}))
        if self.early_usage:
            await self._event(send, _usage_event(call, int(body.get("max_tokens", 1))))
        if call.role == "decoder" and self.metadata_gate is not None and ordinal == self.gate_decoder:
            self.metadata_sent.set()
            await self.metadata_gate.wait()
        if call.role == "long":
            self.long_started.set()
            if not call.cached_tokens:
                # A full freeze resumes decoding on the next token interval after the prefill response.
                resume_delay = 0 if self.post_prefill_tail else self.token_delay_s
                self._blocked_until = max(self._blocked_until, time.monotonic() + self.stall_s + resume_delay)
                await asyncio.sleep(self.stall_s)
            if self.post_prefill_tail:
                # This variant keeps HTTP open long enough to observe decode chunks beside the aggressor.
                await asyncio.sleep(self.token_delay_s * 3)
        count = int(body.get("max_tokens", 1))
        if fault == "early_finish":
            count = 1 if call.role == "decoder" else 0
        if fault == "empty_content":
            count = 0
        for index in range(count):
            if call.role == "decoder":
                await asyncio.sleep(self.token_delay_s)
                await asyncio.sleep(max(0, self._blocked_until - time.monotonic()))
            if fault == "stream_error" and index == min(4, count - 1):
                await self._event(send, {"error": {"code": "engine_unavailable", "message": self.marker}})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            if fault == "hang" and index == min(4, count - 1):
                await asyncio.Event().wait()
            await self._event(send, _chunk({"content": self.marker}))
            call.first_token_at = call.first_token_at or time.monotonic()
            call.tokens += 1
            self.first_token_sent.set()
        await self._event(send, _chunk({}, finish="stop" if fault == "early_finish" else "length"))
        if not self.early_usage:
            await self._event(send, _usage_event(call, call.tokens))
        if fault != "missing_done":
            await self._bytes(send, b"data: [DONE]\r\n\r\n")
            call.complete = True
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _event(self, send: Send, value: dict[str, Any]) -> None:
        await self._bytes(send, b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\r\n\r\n")

    async def _bytes(self, send: Send, data: bytes) -> None:
        chunks: tuple[bytes, ...]
        if self.fragments:
            utf8 = data.find("ё".encode())
            at = utf8 + 1 if utf8 >= 0 else max(1, len(data) // 2)
            chunks = (data[:at], data[at:-1], data[-1:])
        else:
            chunks = (data,)
        for chunk in chunks:
            await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await asyncio.sleep(0)


@dataclass(frozen=True)
class SyntheticGateway:
    url: str
    fake: FakeInterference
    model: str = MODEL
    key: str = KEY


@contextlib.asynccontextmanager
async def synthetic_gateway(fake: FakeInterference | None = None) -> AsyncIterator[SyntheticGateway]:
    fake = fake or FakeInterference()
    sock = bind(Listener("127.0.0.1", 0))
    server = Servers([(fake, sock)], graceful_s=1)
    try:
        await server.start()
        yield SyntheticGateway(f"http://127.0.0.1:{sock.getsockname()[1]}", fake)
    finally:
        await server.stop()


def _count(body: dict[str, Any]) -> int:
    return max(1, math.ceil(sum(len(item.get("content", "")) for item in body.get("messages", [])) / 4))


async def _body(receive: Receive) -> dict[str, Any]:
    chunks = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return {}
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            body: dict[str, Any] = json.loads(b"".join(chunks) or b"{}")
            return body


async def _disconnected(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass


def _chunk(delta: dict[str, Any], *, finish: str | None = None) -> dict[str, Any]:
    return {"id": "synthetic", "object": "chat.completion.chunk", "model": MODEL,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def _usage_event(call: Call, generated: int) -> dict[str, Any]:
    elapsed = max(0, int((time.monotonic() - call.started_at) * 1000))
    first = max(0, int(((call.first_token_at or call.started_at) - call.started_at) * 1000))
    usage = {"prompt_tokens": call.input_tokens, "completion_tokens": generated,
             "total_tokens": call.input_tokens + generated,
             "prompt_tokens_details": {"cached_tokens": call.cached_tokens},
             "simple_serving": {"wait_ms": 0, "first_token_ms": first, "total_ms": elapsed}}
    return {**_chunk({}), "choices": [], "usage": usage}


async def _json(send: Send, status: int, value: dict[str, Any]) -> None:
    await _respond(send, status, json.dumps(value).encode(), b"application/json")


async def _respond(send: Send, status: int, data: bytes, content_type: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type), (b"content-length", str(len(data)).encode())]})
    await send({"type": "http.response.body", "body": data, "more_body": False})

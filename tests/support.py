"""The gateway in front of the fake engine on real loopback sockets, and helpers to call it and read its answers.

Every test starts its own fake engine and gateway, so every case runs on a new boot. Bodies are synthetic.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving.admission import Admission
from simple_serving.asgi import Exchange
from simple_serving.config import Listener, from_service_block
from simple_serving.engine import EngineStream
from simple_serving.fake_engine import Call, FakeEngine, Script
from simple_serving.server import Gateway, Servers, bind

ROOT = Path(__file__).resolve().parent.parent
CASES: dict[str, Any] = json.loads((ROOT / "contract" / "cases-v1.json").read_text(encoding="utf-8"))
SERVICE: dict[str, Any] = CASES["service"]
ALIAS: str = SERVICE["alias"]
BOT, CONTROL, OUTSIDE_A, OUTSIDE_B = "test-key-bot", "test-key-control", "test-key-outside-a", "test-key-outside-b"
CLASS, SCOPE = "X-Simple-Serving-Class", "X-Simple-Serving-Scope"


def service_with(**changes: Any) -> dict[str, Any]:
    """The cases' service block with some fields changed; `limits` is merged class by class."""
    block = copy.deepcopy(SERVICE)
    for name, value in changes.items():
        if name == "limits":
            for cls, fields in value.items():
                block["limits"][cls].update(fields)
        else:
            block[name] = value
    return block


@dataclass
class Stack:
    fake: FakeEngine
    gateway: Gateway
    public: str
    control: str

    @property
    def service(self) -> Any:
        return self.gateway.service


@asynccontextmanager
async def running(service: dict[str, Any] | None = None, *, fake: FakeEngine | None = None,
                  wait_ready: bool = True, **extra: Any) -> AsyncIterator[Stack]:
    """A fake engine and a gateway in front of it, both served by uvicorn on free loopback ports. `extra` holds
    configuration fields that the service block leaves out, such as `health_interval_s`."""
    block = service or SERVICE
    fake = fake or FakeEngine(block["alias"], block["context_tokens"])
    sock = bind(Listener("127.0.0.1", 0))
    engine_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    engine = Servers([(fake, sock)], graceful_s=1)
    await engine.start()
    try:
        config = from_service_block(block, engine_url=engine_url, **extra)
        async with Gateway(config, graceful_s=1) as gateway:
            stack = Stack(fake, gateway, f"http://127.0.0.1:{gateway.ports['public']}",
                          f"http://127.0.0.1:{gateway.ports['control']}")
            if wait_ready:
                await until(lambda: gateway.service.status == "ready")
            yield stack
    finally:
        await engine.stop()


async def until(condition: Callable[[], bool], timeout: float = 5.0, interval: float = 0.005) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("the condition did not come true in time")
        await asyncio.sleep(interval)


def chat_body(**patch: Any) -> dict[str, Any]:
    body = copy.deepcopy(CASES["defaults"]["chat_body"])
    body.update(patch)
    return body


def user_body(text: str, **patch: Any) -> dict[str, Any]:
    """A generation body whose only message names the request, so that a script and a test can tell requests apart."""
    return chat_body(messages=[{"role": "user", "content": text}], **patch)


def request_headers(key: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {} if key is None else {"Authorization": f"Bearer {key}"}
    headers.update(extra or {})
    return headers


def reader(scope: str = "reader.r1aaaaaa") -> dict[str, str]:
    return {CLASS: "reader", SCOPE: scope}


@dataclass
class Answer:
    status: int
    headers: httpx.Headers
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)

    def events(self) -> list[Any]:
        """The `data:` events of a stream, parsed; `[DONE]` stays a string."""
        return parse_events(self.body)

    @property
    def done(self) -> bool:
        events = self.events()
        return bool(events) and events[-1] == "[DONE]"

    @property
    def error_event(self) -> str | None:
        events = self.events()
        last = events[-1] if events else None
        return last["error"]["code"] if isinstance(last, dict) and "error" in last else None


def parse_events(body: bytes) -> list[Any]:
    events = []
    for block in body.decode("utf-8").split("\n\n"):
        data = "\n".join(line[6:] for line in block.split("\n") if line.startswith("data: "))
        if data:
            events.append(data if data == "[DONE]" else json.loads(data))
    return events


async def call(client: httpx.AsyncClient, method: str, url: str, *, key: str | None = BOT,
               headers: dict[str, str] | None = None, body: Any = None, content: Any = None) -> Answer:
    """One request, read to its end."""
    if body is not None:
        content = json.dumps(body).encode()
    async with client.stream(method, url, headers=request_headers(key, headers), content=content) as response:
        data = await response.aread()
    return Answer(response.status_code, response.headers, data)


async def generate(client: httpx.AsyncClient, stack: Stack, body: dict[str, Any], *, key: str | None = BOT,
                   headers: dict[str, str] | None = None) -> Answer:
    return await call(client, "POST", stack.public + "/v1/chat/completions", key=key, headers=headers, body=body)


async def count(client: httpx.AsyncClient, stack: Stack, body: dict[str, Any], *, key: str | None = BOT,
                headers: dict[str, str] | None = None) -> Answer:
    return await call(client, "POST", stack.public + "/v1/chat/completions/input_tokens", key=key,
                      headers=headers, body=body)


async def control(client: httpx.AsyncClient, stack: Stack, path: str, body: Any = None,
                  method: str = "POST") -> Answer:
    return await call(client, method, stack.control + path, key=CONTROL, body=body)


def prompt_of(body: Any) -> str:
    """The text of a request's messages, as the fake engine saw it."""
    return "".join(message["content"] for message in body["messages"])


def by_prompt(scripts: dict[str, Script]) -> Callable[[Any], Script]:
    """A fake engine script for each request, chosen by its prompt; other prompts get the default answers."""
    return lambda body: scripts.get(prompt_of(body), Script())


def engine_call(stack: Stack, kind: str, text: str) -> Call | None:
    """The fake engine's call of `kind` for the request whose prompt is `text`, if it arrived."""
    return next((call for call in stack.fake.calls_of(kind) if prompt_of(call.body) == text), None)


def generate_call(stack: Stack, text: str) -> Call | None:
    return engine_call(stack, "generate", text)


async def reached(stack: Stack, text: str, kind: str = "generate") -> Call:
    """Wait until the request whose prompt is `text` has reached the fake engine."""
    await until(lambda: engine_call(stack, kind, text) is not None)
    call = engine_call(stack, kind, text)
    assert call is not None
    return call


def work_of(stack: Stack, text: str) -> Any:
    """The gateway's work for the request whose prompt is `text`, while it has any."""
    return next((work for work in stack.service.work if prompt_of({"messages": work.request.messages}) == text), None)


async def streaming(stack: Stack, text: str) -> None:
    """Wait until the gateway has started the stream of the request whose prompt is `text`."""
    await until(lambda: (work := work_of(stack, text)) is not None and work.exchange.started)


def record_abort_order(monkeypatch: Any, close_delay_s: float = 0.0) -> list[str]:
    """Note, in order, when an engine stream has finished closing and when an active place is freed. A delay before
    each close makes a place freed too early visible."""
    order: list[str] = []
    close, leave = EngineStream.aclose, Admission.leave

    async def slow_close(self: EngineStream) -> None:
        await asyncio.sleep(close_delay_s)
        await close(self)
        order.append("closed")

    def noting_leave(self: Admission, ticket: Any) -> None:
        if ticket.active:
            order.append("freed")
        leave(self, ticket)

    monkeypatch.setattr(EngineStream, "aclose", slow_close)
    monkeypatch.setattr(Admission, "leave", noting_leave)
    return order


class RawClient:
    """A request over a plain socket, so that a test decides the moment its client leaves."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer
        self.received = b""
        self.left_at: float | None = None

    @classmethod
    async def post(cls, base: str, path: str, body: Any, *, key: str | None = BOT,
                   headers: dict[str, str] | None = None, missing: int = 0) -> RawClient:
        """Send a request. With `missing`, the body is announced that many bytes longer than it is, so that its end
        never arrives."""
        host, port = base.removeprefix("http://").split(":")
        reader_, writer = await asyncio.open_connection(host, int(port))
        payload = json.dumps(body).encode()
        lines = [f"POST {path} HTTP/1.1", f"Host: {host}:{port}", "Content-Type: application/json",
                 f"Content-Length: {len(payload) + missing}"]
        lines += [f"{name}: {value}" for name, value in request_headers(key, headers).items()]
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode() + payload)
        await writer.drain()
        return cls(reader_, writer)

    async def read_some(self, timeout: float) -> bytes:
        """What arrives within `timeout`, or nothing."""
        try:
            data = await asyncio.wait_for(self.reader.read(65536), timeout)
        except TimeoutError:
            return b""
        self.received += data
        return data

    async def read_until_events(self, count: int, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.received.count(b"data: ") < count:
            if time.monotonic() > deadline:
                raise AssertionError("the stream did not arrive in time")
            data = await self.read_some(deadline - time.monotonic())
            if not data and self.reader.at_eof():
                raise AssertionError("the connection ended early")

    async def read_to_end(self, timeout: float = 5.0) -> bytes:
        deadline = time.monotonic() + timeout
        while not self.reader.at_eof() and not self.received.endswith(b"0\r\n\r\n"):
            if time.monotonic() > deadline:
                raise AssertionError("the response did not end in time")
            await self.read_some(deadline - time.monotonic())
        return self.received

    async def leave(self) -> float:
        """Close the connection and return the moment."""
        self.left_at = time.monotonic()
        self.writer.close()
        with contextlib.suppress(OSError):
            await self.writer.wait_closed()
        return self.left_at


async def eventually(awaitable: Awaitable[Any], timeout: float = 5.0) -> Any:
    return await asyncio.wait_for(awaitable, timeout)


class FakeClock:
    """The event loop's clock, moved on by hand. Every timer of the loop, the gateway's and the servers' alike, comes
    due as if that time had passed, so no test waits for one. Requests still take their real milliseconds."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop = asyncio.get_running_loop()
        real = loop.time
        self.offset = 0.0
        monkeypatch.setattr(loop, "time", lambda: real() + self.offset)

    async def advance(self, seconds: float) -> None:
        """Move the clock on and let what came due run."""
        self.offset += seconds
        for _ in range(20):
            await asyncio.sleep(0)


def stall_the_send_of(monkeypatch: pytest.MonkeyPatch, start: bytes) -> asyncio.Event:
    """Make the gateway's send of the first body that begins with `start` wait until it is cancelled, as a send waits
    while its client does not read. The event is set when that send begins."""
    began = asyncio.Event()
    init = Exchange.__init__

    def stalling_init(self: Exchange, scope: Any, receive: Any, send: Any) -> None:
        async def stalling_send(message: Any) -> None:
            if not began.is_set() and message.get("body", b"").startswith(start):
                began.set()
                await asyncio.Future()  # nothing resolves it: only a cancellation ends the wait
            await send(message)

        init(self, scope, receive, stalling_send)

    monkeypatch.setattr(Exchange, "__init__", stalling_init)
    return began

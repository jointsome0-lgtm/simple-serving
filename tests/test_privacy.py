"""The scenario error-bodies-are-safe (contract sections 9 and 10): no error answer and no log line holds any part of a
request or of its output, on every error path.

The gateway runs as its own process, configured from a file with hashed keys, as in production. Every request carries
a marker: in its messages, its schema, its field names and values, its path and query, its headers and its key. The
fake engine echoes the prompt in its errors, as engines may. Afterwards everything the process wrote is searched for
the marker and for the keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving.config import Listener
from simple_serving.fake_engine import FakeEngine, Script
from simple_serving.server import Servers, bind

from .support import (
    ALIAS,
    BOT,
    CLASS,
    CONTROL,
    OUTSIDE_A,
    OUTSIDE_B,
    ROOT,
    SCOPE,
    SERVICE,
    Answer,
    RawClient,
    call,
    chat_body,
    prompt_of,
    reader,
    service_with,
    until,
)

pytestmark = pytest.mark.anyio

MARKER = "Mk" + secrets.token_hex(8)  # letters and digits, so that it also fits a scope and a header
KEYS = (BOT, CONTROL, OUTSIDE_A, OUTSIDE_B)
CHAT, COUNT = "/v1/chat/completions", "/v1/chat/completions/input_tokens"


def text(name: str) -> str:
    return f"{name}: the keeper whispers {MARKER}"


def body(name: str, **patch: Any) -> dict[str, Any]:
    """A generation body whose every message carries the marker."""
    messages = [{"role": "system", "content": f"System {MARKER}."}, {"role": "user", "content": text(name)}]
    return chat_body(**{"messages": messages, **patch})


class Process:
    """The gateway as a child process: `python -m simple_serving` with its configuration file."""

    def __init__(self, config: Path) -> None:
        self.config = config
        self.lines: list[bytes] = []
        self.ports: dict[str, int] = {}

    async def start(self) -> None:
        # Without Vast's two values, a sleep of this process could never stop a real instance.
        env = {name: value for name, value in os.environ.items() if not name.startswith("CONTAINER_")}
        env["SIMPLE_SERVING_CONFIG"] = str(self.config)
        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "simple_serving", cwd=ROOT, env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self._readers = [asyncio.create_task(self._read(stream))
                         for stream in (self.process.stdout, self.process.stderr)]
        await until(lambda: len(self.ports) == 2 or self.process.returncode is not None, timeout=15)
        assert len(self.ports) == 2, b"".join(self.lines)[-2000:]

    async def stop(self) -> None:
        if self.process.returncode is None:
            self.process.send_signal(signal.SIGTERM)
        await asyncio.wait_for(self.process.wait(), 15)
        await asyncio.gather(*self._readers)

    def url(self, listener: str) -> str:
        return f"http://127.0.0.1:{self.ports[listener]}"

    def rows(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.lines if line.strip()]

    async def _read(self, stream: asyncio.StreamReader | None) -> None:
        assert stream is not None
        while line := await stream.readline():
            self.lines.append(line)
            if b'"listening"' in line:
                row = json.loads(line)
                self.ports[row["listener"]] = row["port"]


@asynccontextmanager
async def gateway(tmp_path: Path, fake: FakeEngine) -> AsyncIterator[Process]:
    engine_socket = bind(Listener("127.0.0.1", 0))
    engine = Servers([(fake, engine_socket)], graceful_s=1)
    await engine.start()
    block = service_with(limits={"agent": {"waiting": 0}, "external": {"wall_s": 0.5}}, drain_deadline_s=0.5)
    block["keys"] = [{"sha256": hashlib.sha256(raw.encode()).hexdigest(), **entry}
                     for raw, entry in block["keys"].items()]
    config = {**block, "engine_url": f"http://127.0.0.1:{engine_socket.getsockname()[1]}", "health_interval_s": 0.1,
              "listen": {"public": {"host": "127.0.0.1", "port": 0}, "control": {"host": "127.0.0.1", "port": 0}}}
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(config))
    process = Process(path)
    try:
        await process.start()
        yield process
    finally:
        await process.stop()
        await engine.stop()


class Checks:
    """The requests of the test and what their answers must be."""

    def __init__(self, client: httpx.AsyncClient, gateway: Process) -> None:
        self.client = client
        self.public = gateway.url("public")
        self.control = gateway.url("control")

    async def error(self, status: int, code: str, method: str, url: str, **options: Any) -> None:
        """A plain error: exactly the error body, and the marker nowhere in the answer."""
        answer = await call(self.client, method, url, **options)
        assert (answer.status, answer.json()) == (status, {"error": {"code": code}}), (url, answer.body[:300])
        self.clean(answer)

    async def status(self, wanted: str) -> None:
        """Wait until the gateway reports this status."""
        for _ in range(200):
            if (await call(self.client, "GET", self.public + "/v1/state")).json()["status"] == wanted:
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"the gateway never became {wanted}")

    async def error_event(self, code: str, url: str, **options: Any) -> None:
        """A stream that ends with an error event."""
        answer = await call(self.client, "POST", url, **options)
        assert answer.status == 200 and answer.error_event == code and not answer.done, answer.body[-300:]
        self.clean(answer)

    @staticmethod
    def clean(answer: Answer) -> None:
        assert MARKER.encode() not in answer.body
        assert not any(MARKER in f"{name}: {value}" for name, value in answer.headers.multi_items())


async def test_no_answer_and_no_log_line_holds_the_request(tmp_path: Path) -> None:
    fake = FakeEngine(ALIAS, SERVICE["context_tokens"])
    scripts: dict[str, Script] = {}  # by the name of the request, which ends its prompt
    fake.script = lambda body: next((script for name, script in scripts.items()
                                     if prompt_of(body).endswith(text(name))), Script())
    async with gateway(tmp_path, fake) as process, httpx.AsyncClient(timeout=10) as client:
        checks = Checks(client, process)
        await checks.status("ready")
        await refusals_before_the_body(checks)
        await refusals_of_the_body(checks)
        await refusals_after_counting(checks, scripts)
        await engine_failures(checks, scripts)
        await places_and_wall_time(checks, scripts, fake)
        await clients_that_leave(checks, scripts)
        await answers_that_succeed(checks, scripts)
        await control_routes_and_a_drain(checks, scripts, fake)
        await an_engine_that_stops_answering(checks, fake)
        await malformed_http(checks)

    output = b"".join(process.lines)
    assert MARKER.encode() not in output
    for key in KEYS:
        assert key.encode() not in output
    rows = process.rows()  # every line is a JSON object
    requests = [row for row in rows if row.get("event") == "request"]
    codes = {row.get("code") for row in requests}
    assert codes >= {"unauthorized", "class_not_allowed", "scope_not_allowed", "invalid_request", "unsupported_field",
                     "limit_exceeded", "context_limit", "body_too_large", "not_found", "forbidden", "stale_boot",
                     "queue_full", "timeout", "draining", "engine_unavailable"}, codes
    assert any(row.get("cancelled") for row in requests)
    assert {row["scope"] for row in requests if "scope" in row} <= {"reader", "agent", "internal", "external"}


async def refusals_before_the_body(checks: Checks) -> None:
    public = checks.public
    await checks.error(401, "unauthorized", "POST", public + CHAT, key=None, body=body("no key"))
    await checks.error(401, "unauthorized", "POST", public + CHAT, key=MARKER, body=body("unknown key"))
    await checks.error(401, "unauthorized", "GET", public + "/v1/state", key=None,
                       headers={"Authorization": f"Basic {MARKER}"})
    await checks.error(403, "class_not_allowed", "POST", public + CHAT, headers={CLASS: MARKER}, body=body("class"))
    await checks.error(403, "scope_not_allowed", "POST", public + CHAT, key=OUTSIDE_A,
                       headers={SCOPE: f"reader.{MARKER}"}, body=body("scope"))
    await checks.error(400, "invalid_request", "POST", public + CHAT, headers={SCOPE: MARKER}, body=body("scope"))
    await checks.error(400, "invalid_request", "POST", public + CHAT, headers={CLASS: "reader"}, body=body("reader"))
    await checks.error(404, "not_found", "GET", public + f"/v1/{MARKER}?q={MARKER}")
    await checks.error(404, "not_found", "GET", public + f"{CHAT}?q={MARKER}")
    await checks.error(404, "not_found", "POST", public + "/v1/control/drain", key=CONTROL,
                       body={"boot_id": MARKER})
    for path in ("/v1/models", "/v1/state"):
        answer = await call(checks.client, "GET", public + f"{path}?{MARKER}={MARKER}")
        assert answer.status == 200
        checks.clean(answer)


async def refusals_of_the_body(checks: Checks) -> None:
    url, internal = checks.public + CHAT, {CLASS: "internal"}
    schema = {"type": "object", "properties": {MARKER: {"type": "string"}}}
    bad_bodies = {
        "invalid_request": [
            f'{{"model": "{MARKER}", '.encode(),  # not JSON
            b"\xff\xfe" + MARKER.encode(),  # not UTF-8
            f'{{"model": "{MARKER}", "model": "{MARKER}"}}'.encode(),  # a repeated key
            json.dumps(body("nan")).replace('"max_tokens": 64', '"max_tokens": NaN').encode(),
            ("[" * 100_000 + f'"{MARKER}"' + "]" * 100_000).encode(),  # deeper than the parser goes
            json.dumps(body("model", model=MARKER)).encode(),
            json.dumps(body("temperature", temperature=MARKER)).encode(),
            json.dumps(body("schema", response_format={"type": "json_schema", "json_schema": {
                "name": MARKER, "strict": MARKER, "schema": schema}})).encode(),
            json.dumps(body("role", messages=[{"role": MARKER, "content": MARKER}])).encode(),
        ],
        "unsupported_field": [
            json.dumps(body("field", **{MARKER: MARKER})).encode(),
            json.dumps(body("nested", messages=[{"role": "user", "content": "Hi.", MARKER: MARKER}])).encode(),
            json.dumps(body("options", stream_options={"include_usage": True, MARKER: MARKER})).encode(),
        ],
    }
    for code, contents in bad_bodies.items():
        for content in contents:
            await checks.error(400, code, "POST", url, headers=internal, content=content)
            await checks.error(400, code, "POST", checks.public + COUNT, headers=internal, content=content)
    await checks.error(400, "limit_exceeded", "POST", url, key=OUTSIDE_A, body=body("max_tokens", max_tokens=512))
    oversized = json.dumps(body("large", messages=[{"role": "user", "content": MARKER * 200_000}])).encode()
    await checks.error(413, "body_too_large", "POST", url, headers=internal, content=oversized)

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(40):
            yield MARKER.encode() * 4000

    await checks.error(413, "body_too_large", "POST", url, headers=internal, content=chunks())


async def refusals_after_counting(checks: Checks, scripts: dict[str, Script]) -> None:
    url = checks.public + CHAT
    scripts["over the context"] = Script(input_tokens=4090)
    await checks.error(400, "context_limit", "POST", url, headers=reader(), body=body("over the context"))
    scripts["over the class"] = Script(input_tokens=2000)
    await checks.error(400, "limit_exceeded", "POST", url, key=OUTSIDE_A, body=body("over the class"))


async def engine_failures(checks: Checks, scripts: dict[str, Script]) -> None:
    url, internal = checks.public + CHAT, {CLASS: "internal"}
    failures = {  # what the engine does, and what the client is told
        "refused": (Script(generate_status=400), 400, "invalid_request"),
        "unprocessable": (Script(generate_status=422), 400, "invalid_request"),
        "not found": (Script(generate_status=404), 503, "engine_unavailable"),
        "failed": (Script(generate_status=500), 503, "engine_unavailable"),
        "error first": (Script(events=[{"error": {"message": MARKER, "code": 400}}]), 400, "invalid_request"),
        "raw first": (Script(events=[{"raw_hex": MARKER.encode().hex()}]), 503, "engine_unavailable"),
        "break first": (Script(events=[{"break": True}]), 503, "engine_unavailable"),
    }
    for name, (script, status, code) in failures.items():
        scripts[name] = script
        await checks.error(status, code, "POST", url, headers=internal, body=body(name))
    streams: dict[str, list[dict[str, Any]]] = {
        "error later": [{"content": "Wind. "}, {"error": {"message": MARKER}}],
        "raw later": [{"content": "Wind. "}, {"raw_hex": (b"\xff" + MARKER.encode()).hex()}],
        "break later": [{"content": "Wind. "}, {"break": True}],
        "second finish": [{"content": "Wind. "}, {"finish_reason": "stop"}, {"finish_reason": "stop"}],
        "unknown finish": [{"content": "Wind. "}, {"finish_reason": MARKER}],
    }
    for name, events in streams.items():
        scripts[name] = Script(events=events)
        await checks.error_event("engine_unavailable", url, headers=internal, body=body(name))


async def places_and_wall_time(checks: Checks, scripts: dict[str, Script], fake: FakeEngine) -> None:
    url = checks.public + CHAT
    gate = asyncio.Event()
    scripts["holds the agent place"] = Script(hold_first=gate)
    held = asyncio.ensure_future(call(checks.client, "POST", url, headers={CLASS: "agent"},
                                      body=body("holds the agent place")))
    await until(lambda: any(text("holds the agent place") in json.dumps(c.body) for c in fake.calls_of("generate")))
    await checks.error(429, "queue_full", "POST", url, headers={CLASS: "agent"}, body=body("no room"))
    gate.set()
    assert (await held).done
    scripts["long prompt"] = Script(hold_first=asyncio.Event())
    await checks.error(504, "timeout", "POST", url, key=OUTSIDE_A, body=body("long prompt"))
    scripts["long answer"] = Script(events=[{"content": "On. "}], endless=True, delay_s=0.02)
    await checks.error_event("timeout", url, key=OUTSIDE_A, body=body("long answer"))


async def clients_that_leave(checks: Checks, scripts: dict[str, Script]) -> None:
    scripts["left mid-stream"] = Script(events=[{"content": "On. "}], endless=True, delay_s=0.01)
    client = await RawClient.post(checks.public, CHAT, body("left mid-stream"), headers=reader())
    await client.read_until_events(3)
    await client.leave()
    scripts["left in the prompt"] = Script(hold_first=asyncio.Event())
    client = await RawClient.post(checks.public, CHAT, body("left in the prompt"), headers=reader())
    await asyncio.sleep(0.1)
    await client.leave()
    host, port = checks.public.removeprefix("http://").split(":")
    _, writer = await asyncio.open_connection(host, int(port))  # leaves halfway through its body
    writer.write(f"POST {CHAT} HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {BOT}\r\nContent-Length: 5000\r\n"
                 f"\r\n{{\"messages\": \"{MARKER}".encode())
    await writer.drain()
    await asyncio.sleep(0.1)
    writer.close()
    await asyncio.sleep(0.2)


async def answers_that_succeed(checks: Checks, scripts: dict[str, Script]) -> None:
    scripts["says the marker"] = Script(events=[{"reasoning": f"Think {MARKER}."},
                                                      {"content": f"The keeper says {MARKER}."},
                                                      {"finish_reason": "stop"}])
    answer = await call(checks.client, "POST", checks.public + CHAT, headers=reader(), body=body("says the marker"))
    assert answer.done and MARKER.encode() in answer.body  # the output goes to its client, and nowhere else
    counted = await call(checks.client, "POST", checks.public + COUNT, headers=reader(), body=body("count"))
    assert counted.status == 200
    checks.clean(counted)


async def control_routes_and_a_drain(checks: Checks, scripts: dict[str, Script], fake: FakeEngine) -> None:
    drain, open_ = checks.control + "/v1/control/drain", checks.control + "/v1/control/open"
    await checks.error(403, "forbidden", "POST", drain, body={"boot_id": MARKER})
    await checks.error(409, "stale_boot", "POST", drain, key=CONTROL, body={"boot_id": MARKER})
    await checks.error(400, "unsupported_field", "POST", drain, key=CONTROL, body={MARKER: MARKER})
    await checks.error(400, "invalid_request", "POST", drain, key=CONTROL, content=f"{{{MARKER}".encode())
    await checks.error(400, "invalid_request", "POST", open_, key=CONTROL,
                       body={"boot_id": MARKER, "drain_generation": MARKER})
    await checks.error(404, "not_found", "GET", checks.control + f"/v1/{MARKER}", key=CONTROL)
    await checks.error(404, "not_found", "POST", checks.control + CHAT, body=body("wrong listener"))

    boot = (await call(checks.client, "GET", checks.control + "/v1/state", key=CONTROL)).json()["boot_id"]
    scripts["outside streams"] = Script(events=[{"content": "On. "}], endless=True, delay_s=0.02)
    scripts["ours in its prompt"] = Script(hold_first=asyncio.Event())
    outside = await RawClient.post(checks.public, CHAT, body("outside streams"), key=OUTSIDE_B)
    await outside.read_until_events(2)
    ours = asyncio.ensure_future(call(checks.client, "POST", checks.public + CHAT, headers=reader(),
                                      body=body("ours in its prompt")))
    await until(lambda: any(text("ours in its prompt") in json.dumps(c.body) for c in fake.calls_of("generate")))
    assert (await call(checks.client, "POST", drain, key=CONTROL, body={"boot_id": boot})).status == 202
    await checks.error(503, "draining", "POST", checks.public + CHAT, headers=reader(), body=body("during the drain"))
    cut = await outside.read_to_end()  # at once: an outside stream
    assert b'data: {"error":{"code":"draining"}}' in cut and b"[DONE]" not in cut
    assert MARKER.encode() not in cut
    deadline = await ours  # ours, in its prompt, at the drain deadline
    assert (deadline.status, deadline.json()) == (503, {"error": {"code": "draining"}})
    checks.clean(deadline)
    await checks.error(503, "drained", "POST", checks.public + COUNT, headers=reader(), body=body("drained"))
    state = (await call(checks.client, "GET", checks.control + "/v1/state", key=CONTROL)).json()
    reopened = await call(checks.client, "POST", open_, key=CONTROL,
                          body={"boot_id": boot, "drain_generation": state["drain_generation"]})
    assert reopened.json()["status"] == "ready"


async def an_engine_that_stops_answering(checks: Checks, fake: FakeEngine) -> None:
    fake.healthy = False
    await checks.status("failed")
    await checks.error(503, "engine_unavailable", "POST", checks.public + CHAT, headers=reader(),
                       body=body("engine down"))
    fake.healthy = True
    await checks.status("ready")


async def malformed_http(checks: Checks) -> None:
    """Requests the HTTP server itself refuses; its error log must not quote them either."""
    host, port = checks.public.removeprefix("http://").split(":")
    for request in (f"GET /{MARKER} HTTP/1.1\r\nHost: x\r\n{MARKER} {MARKER}\r\n\r\n",
                    f"{MARKER} /{MARKER} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
                    f"POST {CHAT} HTTP/1.1\r\nHost: x\r\nContent-Length: {MARKER}\r\n\r\n"):
        reader_, writer = await asyncio.open_connection(host, int(port))
        writer.write(request.encode())
        await writer.drain()
        answer = await asyncio.wait_for(reader_.read(), 5)
        assert MARKER.encode() not in answer
        writer.close()

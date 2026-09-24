"""What the cases and the scenarios leave out: the engine's health and context, what the engine receives, the
listeners' answers outside the cases, errors inside the gateway itself, and a client that does not read its terminal
message. Every test runs its own gateway over real sockets (support.py)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import secrets
import time
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from simple_serving import app, log
from simple_serving.asgi import Exchange
from simple_serving.config import CLASSES
from simple_serving.fake_engine import FakeEngine, Script
from simple_serving.service import Service
from simple_serving.stream import Translator
from simple_serving.validation import MAX_DEPTH

from .support import (ALIAS, CLASS, CONTROL, SCOPE, SERVICE, Answer, RawClient, by_prompt, call, chat_body, control,
                      count, eventually, generate, reached, reader, record_abort_order, running, service_with, until,
                      user_body)

pytestmark = pytest.mark.anyio
MARKER = "Mk" + secrets.token_hex(8)
AGENT = {CLASS: "agent"}


def assert_refused(answer: Answer, status: int, code: str) -> None:
    """A plain error of section 9, without Retry-After, which the gateway never sends."""
    assert (answer.status, answer.json()) == (status, {"error": {"code": code}}), answer.body[:200]
    assert "retry-after" not in answer.headers


@pytest.mark.parametrize("name", ["reasoning", "reasoning_content"])
async def test_reasoning_arrives_as_reasoning_content_whatever_the_engine_calls_it(name: str) -> None:
    fake = FakeEngine(ALIAS, SERVICE["context_tokens"], reasoning_field=name)
    fake.script = Script(events=[{"role": "assistant"}, {"reasoning": "The sea is calm."}, {"content": "Calm."},
                                 {"finish_reason": "stop"}])
    async with running(fake=fake) as stack, httpx.AsyncClient(timeout=10) as client:
        answer = await generate(client, stack, chat_body())
    assert answer.done
    deltas = [event["choices"][0]["delta"] for event in answer.events()[:-2]]
    assert deltas == [{"role": "assistant"}, {"reasoning_content": "The sea is calm."}, {"content": "Calm."}, {}]


@pytest.mark.parametrize(("gateway_limit", "engine_length", "effective"),
                         [(4096, 2048, 2048), (4096, 8192, 4096), (None, 8192, 8192)])
async def test_the_context_is_the_engines_length_or_the_gateways_smaller_limit(
        gateway_limit: int | None, engine_length: int, effective: int) -> None:
    fake = FakeEngine(ALIAS, engine_length)
    async with (running(service_with(context_tokens=gateway_limit), fake=fake) as stack,
                httpx.AsyncClient(timeout=10) as client):
        models = (await call(client, "GET", stack.public + "/v1/models")).json()
        state = (await call(client, "GET", stack.public + "/v1/state")).json()
        fake.script = Script(input_tokens=effective - 64)
        fits = await generate(client, stack, chat_body(max_tokens=64))
        fake.script = Script(input_tokens=effective - 63)
        too_long = await generate(client, stack, chat_body(max_tokens=64))
    assert models["data"][0]["max_model_len"] == effective == state["context_tokens"]
    assert fits.done
    assert_refused(too_long, 400, "context_limit")


async def test_the_service_is_starting_until_the_engine_is_verified() -> None:
    fake = FakeEngine(ALIAS, SERVICE["context_tokens"])
    fake.healthy = False
    async with (running(fake=fake, wait_ready=False, health_interval_s=0.05) as stack,
                httpx.AsyncClient(timeout=10) as client):
        await until(lambda: len(fake.calls_of("health")) >= 2)
        state = (await call(client, "GET", stack.public + "/v1/state")).json()
        assert (state["status"], state["context_tokens"]) == ("starting", None)
        assert_refused(await generate(client, stack, chat_body()), 503, "starting")
        assert_refused(await count(client, stack, chat_body()), 503, "starting")
        assert_refused(await call(client, "GET", stack.public + "/v1/models"), 503, "starting")
        assert fake.calls_of("tokenize", "generate") == []
        fake.healthy = True
        await until(lambda: stack.service.status == "ready")
        assert (await generate(client, stack, chat_body())).done


async def test_an_engine_that_serves_another_model_is_not_verified() -> None:
    fake = FakeEngine("another-model", SERVICE["context_tokens"])
    async with (running(fake=fake, wait_ready=False, health_interval_s=0.05) as stack,
                httpx.AsyncClient(timeout=10) as client):
        await until(lambda: len(fake.calls_of("models")) >= 2)
        assert stack.service.status == "starting"
        assert_refused(await generate(client, stack, chat_body()), 503, "starting")


async def test_an_engine_that_stops_answering_fails_the_service_and_lets_accepted_work_go_on() -> None:
    async with running(health_interval_s=0.05) as stack, httpx.AsyncClient(timeout=10) as client:
        prompt_read = asyncio.Event()
        stack.fake.script = Script(hold_first=prompt_read)
        accepted = asyncio.ensure_future(generate(client, stack, chat_body()))
        await until(lambda: bool(stack.fake.calls_of("generate")))
        stack.fake.healthy = False
        await until(lambda: stack.service.status == "failed")
        state = (await call(client, "GET", stack.public + "/v1/state")).json()
        assert (state["status"], state["context_tokens"]) == ("failed", SERVICE["context_tokens"])
        assert_refused(await generate(client, stack, chat_body()), 503, "engine_unavailable")
        assert_refused(await count(client, stack, chat_body()), 503, "engine_unavailable")
        prompt_read.set()
        assert (await accepted).done
        stack.fake.healthy = True
        await until(lambda: stack.service.status == "ready")
        assert (await generate(client, stack, chat_body())).done


@pytest.mark.parametrize("change", ["another model", "another context"])
async def test_an_engine_that_changes_under_a_verified_boot_fails_it_and_ends_all_its_work(change: str) -> None:
    async with running(health_interval_s=0.05) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = by_prompt({"streams": Script(events=[{"content": "On. "}], endless=True, delay_s=0.01),
                                       "in its prompt": Script(hold_first=asyncio.Event()),
                                       "counted": Script(hold_tokenize=asyncio.Event())})
        streaming = asyncio.ensure_future(generate(client, stack, user_body("streams"), headers=reader()))
        stream_call = await reached(stack, "streams")
        await until(lambda: stream_call.events_sent > 0)
        work = [asyncio.ensure_future(generate(client, stack, user_body("in its prompt"), headers=AGENT))]
        await reached(stack, "in its prompt")
        work.append(asyncio.ensure_future(generate(client, stack, user_body("waits"), headers=AGENT)))
        await until(lambda: stack.service.admission.waiting_counts()["agent"] == 1)  # one agent place
        work.append(asyncio.ensure_future(count(client, stack, user_body("counted"), headers=reader())))
        await reached(stack, "counted", "tokenize")

        verified = stack.fake.served_name, stack.fake.max_model_len
        if change == "another model":
            stack.fake.served_name = "another-model"
        else:  # a longer one, past the gateway's own limit: the engine has changed all the same
            stack.fake.max_model_len += 1024
        for answer in await asyncio.gather(*work):
            assert_refused(answer, 503, "engine_unavailable")
        streamed = await streaming
        assert (streamed.status, streamed.error_event, streamed.done) == (200, "engine_unavailable", False)
        assert stack.service.status == "failed" and not stack.service.work
        assert_refused(await generate(client, stack, chat_body()), 503, "engine_unavailable")
        stack.fake.served_name, stack.fake.max_model_len = verified
        await until(lambda: stack.service.status == "ready")
        assert (await generate(client, stack, chat_body())).done


async def test_a_drain_outranks_the_engines_status() -> None:
    async with running(health_interval_s=0.05) as stack, httpx.AsyncClient(timeout=10) as client:
        boot = stack.service.boot_id
        stack.fake.healthy = False
        await until(lambda: stack.service.status == "failed")
        drained = await control(client, stack, "/v1/control/drain", {"boot_id": boot})
        assert (drained.status, drained.json()["status"]) == (202, "drained")
        stack.fake.healthy = True
        await until(lambda: stack.service.engine_status == "ready")
        assert (await call(client, "GET", stack.public + "/v1/state")).json()["status"] == "drained"
        assert_refused(await generate(client, stack, chat_body()), 503, "drained")
        opened = await control(client, stack, "/v1/control/open", {"boot_id": boot, "drain_generation": 1})
        assert opened.json()["status"] == "ready"


async def test_connections_past_the_cap_are_refused_and_closed() -> None:
    async with running(max_connections=2) as stack:
        clients = [httpx.AsyncClient(timeout=10) for _ in range(3)]
        try:
            for client in clients[:2]:  # each keeps its connection open afterwards
                assert (await call(client, "GET", stack.public + "/v1/state")).status == 200
            third = await call(clients[2], "GET", stack.public + "/v1/state")
            assert_refused(third, 429, "queue_full")
            assert third.headers["connection"] == "close"
            await clients[0].aclose()
            for _ in range(200):  # until the server has seen the first connection close
                if (await call(clients[2], "GET", stack.public + "/v1/state")).status == 200:
                    break
                await asyncio.sleep(0.005)
            else:
                raise AssertionError("a closed connection did not free its room")
        finally:
            for client in clients:
                await client.aclose()


async def test_the_engine_receives_only_what_renders_the_prompt_and_what_the_class_sets() -> None:
    schema = {"type": "object", "properties": {"door": {"type": "string"}}, "required": ["door"]}
    body = chat_body(temperature=0.2, top_p=0.95, top_k=64, min_p=0, repetition_penalty=1.1, seed=7,
                     response_format={"type": "json_schema",
                                      "json_schema": {"name": "door", "strict": True, "schema": schema}},
                     chat_template_kwargs={"enable_thinking": True})
    client_headers = {**reader(), "X-Request-Id": "synthetic-7", "Priority": "u=0"}
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        assert (await generate(client, stack, body, headers=client_headers)).done
    (tokenize,), (generation,) = stack.fake.calls_of("tokenize"), stack.fake.calls_of("generate")
    template = {"add_generation_prompt": True, "add_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": True}}
    assert tokenize.body == {"model": ALIAS, "messages": body["messages"], **template}
    salt = generation.body.pop("cache_salt")
    assert isinstance(salt, str) and len(salt) >= 32
    assert generation.body == {
        "model": ALIAS, "messages": body["messages"], "max_tokens": 64, "stream": True,
        "stream_options": {"include_usage": True}, "temperature": 0.2, "top_p": 0.95, "top_k": 64, "min_p": 0,
        "repetition_penalty": 1.1, "seed": 7, "response_format": body["response_format"], "priority": 0, **template}
    client_only = {name.lower() for name in ("Authorization", CLASS, SCOPE, *client_headers)}
    for engine_call in (tokenize, generation):
        assert not client_only & set(engine_call.headers)


async def test_without_optional_fields_thinking_is_off_and_sampling_is_left_to_the_engine() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        assert (await generate(client, stack, chat_body())).done
    (tokenize,), (generation,) = stack.fake.calls_of("tokenize"), stack.fake.calls_of("generate")
    thinking_off = {"enable_thinking": False}
    assert tokenize.body["chat_template_kwargs"] == generation.body["chat_template_kwargs"] == thinking_off
    assert set(generation.body) == {"model", "messages", "max_tokens", "stream", "stream_options", "priority",
                                    "cache_salt", "add_generation_prompt", "add_special_tokens", "chat_template_kwargs"}
    assert generation.body["priority"] == SERVICE["engine_priority"]["internal"]


async def test_a_body_nested_past_the_limit_is_invalid_even_inside_the_schema() -> None:
    def with_schema(schema: Any) -> dict[str, Any]:
        return chat_body(response_format={"type": "json_schema",
                                          "json_schema": {"name": "deep", "strict": True, "schema": schema}})

    def nested(levels: int) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for _ in range(levels - 1):
            value = {"a": value}
        return value

    # The body, `response_format` and `json_schema` are the three levels above the schema.
    deepest = with_schema(nested(MAX_DEPTH - 3))
    far_too_deep = json.dumps(with_schema({"a": "hole"})).replace('"hole"', "[" * 50_000 + "]" * 50_000)
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        assert (await generate(client, stack, deepest)).done
        assert_refused(await generate(client, stack, with_schema(nested(MAX_DEPTH - 2))), 400, "invalid_request")
        answer = await call(client, "POST", stack.public + "/v1/chat/completions", content=far_too_deep.encode())
        assert_refused(answer, 400, "invalid_request")
    (generation,) = stack.fake.calls_of("generate")
    assert generation.body["response_format"] == deepest["response_format"]


async def test_a_route_or_method_outside_section_3_is_not_found() -> None:
    elsewhere = {"public": [("GET", "/v1/chat/completions"), ("PUT", "/v1/chat/completions/input_tokens"),
                            ("POST", "/v1/models"), ("DELETE", "/v1/state"), ("GET", "/v1/models/"),
                            ("GET", "/v1/control/drain")],
                 "control": [("GET", "/v1/control/drain"), ("POST", "/v1/state"), ("GET", "/v1/models"),
                             ("POST", "/v1/chat/completions")]}
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        for listener, requests in elsewhere.items():
            base = stack.public if listener == "public" else stack.control
            for method, path in requests:
                assert_refused(await call(client, method, base + path, key=CONTROL), 404, "not_found")


async def test_the_control_key_sees_places_by_class_and_the_pinned_versions() -> None:
    gate = asyncio.Event()
    async with running(versions={"vllm": "0.0.0-synthetic"}) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = Script(hold_first=gate)
        turns = [asyncio.ensure_future(generate(client, stack, user_body(f"Agent turn {number}."),
                                                headers={CLASS: "agent"})) for number in (1, 2)]
        await until(lambda: stack.service.admission.waiting_counts()["agent"] == 1)
        expected = dict.fromkeys(CLASSES, 0) | {"agent": 1}
        for base in (stack.public, stack.control):
            state = (await call(client, "GET", base + "/v1/state", key=CONTROL)).json()
            assert (state["active"], state["waiting"]) == (expected, expected)
            assert state["versions"]["vllm"] == "0.0.0-synthetic"
            assert {"fastapi", "starlette", "uvicorn", "httpx", "python"} <= set(state["versions"])
        for base in (stack.public, stack.control):
            state = (await call(client, "GET", base + "/v1/state")).json()
            assert set(state) == {"contract", "boot_id", "status", "model", "context_tokens", "drain_generation"}
        gate.set()
        assert all(answer.done for answer in await asyncio.gather(*turns))


ENGINE_HEAD = {"id": "chatcmpl-engine", "object": "chat.completion.chunk", "created": 0, "model": ALIAS}
NOT_TEXT_CHUNKS = {  # of the served model
    "an empty event": {},
    "another model": {**ENGINE_HEAD, "model": "WRONG", "choices": [{"index": 0, "delta": {"content": "Hi."},
                                                                     "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 9, "completion_tokens": 1}},
    "a tool call": {**ENGINE_HEAD, "choices": [{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
        {"index": 0, "id": "call-1", "type": "function", "function": {"name": "open_door", "arguments": "{}"}}]}}]},
    "a function call": {**ENGINE_HEAD, "choices": [{"index": 0, "finish_reason": None, "delta": {
        "function_call": {"name": "open_door", "arguments": "{}"}}}]},
}


@pytest.mark.parametrize("name", NOT_TEXT_CHUNKS)
async def test_an_engine_event_that_is_not_a_text_chunk_of_the_served_model_is_engine_unavailable(name: str) -> None:
    broken = {"raw_hex": json.dumps(NOT_TEXT_CHUNKS[name]).encode().hex()}
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = Script(events=[broken])
        assert_refused(await generate(client, stack, chat_body()), 503, "engine_unavailable")
        stack.fake.script = Script(events=[{"role": "assistant"}, {"content": "The keeper "}, broken,
                                           {"finish_reason": "stop"}])
        later = await generate(client, stack, chat_body())
    assert (later.status, later.error_event, later.done) == (200, "engine_unavailable", False)


@pytest.fixture
def logged() -> Iterator[io.StringIO]:
    """Everything logged during the test, formatted as the gateway writes its log."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(log.JsonFormatter())
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def fail(*args: Any, **kwargs: Any) -> Any:
    raise RuntimeError(f"a synthetic failure that quotes {MARKER}")


def assert_logged_class_only(logged: io.StringIO) -> None:
    """The request's row names the exception's class, and no line holds its text."""
    rows = [json.loads(line) for line in logged.getvalue().splitlines()]
    assert MARKER not in logged.getvalue()
    assert any(row.get("event") == "request" and row.get("code") == "internal_error"
               and row.get("exception") == "RuntimeError" for row in rows), rows


async def test_an_error_before_the_stream_answers_internal_error_and_frees_the_place(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    monkeypatch.setattr(Service, "payload", fail)
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        assert_refused(await generate(client, stack, chat_body()), 500, "internal_error")
        await until(lambda: not stack.service.work)
        assert stack.service.admission.active_counts() == dict.fromkeys(CLASSES, 0)
    assert_logged_class_only(logged)


async def test_an_error_after_the_stream_started_ends_it_with_internal_error(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    monkeypatch.setattr(Translator, "usage_chunk", fail)
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        answer = await generate(client, stack, chat_body())
    assert (answer.status, answer.error_event, answer.done) == (200, "internal_error", False)
    assert_logged_class_only(logged)


async def test_an_error_in_a_route_answers_internal_error(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    monkeypatch.setattr(Service, "state", fail)
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        assert_refused(await call(client, "GET", stack.public + "/v1/state"), 500, "internal_error")
        assert_refused(await call(client, "GET", stack.control + "/v1/state", key=CONTROL), 500, "internal_error")
    assert_logged_class_only(logged)


async def test_an_error_in_the_error_handler_still_gets_an_answer(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    monkeypatch.setattr(Service, "state", fail)
    monkeypatch.setattr(app, "_unexpected_error", fail)  # before the gateway builds its applications
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        assert_refused(await call(client, "GET", stack.public + "/v1/state"), 500, "internal_error")
    assert_logged_class_only(logged)


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


async def stalled_terminal(monkeypatch: pytest.MonkeyPatch, logged: io.StringIO, script: Script, start: bytes,
                           ending: str) -> tuple[bytes, dict[str, Any]]:
    """An internal generation whose terminal message stalls at `start`, until its wall time or a drain's deadline ends
    it. Returns what its client received and the request's log row."""
    order = record_abort_order(monkeypatch)
    began = stall_the_send_of(monkeypatch, start)
    wall_s, deadline_s = (0.4, 60.0) if ending == "timeout" else (60.0, 0.4)
    block = service_with(limits={"internal": {"wall_s": wall_s}}, drain_deadline_s=deadline_s)
    async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = script
        since = time.monotonic()
        raw = await RawClient.post(stack.public, "/v1/chat/completions", chat_body(), headers={CLASS: "internal"})
        await eventually(began.wait())
        # The engine request is closed and its place freed before the terminal message: only the send waits.
        assert order == ["closed", "freed"]
        assert stack.service.admission.active_counts()["internal"] == 0 and stack.service.work
        if ending == "drain deadline":
            since = time.monotonic()
            await control(client, stack, "/v1/control/drain", {"boot_id": stack.service.boot_id})
            assert stack.service.status == "draining"  # our work may finish until the deadline
        await until(lambda: not stack.service.work)
        assert time.monotonic() - since >= min(wall_s, deadline_s)
        assert stack.service.status == ("ready" if ending == "timeout" else "drained")
        received = await raw.read_to_end()  # the server closed the connection
    rows = [json.loads(line) for line in logged.getvalue().splitlines()]
    (row,) = [row for row in rows if row.get("route") == "/v1/chat/completions"]
    return received, row


@pytest.mark.parametrize("ending", ["timeout", "drain deadline"])
async def test_a_done_the_client_does_not_read_holds_no_place_and_a_stop_still_ends_it(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO, ending: str) -> None:
    received, row = await stalled_terminal(monkeypatch, logged, Script(), b"data: [DONE]", ending)
    assert b'"usage"' in received and b"[DONE]" not in received and b'"error"' not in received
    assert (row["status"], row["finish"], row["cancelled"], row.get("code")) == (200, "stop", True, None)


@pytest.mark.parametrize("ending", ["timeout", "drain deadline"])
async def test_an_error_event_the_client_does_not_read_holds_no_place_and_a_stop_still_ends_it(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO, ending: str) -> None:
    script = Script(events=[{"role": "assistant"}, {"content": "The keeper "}, {"error": {"message": "synthetic"}}])
    received, row = await stalled_terminal(monkeypatch, logged, script, b'data: {"error"', ending)
    assert b"The keeper" in received and b'"error"' not in received and b"[DONE]" not in received
    assert (row["status"], row["code"], row["cancelled"]) == (200, "engine_unavailable", True)

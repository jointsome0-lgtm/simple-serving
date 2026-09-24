"""The gateway's parts on their own: validation, keys and scopes, places, configuration, the stream and the log."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from functools import partial
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving import log, vast
from simple_serving.admission import Admission, CountPlaces
from simple_serving.config import ConfigError, Listener, from_service_block, load
from simple_serving.engine import EngineError, refusal
from simple_serving.errors import STATUS, ServiceError
from simple_serving.policy import cache_salt, find_key, resolve
from simple_serving.stream import Translator, Usage
from simple_serving.validation import check_chat, check_drain, check_open, parse_json

from .support import (
    ALIAS,
    BOT,
    CONTROL,
    OUTSIDE_A,
    OUTSIDE_B,
    SERVICE,
    chat_body,
    service_with,
)

MARKER = "Mk7c0ffee"


def config(**changes: Any) -> Any:
    return from_service_block(service_with(**changes), engine_url="http://127.0.0.1:1")


def code_of(action: Any) -> str:
    with pytest.raises(ServiceError) as caught:
        action()
    return caught.value.code


# Validation (contract section 4)

def chat(**patch: Any) -> Any:
    return check_chat(parse_json(json.dumps(chat_body(**patch)).encode()), ALIAS)


@pytest.mark.parametrize("patch", [
    {"max_tokens": 1.0}, {"max_tokens": "64"}, {"max_tokens": 0},
    {"temperature": True}, {"temperature": -0.1}, {"temperature": 2.5},
    {"top_p": 0}, {"top_p": 1.5}, {"top_k": 0}, {"top_k": 1.0}, {"min_p": 1.1}, {"repetition_penalty": 0},
    {"repetition_penalty": 2.1}, {"seed": False}, {"seed": 1.5}, {"stream": 1},
    {"stream_options": {"include_usage": 1}}, {"stream_options": {}}, {"model": "test-model "},
    {"messages": []}, {"messages": [{"role": "user"}]},
    {"messages": [{"role": "user", "content": ["x"]}]}, {"messages": "x"},
    {"response_format": {"type": "json_object"}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": "r", "strict": 1, "schema": {}}}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": []}}},
    {"chat_template_kwargs": {"enable_thinking": "false"}},
])
def test_wrong_types_and_ranges_are_invalid(patch: dict[str, Any]) -> None:
    assert code_of(lambda: chat(**patch)) == "invalid_request"


@pytest.mark.parametrize("patch", [
    {"n": 1}, {"logprobs": False},
    {"stream_options": {"include_usage": True, "continuous_usage_stats": True}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": {}},
                         "extra": 1}},
    {"response_format": {"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": {},
                                                                "description": "x"}}},
    {"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}},
])
def test_fields_outside_the_contract_are_unsupported_at_any_depth(patch: dict[str, Any]) -> None:
    assert code_of(lambda: chat(**patch)) == "unsupported_field"


def test_valid_bodies_pass_as_they_are() -> None:
    schema = {"type": "object", "anything": {"goes": ["here", 1, None]}, "additionalProperties": False}
    request = chat(temperature=0, top_p=1, top_k=64, min_p=0.0, repetition_penalty=2, seed=-3,
                   response_format={"type": "json_schema", "json_schema": {"name": "r", "strict": False,
                                                                           "schema": schema}},
                   chat_template_kwargs={"enable_thinking": True})
    assert request.sampling == {"temperature": 0, "top_p": 1, "top_k": 64, "min_p": 0.0, "repetition_penalty": 2,
                                "seed": -3}
    assert request.response_format["json_schema"]["schema"] == schema
    assert request.enable_thinking is True
    assert chat().enable_thinking is False and chat().sampling == {} and chat().response_format is None


NOT_PLAIN = {
    "NaN": b'{"model": NaN}', "infinity": b'{"model": Infinity}', "overflow": b'{"max_tokens": 1e999}',
    "repeated key": b'{"model": "a", "model": "a"}', "not UTF-8": b"\xff\xfe{}", "empty": b"",
    "trailing data": b'{"model": "x"} trailing', "too deep": b"[" * 65 + b"]" * 65,
    "deeper than the encoder goes": b"[" * 50_000 + b"]" * 50_000,
    "deeper than the parser goes": b"[" * 1_000_000 + b"]" * 1_000_000,
}


@pytest.mark.parametrize("raw", NOT_PLAIN.values(), ids=NOT_PLAIN.keys())
def test_json_that_is_not_plain_is_invalid(raw: bytes) -> None:
    assert code_of(lambda: parse_json(raw)) == "invalid_request"


def test_json_nested_as_deep_as_allowed() -> None:
    assert parse_json(b'{"a":' * 63 + b"[]" + b"}" * 63) is not None


def test_control_bodies() -> None:
    assert check_drain({"boot_id": "b"}) == "b"
    assert check_open({"boot_id": "b", "drain_generation": 0}) == ("b", 0)
    assert code_of(lambda: check_drain({"boot_id": "b", "force": True})) == "unsupported_field"
    assert code_of(lambda: check_drain({"boot_id": 1})) == "invalid_request"
    assert code_of(lambda: check_open({"boot_id": "b", "drain_generation": True})) == "invalid_request"
    assert code_of(lambda: check_open({"boot_id": "b", "drain_generation": -1})) == "invalid_request"
    assert code_of(lambda: check_open({"boot_id": "b"})) == "invalid_request"


# Keys, classes and scopes (section 2)

def test_keys_are_found_by_their_hash_only() -> None:
    settings = config()
    bot, outside = find_key(settings, f"Bearer {BOT}".encode()), find_key(settings, f"bearer {OUTSIDE_A}".encode())
    assert bot is not None and bot.label == "bot"
    assert outside is not None and outside.label == "outside-a"
    for header in (None, b"", BOT.encode(), f"Basic {BOT}".encode(), b"Bearer ", f"Bearer {BOT}x".encode()):
        assert find_key(settings, header) is None
    assert all(len(key.digest) == 32 for key in settings.keys)


def test_classes_and_scopes() -> None:
    keys = {key.label: key for key in config().keys}
    bot, outside = keys["bot"], keys["outside-a"]
    assert resolve(bot, None, None).cls == "internal"
    assert (resolve(bot, "reader", "reader.r1aaaaaa").scope, resolve(bot, "agent", None).scope) == ("reader.r1aaaaaa",
                                                                                                    "agent")
    assert resolve(bot, "internal", "agent").scope == "agent"
    assert resolve(outside, None, None).scope is None and resolve(outside, None, None).scope_kind == "external"
    assert code_of(lambda: resolve(outside, "reader", None)) == "class_not_allowed"
    assert code_of(lambda: resolve(bot, "external", None)) == "class_not_allowed"
    assert code_of(lambda: resolve(bot, "Reader", None)) == "class_not_allowed"
    assert code_of(lambda: resolve(keys["control"], None, None)) == "class_not_allowed"
    assert code_of(lambda: resolve(outside, None, "reader.r1aaaaaa")) == "scope_not_allowed"
    for scope in ("reader.short", "reader." + "a" * 65, "reader.r1aa aaaa", "reader.r1aaaaa!", "external", "x"):
        assert code_of(partial(resolve, bot, "internal", scope)) == "invalid_request", scope
    assert code_of(lambda: resolve(bot, "reader", None)) == "invalid_request"
    assert code_of(lambda: resolve(bot, "reader", "internal")) == "invalid_request"


def test_salts_keep_scopes_and_outside_keys_apart() -> None:
    block = service_with()
    block["keys"]["test-key-outside-agent"] = {"label": "agent", "classes": ["external"], "default": "external",
                                               "scopes": False, "control": False}
    keys = {key.label: key for key in from_service_block(block, engine_url="http://127.0.0.1:1").keys}
    secret = bytes(32)
    salts = {
        "reader r1": cache_salt(secret, resolve(keys["bot"], "reader", "reader.r1aaaaaa")),
        "reader r2": cache_salt(secret, resolve(keys["bot"], "reader", "reader.r2bbbbbb")),
        "agent": cache_salt(secret, resolve(keys["bot"], "agent", None)),
        "internal": cache_salt(secret, resolve(keys["bot"], "internal", None)),
        "outside-a": cache_salt(secret, resolve(keys["outside-a"], None, None)),
        "outside-b": cache_salt(secret, resolve(keys["outside-b"], None, None)),
        "an outside key labelled agent": cache_salt(secret, resolve(keys["agent"], None, None)),
    }
    assert len(set(salts.values())) == len(salts)
    assert cache_salt(secret, resolve(keys["bot"], "internal", "agent")) == salts["agent"]  # the scope decides
    assert cache_salt(bytes([1]) * 32, resolve(keys["bot"], "agent", None)) != salts["agent"]  # a new start
    assert all(len(salt) == 43 for salt in salts.values())


# Places (sections 5 and 7)

@pytest.mark.anyio
async def test_admission_keeps_reader_places_apart_and_orders_the_shared_ones() -> None:
    settings = config()
    keys = {key.label: key for key in settings.keys}
    admission = Admission(settings)
    shared = [admission.enter("internal", None), admission.enter("internal", None), admission.enter("agent", None)]
    assert all(ticket.active for ticket in shared)
    outside = admission.enter("external", keys["outside-a"])
    late_agent = admission.enter("agent", None)
    readers = [admission.enter("reader", None) for _ in range(4)]
    assert outside.waiting and late_agent.waiting and all(ticket.active for ticket in readers)
    assert admission.enter("reader", None).waiting  # past the readers' own four places

    admission.leave(shared[0])  # an internal place frees a shared place: the agent is first, but at its cap
    assert outside.active and late_agent.waiting
    admission.leave(shared[2])
    assert late_agent.active
    assert admission.active_counts() == {"reader": 4, "agent": 1, "internal": 1, "external": 1}
    assert admission.waiting_counts() == {"reader": 1, "agent": 0, "internal": 0, "external": 0}


@pytest.mark.anyio
async def test_admission_caps_for_a_class_and_for_one_outside_key() -> None:
    settings = config()
    keys = {key.label: key for key in settings.keys}
    admission = Admission(settings)
    agents = [admission.enter("agent", None) for _ in range(3)]
    assert [ticket.state for ticket in agents] == ["active", "waiting", "waiting"]
    assert code_of(lambda: admission.enter("agent", None)) == "queue_full"
    first = admission.enter("external", keys["outside-a"])
    queued = [admission.enter("external", keys["outside-a"]) for _ in range(2)]
    assert first.active and all(ticket.waiting for ticket in queued)
    assert code_of(lambda: admission.enter("external", keys["outside-a"])) == "queue_full"
    assert admission.enter("external", keys["outside-b"]).active  # the other key has room, and so has the class
    admission.leave(queued[0])  # a waiting ticket that leaves frees its place in line
    assert admission.enter("external", keys["outside-a"]).waiting
    admission.leave(first)
    assert queued[1].active


@pytest.mark.anyio
async def test_count_places_wait_in_class_order_and_cap_one_outside_key() -> None:
    settings = config()
    keys = {key.label: key for key in settings.keys}
    places = CountPlaces(settings)
    running = [places.enter("external", keys["outside-a"]), places.enter("external", keys["outside-a"]),
               places.enter("internal", None)]
    assert all(ticket.active for ticket in running)
    assert code_of(lambda: places.enter("external", keys["outside-a"])) == "queue_full"
    internal, reader = places.enter("internal", None), places.enter("reader", None)
    assert internal.waiting and reader.waiting and places.waiting == 2
    places.leave(running[0])
    assert reader.active and internal.waiting
    places.leave(reader)
    assert internal.active
    places.leave(internal)
    places.leave(internal)  # leaving twice changes nothing
    assert places.running == 2 and places.enter("external", keys["outside-a"]).active


@pytest.mark.anyio
async def test_a_stopped_request_granted_a_place_frees_it() -> None:
    admission = Admission(config())
    holders = [admission.enter("agent", None)]
    waiter = admission.enter("agent", None)
    waiter.granted.cancel()  # its task was cancelled while it waited
    admission.leave(holders[0])  # the place goes to it all the same
    assert waiter.active
    admission.leave(waiter)
    assert admission.active_counts()["agent"] == 0
    await asyncio.sleep(0)


# Configuration

def test_the_configuration_file_holds_hashes_and_refuses_what_it_does_not_know(tmp_path: Path) -> None:
    block = service_with()
    keys = [{"sha256": hashlib.sha256(raw.encode()).hexdigest(), **entry} for raw, entry in block["keys"].items()]
    data = {**block, "keys": keys, "engine_url": "http://127.0.0.1:8000/",
            "listen": {"public": {"host": "0.0.0.0", "port": 8443}, "control": {"host": "127.0.0.1", "port": 9000}}}
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(data))
    loaded = load(str(path))
    assert loaded.engine_url == "http://127.0.0.1:8000" and loaded.public.port == 8443
    assert {key.label for key in loaded.keys} == {"bot", "control", "outside-a", "outside-b"}
    control_key = find_key(loaded, f"Bearer {CONTROL}".encode())
    assert control_key is not None and control_key.control

    def refused(change: Any) -> str:
        broken = json.loads(json.dumps(data))
        change(broken)
        path.write_text(json.dumps(broken))
        with pytest.raises(ConfigError) as caught:
            load(str(path))
        return str(caught.value)

    assert "engine" in refused(lambda d: d["limits"].update(engine={"active": 8}))  # the old name of shared
    assert "limits.reader" in refused(lambda d: d["limits"]["reader"].update(activ=4))
    assert "sha256" in refused(lambda d: d["keys"][0].update(sha256=BOT))
    assert "labels" in refused(lambda d: d["keys"][1].update(label="bot"))
    assert "outside key" in refused(lambda d: d["keys"][2].update(scopes=True))
    assert "outside key" in refused(lambda d: d["keys"][2].update(classes=["external", "internal"]))
    assert "default" in refused(lambda d: d["keys"][0].update(default="external"))
    assert "wall_s" in refused(lambda d: d["limits"]["agent"].update(wall_s=0))
    assert "active" in refused(lambda d: d["limits"]["agent"].update(active=True))
    message = refused(lambda d: d.update(secret=MARKER))
    assert "secret" in message and MARKER not in message  # a message names a field, never its value
    path.write_text("{" + MARKER)
    with pytest.raises(ConfigError) as caught:
        load(str(path))
    assert MARKER not in str(caught.value)


def test_the_control_listener_and_the_engine_are_on_loopback_ip_addresses() -> None:
    def refused(**options: Any) -> str:
        with pytest.raises(ConfigError) as caught:
            from_service_block(service_with(), **{"engine_url": "http://127.0.0.1:8000", **options})
        return str(caught.value)

    for host in ("0.0.0.0", "::", "localhost", "192.168.1.5", "::ffff:127.0.0.1", "::1%lo"):
        assert "listen.control.host" in refused(control=Listener(host, 9000)), host
    for url in ("http://localhost:8000", "http://192.168.1.5:8000", "http://127.0.0.1.nip.io:8000",
                "http://127.0.0.1@example.com:8000", "http://[::1:8000", "ftp://127.0.0.1:8000"):
        assert "engine_url" in refused(engine_url=url), url
    settings = from_service_block(service_with(), engine_url="http://[::1]:8000", public=Listener("0.0.0.0", 443),
                                  control=Listener("127.0.0.2", 9000))
    assert (settings.public.host, settings.control.host) == ("0.0.0.0", "127.0.0.2")
    assert from_service_block(service_with(), engine_url="https://127.8.9.10", control=Listener("::1", 0))


# The engine and its stream

@pytest.mark.parametrize(("status", "code"), [
    (401, "engine_unavailable"), (403, "engine_unavailable"), (429, "engine_unavailable"), (503, "engine_unavailable"),
])
def test_engine_statuses(status: int, code: str) -> None:
    assert refusal(status).code == code


def translate(*events: dict[str, Any]) -> tuple[Translator, list[dict[str, Any]]]:
    translator = Translator(ALIAS)
    chunks = [chunk for event in events for chunk in translator.chunks(event)]
    return translator, [{"delta": c["choices"][0]["delta"], "finish": c["choices"][0]["finish_reason"]}
                        for c in chunks]


def event(**fields: Any) -> dict[str, Any]:
    """An engine event as vLLM sends it."""
    return {"id": "chatcmpl-engine", "object": "chat.completion.chunk", "created": 0, "model": ALIAS, **fields}


def choice(delta: dict[str, Any] | None = None, finish: str | None = None) -> dict[str, Any]:
    return event(choices=[{"index": 0, "delta": delta or {}, "finish_reason": finish}])


def test_a_delta_with_several_fields_becomes_one_chunk_for_each() -> None:
    _, chunks = translate(choice({"role": "assistant", "reasoning": "Hm.", "content": "Hi."}, "stop"))
    assert chunks == [{"delta": {"role": "assistant"}, "finish": None},
                      {"delta": {"reasoning_content": "Hm."}, "finish": None},
                      {"delta": {"content": "Hi."}, "finish": "stop"}]


def test_reasoning_under_either_name_and_empty_text() -> None:
    translator, chunks = translate(choice({"reasoning_content": "A.", "reasoning": "A."}),
                                   choice({"role": "assistant", "content": ""}), choice({"content": None}),
                                   choice(None))
    assert chunks == [{"delta": {"reasoning_content": "A."}, "finish": None},
                      {"delta": {"role": "assistant"}, "finish": None},
                      {"delta": {}, "finish": None}, {"delta": {}, "finish": None}]
    assert translator.generated


def test_what_vllm_leaves_out_or_sends_as_null_means_none() -> None:
    text = event(choices=[{"index": 0, "delta": {"content": "Hi.", "reasoning_content": None, "tool_calls": None}}])
    finish = event(choices=[{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop", "stop_reason": None}],
                   usage=None)
    usage = event(choices=[], usage={"prompt_tokens": 3, "completion_tokens": 1, "prompt_tokens_details": None})
    translator, chunks = translate(text, finish, usage)
    assert chunks == [{"delta": {"content": "Hi."}, "finish": None}, {"delta": {}, "finish": "stop"}]
    assert translator.end() == Usage(prompt_tokens=3, completion_tokens=1, cached_tokens=None)


@pytest.mark.parametrize("events", [
    [event(choices=[choice()["choices"][0], choice()["choices"][0]])],
    [event(choices=[{"delta": {"content": "x"}}])],
    [event(choices=[{"index": 0, "finish_reason": "stop"}])],
    [{**choice({"content": "x"}), "object": None}],
    [choice({"role": "user"})],
    [choice({"content": 5})],
    [choice({}, "abort")],
    [choice({}, "stop"), choice({"content": "more"})],
    [{"object": "error", "message": MARKER, "code": 400}],
    [event(choices=[], usage={"prompt_tokens": "1", "completion_tokens": 1})],
    [event(choices=[], usage=5)],
    [event(choices=[], usage={"prompt_tokens": 1, "completion_tokens": 1,
                              "prompt_tokens_details": {"cached_tokens": -1}})],
    [event(choices="x")],
])
def test_events_that_break_the_rules(events: list[dict[str, Any]]) -> None:
    with pytest.raises(EngineError) as caught:
        translate(*events)
    assert caught.value.code == "engine_unavailable" and MARKER not in str(caught.value)


def test_the_end_needs_a_finish_and_usage() -> None:
    usage = event(choices=[], usage={"prompt_tokens": 5, "completion_tokens": 2,
                                     "prompt_tokens_details": {"cached_tokens": None}})
    translator, _ = translate(choice({"content": "x"}, "stop"), usage)
    assert translator.end().cached_tokens is None
    chunk = translator.usage_chunk(translator.end(), {"wait_ms": 0, "first_token_ms": 1, "total_ms": 2})
    assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 2,
                              "simple_serving": {"wait_ms": 0, "first_token_ms": 1, "total_ms": 2}}
    for events in ([choice({"content": "x"}, "stop")], [choice({"content": "x"}), usage]):
        with pytest.raises(EngineError):
            translate(*events)[0].end()


# The log (section 10)

def formatted(record: logging.LogRecord) -> dict[str, Any]:
    return json.loads(log.JsonFormatter().format(record))


def test_log_rows_keep_named_fields_and_the_class_of_an_exception_only() -> None:
    try:
        raise ValueError(MARKER)
    except ValueError as error:
        exc_info = (type(error), error, error.__traceback__)
    ours = logging.LogRecord("simple_serving", logging.INFO, __file__, 1, "request", None, exc_info)
    ours.fields = {"event": "request", "code": "timeout", "body": MARKER, "messages": [MARKER]}
    row = formatted(ours)
    assert row["event"] == "request" and row["code"] == "timeout" and row["exception"] == "ValueError"
    assert MARKER not in json.dumps(row) and "body" not in row
    other = logging.LogRecord("httpx", logging.WARNING, __file__, 1, "sent %s", (MARKER,), exc_info)
    assert MARKER not in json.dumps(formatted(other))
    server = logging.LogRecord("uvicorn.error", logging.ERROR, __file__, 1, "Exception in %s", (MARKER,), None)
    assert formatted(server) == {"time": formatted(server)["time"], "logger": "uvicorn.error", "level": "ERROR"}


def test_request_rows_drop_nothing_but_missing_values() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(log.JsonFormatter())
    level = log.LOGGER.level
    log.LOGGER.addHandler(handler)
    log.LOGGER.setLevel(logging.INFO)
    try:
        log.RequestRecord(listener="public", count_matches=False, cancelled=False).emit()
    finally:
        log.LOGGER.removeHandler(handler)
        log.LOGGER.setLevel(level)
    row = json.loads(stream.getvalue())
    assert row["count_matches"] is False and row["cancelled"] is False and "code" not in row


def test_every_code_has_its_status() -> None:
    assert STATUS["internal_error"] == 500 and STATUS["queue_full"] == 429 and STATUS["timeout"] == 504
    with pytest.raises(ValueError):
        ServiceError(MARKER)


def test_the_service_block_of_the_cases_is_a_valid_configuration() -> None:
    settings = config()
    assert settings.shared_places == SERVICE["limits"]["shared"]["active"]
    assert settings.context_tokens == 4096 and settings.alias == ALIAS
    assert {key.label: key.outside for key in settings.keys} == {"bot": False, "control": False, "outside-a": True,
                                                                 "outside-b": True}
    assert OUTSIDE_B not in repr(settings)


# The stop of the instance (section 8)

@pytest.mark.anyio
async def test_a_stop_puts_stopped_to_the_fixed_endpoint_with_the_key_in_a_header() -> None:
    requests: list[httpx.Request] = []

    def accept(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"success": True})

    await vast.stop("123", MARKER, transport=httpx.MockTransport(accept))
    (request,) = requests
    assert (request.method, str(request.url)) == ("PUT", "https://console.vast.ai/api/v0/instances/123/")
    assert request.headers["authorization"] == f"Bearer {MARKER}"
    assert json.loads(request.content) == {"state": "stopped"}


@pytest.mark.anyio
@pytest.mark.parametrize(("answer", "code", "status"), [
    (httpx.Response(200, json={"success": False}), "answer", None),
    (httpx.Response(200, content=b"<html>"), "answer", None),
    (httpx.Response(200, content=b" " * (vast.MAX_ANSWER_BYTES + 1)), "answer", None),
    (httpx.Response(401), "forbidden", 401),
    (httpx.Response(403), "forbidden", 403),
    (httpx.Response(503), "http", 503),
    (httpx.Response(301, headers={"location": "https://console.vast.ai/elsewhere/"}), "http", 301),
    (httpx.ConnectError("synthetic"), "network", None),
    (httpx.ReadTimeout("synthetic"), "timeout", None),
])
async def test_anything_but_success_is_a_failure_of_a_fixed_category(
        answer: httpx.Response | Exception, code: str, status: int | None) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if isinstance(answer, Exception):
            raise answer
        return answer

    with pytest.raises(vast.VastError) as caught:
        await vast.stop("123", MARKER, transport=httpx.MockTransport(respond))
    assert (caught.value.code, caught.value.status, str(caught.value)) == (code, status, code)


def test_a_stop_needs_the_containers_instance_and_its_key() -> None:
    for environ in ({}, {"CONTAINER_ID": "123"}, {"CONTAINER_ID": "0", "CONTAINER_API_KEY": MARKER},
                    {"CONTAINER_ID": "12/", "CONTAINER_API_KEY": MARKER},
                    {"CONTAINER_ID": "123", "CONTAINER_API_KEY": f"{MARKER}\r\n"}):
        assert vast.from_environment(environ) is vast.unconfigured
    assert vast.from_environment({"CONTAINER_ID": "123", "CONTAINER_API_KEY": MARKER}) is not vast.unconfigured

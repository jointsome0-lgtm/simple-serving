"""Every case of contract/cases-v2.json, each on a fresh gateway in front of the fake engine (contract/README.md)."""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from simple_serving.fake_engine import Script

from .support import ALIAS, CASES, Answer, Stack, parse_events, request_headers, running

pytestmark = pytest.mark.anyio
ENGINE_CALLS = ("tokenize", "generate")


@pytest.mark.parametrize("case", CASES["cases"], ids=[case["name"] for case in CASES["cases"]])
async def test_case(case: dict[str, Any]) -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        run = CaseRun(stack, client)
        for number, step in enumerate(case["steps"], 1):
            await run.step(step, where=f"{case['name']}, step {number}")


class CaseRun:
    """The steps of one case, which share a gateway, captured values and the salts of named steps."""

    def __init__(self, stack: Stack, client: httpx.AsyncClient) -> None:
        self.stack = stack
        self.client = client
        self.captured: dict[str, Any] = {}
        self.salts: dict[str, str] = {}

    async def step(self, step: dict[str, Any], where: str) -> None:
        self.stack.fake.script = Script.from_case(step["engine"]) if step["engine"] else None
        first_call = len(self.stack.fake.calls)
        answer = await self.send(step["request"])
        calls = [call for call in self.stack.fake.calls[first_call:] if call.kind in ENGINE_CALLS]
        check_response(step["response"], answer, self.captured, where)
        self.check_engine(step, calls, where)
        for name, key in step.get("capture", {}).items():
            self.captured[name] = answer.json()[key]

    async def send(self, request: dict[str, Any]) -> Answer:
        base = self.stack.public if request["base"] == "public" else self.stack.control
        headers = request_headers(request["key"], request.get("headers"))
        content: Any = None
        if "body_patch" in request:
            body = copy.deepcopy(CASES["defaults"]["chat_body"])
            body.update(request["body_patch"])
            content = json.dumps(body).encode()
        elif "body" in request:
            content = json.dumps(substitute(request["body"], self.captured)).encode()
        elif "raw_body" in request:
            content = request["raw_body"].encode()
        elif "raw_body_bytes" in request:
            size = request["raw_body_bytes"]
            content = chunked(size) if request.get("chunked") else b"a" * size
        if content is not None and not isinstance(content, AsyncIterator):
            headers.setdefault("Content-Type", "application/json")
        async with self.client.stream(request["method"], base + request["path"], headers=headers,
                                      content=content) as response:
            data = await response.aread()
        return Answer(response.status_code, response.headers, data)

    def check_engine(self, step: dict[str, Any], calls: list[Any], where: str) -> None:
        if step["engine"] is None:
            assert calls == [], f"{where}: the engine was called"
            return
        expected = step.get("engine_receives", {})
        assert [call.kind for call in calls] == expected.get("calls", []), f"{where}: engine calls"
        for name in expected.get("headers_absent", []):
            for call in calls:
                assert name.lower() not in call.headers, f"{where}: {name} reached the engine"
        generate = next((call for call in calls if call.kind == "generate"), None)
        if generate is None:
            assert not expected.keys() & {"priority", "cache_salt", "response_format"}, f"{where}: no generate call"
            return
        if "priority" in expected:
            assert generate.body["priority"] == expected["priority"], f"{where}: priority"
        if "cache_salt" in expected:
            salt = generate.body.get("cache_salt")
            assert isinstance(salt, str) and salt, f"{where}: no cache_salt"
            rule = expected["cache_salt"]
            if isinstance(rule, dict) and "same_as" in rule:
                assert salt == self.salts[rule["same_as"]], f"{where}: salt differs from {rule['same_as']}"
            if isinstance(rule, dict) and "differs_from" in rule:
                for other in rule["differs_from"]:
                    assert salt != self.salts[other], f"{where}: salt equals that of {other}"
            if "name" in step:
                self.salts[step["name"]] = salt
        if expected.get("response_format") == "json_schema":
            sent = step["request"]["body_patch"]["response_format"]
            assert generate.body["response_format"] == sent, f"{where}: response_format changed"


def check_response(expected: dict[str, Any], answer: Answer, captured: dict[str, Any], where: str) -> None:
    assert answer.status == expected["status"], f"{where}: status {answer.status}, body {answer.body[:200]!r}"
    if "error" in expected:
        assert answer.json() == {"error": {"code": expected["error"]}}, f"{where}: {answer.body!r}"
    if "json" in expected:
        actual = answer.json()
        if expected.get("json_subset"):
            actual = {key: actual.get(key) for key in expected["json"]}
        compare(expected["json"], actual, captured, f"{where}: body")
    if "chunks" in expected:
        assert answer.headers["content-type"].startswith("text/event-stream"), where
        events = parse_events(answer.body)
        ending = events.pop() if events and (events[-1] == "[DONE]" or "error" in events[-1]) else None
        assert (ending == "[DONE]") == expected["done"], f"{where}: the stream ended with {ending!r}"
        if "error_event" in expected:
            assert ending == {"error": {"code": expected["error_event"]}}, f"{where}: {ending!r}"
        assert len(events) == len(expected["chunks"]), f"{where}: chunks {events!r}"
        for number, (want, chunk) in enumerate(zip(expected["chunks"], events, strict=True), 1):
            assert chunk.get("model") == ALIAS, f"{where}: chunk {number} has model {chunk.get('model')!r}"
            compare(want, comparable(chunk), captured, f"{where}: chunk {number}")


def comparable(chunk: dict[str, Any]) -> dict[str, Any]:
    """What the cases compare of a chunk: `choices`, where a null `finish_reason` counts as absent, and `usage`."""
    choices = [{key: value for key, value in choice.items() if not (key == "finish_reason" and value is None)}
               for choice in chunk.get("choices", [])]
    return {"choices": choices} | ({"usage": chunk["usage"]} if "usage" in chunk else {})


def compare(expected: Any, actual: Any, captured: dict[str, Any], where: str) -> None:
    """Compare with the placeholders of contract/README.md: "present", "measurements" and "$<name>"."""
    if expected == "present":
        assert actual is not None, f"{where}: missing"
    elif expected == "measurements":
        assert isinstance(actual, dict) and set(actual) == {"wait_ms", "first_token_ms", "total_ms"}, where
        assert all(type(value) is int and value >= 0 for value in actual.values()), f"{where}: {actual}"
        assert actual["wait_ms"] <= actual["first_token_ms"] <= actual["total_ms"], f"{where}: {actual}"
    elif isinstance(expected, str) and expected.startswith("$"):
        assert actual == captured[expected[1:]], f"{where}: {actual!r} is not {expected}"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), f"{where}: keys {actual!r}"
        for key, value in expected.items():
            compare(value, actual[key], captured, f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{where}: {actual!r}"
        for index, (want, item) in enumerate(zip(expected, actual, strict=True)):
            compare(want, item, captured, f"{where}[{index}]")
    else:
        assert type(actual) is type(expected) and actual == expected, f"{where}: {actual!r} is not {expected!r}"


def substitute(value: Any, captured: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        return captured[value[1:]]
    if isinstance(value, dict):
        return {key: substitute(item, captured) for key, item in value.items()}
    if isinstance(value, list):
        return [substitute(item, captured) for item in value]
    return value


async def chunked(size: int, piece: int = 65536) -> AsyncIterator[bytes]:
    """`size` bytes of `a`, sent without Content-Length, so the gateway can count them only as they arrive."""
    for start in range(0, size, piece):
        yield b"a" * min(piece, size - start)

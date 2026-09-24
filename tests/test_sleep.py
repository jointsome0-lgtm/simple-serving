"""What keeps the service awake, and how it falls asleep and stops its instance (contract section 8). The clock is a
fake one, so no test waits for the idle interval or a deadline, and the stop goes to a fake Vast."""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import httpx
import pytest

from simple_serving.fake_engine import FakeEngine, Script
from simple_serving.routes import PREPARE_S
from simple_serving.service import FORCED_STOP_S
from simple_serving.vast import ATTEMPT_S, RETRY_S, VastError

from .support import (
    ALIAS,
    CLASS,
    CONTROL,
    HANG,
    OUTSIDE_A,
    Answer,
    FakeClock,
    FakeVast,
    RawClient,
    Stack,
    by_prompt,
    call,
    control,
    count,
    generate,
    reached,
    reader,
    running,
    service_with,
    stall_the_send_of,
    until,
    user_body,
)

pytestmark = pytest.mark.anyio
CHAT, COUNT = "/v1/chat/completions", "/v1/chat/completions/input_tokens"
IDLE_S = 780  # the default idle_timeout_s


async def sleep(client: httpx.AsyncClient, stack: Stack) -> Answer:
    return await control(client, stack, "/v1/control/sleep", {"boot_id": stack.service.boot_id})


def refused(answer: Answer, code: str) -> bool:
    return (answer.status, answer.json()) == (503, {"error": {"code": code}})


def rows(logged: io.StringIO, *events: str) -> list[dict[str, Any]]:
    lines = (json.loads(line) for line in logged.getvalue().splitlines())
    return [row for row in lines if row.get("event") in events]


# Idle.


async def test_an_interval_after_the_first_ready_the_service_falls_asleep_and_a_later_request_is_refused(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running() as stack, httpx.AsyncClient(timeout=None) as client:
        boot = stack.service.boot_id
        await clock.advance(IDLE_S - 2)
        assert not stack.service.sleep_requested and stack.vast.attempts == []
        await clock.advance(2)
        assert stack.service.sleep_requested and stack.service.status == "drained"
        assert refused(await generate(client, stack, user_body("late"), headers=reader()), "drained")
        await until(lambda: stack.vast.stopped)
        state = (await call(client, "GET", stack.control + "/v1/state", key=CONTROL)).json()
        assert (state["boot_id"], state["status"], state["sleep_requested"]) == (boot, "drained", True)
        assert len(stack.vast.attempts) == 1


async def test_a_request_of_ours_that_comes_before_the_idle_decision_holds_the_service_while_its_body_arrives(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running() as stack:
        await clock.advance(IDLE_S - 2)
        slow = await RawClient.post(stack.public, CHAT, user_body("ours"), headers=reader(), missing=10)
        await until(lambda: stack.service.ours_inflight == 1)
        await clock.advance(2)
        assert not stack.service.sleep_requested
        await clock.advance(PREPARE_S - 2)  # its body never ends: the request ends at the bound
        await until(lambda: stack.service.ours_inflight == 0)
        assert (await slow.read_some(5)).startswith(b"HTTP/1.1 504")
        await clock.advance(IDLE_S - 2)  # a full interval runs from its end
        assert not stack.service.sleep_requested
        await clock.advance(2)
        assert stack.service.sleep_requested


async def test_work_of_ours_longer_than_the_interval_holds_the_service_until_it_ends(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    gate = asyncio.Event()
    async with running() as stack, httpx.AsyncClient(timeout=None) as client:
        stack.fake.script = Script(hold_first=gate)
        turn = asyncio.ensure_future(generate(client, stack, user_body("long")))  # internal: 900 s of wall time
        await reached(stack, "long")
        await clock.advance(IDLE_S + 60)
        assert not stack.service.sleep_requested
        gate.set()
        assert (await turn).done
        await clock.advance(IDLE_S - 2)
        assert not stack.service.sleep_requested
        await clock.advance(2)
        assert stack.service.sleep_requested


async def test_outside_work_models_state_and_refusals_before_authorization_never_hold_the_service(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running() as stack, httpx.AsyncClient(timeout=None) as client:
        await clock.advance(IDLE_S - 10)
        assert (await generate(client, stack, user_body("outside"), key=OUTSIDE_A)).done
        assert (await count(client, stack, user_body("outside"), key=OUTSIDE_A)).status == 200
        for url, key in ((stack.public + "/v1/models", "test-key-bot"), (stack.public + "/v1/state", "test-key-bot"),
                         (stack.control + "/v1/state", CONTROL)):
            assert (await call(client, "GET", url, key=key)).status == 200
        assert (await generate(client, stack, user_body("no key"), key=None)).status == 401
        assert (await generate(client, stack, user_body("no scope"), headers={CLASS: "reader"})).status == 400
        await clock.advance(10)
        assert stack.service.sleep_requested


async def test_the_interval_starts_at_the_first_ready_and_a_failed_engine_holds_nothing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    fake = FakeEngine(ALIAS, 4096)
    fake.healthy = False
    async with running(fake=fake, wait_ready=False, health_interval_s=0.05) as stack, \
            httpx.AsyncClient(timeout=None) as client:
        await clock.advance(IDLE_S + 60)
        assert stack.service.status == "starting" and not stack.service.sleep_requested
        fake.healthy = True
        await clock.advance(1)  # the next check of a starting service
        await until(lambda: stack.service.status == "ready")
        fake.healthy = False
        await until(lambda: stack.service.status == "failed")
        assert refused(await generate(client, stack, user_body("refused"), headers=reader()), "engine_unavailable")
        await clock.advance(IDLE_S - 2)
        assert not stack.service.sleep_requested
        await clock.advance(2)
        assert stack.service.sleep_requested
        await until(lambda: stack.vast.stopped)


async def test_our_request_counts_from_before_its_body_and_an_outside_one_never(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running() as stack, httpx.AsyncClient(timeout=None) as client:
        gate = asyncio.Event()
        stack.fake.script = by_prompt({"outside": Script(hold_first=gate)})
        slow = await RawClient.post(stack.public, CHAT, user_body("ours"), headers=reader(), missing=10)
        await until(lambda: stack.service.ours_inflight == 1)  # its body has not all arrived
        outside = asyncio.ensure_future(generate(client, stack, user_body("outside"), key=OUTSIDE_A))
        await reached(stack, "outside")
        assert stack.service.ours_inflight == 1

        await clock.advance(PREPARE_S - 1)
        assert stack.service.ours_inflight == 1
        await clock.advance(1)
        await until(lambda: stack.service.ours_inflight == 0)
        answer = await slow.read_some(5)
        assert answer.startswith(b"HTTP/1.1 504") and answer.endswith(b'{"error": {"code": "timeout"}}')
        ended = stack.service.idle_since
        gate.set()
        assert (await outside).done
        assert (stack.service.ours_inflight, stack.service.idle_since) == (0, ended)


async def test_a_refusal_that_its_client_does_not_read_is_given_up_at_the_bound(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    began = stall_the_send_of(monkeypatch, b'{"error"')
    async with running() as stack:
        await RawClient.post(stack.public, CHAT, user_body("refused", max_tokens=0), headers=reader())
        await began.wait()
        await clock.advance(PREPARE_S - 1)
        assert stack.service.ours_inflight == 1
        await clock.advance(1)
        await until(lambda: stack.service.ours_inflight == 0)


async def test_a_client_that_leaves_in_the_middle_of_a_count_or_a_stream_changes_neither_the_boot_nor_the_sleep(
        ) -> None:
    hold = asyncio.Event()
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        boot = stack.service.boot_id
        endless = Script(events=[{"role": "assistant"}], endless=True, delay_s=0.01)
        stack.fake.script = by_prompt({"counted": Script(hold_tokenize=hold), "streamed": endless})
        counting = await RawClient.post(stack.public, COUNT, user_body("counted"), headers=reader())
        await reached(stack, "counted", "tokenize")
        await counting.leave()
        streaming = await RawClient.post(stack.public, CHAT, user_body("streamed"), headers=reader())
        await streaming.read_until_events(2)
        await streaming.leave()
        await until(lambda: stack.service.ours_inflight == 0)
        assert (stack.service.boot_id, stack.service.sleep_requested, stack.service.status) == (boot, False, "ready")

        streaming = await RawClient.post(stack.public, CHAT, user_body("streamed"), headers=reader())
        await streaming.read_until_events(2)
        assert (await sleep(client, stack)).json()["status"] == "draining"
        await streaming.leave()
        await until(lambda: stack.vast.stopped)
        assert (stack.service.boot_id, stack.service.sleep_requested, stack.service.status) == (boot, True, "drained")


# Manual sleep.


async def test_a_sleep_lets_our_work_finish_refuses_new_work_and_an_open_and_then_stops_the_instance() -> None:
    gate = asyncio.Event()
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        boot = stack.service.boot_id
        stack.fake.script = by_prompt({"ours": Script(hold_first=gate)})
        ours = asyncio.ensure_future(generate(client, stack, user_body("ours"), headers=reader()))
        await reached(stack, "ours")
        answer = await sleep(client, stack)
        assert (answer.status, answer.json()) == (202, {"status": "draining", "boot_id": boot, "drain_generation": 1})
        assert refused(await generate(client, stack, user_body("new"), headers=reader()), "draining")
        opened = await control(client, stack, "/v1/control/open", {"boot_id": boot, "drain_generation": 1})
        assert (opened.status, opened.json()) == (409, {"error": {"code": "sleep_pending"}})
        assert stack.vast.attempts == []
        gate.set()
        assert (await ours).done
        await until(lambda: stack.vast.stopped)
        assert stack.service.status == "drained" and len(stack.vast.attempts) == 1


async def test_a_sleep_whose_answer_was_lost_is_asked_again_and_nothing_starts_twice(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running() as stack, httpx.AsyncClient(timeout=None) as client:
        boot = stack.service.boot_id
        lost = await RawClient.post(stack.control, "/v1/control/sleep", {"boot_id": boot}, key=CONTROL)
        await until(lambda: stack.service.sleep_requested)
        await lost.leave()  # before its answer was read
        for _ in range(2):
            again = await sleep(client, stack)
            assert (again.status, again.json()) == (202, {"status": "drained", "boot_id": boot, "drain_generation": 1})
        await clock.advance(IDLE_S)  # the idle timer is gone with the sleep
        await until(lambda: stack.vast.stopped)
        assert (len(stack.vast.attempts), stack.vast.most) == (1, 1)


# The stop.


async def test_a_stop_that_fails_is_tried_again_one_attempt_at_a_time_and_the_service_never_reopens(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    clock = FakeClock(monkeypatch)
    vast = FakeVast(HANG, VastError("http", 503), VastError("forbidden", 401), 5.0)
    async with running(vast=vast) as stack, httpx.AsyncClient(timeout=None) as client:
        boot = stack.service.boot_id
        await sleep(client, stack)
        for wait in (ATTEMPT_S, RETRY_S, RETRY_S, RETRY_S, 5.0):  # a hang, a 503, a 401, then a success 5 s late
            assert not vast.stopped
            assert refused(await generate(client, stack, user_body("new"), headers=reader()), "drained")
            opened = await control(client, stack, "/v1/control/open", {"boot_id": boot, "drain_generation": 1})
            assert opened.status == 409
            assert (await call(client, "GET", stack.public + "/v1/state")).json()["status"] == "drained"
            await clock.advance(wait)
        await until(lambda: vast.stopped)
        began = vast.attempts[0]
        assert [round(moment - began) for moment in vast.attempts] == [0, 50, 80, 110]
        assert vast.most == 1
    failures = [(row["code"], row.get("status")) for row in rows(logged, "stop_failed")]
    assert failures == [("timeout", None), ("http", 503), ("forbidden", 401)]
    assert len(rows(logged, "stop_accepted")) == 1


async def test_the_instance_stops_at_the_forced_bound_under_a_drain_that_has_not_ended(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    clock = FakeClock(monkeypatch)
    gate = asyncio.Event()
    async with running(service_with(drain_deadline_s=300)) as stack, httpx.AsyncClient(timeout=None) as client:
        stack.fake.script = Script(hold_first=gate)
        turn = asyncio.ensure_future(generate(client, stack, user_body("long")))  # internal: 900 s of wall time
        await reached(stack, "long")
        await sleep(client, stack)
        await clock.advance(FORCED_STOP_S - 2)
        assert stack.vast.attempts == []
        await clock.advance(2)
        await until(lambda: stack.vast.stopped)
        assert stack.service.status == "draining" and [row["event"] for row in rows(logged, "sleep_forced")]
        gate.set()
        await turn

"""The scenarios of contract/cases-v2.json, which a static case cannot describe: cancellation, places, wall time and
drains, over real sockets. The one about error bodies is in test_privacy.py."""

from __future__ import annotations

import asyncio
import random
import time
from functools import partial
from typing import Any

import httpx
import pytest

from simple_serving.fake_engine import Script

from .support import (
    BOT,
    CLASS,
    OUTSIDE_A,
    OUTSIDE_B,
    Answer,
    RawClient,
    Stack,
    by_prompt,
    call,
    control,
    count,
    engine_call,
    eventually,
    generate,
    generate_call,
    reached,
    reader,
    record_abort_order,
    running,
    service_with,
    streaming,
    until,
    user_body,
)

pytestmark = pytest.mark.anyio

CHAT = "/v1/chat/completions"
AGENT, INTERNAL = {CLASS: "agent"}, {CLASS: "internal"}
NOTICE_S = 0.25  # how soon the fake engine must see the gateway's engine request end; it takes milliseconds


class Engine:
    """Scripts of the fake engine by prompt: requests it holds before their first event, as if it read a long
    prompt, and streams that go on until their client leaves."""

    def __init__(self, stack: Stack) -> None:
        self.scripts: dict[str, Script] = {}
        self.gates: dict[str, asyncio.Event] = {}
        stack.fake.script = by_prompt(self.scripts)

    def hold(self, *texts: str) -> None:
        for text in texts:
            self.gates[text] = asyncio.Event()
            self.scripts[text] = Script(hold_first=self.gates[text])

    def release(self, text: str) -> None:
        self.gates[text].set()

    def endless(self, *texts: str) -> None:
        for text in texts:
            self.scripts[text] = Script(events=[{"role": "assistant"}], endless=True, delay_s=0.01)


def active(stack: Stack) -> dict[str, int]:
    return stack.service.admission.active_counts()


def waiting(stack: Stack) -> dict[str, int]:
    return stack.service.admission.waiting_counts()


def waiting_is(stack: Stack, cls: str, number: int) -> bool:
    return waiting(stack)[cls] == number


async def settled(stack: Stack) -> None:
    """Wait until the gateway has no work left."""
    await until(lambda: not stack.service.work)


def start(coroutine: Any) -> asyncio.Task[Answer]:
    return asyncio.ensure_future(coroutine)


# cancel-while-waiting, cancel-during-prompt, cancel-during-stream

async def test_cancel_while_waiting() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.hold("first")
        first = start(generate(client, stack, user_body("first"), headers=AGENT))
        await reached(stack, "first")
        second = await RawClient.post(stack.public, CHAT, user_body("second"), headers=AGENT)
        await until(lambda: waiting(stack)["agent"] == 1)  # the agent class has one place

        await second.leave()
        await until(lambda: waiting(stack)["agent"] == 0)
        engine.release("first")
        assert (await first).done
        await settled(stack)
        assert generate_call(stack, "second") is None


async def test_cancel_during_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    order = record_abort_order(monkeypatch, close_delay_s=0.2)
    async with running() as stack:
        Engine(stack).hold("a long prompt")
        client = await RawClient.post(stack.public, CHAT, user_body("a long prompt"), headers=reader())
        engine_request = await reached(stack, "a long prompt")
        assert await client.read_some(0.1) == b""  # nothing is sent while the engine reads the prompt

        await client.leave()
        await asyncio.sleep(0.1)  # the gateway is closing its engine request, slowly
        assert active(stack)["reader"] == 1 and order == []
        await eventually(engine_request.client_left.wait())
        await settled(stack)
        assert active(stack)["reader"] == 0
        assert order == ["closed", "freed"]


async def test_cancel_during_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    order = record_abort_order(monkeypatch)
    async with running() as stack:
        Engine(stack).endless("a long answer")
        client = await RawClient.post(stack.public, CHAT, user_body("a long answer"), headers=reader())
        await client.read_until_events(5)
        engine_request = generate_call(stack, "a long answer")
        assert engine_request is not None

        left_at = await client.leave()
        await eventually(engine_request.ended.wait())
        assert engine_request.client_left_at is not None and engine_request.last_event_at is not None
        noticed = engine_request.client_left_at - left_at
        went_on = engine_request.last_event_at - left_at
        print(f"the fake engine noticed after {noticed * 1000:.1f} ms and sent its last event "
              f"{went_on * 1000:.1f} ms after the client left")
        assert noticed < NOTICE_S and went_on < NOTICE_S
        await settled(stack)
        assert order == ["closed", "freed"] and active(stack)["reader"] == 0


# queue-full

async def test_queue_full_for_a_class() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.hold("a1", "a2", "a3")
        held = [start(generate(client, stack, user_body("a1"), headers=AGENT))]
        await reached(stack, "a1")
        for number, text in enumerate(("a2", "a3"), 1):  # the agent class waits for two
            held.append(start(generate(client, stack, user_body(text), headers=AGENT)))
            await until(partial(waiting_is, stack, "agent", number))

        refused = await generate(client, stack, user_body("a4"), headers=AGENT)
        assert (refused.status, refused.json()) == (429, {"error": {"code": "queue_full"}})
        for text in ("a1", "a2", "a3"):
            engine.release(text)
        assert all(answer.done for answer in await asyncio.gather(*held))
        assert generate_call(stack, "a4") is None


async def test_queue_full_for_one_outside_key() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.hold("o1", "o2", "o3")
        held = [start(generate(client, stack, user_body("o1"), key=OUTSIDE_A))]
        await reached(stack, "o1")
        for number, text in enumerate(("o2", "o3"), 1):  # one outside key: one active, two waiting
            held.append(start(generate(client, stack, user_body(text), key=OUTSIDE_A)))
            await until(partial(waiting_is, stack, "external", number))

        refused = await generate(client, stack, user_body("o4"), key=OUTSIDE_A)
        assert (refused.status, refused.json()) == (429, {"error": {"code": "queue_full"}})
        other = await generate(client, stack, user_body("b1"), key=OUTSIDE_B)  # another key has its own caps
        assert other.done
        for text in ("o1", "o2", "o3"):
            engine.release(text)
        assert all(answer.done for answer in await asyncio.gather(*held))
        assert generate_call(stack, "o4") is None


# readers-reserved, shared-order

async def test_readers_have_places_of_their_own() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.hold("a1", "i1", "i2", "i3", "e1")
        held = []
        for text, headers in (("a1", AGENT), ("i1", INTERNAL), ("i2", INTERNAL)):  # the three shared places
            held.append(start(generate(client, stack, user_body(text), headers=headers)))
            await reached(stack, text)
        held.append(start(generate(client, stack, user_body("e1"), key=OUTSIDE_A)))
        held.append(start(generate(client, stack, user_body("i3"), headers=INTERNAL)))
        await until(lambda: waiting(stack)["external"] == 1 and waiting(stack)["internal"] == 1)

        answer = await generate(client, stack, user_body("r1"), headers=reader())
        assert answer.done  # at once, while the shared places stay full
        assert waiting(stack)["external"] == 1 and waiting(stack)["internal"] == 1
        for text in ("a1", "i1", "i2", "i3", "e1"):
            engine.release(text)
        assert all(answer.done for answer in await asyncio.gather(*held))


async def test_a_freed_shared_place_goes_by_class_order_then_arrival() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        texts = {"i1": INTERNAL, "i2": INTERNAL, "e0": {}, "e1": {}, "a1": AGENT, "i3": INTERNAL, "i4": INTERNAL}
        keys = {"e0": OUTSIDE_A, "e1": OUTSIDE_B}
        engine.hold(*texts)
        tasks = {}
        for text, headers in texts.items():
            tasks[text] = start(generate(client, stack, user_body(text), key=keys.get(text, BOT), headers=headers))
            if text in ("i1", "i2", "e0"):  # the three shared places
                await reached(stack, text)
            else:
                await until(lambda: sum(waiting(stack).values()) == len(tasks) - 3)

        # The outside request e1 waited longest, but a freed shared place goes to the first class in order that has
        # room; within a class, to the request that came first.
        started = ["i1", "i2", "e0"]
        for released, next_up in (("e0", "a1"), ("i1", "i3"), ("i2", "i4"), ("a1", "e1")):
            engine.release(released)
            assert (await tasks[released]).done
            started.append(next_up)
            await reached(stack, next_up)
            assert {text for text in texts if generate_call(stack, text) is not None} == set(started)
        for text in ("i3", "i4", "e1"):
            engine.release(text)
        assert all(answer.done for answer in await asyncio.gather(*tasks.values()))


# count-quota, tokenize-hangs

async def test_count_places() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        gates = {text: asyncio.Event() for text in ("c1", "c2", "c3", "c4", "a reader waits")}
        stack.fake.script = by_prompt({text: Script(hold_tokenize=gate) for text, gate in gates.items()})
        places = stack.service.count_places
        held = [start(count(client, stack, user_body(text), key=OUTSIDE_A)) for text in ("c1", "c2")]
        await until(lambda: len(stack.fake.calls_of("tokenize")) == 2)

        third = await count(client, stack, user_body("c3"), key=OUTSIDE_A)  # one outside key has two at once
        assert (third.status, third.json()) == (429, {"error": {"code": "queue_full"}})
        held.append(start(count(client, stack, user_body("c4"), headers=INTERNAL)))  # the last of three places
        await until(lambda: places.running == 3)
        internal = start(count(client, stack, user_body("waits too"), headers=INTERNAL))
        await until(lambda: places.waiting == 1)
        readers = start(count(client, stack, user_body("a reader waits"), headers=reader()))
        await until(lambda: places.waiting == 2)  # waiting, not refused

        gates["c1"].set()  # the freed place goes to the reader, first in the class order
        await reached(stack, "a reader waits", "tokenize")
        assert places.waiting == 1 and engine_call(stack, "tokenize", "waits too") is None
        gates["a reader waits"].set()
        assert (await readers).json() == {"input_tokens": 4}
        assert (await internal).status == 200
        gates["c2"].set()
        gates["c4"].set()
        assert [answer.status for answer in await asyncio.gather(*held)] == [200, 200, 200]
        assert engine_call(stack, "tokenize", "c3") is None


async def test_a_count_the_engine_does_not_answer_times_out() -> None:
    block = service_with(limits={"reader": {"wall_s": 0.3}})
    async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = Script(hold_tokenize=asyncio.Event())  # never answered
        began = time.monotonic()
        answer = await count(client, stack, user_body("no answer"), headers=reader())
        assert (answer.status, answer.json()) == (504, {"error": {"code": "timeout"}})
        assert 0.3 <= time.monotonic() - began < 1.3
        await eventually(stack.fake.calls_of("tokenize")[0].client_left.wait())
        assert stack.service.count_places.running == 0
        stack.fake.script = None
        assert (await count(client, stack, user_body("an answer"), headers=reader())).status == 200


# wall-time

async def test_wall_time_while_waiting_before_the_stream_and_during_it() -> None:
    block = service_with(limits={"external": {"wall_s": 0.3}, "reader": {"wall_s": 0.3}})
    async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.hold("a1", "i1", "i2", "a long prompt")
        engine.endless("a long answer")
        held = []
        for text, headers in (("a1", AGENT), ("i1", INTERNAL), ("i2", INTERNAL)):  # the shared places, 900 s each
            held.append(start(generate(client, stack, user_body(text), headers=headers)))
            await reached(stack, text)

        began = time.monotonic()
        waited, prompt, streamed = await asyncio.gather(
            generate(client, stack, user_body("waits"), key=OUTSIDE_A),
            generate(client, stack, user_body("a long prompt"), headers=reader()),
            generate(client, stack, user_body("a long answer"), headers=reader(scope="reader.r2bbbbbb")))
        assert 0.3 <= time.monotonic() - began < 1.3
        assert (waited.status, waited.json()) == (504, {"error": {"code": "timeout"}})
        assert generate_call(stack, "waits") is None
        assert (prompt.status, prompt.json()) == (504, {"error": {"code": "timeout"}})
        assert streamed.status == 200 and streamed.error_event == "timeout" and not streamed.done
        for text in ("a long prompt", "a long answer"):
            engine_request = generate_call(stack, text)
            assert engine_request is not None
            await eventually(engine_request.ended.wait())
        for text in ("a1", "i1", "i2"):
            engine.release(text)
        assert all(answer.done for answer in await asyncio.gather(*held))


# drain-races-request, drain-cancels-outside

async def test_drain_races_requests() -> None:
    block = service_with(drain_deadline_s=0.6, limits={"shared": {"active": 6}})
    async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.hold("agent in its prompt", "reader in its prompt", "outside in its prompt", "internal finishing")
        engine.endless("internal past the deadline", "outside streaming")
        boot = (await control(client, stack, "/v1/state", method="GET")).json()["boot_id"]
        # The request, how it is sent, and whether it has a place when the drain begins.
        accepted: dict[str, tuple[dict[str, Any], str]] = {
            "agent in its prompt": ({"headers": AGENT}, "held"),
            "reader in its prompt": ({"headers": reader()}, "held"),
            "outside in its prompt": ({"key": OUTSIDE_A}, "held"),
            "internal finishing": ({"headers": INTERNAL}, "held"),
            "internal past the deadline": ({"headers": INTERNAL}, "streaming"),
            "outside streaming": ({"key": OUTSIDE_B}, "streaming"),
            "agent waiting": ({"headers": AGENT}, "waiting"),
            "internal waiting": ({"headers": INTERNAL}, "waiting"),
            "outside waiting": ({"key": OUTSIDE_A}, "waiting"),
        }
        tasks = {}
        for text, (options, phase) in accepted.items():
            tasks[text] = start(generate(client, stack, user_body(text), **options))
            if phase == "held":
                await reached(stack, text)
            elif phase == "streaming":
                await streaming(stack, text)
            else:
                await until(lambda: sum(waiting(stack).values()) == len(tasks) - 6)

        drained_at = time.monotonic()
        answer = await control(client, stack, "/v1/control/drain", {"boot_id": boot})
        assert (answer.status, answer.json()) == (202, {"status": "draining", "boot_id": boot, "drain_generation": 1})
        engine.release("internal finishing")  # ours finishes during the drain
        late = await generate(client, stack, user_body("arrives during the drain"), headers=INTERNAL)
        late_count = await count(client, stack, user_body("arrives during the drain"), headers=INTERNAL)
        assert [late.status, late_count.status] == [503, 503]
        assert late.json() == late_count.json() == {"error": {"code": "draining"}}

        answers = dict(zip(tasks, await asyncio.gather(*tasks.values()), strict=True))
        assert time.monotonic() - drained_at < 3
        for text in ("agent waiting", "internal waiting", "outside waiting", "outside in its prompt",
                     "agent in its prompt", "reader in its prompt"):  # the last two at the deadline
            assert (answers[text].status, answers[text].json()) == (503, {"error": {"code": "draining"}}), text
        for text in ("outside streaming", "internal past the deadline"):
            assert answers[text].status == 200 and answers[text].error_event == "draining", text
        assert answers["internal finishing"].done
        for text in ("agent waiting", "internal waiting", "outside waiting"):
            assert generate_call(stack, text) is None

        await settled(stack)
        state = (await control(client, stack, "/v1/state", method="GET")).json()
        assert state["status"] == "drained"
        assert set(state["active"].values()) == set(state["waiting"].values()) == {0}
        after = await generate(client, stack, user_body("arrives after the drain"), headers=reader())
        assert (after.status, after.json()) == (503, {"error": {"code": "drained"}})
        for engine_request in stack.fake.calls_of("generate", "tokenize"):
            await eventually(engine_request.ended.wait())


async def test_a_drain_racing_a_burst_leaves_no_request_without_an_answer() -> None:
    block = service_with(drain_deadline_s=0.5)
    options: list[dict[str, Any]] = [{"headers": reader()}, {"headers": AGENT}, {"headers": INTERNAL},
                                     {"key": OUTSIDE_A}, {"key": OUTSIDE_B}]
    for seed in range(3):
        rng = random.Random(seed)
        async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
            stack.fake.script = lambda body, rng=rng: Script(delay_s=rng.choice((0.0, 0.02, 0.05)))
            boot = (await control(client, stack, "/v1/state", method="GET")).json()["boot_id"]
            burst = [start(generate(client, stack, user_body(f"burst {n}"), **rng.choice(options)))
                     for n in range(16)]
            await asyncio.sleep(rng.uniform(0, 0.05))
            await control(client, stack, "/v1/control/drain", {"boot_id": boot})
            for answer in await asyncio.gather(*burst):
                outcome = ("done" if answer.done else f"event {answer.error_event}" if answer.status == 200
                           else f"{answer.status} {answer.json()['error']['code']}")
                assert outcome in ("done", "event draining", "503 draining", "503 drained", "429 queue_full"), outcome
            await settled(stack)
            assert stack.service.status == "drained"


async def test_a_drain_cancels_outside_work_and_lets_ours_finish_until_the_deadline() -> None:
    block = service_with(drain_deadline_s=0.5)
    async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
        engine = Engine(stack)
        engine.endless("outside", "internal past the deadline")
        engine.hold("internal")
        boot = (await control(client, stack, "/v1/state", method="GET")).json()["boot_id"]
        outside = start(generate(client, stack, user_body("outside"), key=OUTSIDE_A))
        ours = start(generate(client, stack, user_body("internal"), headers=INTERNAL))
        long = start(generate(client, stack, user_body("internal past the deadline"), headers=INTERNAL))
        for text in ("outside", "internal past the deadline"):
            await streaming(stack, text)
        await reached(stack, "internal")

        await control(client, stack, "/v1/control/drain", {"boot_id": boot})
        engine.release("internal")
        cancelled = await outside
        assert cancelled.status == 200 and cancelled.error_event == "draining" and not cancelled.done
        assert (await ours).done
        assert stack.service.status == "draining"
        cut = await long
        assert cut.status == 200 and cut.error_event == "draining"
        await settled(stack)
        assert stack.service.status == "drained"


# late-open-after-restart

async def test_an_open_from_before_a_restart_is_stale() -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        async with running() as stack:
            before = (await control(client, stack, "/v1/state", method="GET")).json()["boot_id"]
            assert (await control(client, stack, "/v1/control/drain", {"boot_id": before})).status == 202
        async with running() as stack:
            state = (await control(client, stack, "/v1/state", method="GET")).json()
            assert state["boot_id"] != before and state["drain_generation"] == 0
            late = await control(client, stack, "/v1/control/open", {"boot_id": before, "drain_generation": 1})
            assert (late.status, late.json()) == (409, {"error": {"code": "stale_boot"}})
            assert (await call(client, "GET", stack.public + "/v1/state")).json()["status"] == "ready"

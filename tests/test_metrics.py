"""The gateway's metrics (docs/metrics.md), read over HTTP from the control listener in front of the fake engine.

Each scenario checks four things: the request ends with exactly one outcome, the gateway's queue gauges return to where
they were, no time to first token is observed without a first token, and no label leaves its set. A scrape holds
nothing awake.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from simple_serving import metrics
from simple_serving.errors import STATUS
from simple_serving.fake_engine import Script
from simple_serving.log import RequestRecord

from .support import (
    BOT,
    CLASS,
    CONTROL,
    OUTSIDE_A,
    SERVICE,
    FakeClock,
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
GENERATION, COUNT = metrics.GENERATION, metrics.COUNT
IDLE_S = 780  # the default idle_timeout_s
METRICS_KEY = "test-key-metrics"
KEYS = {**SERVICE["keys"], METRICS_KEY: {"label": "metrics", "classes": [], "default": None, "scopes": False,
                                        "control": False, "metrics": True}}
QUEUES = ("generations_active", "generations_waiting", "counts_active", "counts_waiting", "accepted_in_flight",
          "ours_in_flight")
Labels = tuple[tuple[str, str], ...]
Samples = dict[tuple[str, Labels], float]


def parse(text: str) -> Samples:
    """The samples of an exposition, checked on the way: each family has its HELP and TYPE first, and a histogram's
    buckets only grow, up to its count."""
    samples: Samples = {}
    typed: dict[str, str] = {}
    helped: set[str] = set()
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helped.add(line.split(" ")[2])
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split(" ")
            assert name in helped
            typed[name] = kind
            continue
        head, _, number = line.rpartition(" ")
        name, _, rest = head.partition("{")
        labels = tuple(sorted(tuple(pair.split("=", 1)) for pair in rest.removesuffix("}").split(",") if pair))
        family = name.removesuffix("_bucket").removesuffix("_sum").removesuffix("_count")
        assert typed.get(name) or typed.get(family) == "histogram", name
        samples[name, tuple((label, raw.strip('"')) for label, raw in labels)] = float(number)
    for (name, labels), value in samples.items():
        if name.endswith("_count") and typed.get(name.removesuffix("_count")) == "histogram":
            family = name.removesuffix("_count")
            buckets = [samples[key] for key in samples if key[0] == family + "_bucket"
                       and {pair for pair in key[1] if pair[0] != "le"} == set(labels)]
            assert buckets == sorted(buckets) and buckets[-1] == value, family
    return samples


def check_labels(samples: Samples) -> None:
    """Every label value is one of its set; a status is a service status on `status`, an HTTP one elsewhere."""
    for name, labels in samples:
        assert name.startswith(metrics.PREFIX)
        for label, value in labels:
            if label == "le":
                continue
            if label == "status":
                allowed = metrics.SERVICE_STATUSES if name == "simple_serving_status" else metrics.HTTP_STATUSES
            else:
                allowed = metrics.LABELS[label]
            assert value in allowed, (name, label, value)


async def scrape(client: httpx.AsyncClient, stack: Stack, key: str = METRICS_KEY) -> Samples:
    answer = await call(client, "GET", stack.control + "/metrics", key=key)
    assert (answer.status, answer.headers["content-type"]) == (200, metrics.CONTENT_TYPE)
    samples = parse(answer.body.decode())
    check_labels(samples)
    return samples


def changes(before: Samples, after: Samples, name: str) -> dict[Labels, float]:
    """The samples of `name` that changed, by their labels, and by how much."""
    return {labels: value - before.get((sample, labels), 0.0) for (sample, labels), value in after.items()
            if sample == name and value != before.get((sample, labels), 0.0)}


def outcome(route: str, cls: str, result: str, code: str = "none") -> Labels:
    return tuple(sorted({"route": route, "class": cls, "outcome": result, "code": code}.items()))


def of_class(cls: str) -> Labels:
    return (("class", cls),)


def queues(samples: Samples) -> dict[tuple[str, Labels], float]:
    return {key: value for key, value in samples.items() if key[0].removeprefix(metrics.PREFIX) in QUEUES}


async def settled(client: httpx.AsyncClient, stack: Stack, before: Samples, ended: int) -> Samples:
    """Wait until `ended` more requests have been counted and no work is left, then scrape. The queues are back where
    they were before, and nothing more is counted a moment later."""
    counted = sum(value for (name, _), value in before.items() if name == "simple_serving_requests_total") + ended
    service = stack.service
    await until(lambda: sum(service.metrics.requests.values()) == counted and not service.work
                and service.ours_inflight == 0)
    after = await scrape(client, stack)
    assert queues(after) == queues(before)
    await asyncio.sleep(0.05)
    assert sum(service.metrics.requests.values()) == counted
    return after


def ttft(before: Samples, after: Samples) -> dict[Labels, float]:
    return changes(before, after, "simple_serving_time_to_first_token_seconds_count")


# The scenarios.


async def test_a_generation_and_a_count_that_succeed_are_counted_once_with_each_stage() -> None:
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=10) as client:
        # A stream without a token: the log measures its first token at its finish, the metrics observe none.
        stack.fake.script = by_prompt({"empty": Script(events=[{"role": "assistant"}, {"finish_reason": "stop"}])})
        before = await scrape(client, stack)
        assert (await generate(client, stack, user_body("fine"), headers=reader())).done
        assert (await count(client, stack, user_body("fine"), headers=reader())).status == 200
        assert (await generate(client, stack, user_body("empty"), headers=reader())).done
        after = await settled(client, stack, before, 3)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "reader", "success"): 2, outcome(COUNT, "reader", "success"): 1}
        assert ttft(before, after) == {of_class("reader"): 1}
        assert changes(before, after, "simple_serving_count_place_wait_seconds_count") == {
            (("class", "reader"), ("route", GENERATION)): 2, (("class", "reader"), ("route", COUNT)): 1}
        assert changes(before, after, "simple_serving_count_seconds_count") == {
            (("class", "reader"), ("route", GENERATION)): 2, (("class", "reader"), ("route", COUNT)): 1}
        for stage in ("generation_place_wait", "wait", "total"):
            assert changes(before, after, f"simple_serving_{stage}_seconds_count") == {of_class("reader"): 2}
        assert changes(before, after, "simple_serving_output_tokens_total")[of_class("reader")] > 0
        assert changes(before, after, "simple_serving_count_mismatches_total") == {}
        responses = changes(before, after, "simple_serving_http_responses_total")
        assert (responses[(("listener", "public"), ("route", GENERATION), ("status", "200"))],
                responses[(("listener", "public"), ("route", COUNT), ("status", "200"))]) == (2, 1)


async def test_a_full_queue_is_rejected_once_and_the_gateways_queue_shows_apart_from_the_engine() -> None:
    gate = asyncio.Event()
    block = service_with(keys=KEYS, limits={"agent": {"active": 1, "waiting": 0}})
    async with running(block) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = by_prompt({"first": Script(hold_first=gate)})
        before = await scrape(client, stack)
        first = asyncio.ensure_future(generate(client, stack, user_body("first"), headers={CLASS: "agent"}))
        await reached(stack, "first")
        refused = await generate(client, stack, user_body("second"), headers={CLASS: "agent"})
        assert (refused.status, refused.json()) == (429, {"error": {"code": "queue_full"}})
        await until(lambda: sum(stack.service.metrics.requests.values()) == sum(
            value for (name, _), value in before.items() if name == "simple_serving_requests_total") + 1)
        during = await scrape(client, stack)
        assert changes(before, during, "simple_serving_requests_total") == {
            outcome(GENERATION, "agent", "rejected", "queue_full"): 1}
        assert during["simple_serving_generations_active", of_class("agent")] == 1
        assert during["simple_serving_accepted_in_flight", (("route", GENERATION),)] == 1
        engine = await call(client, "GET", stack.control + "/metrics/engine", key=METRICS_KEY)
        assert b"vllm:num_requests_running{" in engine.body  # the engine's queue, from its own metrics
        gate.set()
        assert (await first).done
        after = await settled(client, stack, before, 2)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "agent", "rejected", "queue_full"): 1, outcome(GENERATION, "agent", "success"): 1}
        assert ttft(before, after) == {of_class("agent"): 1}
        assert changes(before, after, "simple_serving_generation_place_wait_seconds_count") == {of_class("agent"): 1}


async def test_a_timeout_before_the_first_token_observes_no_first_token(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=None) as client:
        stack.fake.script = by_prompt({"slow": Script(hold_first=asyncio.Event())})  # the prompt is never read
        before = await scrape(client, stack)
        turn = asyncio.ensure_future(generate(client, stack, user_body("slow"), headers=reader()))
        await reached(stack, "slow")
        await clock.advance(SERVICE["limits"]["reader"]["wall_s"])
        answer = await turn
        assert (answer.status, answer.json()) == (504, {"error": {"code": "timeout"}})
        after = await settled(client, stack, before, 1)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "reader", "timeout", "timeout"): 1}
        assert ttft(before, after) == {}
        assert changes(before, after, "simple_serving_total_seconds_count") == {}
        assert changes(before, after, "simple_serving_wait_seconds_count") == {of_class("reader"): 1}


async def test_an_engine_error_after_the_200_is_an_engine_error_and_its_status_stays_200() -> None:
    breaks = Script(events=[{"role": "assistant"}, {"content": "The keeper "},
                            {"error": {"message": "synthetic", "type": "InternalServerError", "code": 500}}])
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = by_prompt({"breaks": breaks})
        before = await scrape(client, stack)
        answer = await generate(client, stack, user_body("breaks"), headers=reader())
        assert (answer.status, answer.error_event, answer.done) == (200, "engine_unavailable", False)
        after = await settled(client, stack, before, 1)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "reader", "engine_error", "engine_unavailable"): 1}
        assert changes(before, after, "simple_serving_http_responses_total")[
            (("listener", "public"), ("route", GENERATION), ("status", "200"))] == 1
        assert ttft(before, after) == {of_class("reader"): 1}  # a token came before the error
        assert changes(before, after, "simple_serving_total_seconds_count") == {}


async def test_a_client_that_leaves_is_a_disconnect_with_a_first_token_only_if_one_came() -> None:
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=10) as client:
        endless = Script(events=[{"role": "assistant"}, {"content": "and "}], endless=True, delay_s=0.01)
        stack.fake.script = by_prompt({"early": Script(hold_first=asyncio.Event()), "late": endless})
        before = await scrape(client, stack)
        early = await RawClient.post(stack.public, GENERATION, user_body("early"), headers=reader())
        await reached(stack, "early")
        await early.leave()
        late = await RawClient.post(stack.public, GENERATION, user_body("late"), headers=reader())
        await late.read_until_events(3)
        await late.leave()
        after = await settled(client, stack, before, 2)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "reader", "client_disconnect"): 2}
        assert ttft(before, after) == {of_class("reader"): 1}  # the late one's
        responses = changes(before, after, "simple_serving_http_responses_total")
        assert responses[(("listener", "public"), ("route", GENERATION), ("status", "none"))] == 1
        assert responses[(("listener", "public"), ("route", GENERATION), ("status", "200"))] == 1


async def test_a_done_that_waits_for_its_client_past_the_wall_time_is_a_timeout_not_a_success(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    began = stall_the_send_of(monkeypatch, b"data: [DONE]")
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=None) as client:
        before = await scrape(client, stack)
        unread = await RawClient.post(stack.public, GENERATION, user_body("unread"), headers=reader())
        await began.wait()  # the engine's stream ended well; its [DONE] waits for the client
        await clock.advance(SERVICE["limits"]["reader"]["wall_s"])
        after = await settled(client, stack, before, 1)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "reader", "timeout", "timeout"): 1}
        assert ttft(before, after) == {of_class("reader"): 1}
        assert changes(before, after, "simple_serving_total_seconds_count") == {of_class("reader"): 1}
        await unread.leave()


async def test_a_drain_stops_the_waiting_and_refuses_the_new_while_our_active_work_ends_well() -> None:
    gate = asyncio.Event()
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=10) as client:
        stack.fake.script = by_prompt({"active": Script(hold_first=gate)})
        before = await scrape(client, stack)
        active = asyncio.ensure_future(generate(client, stack, user_body("active"), headers={CLASS: "agent"}))
        await reached(stack, "active")
        waiting = asyncio.ensure_future(generate(client, stack, user_body("waiting"), headers={CLASS: "agent"}))
        await until(lambda: stack.service.admission.waiting_counts()["agent"] == 1)
        during = await scrape(client, stack)
        assert (during["simple_serving_generations_active", of_class("agent")],
                during["simple_serving_generations_waiting", of_class("agent")]) == (1, 1)
        boot = stack.service.boot_id
        assert (await control(client, stack, "/v1/control/drain", {"boot_id": boot})).status == 202
        assert ((await waiting).status, (await waiting).json()) == (503, {"error": {"code": "draining"}})
        late = await generate(client, stack, user_body("late"), headers=reader())
        assert (late.status, late.json()) == (503, {"error": {"code": "draining"}})
        gate.set()
        assert (await active).done
        after = await settled(client, stack, before, 3)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(GENERATION, "agent", "draining", "draining"): 1,
            outcome(GENERATION, "none", "draining", "draining"): 1,  # refused before its class was checked
            outcome(GENERATION, "agent", "success"): 1}
        assert ttft(before, after) == {of_class("agent"): 1}
        assert after["simple_serving_status", (("status", "drained"),)] == 1
        assert after["simple_serving_drain_generation", ()] == 1


# A scrape and its rights.


async def test_a_scrape_never_holds_the_card_awake(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=None) as client:
        for _ in range(3):
            await clock.advance((IDLE_S - 2) / 3)
            samples = await scrape(client, stack)
            assert (await call(client, "GET", stack.control + "/metrics/engine", key=METRICS_KEY)).status == 200
            assert (await scrape(client, stack, CONTROL)) is not None
            assert not stack.service.sleep_requested
        assert IDLE_S - 3 < samples["simple_serving_idle_seconds", ()] < IDLE_S
        assert samples["simple_serving_idle_timeout_seconds", ()] == IDLE_S
        await clock.advance(2)
        assert stack.service.sleep_requested
        await until(lambda: stack.vast.stopped)


async def test_only_the_metrics_right_or_the_control_key_reads_the_metrics_and_only_on_the_control_listener(
        ) -> None:
    async with running(service_with(keys=KEYS)) as stack, httpx.AsyncClient(timeout=10) as client:
        for path in ("/metrics", "/metrics/engine"):
            for key, status, code in ((None, 401, "unauthorized"), ("test-key-unknown", 401, "unauthorized"),
                                      (BOT, 403, "forbidden"), (OUTSIDE_A, 403, "forbidden")):
                answer = await call(client, "GET", stack.control + path, key=key)
                assert (answer.status, answer.json()) == (status, {"error": {"code": code}})
            for key in (METRICS_KEY, CONTROL):
                assert (await call(client, "GET", stack.control + path, key=key)).status == 200
            answer = await call(client, "GET", stack.public + path, key=METRICS_KEY)
            assert (answer.status, answer.json()) == (404, {"error": {"code": "not_found"}})
        # The metrics right is the only right of that key.
        drain = await call(client, "POST", stack.control + "/v1/control/drain", key=METRICS_KEY,
                           body={"boot_id": stack.service.boot_id})
        assert (drain.status, drain.json()) == (403, {"error": {"code": "forbidden"}})
        refused = await generate(client, stack, user_body("metrics"), key=METRICS_KEY)
        assert (refused.status, refused.json()) == (403, {"error": {"code": "class_not_allowed"}})
        # The engine's own metrics come as the engine gave them, and a failed read is the engine's.
        engine = await call(client, "GET", stack.control + "/metrics/engine", key=METRICS_KEY)
        assert engine.headers["content-type"].startswith("text/plain")
        assert f'vllm:num_requests_running{{engine="0",model_name="{SERVICE["alias"]}"}}'.encode() in engine.body
        stack.fake.healthy = False
        failed = await call(client, "GET", stack.control + "/metrics/engine", key=METRICS_KEY)
        assert (failed.status, failed.json()) == (503, {"error": {"code": "engine_unavailable"}})
        samples = await scrape(client, stack)
        assert samples["simple_serving_requests_total", outcome(GENERATION, "none", "rejected",
                                                                "class_not_allowed")] == 1


# The outcome of a row.


def test_every_code_has_one_outcome_that_lists_it() -> None:
    for code in STATUS:
        for accepted in (False, True):
            result = metrics.outcome_of(code, accepted)
            assert result in metrics.OUTCOMES
            if code not in ("forbidden", "not_found", "stale_boot", "stale_generation", "sleep_pending"):
                assert code in metrics.OUTCOMES[result], (code, accepted)  # the codes an inference route answers


@pytest.mark.parametrize(("fields", "expected"), [
    ({}, ("success", "none")),
    ({"code": "queue_full", "cancelled": True}, ("rejected", "queue_full")),  # a refusal its client did not take
    ({"code": "timeout"}, ("rejected", "timeout")),  # a body that did not arrive in 30 s
    ({"code": "timeout", "accepted": True}, ("timeout", "timeout")),
    ({"code": "engine_unavailable"}, ("unavailable", "engine_unavailable")),
    ({"code": "engine_unavailable", "accepted": True}, ("engine_error", "engine_unavailable")),
    ({"code": "drained"}, ("draining", "drained")),
    ({"code": "starting"}, ("unavailable", "starting")),
    ({"cancelled": True, "accepted": True, "stop": "disconnect"}, ("client_disconnect", "none")),
    ({"cancelled": True}, ("client_disconnect", "none")),  # gone while its body was read
    # A terminal message that waited for its client, given up at a stop: the stop's code, as the client would have
    # been told.
    ({"cancelled": True, "accepted": True, "stop": "timeout"}, ("timeout", "timeout")),
    ({"cancelled": True, "accepted": True, "stop": "draining"}, ("draining", "draining")),
    ({"cancelled": True, "accepted": True, "stop": "engine_changed"}, ("engine_error", "engine_unavailable")),
    ({"cancelled": True, "accepted": True, "stop": "shutdown"}, ("unavailable", "none")),
    ({"code": "internal_error", "accepted": True}, ("internal_error", "internal_error")),
])
def test_a_rows_outcome(fields: dict[str, Any], expected: tuple[str, str]) -> None:
    record = RequestRecord(listener="public", route=GENERATION, method="POST", **fields)
    assert metrics.request_outcome(record) == expected


def test_an_exception_that_reached_the_last_resort_is_an_internal_error_even_after_the_200() -> None:
    record = RequestRecord(listener="public", route=GENERATION, method="POST", status=200, accepted=True)
    assert metrics.request_outcome(record, escaped=True) == ("internal_error", "internal_error")

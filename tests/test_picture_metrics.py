"""Picture metrics over HTTP: access, outcomes, unresolved work and bounded labels, against fake ComfyUI.

The scrape must observe the lane without keeping its card awake or changing its admission or drain.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import uuid

import httpx
import pytest

from simple_serving import metrics
from simple_serving.pictures import comfy
from simple_serving.pictures.fake_comfy import Script
from simple_serving.pictures.uploads import BLANK

from .support import BOT, CONTROL, FakeClock, call, until
from .test_metrics import Samples, changes, check_labels, of_class, outcome, parse
from .test_monitoring import gateway_catalog, served
from .test_pictures import AGENT, GRAPH, INTERNAL, KEYS, PICTURES, A, Stack, Stream, running

pytestmark = pytest.mark.anyio
METRICS_KEY = "synthetic-private-metrics-key"
PRIVATE_LABEL = "synthetic-private-key-label"
MARKER = "synthetic-private-graph-marker"
METRICS_KEYS = {**KEYS, METRICS_KEY: {"label": PRIVATE_LABEL, "classes": [], "default": None, "scopes": False,
                                   "control": False, "metrics": True}}
STOP = "/v1/pictures/{job}/stop"
UPLOAD = "/v1/pictures/uploads/{name}"
INPUT = "/v1/pictures/inputs/{name}"


async def scrape(client: httpx.AsyncClient, stack: Stack, *, key: str = METRICS_KEY,
                 private: tuple[str, ...] = ()) -> Samples:
    answer = await call(client, "GET", stack.control + "/metrics", key=key)
    assert (answer.status, answer.headers["content-type"]) == (200, metrics.CONTENT_TYPE)
    text = answer.body.decode()
    assert all(value not in text for value in (BOT, CONTROL, METRICS_KEY, PRIVATE_LABEL, MARKER, *private))
    assert served(text) == gateway_catalog("pictures")
    samples = parse(text)
    check_labels(samples)
    return samples


async def finished(stack: Stack, requests: int) -> None:
    await until(lambda: not stack.service.work and stack.service.ours_inflight == 0
                and sum(stack.service.metrics.requests.values()) == requests)


def empty_jobs(samples: Samples) -> bool:
    return all(value == 0 for (name, _), value in samples.items()
               if name in ("simple_serving_picture_jobs_active", "simple_serving_picture_jobs_waiting",
                           "simple_serving_picture_stops_unresolved", "simple_serving_accepted_in_flight"))


async def test_picture_scrapes_keep_their_rights_labels_and_outcomes_without_holding_the_card_awake(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    graph = copy.deepcopy(GRAPH)
    graph["2"]["inputs"]["text"] = MARKER
    upload_name = f"{uuid.uuid4()}.png"
    input_name = f"ref-{hashlib.sha256(BLANK).hexdigest()[:16]}.png"
    async with (running(Script(outcomes={2: "oom"}, marker=MARKER), keys=METRICS_KEYS) as stack,
                httpx.AsyncClient(timeout=None) as client):
        for key, status, code in ((None, 401, "unauthorized"), ("unknown-metrics-key", 401, "unauthorized"),
                                  (BOT, 403, "forbidden")):
            refused = await call(client, "GET", stack.control + "/metrics", key=key)
            assert (refused.status, refused.json()) == (status, {"error": {"code": code}})
        for key in (METRICS_KEY, CONTROL):
            await scrape(client, stack, key=key)
            for origin, path in ((stack.public, "/metrics"), (stack.public, "/metrics/engine"),
                                 (stack.control, "/metrics/engine")):
                missing = await call(client, "GET", origin + path, key=key)
                assert (missing.status, missing.json()) == (404, {"error": {"code": "not_found"}})
        denied = await call(client, "POST", stack.control + "/v1/control/drain", key=METRICS_KEY,
                            body={"boot_id": stack.service.boot_id})
        assert (denied.status, denied.json()) == (403, {"error": {"code": "forbidden"}})
        denied = await call(client, "POST", stack.public + PICTURES, key=METRICS_KEY, body={"graph": graph})
        assert (denied.status, denied.json()) == (403, {"error": {"code": "class_not_allowed"}})
        assert stack.fake.jobs == [] and not stack.service.draining
        await finished(stack, 1)
        before = await scrape(client, stack)

        assert (await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": graph})).done
        failed = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": graph})
        assert failed.status == 200 and failed.error_event == "picture_failed"
        upload_url = f"{stack.public}{PICTURES}/uploads/{upload_name}"
        assert (await call(client, "PUT", upload_url, headers={**A, "Content-Type": "image/png"},
                           content=BLANK)).status == 200
        assert (await call(client, "DELETE", upload_url, headers=A)).status == 204
        assert (await call(client, "PUT", f"{stack.public}{PICTURES}/inputs/{input_name}",
                           headers={**INTERNAL, "Content-Type": "image/png"}, content=BLANK)).status == 200
        await finished(stack, 6)
        private = (upload_name, input_name, "reader.aaaaaaaa", *(job.id for job in stack.fake.jobs))
        after = await scrape(client, stack, private=private)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(PICTURES, "reader", "success"): 1,
            outcome(PICTURES, "reader", "engine_error", "picture_failed"): 1,
            outcome(UPLOAD, "reader", "success"): 2, outcome(INPUT, "internal", "success"): 1}
        responses = changes(before, after, "simple_serving_http_responses_total")
        assert responses[(("listener", "public"), ("route", PICTURES), ("status", "200"))] == 2
        assert responses[(("listener", "public"), ("route", UPLOAD), ("status", "204"))] == 1
        assert empty_jobs(after)
        idle_since, idle_s = stack.service.idle_since, stack.service.config.idle_timeout_s
        for _ in range(3):
            await clock.advance((idle_s - 2) / 3)
            after = await scrape(client, stack, private=private)
            await scrape(client, stack, key=CONTROL)
            assert stack.service.idle_since == idle_since and stack.service.ours_inflight == 0
            assert not stack.service.sleep_requested
        assert idle_s - 3 < after["simple_serving_idle_seconds", ()] < idle_s
        await clock.advance(2)
        assert stack.service.sleep_requested


async def test_picture_metrics_follow_active_waiting_and_draining_work_without_changing_it() -> None:
    hold = asyncio.Event()
    async with running(Script(hold=hold), keys=METRICS_KEYS) as stack, httpx.AsyncClient(timeout=10) as client:
        before = await scrape(client, stack)
        drawn = Stream(client, stack, INTERNAL)
        drawn_id = await drawn.accepted()
        await until(lambda: stack.service.jobs[drawn_id].phase == "drawing")
        waiting = Stream(client, stack, A)
        waiting_id = await waiting.accepted()
        await until(lambda: stack.service.jobs[waiting_id].phase == "pending")
        during = await scrape(client, stack, private=(drawn_id, waiting_id))
        assert during["simple_serving_picture_jobs_active", of_class("internal")] == 1
        assert during["simple_serving_picture_jobs_waiting", of_class("reader")] == 1
        assert during["simple_serving_accepted_in_flight", (("route", PICTURES),)] == 2
        assert during["simple_serving_ours_in_flight", ()] == 2
        assert during["simple_serving_idle_seconds", ()] == 0
        assert set(stack.service.jobs) == {drawn_id, waiting_id} and not stack.service.draining
        drain = await call(client, "POST", stack.control + "/v1/control/drain", key=CONTROL,
                           body={"boot_id": stack.service.boot_id})
        assert drain.status == 202
        await waiting.task
        assert waiting.events[-1] == {"error": {"code": "draining"}}
        during = await scrape(client, stack)
        assert during["simple_serving_status", (("status", "draining"),)] == 1
        assert during["simple_serving_picture_jobs_active", of_class("internal")] == 1
        assert during["simple_serving_picture_jobs_waiting", of_class("reader")] == 0
        assert during["simple_serving_accepted_in_flight", (("route", PICTURES),)] == 1
        assert not drawn.task.done() and stack.service.draining
        late = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": GRAPH})
        assert (late.status, late.json()) == (503, {"error": {"code": "draining"}})
        hold.set()
        await drawn.task
        assert drawn.events[-1] == "[DONE]"
        await finished(stack, 3)
        after = await scrape(client, stack)
        assert changes(before, after, "simple_serving_requests_total") == {
            outcome(PICTURES, "reader", "draining", "draining"): 1,
            outcome(PICTURES, "none", "draining", "draining"): 1,
            outcome(PICTURES, "internal", "success"): 1}
        assert empty_jobs(after)
        assert after["simple_serving_status", (("status", "drained"),)] == 1
        assert after["simple_serving_drain_generation", ()] == 1


async def test_unconfirmed_picture_stops_are_separate_from_jobs_and_keep_the_cap_until_resolved(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfy, "STOP_POLL_S", 0.01)
    limits = {"agent": {"jobs": 1, "wall_s": 1860, "uploads": 16},
              "internal": {"jobs": 32, "wall_s": 0.5, "uploads": 0}}
    async with (running(Script(outcomes={1: "hang", 2: "hang"}), keys=METRICS_KEYS, limits=limits) as stack,
                httpx.AsyncClient(timeout=10) as client):
        before = await scrape(client, stack)
        cancel = stack.fake._cancel
        monkeypatch.setattr(stack.fake, "_cancel", lambda prompt_id: False)
        timed = await call(client, "POST", stack.public + PICTURES, headers=INTERNAL, body={"graph": GRAPH})
        assert timed.status == 200 and timed.error_event == "stop_unconfirmed"
        drawing = Stream(client, stack, AGENT)
        job_id = await drawing.accepted()
        await until(lambda: len(stack.fake.pending) == 1)
        stop_url = f"{stack.public}{PICTURES}/{job_id}/stop"
        for _ in range(2):
            stopped = await call(client, "POST", stop_url, headers=AGENT)
            assert (stopped.status, stopped.json()) == (200, {"stopped": False})
        await drawing.task
        full = await call(client, "POST", stack.public + PICTURES, headers=AGENT, body={"graph": GRAPH})
        assert (full.status, full.json()) == (429, {"error": {"code": "queue_full"}})
        await finished(stack, 5)
        unresolved = await scrape(client, stack, private=(job_id,))
        assert unresolved["simple_serving_picture_jobs_active", of_class("agent")] == 0
        assert unresolved["simple_serving_picture_jobs_waiting", of_class("agent")] == 0
        assert unresolved["simple_serving_picture_stops_unresolved", of_class("agent")] == 1
        assert unresolved["simple_serving_picture_stops_unresolved", of_class("internal")] == 1
        assert unresolved["simple_serving_picture_jobs_active", of_class("internal")] == 0
        assert unresolved["simple_serving_picture_job_limit", of_class("agent")] == 1
        assert unresolved["simple_serving_accepted_in_flight", (("route", PICTURES),)] == 0
        assert changes(before, unresolved, "simple_serving_requests_total") == {
            outcome(PICTURES, "internal", "engine_error", "stop_unconfirmed"): 1,
            outcome(PICTURES, "agent", "client_cancelled"): 1,
            outcome(STOP, "agent", "success"): 2,
            outcome(PICTURES, "agent", "rejected", "queue_full"): 1}
        monkeypatch.setattr(stack.fake, "_cancel", cancel)
        await until(lambda: not stack.service.unresolved)
        assert empty_jobs(await scrape(client, stack))
        assert (await call(client, "POST", stack.public + PICTURES, headers=AGENT, body={"graph": GRAPH})).done
        await finished(stack, 6)
        after = await scrape(client, stack)
        assert changes(unresolved, after, "simple_serving_requests_total") == {
            outcome(PICTURES, "agent", "success"): 1}
        assert empty_jobs(after)

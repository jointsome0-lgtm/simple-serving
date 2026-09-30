"""The picture lane's gateway (docs/pictures.md, sections 3 to 8 and 11) in front of the fake ComfyUI
(`simple_serving/pictures/fake_comfy.py`), both on loopback sockets and the fake with its websocket: what reaches the
card and in which order, a picture's stream, its stops and whether the card confirms them, the drain, the engine the
gateway holds itself to, the readers' uploads, and, with the gateway as its own process, answers and a log that hold
nothing of a graph, an upload or the engine's words. Graphs and pictures are synthetic."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import secrets
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, MutableMapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving.asgi import RECORD, Exchange
from simple_serving.config import Listener
from simple_serving.errors import ServiceError
from simple_serving.log import RequestRecord
from simple_serving.pictures import comfy, jobs, routes
from simple_serving.pictures import uploads as uploads_module
from simple_serving.pictures.config import CARD_ENGINE, PROVISIONAL, READER_NODES, from_block
from simple_serving.pictures.fake_comfy import FakeComfy, Script
from simple_serving.pictures.graphs import PNG_SIGNATURE, strip_png
from simple_serving.pictures.server import PictureGateway
from simple_serving.pictures.service import PictureService
from simple_serving.pictures.uploads import BLANK, Uploads
from simple_serving.policy import resolve
from simple_serving.server import Servers, bind

from .support import BOT, CLASS, CONTROL, FakeVast, Process, RawClient, call, reader, until

pytestmark = pytest.mark.anyio

PICTURES, ENGINE = "/v1/pictures", "/v1/pictures/engine"
KEYS: dict[str, dict[str, Any]] = {
    BOT: {"label": "bot", "classes": ["reader", "agent", "internal"], "default": "internal", "scopes": True,
          "control": False},
    CONTROL: {"label": "control", "classes": [], "default": None, "scopes": False, "control": True},
}
# The bot's plain graph, shortened, with its saving node turned into a preview as the bot sends it.
GRAPH: dict[str, dict[str, Any]] = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "synthetic.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a synthetic pier at dusk", "clip": ["1", 0]}},
    "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "seed": 7}},
    "4": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0]}},
    "5": {"class_type": "PreviewImage", "inputs": {"images": ["4", 0]}},
}
A, B = reader("reader.aaaaaaaa"), reader("reader.bbbbbbbb")
AGENT, INTERNAL = {CLASS: "agent"}, {CLASS: "internal"}
REFERENCE = PNG_SIGNATURE + b"synthetic reference"


@dataclass
class Stack:
    fake: FakeComfy
    gateway: PictureGateway
    public: str
    control: str

    @property
    def service(self) -> PictureService:
        return self.gateway.service


@asynccontextmanager
async def running(script: Script | None = None, *, wait_ready: bool = True, **block: Any) -> AsyncIterator[Stack]:
    """The fake ComfyUI and a picture gateway in front of it, with the test keys, the fields of `block`, and a fake
    stop of the instance."""
    fake = FakeComfy(script)
    sock = bind(Listener("127.0.0.1", 0))
    engine = Servers([(fake, sock)], graceful_s=1, ws="websockets-sansio")
    await engine.start()
    try:
        config = from_block({"keys": KEYS, "health_interval_s": 0.05, **block},
                            engine_url=f"http://127.0.0.1:{sock.getsockname()[1]}")
        async with PictureGateway(config, stop=FakeVast(), graceful_s=1) as gateway:
            stack = Stack(fake, gateway, f"http://127.0.0.1:{gateway.ports['public']}",
                          f"http://127.0.0.1:{gateway.ports['control']}")
            if wait_ready:
                await until(lambda: gateway.service.status == "ready")
            yield stack
    finally:
        await engine.stop()


class Stream:
    """A picture's request, read in a task of its own: its events land in `events` as they come."""

    def __init__(self, client: httpx.AsyncClient, stack: Stack, headers: dict[str, str],
                 body: dict[str, Any] | None = None) -> None:
        self.events: list[Any] = []
        self.task = asyncio.ensure_future(self._read(client, stack.public + PICTURES, headers,
                                                     body or {"graph": GRAPH}))

    async def _read(self, client: httpx.AsyncClient, url: str, headers: dict[str, str], body: dict[str, Any]) -> None:
        async with client.stream("POST", url, headers={"Authorization": f"Bearer {BOT}", **headers},
                                 json=body) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line.removeprefix("data: ")
                    self.events.append(data if data == "[DONE]" else json.loads(data))

    def named(self, name: str) -> list[dict[str, Any]]:
        return named(self.events, name)

    async def accepted(self) -> str:
        """The job's id, once the card has taken it."""
        await until(lambda: bool(self.named("accepted")))
        job: str = self.named("accepted")[0]["job"]
        return job


def named(events: list[Any], name: str) -> list[dict[str, Any]]:
    return [event for event in events if isinstance(event, dict) and event.get("event") == name]


def rows(logged: io.StringIO, *events: str) -> list[dict[str, Any]]:
    found = (json.loads(line) for line in logged.getvalue().splitlines())
    return [{name: value for name, value in row.items() if name != "time"} for row in found
            if row.get("event") in events]


def pictures(logged: io.StringIO) -> list[dict[str, Any]]:
    """The rows of the requests for pictures, in the order they ended."""
    return [row for row in rows(logged, "request") if row.get("route") == PICTURES]


def placed(stream: Stream, before: int) -> Callable[[], bool]:
    """Whether the stream has been told that `before` jobs will be drawn before its own, while another is drawn."""
    return lambda: any(place["before"] == before and place["drawing"] for place in stream.named("place"))


async def test_a_readers_picture_reaches_the_card_as_sent_and_comes_back_stripped_with_its_course(
        logged: io.StringIO) -> None:
    async with running(Script(job_s=0.2)) as stack, httpx.AsyncClient(timeout=10) as client:
        facts = await call(client, "GET", stack.public + ENGINE, headers=A)
        answer = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": GRAPH})
        await until(lambda: not stack.fake.history)  # the job's record, deleted once its picture was read
        client_id = stack.service.socket.client_id
    assert facts.json() == {"comfyui": "0.37.0", "pytorch": "2.11.0+cu130",
                            "card": "cuda:0 Fake Card : cudaMallocAsync", "triton": True, "kitchen_attention": True}
    (job,) = stack.fake.jobs
    nonce = job.graph["5"]["inputs"].pop("nonce")
    assert str(uuid.UUID(nonce)) == nonce  # ComfyUI's per-job preview nonce, fresh for every job, and nothing else
    assert (job.graph, job.number, job.client_id) == (GRAPH, -(10**9 + 1), client_id)
    events = answer.events()
    assert events[0] == {"event": "accepted", "job": job.id, "socket": True} and events[-1] == "[DONE]"
    course = [event["event"] for event in events[1:-1]]
    assert course[:2] == ["started", "cached"] and course[-2:] == ["over", "image"]
    assert [event["node"] for event in named(events, "node")] == list(GRAPH)
    assert [(event["value"], event["max"]) for event in named(events, "progress")] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    (over,) = named(events, "over")
    assert over["heard"] and over["stamps"]["start"] <= over["stamps"]["end"] and over["memory"]["samples"] >= 1
    assert over["memory"]["vram"] == [{"index": 0, "total_mib": 32768, "used_mib_max": 22528,
                                       "occupied_mib_max": 23552}]
    assert over["memory"]["ram_mib"] == {"min": 24576, "max": 24576}
    (image,) = named(events, "image")
    png, drawn = base64.b64decode(image["png"]), stack.fake.files[("temp", "", "ComfyUI_temp_00001_5_.png")]
    assert image["node"] == "5" and png == strip_png(drawn) and b"tEXt" in drawn and b"tEXt" not in png
    (row,) = pictures(logged)
    assert {name: row.get(name) for name in ("key", "class", "scope", "status", "nodes", "images", "image_bytes",
                                            "references", "socket")} == {
        "key": "bot", "class": "reader", "scope": "reader", "status": 200, "nodes": 5, "images": 1,
        "image_bytes": len(png), "references": 0, "socket": True}
    assert row["queue_ms"] >= 0 and row["draw_ms"] >= 0


async def test_the_card_draws_the_readers_first_the_later_first_then_the_agents_then_the_experiments(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "PLACE_S", 0.05)
    hold = asyncio.Event()  # the experiment being drawn goes on until the test lets it end
    async with running(Script(hold=hold)) as stack, httpx.AsyncClient(timeout=10) as client:
        drawn = Stream(client, stack, INTERNAL)
        await until(lambda: stack.fake.running is not None)
        waiting: dict[str, Stream] = {}
        for name, headers in (("reader a", A), ("agent", AGENT), ("reader b", B), ("experiment", INTERNAL)):
            waiting[name] = Stream(client, stack, headers)
            await until(lambda: len(stack.fake.pending) == len(waiting))  # each taken before the next is sent
        order = ("reader b", "reader a", "agent", "experiment")
        assert [job.id for job in sorted(stack.fake.pending)] == [await waiting[name].accepted() for name in order]
        for before, name in enumerate(order):  # and each waiting client is told so
            await until(placed(waiting[name], before))
        hold.set()
        for stream in (drawn, *waiting.values()):
            await stream.task
            assert stream.named("image") and stream.events[-1] == "[DONE]"
    assert [job.number for job in stack.fake.jobs] == [10**9 + 1, -(10**9 + 2), 3, -(10**9 + 4), 10**9 + 5]
    stamps = [waiting[name].named("started")[0]["stamp"] for name in order]
    assert stamps == sorted(set(stamps))  # drawn in that order, one after another


async def test_a_stop_takes_the_job_off_the_card_whether_it_waits_or_is_drawn_and_so_does_a_client_that_leaves(
        logged: io.StringIO) -> None:
    hold = asyncio.Event()  # never set: a job being drawn ends only when it is stopped
    async with running(Script(hold=hold)) as stack, httpx.AsyncClient(timeout=10) as client:
        drawn = Stream(client, stack, A)
        await until(lambda: stack.fake.running is not None)
        waiting = Stream(client, stack, A)
        await until(lambda: len(stack.fake.pending) == 1)
        for stream in (waiting, drawn):
            url = f"{stack.public}{PICTURES}/{await stream.accepted()}/stop"
            others = await call(client, "POST", url, headers=B)  # another reader's stop finds no such job
            assert (others.status, others.json()) == (404, {"error": {"code": "not_found"}})
            stopped = await call(client, "POST", url, headers=A)
            assert (stopped.status, stopped.json()) == (200, {"stopped": True})
            await stream.task  # it ends without an event: its client asked for the stop
            assert not stream.named("image") and not stream.named("over") and "[DONE]" not in stream.events
        assert (stack.fake.pending, stack.fake.running) == ([], None)
        again = await call(client, "POST", url, headers=A)  # a job the gateway no longer knows, which is stopped
        assert again.status == 404
        left = await RawClient.post(stack.public, PICTURES, {"graph": GRAPH}, headers=A)
        await until(lambda: stack.fake.running is not None)
        await left.leave()
        await until(lambda: stack.fake.running is None and not stack.service.jobs)
        await until(lambda: not stack.fake.history)  # the records of the stops, deleted once the stops were confirmed
    assert [(row["stop_confirmed"], row["cancelled"]) for row in pictures(logged)] == [(True, True)] * 3
    lines = [json.loads(line) for line in logged.getvalue().splitlines()]
    assert not [line for line in lines if line.get("level") == "ERROR"]  # and every response ended whole


async def test_a_stop_the_card_does_not_confirm_is_told_and_holds_its_classes_room_until_it_resolves(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    monkeypatch.setattr(comfy, "STOP_POLL_S", 0.01)
    limits = {"agent": {"jobs": 1, "wall_s": 1860, "uploads": 16},
              "internal": {"jobs": 32, "wall_s": 0.5, "uploads": 0}}
    async with (running(Script(outcomes={1: "hang", 2: "hang"}), limits=limits) as stack,
                httpx.AsyncClient(timeout=10) as client):
        cancel = stack.fake._cancel
        monkeypatch.setattr(stack.fake, "_cancel", lambda prompt_id: False)  # a card that lets no job go
        # An experiment past its wall time, whose stream still listens: the stop is not confirmed, and it says so.
        timed = await call(client, "POST", stack.public + PICTURES, headers=INTERNAL, body={"graph": GRAPH})
        assert timed.events()[-1] == {"error": {"code": "stop_unconfirmed"}}
        # An agent's job, waiting behind it, that its client stops.
        waiting = Stream(client, stack, AGENT)
        url = f"{stack.public}{PICTURES}/{await waiting.accepted()}/stop"
        for _ in range(2):  # the stop, and the same stop asked again
            stopped = await call(client, "POST", url, headers=AGENT)
            assert (stopped.status, stopped.json()) == (200, {"stopped": False})
        await waiting.task
        state = (await call(client, "GET", stack.control + "/v1/state", key=CONTROL)).json()
        assert (state["active"], sum(state["waiting"].values())) == (
            {"reader": 0, "agent": 1, "internal": 1, "external": 0}, 0)  # both may still be on the card
        full = await call(client, "POST", stack.public + PICTURES, headers=AGENT, body={"graph": GRAPH})
        assert (full.status, full.json()) == (429, {"error": {"code": "queue_full"}})
        monkeypatch.setattr(stack.fake, "_cancel", cancel)  # the card lets them go at last
        await until(lambda: len(rows(logged, "stop_resolved")) == 2)
        state = (await call(client, "GET", stack.control + "/v1/state", key=CONTROL)).json()
        assert sum(state["active"].values()) == 0
        room = await call(client, "POST", stack.public + PICTURES, headers=AGENT, body={"graph": GRAPH})
        assert room.events()[-1] == "[DONE]"
        await until(lambda: not stack.fake.history)
    assert [(row.get("code"), row["stop_confirmed"]) for row in pictures(logged)[:2]] == [
        ("stop_unconfirmed", False), (None, False)]
    assert sorted(row["class"] for row in rows(logged, "stop_resolved")) == ["agent", "internal"]


async def test_a_drain_stops_the_waiting_picture_at_once_and_lets_the_drawn_one_finish() -> None:
    hold = asyncio.Event()
    async with running(Script(hold=hold)) as stack, httpx.AsyncClient(timeout=10) as client:
        drawn = Stream(client, stack, A)
        await until(lambda: stack.fake.running is not None)
        waiting = Stream(client, stack, B)
        await until(lambda: len(stack.fake.pending) == 1)
        boot = (await call(client, "GET", stack.control + "/v1/state", key=CONTROL)).json()["boot_id"]
        drain = await call(client, "POST", stack.control + "/v1/control/drain", key=CONTROL, body={"boot_id": boot})
        assert drain.status == 202
        await waiting.task
        assert waiting.events[-1] == {"error": {"code": "draining"}} and stack.fake.pending == []
        refused = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": GRAPH})
        assert (refused.status, refused.json()) == (503, {"error": {"code": "draining"}})
        hold.set()
        await drawn.task
        assert drawn.named("image") and drawn.events[-1] == "[DONE]"
        await until(lambda: stack.service.status == "drained")


async def test_the_gateway_serves_the_engine_it_expects_and_stops_its_work_when_the_engine_changes(
        logged: io.StringIO) -> None:
    async with (running(Script(pytorch="2.10.0+cu128", outcomes={1: "hang"}), wait_ready=False) as stack,
                httpx.AsyncClient(timeout=10) as client):
        await until(lambda: stack.fake.calls.count(("GET", "/system_stats")) >= 3)
        for method, path, body in (("GET", ENGINE, None), ("POST", PICTURES, {"graph": GRAPH})):
            refused = await call(client, method, stack.public + path, headers=A, body=body)
            assert (refused.status, refused.json()) == (503, {"error": {"code": "starting"}})
        stack.fake.script.pytorch = "2.11.0+cu130"  # the pinned torch, as the card's engine has it
        await until(lambda: stack.service.status == "ready")
        drawn = Stream(client, stack, A)
        await until(lambda: stack.fake.running is not None)
        stack.fake.script.argv = ("main.py",)  # ComfyUI started again, without comfy-kitchen's Triton backend
        await drawn.task
        assert drawn.events[-1] == {"error": {"code": "engine_unavailable"}} and stack.fake.running is None
        refused = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": GRAPH})
        assert (refused.status, refused.json()) == (503, {"error": {"code": "engine_unavailable"}})
        stack.fake.script.argv = Script().argv
        await until(lambda: stack.service.status == "ready")
    assert [(row["event"], row["service_status"]) for row in rows(logged, "engine", "engine_changed")] == [
        ("engine", "ready"), ("engine_changed", "failed"), ("engine", "ready")]


async def test_without_its_socket_a_picture_comes_by_the_polls_unless_the_socket_is_required(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "SOCKET_WAIT_S", 0.1)
    async with running(Script(sockets=False)) as stack, httpx.AsyncClient(timeout=10) as client:
        required = await call(client, "POST", stack.public + PICTURES, headers=A,
                              body={"graph": GRAPH, "require_socket": True})
        assert (required.status, required.json()) == (503, {"error": {"code": "socket_unavailable"}})
        assert stack.fake.jobs == []  # nothing reached the card
        answer = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": GRAPH})
    events = answer.events()
    assert events[0]["socket"] is False and events[-1] == "[DONE]"
    assert [event["event"] for event in events[1:-1]] == ["started", "over", "image"]  # no node, no progress
    assert "stamp" not in events[1] and events[2]["heard"] is False


async def test_an_upload_is_its_readers_alone_and_is_erased_on_their_word_and_when_its_answer_is_lost(
        monkeypatch: pytest.MonkeyPatch, logged: io.StringIO) -> None:
    name, lost = f"{uuid.uuid4()}.png", f"{uuid.uuid4()}.png"
    graph = {**GRAPH, "6": {"class_type": "LoadImage", "inputs": {"image": f"bot-references/{name} [temp]"}}}
    png = {"Content-Type": "image/png"}
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        url = f"{stack.public}{PICTURES}/uploads/{name}"
        put = await call(client, "PUT", url, headers={**A, **png}, content=REFERENCE)
        assert (put.status, put.json()) == (200, {"name": f"bot-references/{name} [temp]"})
        assert stack.fake.files[("temp", "bot-references", name)] == REFERENCE
        for method in ("PUT", "DELETE"):  # another reader's send and erasure find no such name
            theirs = await call(client, method, url, headers={**B, **png},
                                content=REFERENCE if method == "PUT" else None)
            assert (theirs.status, theirs.json()) == (404, {"error": {"code": "not_found"}})
        theirs = await call(client, "POST", stack.public + PICTURES, headers=B, body={"graph": graph})
        assert (theirs.status, theirs.json()) == (400, {"error": {"code": "graph_refused"}})
        ours = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": graph})
        assert ours.events()[-1] == "[DONE]"
        erased = await call(client, "DELETE", url, headers=A)
        assert erased.status == 204 and stack.fake.files[("temp", "bot-references", name)] == BLANK
        gone = await call(client, "POST", stack.public + PICTURES, headers=A, body={"graph": graph})
        assert (gone.status, gone.json()) == (400, {"error": {"code": "graph_refused"}})
        for content, kind in ((b"\xff\xd8\xff synthetic", "image/png"), (REFERENCE, "image/jpeg")):
            wrong = await call(client, "PUT", url, headers={**A, "Content-Type": kind}, content=content)
            assert (wrong.status, wrong.json()) == (400, {"error": {"code": "invalid_request"}})
        upload = stack.fake._upload

        def answer_lost(content_type: str, body: bytes) -> tuple[int, Any]:
            upload(content_type, body)  # ComfyUI holds the file, and its answer does not come
            monkeypatch.setattr(stack.fake, "_upload", upload)
            return 500, {}

        monkeypatch.setattr(stack.fake, "_upload", answer_lost)
        failed = await call(client, "PUT", f"{stack.public}{PICTURES}/uploads/{lost}", headers={**A, **png},
                            content=REFERENCE)
        assert (failed.status, failed.json()) == (503, {"error": {"code": "engine_unavailable"}})
        await until(lambda: stack.fake.files.get(("temp", "bot-references", lost)) == BLANK)
    assert [row.get("references") for row in pictures(logged) if row["status"] == 200] == [1]


class UploadComfy(comfy.Comfy):
    """An upload can reach the card before its answer fails, or wait until this test lets it finish."""

    def __init__(self, hold: bytes | None = None, *, fail: bool = False) -> None:
        self.hold, self.fail = hold, fail
        self.started, self.release = asyncio.Event(), asyncio.Event()
        self.files: dict[str, bytes] = {}
        self.calls: list[bytes] = []

    async def upload(self, name: str, data: bytes, content_type: str, *, kind: str, subfolder: str) -> None:
        self.calls.append(data)
        if data == self.hold:
            self.started.set()
            await self.release.wait()
        self.files[name] = data
        if data == self.hold and self.fail:
            raise comfy.ComfyError("synthetic upload failure")


@pytest.mark.parametrize("older", ["failed_send", "expired_owner", "sweep"])
async def test_old_upload_cleanup_cannot_erase_a_newer_send(
        older: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = from_block({"keys": KEYS}, engine_url="http://127.0.0.1:1")
    caller, other = (resolve(config.keys[0], "reader", scope) for scope in ("reader.aaaaaaaa", "reader.bbbbbbbb"))
    first, second, name = REFERENCE + b"old", REFERENCE + b"new", f"{uuid.uuid4()}.png"
    fake = UploadComfy(first, fail=older != "sweep")
    background: list[Coroutine[Any, Any, None]] = []  # delay cleanup until the newer request has succeeded
    uploads = Uploads(fake, config.limits, background.append)
    absent = asyncio.Event()

    async def answered() -> None:
        return

    async def gone() -> None:
        await absent.wait()

    if older in {"expired_owner", "sweep"}:
        monkeypatch.setattr(uploads_module, "TTL_S", -1)
    initial = asyncio.create_task(uploads.put(caller, name, first, "image/png", gone, answered))
    await fake.started.wait()
    if older == "expired_owner":
        fake.release.set()
        with pytest.raises(ServiceError):
            await initial
        monkeypatch.setattr(uploads_module, "TTL_S", 600)
        await uploads.put(other, name, second, "image/png", gone, answered)
        current = other  # the expired name belongs to a different Upload object and scope now
    else:
        sweep = asyncio.create_task(uploads.sweep()) if older == "sweep" else None
        if sweep is not None:
            await asyncio.sleep(0)  # the sweep has captured the old expiry and waits for the send's lock
            monkeypatch.setattr(uploads_module, "TTL_S", 600)
        newer = asyncio.create_task(uploads.put(caller, name, second, "image/png", gone, answered))
        await asyncio.sleep(0)  # the new PUT claims the name before the old backend request completes
        fake.release.set()
        outcomes = await asyncio.gather(initial, newer, return_exceptions=True)
        assert outcomes[1] is None
        assert isinstance(outcomes[0], ServiceError) if older == "failed_send" else outcomes[0] is None
        if sweep is not None:
            await sweep
        current = caller
    await asyncio.gather(*background)
    assert uploads.owner(current)(name) and fake.files[name] == second
    assert BLANK not in fake.calls
    if older == "expired_owner":
        assert not uploads.owner(caller)(name)


@pytest.mark.parametrize("ending", ["response_complete", "response_failure", "response_disconnect",
                                    "cancel_during_send"])
async def test_an_upload_is_kept_only_if_its_http_answer_finishes(ending: str) -> None:
    config = from_block({"keys": KEYS}, engine_url="http://127.0.0.1:1")
    caller, name = resolve(config.keys[0], "reader", "reader.aaaaaaaa"), f"{uuid.uuid4()}.png"
    service = PictureService(config, FakeVast())  # never started: no health checks, sockets or card stop
    service.engine_status = "ready"
    fake = UploadComfy(REFERENCE if ending == "cancel_during_send" else None)
    background: list[Coroutine[Any, Any, None]] = []
    service.uploads = Uploads(fake, config.limits, background.append)
    disconnect, response_body, sender_stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    read = False
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal read
        if not read:
            read = True
            return {"type": "http.request", "body": REFERENCE, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(dict(message))
        if message["type"] == "http.response.body":
            response_body.set()
            if ending == "response_complete":
                disconnect.set()  # ASGI's disconnect can arrive with a fully completed response
                return
            if ending == "response_failure":
                raise BrokenPipeError("synthetic response failure")
            try:
                await asyncio.Event().wait()  # response completion is still outstanding when the client leaves
            finally:
                sender_stopped.set()

    exchange = Exchange({"headers": [(b"authorization", f"Bearer {BOT}".encode()),
                                      (b"content-type", b"image/png"),
                                      *((key.lower().encode(), value.encode()) for key, value in A.items())],
                         "path_params": {"name": name}, RECORD: RequestRecord(listener="public", method="PUT")},
                        receive, send)
    try:
        request = asyncio.create_task(routes.put_upload(service, exchange))
        if ending == "response_failure":
            with pytest.raises(BrokenPipeError):
                await request
        elif ending == "response_complete":
            await request
        elif ending == "response_disconnect":
            await response_body.wait()
            disconnect.set()
            await request
            assert exchange.record.cancelled and sender_stopped.is_set()
        else:
            await fake.started.wait()
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            cleanup = asyncio.gather(*background)
            await asyncio.sleep(0)
            assert fake.calls == [REFERENCE] and sent == []  # the submitted upload is allowed to finish first
            fake.release.set()
            await cleanup
            background.clear()
        await asyncio.gather(*background)
        kept = ending == "response_complete"
        assert fake.files[name] == (REFERENCE if kept else BLANK)
        assert service.uploads.owner(caller)(name) == kept
        assert fake.calls == ([REFERENCE] if kept else [REFERENCE, BLANK])
    finally:
        await service.comfy.aclose()


# Privacy: the gateway as its own process, as the card's launcher starts it.

async def test_no_answer_and_no_log_line_holds_a_graph_an_upload_or_the_engines_words(tmp_path: Path) -> None:
    marker = "Mk" + secrets.token_hex(8)  # letters and digits, so that it also fits a scope and a name
    scope = reader(f"reader.{marker}")
    fake = FakeComfy(Script(outcomes={2: "error", 3: "oom"}, unknown=frozenset({f"No{marker}"}), marker=marker))
    sock = bind(Listener("127.0.0.1", 0))
    engine = Servers([(fake, sock)], graceful_s=1, ws="websockets-sansio")
    await engine.start()
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps({
        **PROVISIONAL, "lane": "pictures", "alias": "dev-pictures", "reader_nodes": list(READER_NODES),
        "expect": CARD_ENGINE, "engine_url": f"http://127.0.0.1:{sock.getsockname()[1]}", "health_interval_s": 0.1,
        "keys": [{"sha256": hashlib.sha256(raw.encode()).hexdigest(), **entry} for raw, entry in KEYS.items()],
        "listen": {"public": {"host": "127.0.0.1", "port": 0}, "control": {"host": "127.0.0.1", "port": 0}}}))
    process = Process(path, "simple_serving.pictures")
    answers: list[bytes] = []
    try:
        await process.start()
        public = process.url("public")
        async with httpx.AsyncClient(timeout=10) as client:
            await until(lambda: (fake.calls.count(("GET", "/system_stats")) > 1))

            async def ask(status: int, method: str, route: str, headers: dict[str, str], **options: Any) -> Any:
                answer = await call(client, method, public + route, headers=headers, **options)
                assert answer.status == status, (route, answer.body[-300:])
                answers.append(answer.body + repr(answer.headers.multi_items()).encode())
                return answer

            # Pictures that end well, whose prompt and graph the card writes into the picture's text chunks.
            prompt = {**GRAPH, "2": {**GRAPH["2"], "inputs": {**GRAPH["2"]["inputs"], "text": f"a pier {marker}"}}}
            drawn = await ask(200, "POST", PICTURES, scope, body={"graph": prompt})
            png = base64.b64decode(named(drawn.events(), "image")[0]["png"])
            assert marker.encode() in fake.files[("temp", "", "ComfyUI_temp_00001_5_.png")]
            assert marker.encode() not in png
            # One whose engine raises and one out of memory, whose errors quote the prompt and the inputs.
            for oom in (False, True):
                failed = await ask(200, "POST", PICTURES, INTERNAL, body={"graph": prompt})
                assert failed.events()[-1] == {"error": {"code": "picture_failed", **({"oom": True} if oom else {})}}
            # Graphs the gateway refuses, and one ComfyUI rejects with its inputs quoted.
            for graph in ({"1": {"class_type": f"Save{marker}", "inputs": {"text": marker}}},
                          {"1": {"class_type": "LoadImage",
                                 "inputs": {"image": f"bot-references/{marker}.png [temp]"}}},
                          {marker: {"class_type": "KSampler", "inputs": {"text": f"{marker} [temp]"}}}):
                await ask(400, "POST", PICTURES, scope, body={"graph": graph})
            await ask(400, "POST", PICTURES, scope, body={"graph": GRAPH, marker: marker})
            await ask(400, "POST", PICTURES, scope, content=f'{{"graph": "{marker}'.encode())
            rejected = {"1": {"class_type": f"No{marker}", "inputs": {"text": marker}}}
            assert (await ask(400, "POST", PICTURES, INTERNAL, body={"graph": rejected})).json() == {
                "error": {"code": "graph_rejected"}}
            # Uploads: one whose bytes hold the marker, and malformed ones.
            uploads = f"{PICTURES}/uploads"
            await ask(200, "PUT", f"{uploads}/{uuid.uuid4()}.png", {**scope, "Content-Type": "image/png"},
                      content=PNG_SIGNATURE + marker.encode())
            await ask(400, "PUT", f"{uploads}/{uuid.uuid4()}.png", {**scope, "Content-Type": "image/png"},
                      content=marker.encode())
            await ask(400, "PUT", f"{uploads}/{marker}.png", {**scope, "Content-Type": "image/png"},
                      content=REFERENCE)
            await ask(400, "PUT", f"{PICTURES}/inputs/ref-{marker}.png", {**INTERNAL, "Content-Type": "image/png"},
                      content=PNG_SIGNATURE + marker.encode())
            await ask(404, "DELETE", f"{uploads}/{marker}.png", scope)
            await ask(404, "POST", f"{PICTURES}/{marker}/stop", scope)
    finally:
        await process.stop()
        await engine.stop()
    output = b"".join(process.lines)
    for text in (marker, BOT, CONTROL):
        assert text.encode() not in output
        assert not any(text.encode() in answer for answer in answers)
    requests = [row for row in process.rows() if row.get("event") == "request"]
    assert {row.get("code") for row in requests} >= {"graph_refused", "graph_rejected", "picture_failed",
                                                     "invalid_request", "unsupported_field", "not_found"}
    assert {row["scope"] for row in requests if "scope" in row} <= {"reader", "internal"}

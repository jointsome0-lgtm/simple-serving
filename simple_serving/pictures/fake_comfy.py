"""A stand-in for the pinned ComfyUI (v0.37.0 at 73c9bad4) on loopback, for the tests and the dev launcher: the routes
the picture gateway calls, one job at a time in the order the pinned server takes them, and the socket's messages in
the order it sends them, only to a socket that is open when they are sent. It models no card: a job's time, its
failure and the numbers on /system_stats are set by `Script`.

What a request asks is kept only as far as a test reads it: the jobs' ids, numbers and client ids, the uploads by
name, and the method and path of each call. A picture is a small grey PNG with the job's graph in a text chunk beside
its pixels, as a node that saves writes one, so that a test can see the gateway strip it; a refusal and an error quote
the graph's inputs, as ComfyUI's do, so that a test can see the gateway drop them.
"""

from __future__ import annotations

import asyncio
import email.parser
import email.policy
import heapq
import json
import re
import struct
import time
import uuid
import zlib
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

from starlette.types import Receive, Scope, Send

GIB = 1 << 30
PROMPT_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
CANCEL = re.compile(r"/api/jobs/([^/]+)/cancel")
SAVING = {"PreviewImage": "temp", "SaveImage": "output"}


@dataclass
class Script:
    """What the fake does. `outcomes` by the job's order of submit, from 1: "oom" and "error" fail at the sampler,
    "hang" draws until it is cancelled, "no_picture" succeeds without one. `unknown`: node classes it refuses, with
    400 and node errors that quote the inputs. `partial_loads`: the jobs that log a partial load. `hold`, when set,
    keeps every sampler from finishing until it is set again."""

    job_s: float = 0.05
    steps: int = 4
    outcomes: dict[int, str] = field(default_factory=dict)
    unknown: frozenset[str] = frozenset({"NoSuchNode"})
    partial_loads: frozenset[int] = frozenset()
    comfyui: str = "0.37.0"
    pytorch: str = "2.11.0+cu130"
    argv: tuple[str, ...] = ("main.py", "--enable-triton-backend")
    card: str = "cuda:0 Fake Card : cudaMallocAsync"
    attention: bool | None = True  # whether ModelAttentionBackend offers the kitchen's attention; None: no such node
    sockets: bool = True  # whether a socket may open
    marker: str = "fake-marker"  # in every picture's text chunk and every error
    hold: asyncio.Event | None = None


@dataclass(order=True)
class Job:
    number: float
    id: str
    ordinal: int = field(compare=False)
    graph: dict[str, Any] = field(compare=False)
    client_id: str = field(compare=False)
    interrupt: asyncio.Event = field(compare=False, default_factory=asyncio.Event)


class FakeComfy:
    """The raw ASGI application of the fake. Serve it with websockets on (`Servers(..., ws="websockets-sansio")`)."""

    def __init__(self, script: Script | None = None) -> None:
        self.script = script or Script()
        self.pending: list[Job] = []  # a heap, as the pinned server's
        self.running: Job | None = None
        self.history: dict[str, dict[str, Any]] = {}
        self.files: dict[tuple[str, str, str], bytes] = {}  # by type, subfolder and name
        self.jobs: list[Job] = []  # every job taken, in the order of its submit
        self.calls: list[tuple[str, str]] = []
        self.log: list[dict[str, str]] = []
        self._sockets: dict[str, asyncio.Queue[str | None]] = {}
        self._wake = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None

    # What a test does to the fake.

    def drop_sockets(self) -> None:
        """Close every socket from the server's side."""
        for queue in self._sockets.values():
            queue.put_nowait(None)

    @property
    def open_sockets(self) -> int:
        return len(self._sockets)

    def say(self, line: str) -> None:
        """A line of the server's log (app/logger.py keeps the last 300)."""
        self.log = [*self.log[-299:], {"t": f"{time.time():.6f}", "m": line + "\n"}]

    # The application.

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._work())
        if scope["type"] == "websocket":
            await self._socket(scope, receive, send)
        elif scope["type"] == "http":
            self.calls.append((scope["method"], scope["path"]))
            status, answer = await self._http(scope, receive)
            body = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
            kind = b"image/png" if isinstance(answer, bytes) else b"application/json"
            await send({"type": "http.response.start", "status": status,
                        "headers": [(b"content-type", kind), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})

    async def _http(self, scope: Scope, receive: Receive) -> tuple[int, Any]:
        method, path = scope["method"], scope["path"]
        query = {name: values[0] for name, values in parse_qs(scope["query_string"].decode()).items()}
        body = await _body(receive)
        if method == "POST" and path == "/prompt":
            return self._prompt(json.loads(body))
        if method == "GET" and path == "/queue":
            running = [[job.number, job.id, job.graph, {}, []] for job in [self.running] if job is not None]
            return 200, {"queue_running": running,
                         "queue_pending": [[job.number, job.id, job.graph, {}, []] for job in sorted(self.pending)]}
        if method == "GET" and path.startswith("/history/"):
            prompt_id = path.removeprefix("/history/")
            return 200, {prompt_id: self.history[prompt_id]} if prompt_id in self.history else {}
        if method == "POST" and path == "/history":
            for prompt_id in json.loads(body).get("delete", []):
                self.history.pop(prompt_id, None)
            return 200, {}
        if method == "POST" and (cancel := CANCEL.fullmatch(path)):
            return 200, {"cancelled": self._cancel(cancel[1])}
        if method == "POST" and path == "/upload/image":
            return self._upload(dict(scope["headers"]).get(b"content-type", b"").decode(), body)
        if method == "GET" and path == "/view":
            found = self.files.get((query.get("type", "output"), query.get("subfolder", ""), query.get("filename", "")))
            return (200, found) if found is not None else (404, {})
        if method == "GET" and path == "/system_stats":
            return 200, self._stats()
        if method == "GET" and path.startswith("/object_info/"):
            return 200, self._object_info(path.removeprefix("/object_info/"))
        if method == "GET" and path == "/internal/logs/raw":
            return 200, {"entries": self.log, "size": {"cols": 120, "rows": 40}}
        return 404, {}

    def _prompt(self, asked: dict[str, Any]) -> tuple[int, Any]:
        graph, prompt_id, number = asked.get("prompt"), asked.get("prompt_id"), asked.get("number")
        if not isinstance(prompt_id, str) or not PROMPT_ID.fullmatch(prompt_id) or not isinstance(graph, dict):
            return 400, {"error": {"type": "invalid_prompt", "message": "invalid prompt_id"}, "node_errors": {}}
        refused = {node_id: {"errors": [{"type": "invalid_node", "details": json.dumps(node.get("inputs"))}],
                             "class_type": node.get("class_type")}
                   for node_id, node in graph.items() if node.get("class_type") in self.script.unknown}
        if refused:
            return 400, {"error": {"type": "prompt_outputs_failed_validation", "message": "invalid"},
                         "node_errors": refused}
        job = Job(number=float(number) if isinstance(number, int | float) else float(len(self.jobs) + 1),
                  id=prompt_id, ordinal=len(self.jobs) + 1, graph=graph, client_id=str(asked.get("client_id", "")))
        self.jobs.append(job)
        heapq.heappush(self.pending, job)
        self.say("got prompt")
        self._wake.set()
        return 200, {"prompt_id": job.id, "number": job.number, "node_errors": {}}

    def _cancel(self, prompt_id: str) -> bool:
        waiting = next((job for job in self.pending if job.id == prompt_id), None)
        if waiting is not None:
            self.pending.remove(waiting)
            heapq.heapify(self.pending)
            return True
        if self.running is not None and self.running.id == prompt_id:
            self.running.interrupt.set()
            return True
        return False

    def _upload(self, content_type: str, body: bytes) -> tuple[int, Any]:
        parsed = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body)
        form: dict[str, tuple[str | None, bytes]] = {}
        for part in parsed.iter_parts():
            payload = part.get_payload(decode=True)
            form[str(part.get_param("name", header="content-disposition"))] = (
                part.get_filename(), payload if isinstance(payload, bytes) else b"")
        name, data = form.get("image", (None, b""))
        if not name:
            return 400, {}
        kind = form.get("type", (None, b"input"))[1].decode()
        subfolder = form.get("subfolder", (None, b""))[1].decode()
        self.files[(kind, subfolder, name)] = data
        return 200, {"name": name, "subfolder": subfolder, "type": kind}

    def _stats(self) -> dict[str, Any]:
        script = self.script
        return {"system": {"os": "posix", "ram_total": 64 * GIB, "ram_free": 40 * GIB,
                           "comfyui_version": script.comfyui, "python_version": "3.12",
                           "pytorch_version": script.pytorch, "embedded_python": False, "argv": list(script.argv)},
                "devices": [{"name": script.card, "type": "cuda", "index": 0, "vram_total": 32 * GIB,
                             "vram_free": 10 * GIB, "torch_vram_total": 23 * GIB, "torch_vram_free": 1 * GIB}]}

    def _object_info(self, name: str) -> dict[str, Any]:
        if name != "ModelAttentionBackend" or self.script.attention is None:
            return {}
        options = ["pytorch attention", *(["comfy kitchen attention"] if self.script.attention else [])]
        return {name: {"input": {"required": {"model": ["MODEL", {}], "attention": ["COMBO", {"options": options}]}}}}

    # The card.

    async def _work(self) -> None:
        while True:
            while not self.pending:
                self._wake.clear()
                await self._wake.wait()
            self.running = heapq.heappop(self.pending)
            await self._draw(self.running)

    async def _draw(self, job: Job) -> None:
        script, messages = self.script, []
        outcome = script.outcomes.get(job.ordinal, "success")

        def tell(kind: str, data: dict[str, Any], *, kept: bool = False) -> None:
            data = {**data, "prompt_id": job.id}
            if kept:
                messages.append([kind, data])
            self._tell(job.client_id, kind, data)

        tell("execution_start", {"timestamp": _now_ms()}, kept=True)
        tell("execution_cached", {"nodes": [], "timestamp": _now_ms()}, kept=True)
        outputs: dict[str, Any] = {}
        status = "success"
        for node_id, node in job.graph.items():
            kind = node.get("class_type")
            tell("executing", {"node": node_id, "display_node": node_id})
            if kind == "KSampler":
                if outcome in ("oom", "error"):
                    oom = outcome == "oom"
                    tell("execution_error", {
                        "node_id": node_id, "node_type": kind, "executed": [],
                        "exception_type": "torch.OutOfMemoryError" if oom else "RuntimeError",
                        "exception_message": f"{'Allocation on device' if oom else 'failed'} {script.marker}",
                        "traceback": [script.marker], "current_inputs": {"inputs": json.dumps(job.graph)},
                        "current_outputs": {}, "timestamp": _now_ms()}, kept=True)
                    status = "error"
                    break
                if job.ordinal in script.partial_loads:
                    self.say("loaded partially; 21000.00 MB usable, 10000.00 MB loaded, 2000.00 MB offloaded")
                if not await self._sample(job, node_id, hang=outcome == "hang"):
                    tell("execution_interrupted", {"node_id": node_id, "node_type": kind, "executed": [],
                                                   "timestamp": _now_ms()}, kept=True)
                    status = "error"
                    break
            elif kind in SAVING and outcome != "no_picture":
                image = {"filename": f"ComfyUI_temp_{job.ordinal:05d}_{node_id}_.png", "subfolder": "",
                         "type": SAVING[kind]}
                self.files[(image["type"], "", image["filename"])] = _png(json.dumps(job.graph) + script.marker)
                outputs[node_id] = {"images": [image]}
                tell("executed", {"node": node_id, "display_node": node_id, "output": outputs[node_id]})
        if status == "success":
            tell("execution_success", {"timestamp": _now_ms()}, kept=True)
        # The record is written as the job leaves the queue, before the socket hears it is over (main.py:367-374).
        self.history[job.id] = {"prompt": [job.number, job.id, job.graph, {"client_id": job.client_id}, []],
                                "outputs": outputs, "meta": {},
                                "status": {"status_str": status, "completed": status == "success",
                                           "messages": messages}}
        self.running = None
        self.say("Prompt executed")
        tell("executing", {"node": None})

    async def _sample(self, job: Job, node_id: str, *, hang: bool) -> bool:
        """The sampler's steps, or False once the job is cancelled."""
        steps = self.script.steps
        for step in range(1, steps + 1):
            try:
                async with asyncio.timeout(None if hang else self.script.job_s / steps):
                    await job.interrupt.wait()
                return False
            except TimeoutError:
                pass
            self._tell(job.client_id, "progress", {"value": step, "max": steps, "prompt_id": job.id, "node": node_id})
        while self.script.hold is not None and not self.script.hold.is_set():
            waits = {asyncio.ensure_future(self.script.hold.wait()), asyncio.ensure_future(job.interrupt.wait())}
            _, left = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            for one in left:
                one.cancel()
            if job.interrupt.is_set():
                return False
        return True

    # The socket.

    async def _socket(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (await receive())["type"] != "websocket.connect":
            return
        if not self.script.sockets:
            await send({"type": "websocket.close", "code": 1011})
            return
        await send({"type": "websocket.accept"})
        client_id = parse_qs(scope["query_string"].decode()).get("clientId", [""])[0] or uuid.uuid4().hex
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._sockets[client_id] = queue  # the pinned server files a socket under its id alone
        queue.put_nowait(json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 0}},
                                                                "sid": client_id}}))

        async def speak() -> None:
            while (text := await queue.get()) is not None:
                await send({"type": "websocket.send", "text": text})
            await send({"type": "websocket.close", "code": 1001})

        speaker = asyncio.create_task(speak())
        listener = asyncio.create_task(_until_disconnect(receive))
        try:
            await asyncio.wait({speaker, listener}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            speaker.cancel()
            listener.cancel()
            if self._sockets.get(client_id) is queue:
                del self._sockets[client_id]

    def _tell(self, client_id: str, kind: str, data: dict[str, Any]) -> None:
        queue = self._sockets.get(client_id)
        if queue is not None:
            queue.put_nowait(json.dumps({"type": kind, "data": data}))


async def _body(receive: Receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            return b"".join(chunks)


async def _until_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "websocket.disconnect":
        pass


def _now_ms() -> int:
    return int(time.time() * 1000)


def _png(text: str) -> bytes:
    """An 8 by 8 grey PNG with `text` in a tEXt chunk, where ComfyUI writes the prompt."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    rows = b"".join(b"\x00" + b"\x80" * 8 for _ in range(8))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 0, 0, 0, 0))
            + chunk(b"tEXt", b"prompt\x00" + text.encode("latin-1", "replace")) + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b""))

"""A picture, from its acceptance to its last event (docs/pictures.md, section 4).

The job goes to ComfyUI at once, under an id and a number of the gateway's, and is followed there: by the gateway's
one socket, which tells its start, its nodes, its steps and its over; by the service's shared read of the queue, which
tells its place while it waits; and by reads of its record while it draws, for what the socket cannot vouch for. The
card's memory is sampled while it draws. Once its record is written, `over` goes out with the memory, then the frame
and its copies, stripped, and the record is deleted. A job that ends any other way while ComfyUI may hold it is
stopped there by its id before its client hears the end, and the end says whether the stop is confirmed.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

from ..asgi import Exchange
from ..errors import ServiceError
from ..inference import Work
from ..policy import Caller
from . import graphs
from .comfy import ComfyError, Memory, Rejected, log_cursor, partial_loads_after, stop_job

if TYPE_CHECKING:
    from .service import PictureService, QueueView

SOCKET_WAIT_S = 2.0  # how long a job waits for the gateway's socket to be open before it is submitted
PLACE_S = 2.0  # a waiting job's place, every PLACE_S from its submit (simple-story-chat's PLACE_MS)
SAMPLE_S = 0.5  # the card's memory while the job draws
POLL_S = 0.5  # the job's record while it draws, when the socket cannot vouch for its news
VOUCHED_POLL_S = 5.0  # and when it can: the socket tells the over, and this read is the net under it
START_GRACE_S = 2.0  # a read of the queue without the job waits this long for the socket's word of its start
RECORD_TRIES = 3  # reads of the queue and the record that may fail in a row (simple-story-chat's POLL_RETRIES)
OUT_OF_MEMORY = re.compile(r"OutOfMemory|out of memory", re.IGNORECASE)  # simple-story-chat's `outOfMemory`
WAITING = ("preparing", "submitting", "pending")  # the phases a drain stops at once
# The stops whose client hears no end: it left, it asked for the stop itself, or the gateway stops.
UNHEARD = ("disconnect", "client", "shutdown")


class PictureFailed(ServiceError):
    """The job's record says it failed or names no picture, or its picture is not a PNG. `oom`: the card ran out of
    memory."""

    def __init__(self, *, oom: bool) -> None:
        super().__init__("picture_failed")
        self.oom = oom


@dataclass
class Heard:
    """What the socket told of the job, on the socket it was submitted under: its start and its success with ComfyUI's
    stamps, whether it ran out of memory, its outputs, and its over."""

    started: bool = False
    start_stamp: float | None = None
    succeeded: bool = False
    end_stamp: float | None = None
    over: bool = False
    oom: bool = False
    outputs: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any] | None:
        """The job's record without a read of it, once the socket has heard the whole job succeed: its outputs are the
        very objects the `executed` messages carried (simple-story-chat's `hearing`)."""
        if self.started and self.succeeded and self.over:
            return {"status": {"completed": True, "status_str": "success"}, "outputs": self.outputs}
        return None


class Picture(Work):
    """`POST /v1/pictures` once its checks have passed."""

    service: PictureService

    def __init__(self, service: PictureService, exchange: Exchange, caller: Caller, graph: dict[str, Any],
                 copies: list[str], references: int, require_socket: bool) -> None:
        super().__init__(service, exchange, caller)
        self.graph = graph
        self.copies = copies
        self.references = references
        self.require_socket = require_socket
        self.prompt_id = str(uuid.uuid4())
        self.phase = "preparing"  # then "submitting", "pending", "drawing" and "over"
        self.stop_confirmed: bool | None = None
        self._stopped: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        self._submit: asyncio.Future[None] | None = None
        self._sent = False  # ComfyUI may hold the job
        self._lost = False  # ComfyUI holds it nowhere
        self._forgotten = False
        self._epoch: int | None = None  # the socket's the job was submitted under, when it was open
        self._heard = Heard()
        self._inbox: collections.deque[dict[str, Any]] = collections.deque()
        self._news = asyncio.Event()
        self._cursor: str | None = None
        self._memory = Memory()
        self._drawn = asyncio.Event()
        self._sampler: asyncio.Task[None] | None = None
        self._submitted_at: float | None = None
        self._started_at: float | None = None
        self._over_at: float | None = None
        self._told_started = False
        # The queue as this job last read it: its place, and the job being drawn with since when this wait has seen
        # it (simple-story-chat's `readPlace`); when a read first no longer held it as waiting; whether its record is
        # to be read at once; and the reads that found it nowhere, or failed, in a row.
        self._view: QueueView | None = None
        self._before: int | None = None
        self._drawing: tuple[str, float] | None = None
        self._left_at: float | None = None
        self._look = False
        self._missing = 0
        self._failures = 0

    @property
    def active(self) -> bool:
        return self.phase in ("drawing", "over")

    def owned_by(self, caller: Caller) -> bool:
        return caller.key.label == self.caller.key.label and caller.scope == self.caller.scope

    def hear(self, message: dict[str, Any], epoch: int) -> None:
        """A message of the socket that names the job."""
        if epoch == self._epoch:
            self._inbox.append(message)
            self._news.set()

    def wake(self) -> None:
        """The service has read the queue again."""
        self._news.set()

    def stop(self, reason: str) -> None:
        if reason == "draining" and self.phase == "over":
            return  # its pictures are on their way: it ends as it would
        super().stop(reason)

    async def stop_by_client(self) -> bool:
        """The client's own stop (`POST /v1/pictures/{job}/stop`): whether the job is off the card, once that is
        known."""
        self.stop("client")
        return await asyncio.shield(self._stopped)

    # The job.

    async def run(self) -> None:
        service, record = self.service, self.exchange.record
        loop = asyncio.get_running_loop()
        service.jobs[self.prompt_id] = self
        socket = service.socket
        if not await socket.wait_open(SOCKET_WAIT_S) and self.require_socket:
            raise ServiceError("socket_unavailable")
        if self.caller.cls == "internal":
            with contextlib.suppress(ComfyError):
                self._cursor = log_cursor(await service.comfy.log_lines())
        if socket.is_open:
            self._epoch, client_id = socket.epoch, socket.client_id
        else:
            client_id = str(uuid.uuid4())  # ComfyUI tells this job's news to nobody
        record.socket = self._epoch is not None
        graph = graphs.with_nonce(self.graph, str(uuid.uuid4()))
        self.phase, self._sent = "submitting", True
        self._submit = asyncio.ensure_future(service.comfy.submit(
            graph, prompt_id=self.prompt_id, number=service.number(self.caller.cls), client_id=client_id))
        try:
            await asyncio.shield(self._submit)  # never cut short: a stop meanwhile waits for it in `cleanup`
        except ComfyError as error:
            self._sent = not isinstance(error, Rejected) and error.kind != "unreachable"
            raise ServiceError("graph_rejected" if isinstance(error, Rejected) else "engine_unavailable") from None
        self._submitted_at = loop.time()
        self.phase = "pending"
        service.read_queue()
        await self.exchange.start_stream()
        await self._send({"event": "accepted", "job": self.prompt_id, "socket": self._epoch is not None})
        await self._deliver(await self._follow())

    async def answer(self) -> None:
        await self.exchange.send_event("[DONE]", last=True)

    async def _follow(self) -> dict[str, Any]:
        """Follow the job until its record says it is over, and return the record."""
        loop = asyncio.get_running_loop()
        assert self._submitted_at is not None
        place_at = self._submitted_at + PLACE_S
        poll_at: float | None = None
        while True:
            self._news.clear()
            await self._take_news()
            record = self._heard.record()
            if record is not None:
                return record
            self._take_view()
            now = loop.time()
            if self.phase == "pending":
                if self._left_at is not None and (not self._vouched() or now >= self._left_at + START_GRACE_S):
                    await self._start(None)
                elif now >= place_at:
                    await self._place(now)
                    place_at += PLACE_S * math.floor((now - place_at) / PLACE_S + 1)
            if poll_at is None and (self.phase == "drawing" or self._missing):
                poll_at = now + (POLL_S if self._missing or not self._vouched() else VOUCHED_POLL_S)
            if self._look or (poll_at is not None and now >= poll_at):
                self._look, poll_at = False, None
                record = await self._poll()
                if record is not None:
                    return record
                continue
            moments = [poll_at] if poll_at is not None else []
            if self.phase == "pending":
                moments.append(place_at)
                if self._left_at is not None:
                    moments.append(self._left_at + START_GRACE_S)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout_at(min(moments) if moments else None):
                    await self._news.wait()

    def _vouched(self) -> bool:
        """Whether the socket can still tell the job's news: it was open at the submit and has not dropped since."""
        socket = self.service.socket
        return self._epoch is not None and socket.is_open and socket.epoch == self._epoch

    async def _take_news(self) -> None:
        """Act on what the socket told of the job since the last look, in the order it told it."""
        heard = self._heard
        while self._inbox:
            message = self._inbox.popleft()
            kind, data = message.get("type"), message.get("data")
            if not isinstance(data, dict):
                continue
            stamp = data.get("timestamp") if _number(data.get("timestamp")) else None
            if kind == "execution_start":
                heard.started, heard.start_stamp = True, stamp
                await self._start(stamp)
            elif kind == "execution_cached":
                nodes = data.get("nodes")
                if isinstance(nodes, list):
                    await self._send({"event": "cached", "nodes": [str(node) for node in nodes]})
            elif kind == "executing":
                node = data.get("node")
                if isinstance(node, str):
                    await self._send({"event": "node", "node": node})
                elif node is None:  # the record is written: the one message that means over (main.py:367-374)
                    heard.over = True
                    self._look = True
            elif kind == "executed":
                node, output = data.get("node"), data.get("output")
                if isinstance(node, str) and isinstance(output, dict):
                    heard.outputs[node] = output
            elif kind == "execution_success":
                heard.succeeded, heard.end_stamp = True, stamp
            elif kind in ("execution_error", "execution_interrupted"):
                heard.oom = heard.oom or _out_of_memory(data)
                self._look = True  # the record says what became of the job
            elif kind == "progress":
                value, most = data.get("value"), data.get("max")
                if _count(value) and _count(most) and most > 0 and value <= most:
                    await self._send({"event": "progress", "value": value, "max": most})

    def _take_view(self) -> None:
        """The service's latest read of the queue, if it began after the job was taken."""
        view = self.service.queue_view
        if view is None or view is self._view or self._submitted_at is None or view.began < self._submitted_at:
            return
        self._view = view
        queue = view.queue
        own = next((number for number, one in queue.pending if one == self.prompt_id), None)
        if own is None:
            if self._left_at is None:
                self._left_at = view.done
            if self.prompt_id not in queue.running:
                self._look = True  # over, or lost: the record says which
            return
        running = queue.running[0] if queue.running else None
        if running is None:
            self._drawing = None
        elif self._drawing is None or self._drawing[0] != running:
            self._drawing = (running, view.done)
        self._before = sum(1 for number, one in queue.pending if (number, one) < (own, self.prompt_id))

    async def _place(self, now: float) -> None:
        if self._before is None:
            return  # no read of the queue has held the job yet
        drawing = self._drawing
        await self._send({"event": "place", "before": self._before, "drawing": drawing is not None,
                          "drawing_ms": round((now - drawing[1]) * 1000) if drawing is not None else 0})

    async def _start(self, stamp: float | None) -> None:
        """The job waits no longer: `started`, with ComfyUI's stamp when the socket heard it, and the samples begin."""
        if self._told_started:
            return
        self._told_started = True
        if self.phase in WAITING:
            self._started_at = asyncio.get_running_loop().time()
            self.phase = "drawing"
            self._sampler = asyncio.create_task(self._sample())
        await self._send({"event": "started"} if stamp is None else {"event": "started", "stamp": stamp})

    async def _sample(self) -> None:
        """The card's memory every SAMPLE_S while the job draws, one read at a time."""
        while not self._drawn.is_set():
            with contextlib.suppress(ComfyError):
                self._memory.merge(await self.service.comfy.system_stats(), during=True)
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(SAMPLE_S):
                    await self._drawn.wait()

    async def _poll(self) -> dict[str, Any] | None:
        """A read of the queue and, when it does not hold the job, of the job's record: the record, once it says the
        job is over. A job in neither twice in a row is lost; reads that fail more than RECORD_TRIES times in a row end
        the job."""
        comfy = self.service.comfy
        try:
            queue = await comfy.queue()
            held = queue.holds(self.prompt_id)
            record = None if held else await comfy.history(self.prompt_id)
        except ComfyError:
            self._failures += 1
            if self._failures > RECORD_TRIES:
                raise ServiceError("engine_unavailable") from None
            return None
        self._failures = 0
        if held:
            self._missing = 0
            if self.prompt_id in queue.running and self._left_at is None:
                self._left_at = asyncio.get_running_loop().time()
            return None
        if record is None:
            self._missing += 1
            if self._missing > 1:
                self._lost = True
                raise ServiceError("engine_unavailable")
            return None
        self._missing = 0
        status = record.get("status")
        if isinstance(status, dict) and (status.get("completed") is True or status.get("status_str") == "error"):
            return record
        return None

    async def _deliver(self, record: dict[str, Any]) -> None:
        """The job is over: `over`, then the frame and its copies, and the record deleted once they are read."""
        exchange = self.exchange
        self.phase = "over"
        self._over_at = asyncio.get_running_loop().time()
        self._drawn.set()
        await self._start(None)
        status = record.get("status")
        status = status if isinstance(status, dict) else {}
        frame, copies = graphs.pictures_of(record, self.copies)
        if frame is None or status.get("status_str") == "error":
            raise PictureFailed(oom=self._heard.oom or _recorded_oom(status))
        _, loads = await asyncio.gather(self._last_sample(), self._partial_loads())
        heard = self._heard
        over: dict[str, Any] = {"event": "over", "heard": heard.started and heard.over, "memory": self._memory.event()}
        if heard.start_stamp is not None and heard.end_stamp is not None:
            over["stamps"] = {"start": heard.start_stamp, "end": heard.end_stamp}
        if self.caller.cls == "internal":
            over["partial_loads"] = exchange.record.partial_loads = loads
        await self._send(over)
        shown = [frame, *copies]
        exchange.record.images = exchange.record.image_bytes = 0
        for index, (node, image) in enumerate(shown):
            try:
                data = graphs.strip_png(await self.service.comfy.view(image))
            except ComfyError:
                raise ServiceError("engine_unavailable") from None
            except graphs.NotPng:
                raise PictureFailed(oom=False) from None
            if index == len(shown) - 1:
                self._forget()
            exchange.record.images += 1
            exchange.record.image_bytes += len(data)
            await self._send({"event": "image", "node": node, "png": base64.b64encode(data).decode("ascii")})

    async def _last_sample(self) -> None:
        """The job's last read of the memory, after the one still out if there is one: it adds to the maxima and not to
        the samples (simple-story-chat's `lastSample`)."""
        if self._sampler is not None:
            await asyncio.wait({self._sampler})
        with contextlib.suppress(ComfyError):
            self._memory.merge(await self.service.comfy.system_stats(), during=False)

    async def _partial_loads(self) -> int | None:
        """An experiment's partial loads since the read of the log before its submit; None when either read failed."""
        if self.caller.cls != "internal" or self._cursor is None:
            return None
        try:
            return partial_loads_after(self._cursor, await self.service.comfy.log_lines())
        except ComfyError:
            return None

    def _forget(self) -> None:
        """Delete the job's record, once, without waiting for its answer."""
        if not self._forgotten:
            self._forgotten = True
            self.service.track(self.service.comfy.forget(self.prompt_id))

    async def _send(self, event: dict[str, Any]) -> None:
        await self.exchange.send_event(event)

    # Its end.

    async def cleanup(self) -> None:
        """Take the job off the card unless its record said it was over or it is nowhere, and say whether that is
        confirmed. Only a job that is over, or whose stop is confirmed, has its record deleted."""
        self._drawn.set()
        try:
            if self._sampler is not None and not self._sampler.done():
                self._sampler.cancel()
                await asyncio.wait({self._sampler})
            if self._submit is not None:
                await asyncio.wait({self._submit})
                error = None if self._submit.cancelled() else self._submit.exception()
                if isinstance(error, Rejected) or (isinstance(error, ComfyError) and error.kind == "unreachable"):
                    self._sent = False
            if self._sent and self.phase != "over" and not self._lost:
                self.stop_confirmed = await stop_job(self.service.comfy, self.prompt_id)
                self.exchange.record.stop_confirmed = self.stop_confirmed
                if self.stop_confirmed:
                    self._forget()
                else:
                    self.service.unresolved_stop(self)
            elif self.phase == "over":
                self._forget()
        finally:
            self.service.jobs.pop(self.prompt_id, None)
            if not self._stopped.done():
                self._stopped.set_result(self.stop_confirmed is not False)
            self._record_times()

    async def _answer_outcome(self, task: asyncio.Task[None]) -> None:
        """A stop that was not confirmed ends a stream whose client still listens with `stop_unconfirmed`, whatever
        ended it; a card out of memory says so beside `picture_failed`; and the stream of a job its client stopped
        itself ends without an event."""
        error = None if task.cancelled() else task.exception()
        if isinstance(error, PictureFailed):
            self.exchange.record.oom = error.oom
        if self.stop_confirmed is False and self.stop_reason not in UNHEARD:
            self.exchange.record.cancelled = task.cancelled()
            await self.exchange.send_error("stop_unconfirmed")
        elif isinstance(error, PictureFailed) and error.oom and not self.exchange.ending:
            self.exchange.record.code = error.code
            failure = {"error": {"code": error.code, "oom": True}}
            if self.exchange.started:
                await self.exchange.send_event(failure, last=True)
            else:
                await self.exchange.send_json(error.status, failure)
        else:
            await super()._answer_outcome(task)
            if self.stop_reason == "client":
                await self.exchange.end_stream()

    def _record_times(self) -> None:
        record = self.exchange.record
        if self._submitted_at is not None and self._started_at is not None:
            record.queue_ms = int((self._started_at - self._submitted_at) * 1000)
        if self._started_at is not None and self._over_at is not None:
            record.draw_ms = int((self._over_at - self._started_at) * 1000)


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _count(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _out_of_memory(data: dict[str, Any]) -> bool:
    """Only this is taken from an error: the rest of it carries the node's inputs, which are the prompt."""
    return OUT_OF_MEMORY.search(f"{data.get('exception_type')} {data.get('exception_message')}") is not None


def _recorded_oom(status: dict[str, Any]) -> bool:
    messages = status.get("messages")
    return any(isinstance(message, list) and len(message) > 1 and message[0] == "execution_error"
               and isinstance(message[1], dict) and _out_of_memory(message[1])
               for message in (messages if isinstance(messages, list) else []))

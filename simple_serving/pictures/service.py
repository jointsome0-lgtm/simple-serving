"""The picture lane's service (docs/pictures.md, sections 4, 6 and 7): the text lane's boot, status, drain and sleep
(`Lifecycle`) in front of ComfyUI, with the engine's fingerprint as its health, one socket and one read of the queue
for all its jobs, the order of the classes on the card, the cap of each class, the stops not yet confirmed and the
readers' uploads.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from .. import CONTRACT, log, vast
from ..config import CLASSES
from ..errors import ServiceError
from ..inference import Work
from ..policy import Caller
from ..service import PINNED_PACKAGES, Lifecycle
from . import LANE
from .comfy import Comfy, ComfyError, Fingerprint, Queue, Socket
from .config import PictureConfig
from .jobs import WAITING, Picture
from .uploads import Uploads

ORDER = 10**9  # a reader's job below every other class's, an experiment's above (section 4, "Order on the card")
QUEUE_S = 1.0  # a read of the queue, while any job waits
CLOSE_S = 5.0  # how long a stop of the service waits for the deletes and the erasures still out


@dataclass(frozen=True)
class QueueView:
    """A read of ComfyUI's queue, with the event loop's time when it was asked and when it came."""

    began: float
    done: float
    queue: Queue


@dataclass
class Unresolved:
    """A job whose stop was not confirmed, which may still be on the card: its class, whose it was, and the looks in a
    row that found it nowhere."""

    cls: str
    key: str
    scope: str | None
    missing: int = 0

    def owned_by(self, caller: Caller) -> bool:
        return caller.key.label == self.key and caller.scope == self.scope


class PictureService(Lifecycle):
    config: PictureConfig
    pinned_packages = (*PINNED_PACKAGES, "websockets")

    def __init__(self, config: PictureConfig, stop: vast.Stop) -> None:
        super().__init__(config, stop)
        self.comfy = Comfy(config.engine_url)
        self.socket = Socket(config.engine_url, self._hear, self._socket_opened)
        self.uploads = Uploads(self.comfy, config.limits, self.track)
        self.jobs: dict[str, Picture] = {}  # by ComfyUI's id, from before the submit to the end of the cleanup
        self.unresolved: dict[str, Unresolved] = {}
        self.queue_view: QueueView | None = None
        self.fingerprint: Fingerprint | None = None  # the engine this boot verified
        self.kitchen_attention: bool | None = None
        self._numbered = 0
        self._background: set[asyncio.Task[None]] = set()
        self._read_queue = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._sweep: asyncio.Task[None] | None = None

    def start(self) -> None:
        super().start()
        self._tasks = [asyncio.create_task(self.socket.run()), asyncio.create_task(self._watch_queue())]

    async def _close_engine(self) -> None:
        tasks = [*self._tasks, *([self._sweep] if self._sweep is not None else [])]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(tasks)
        if self._background:
            await asyncio.wait(self._background, timeout=CLOSE_S)
        await self.comfy.aclose()

    def track(self, work: Coroutine[Any, Any, None]) -> None:
        """Run a request to ComfyUI that nobody waits for, such as the delete of a record; a stop of the service waits
        CLOSE_S for those still out. Its failure is dropped: the card's sweeper deletes what is left."""

        async def quietly() -> None:
            with contextlib.suppress(ComfyError):
                await work

        task = asyncio.create_task(quietly())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # Jobs.

    def number(self, cls: str) -> int:
        """The job's number on the card: ComfyUI draws the lowest first, and the later reader's picture before the
        earlier one's, as `front` puts the bot's today."""
        self._numbered += 1
        n = self._numbered
        return -(ORDER + n) if cls == "reader" else n if cls == "agent" else ORDER + n

    def accept(self, work: Work) -> None:
        """Take on a checked picture, if the service still serves and its class has room on the card: its accepted jobs
        and its stops not yet confirmed count."""
        self.check_serving()
        cls = work.caller.cls
        held = sum(1 for one in self.work if one.caller.cls == cls)
        held += sum(1 for one in self.unresolved.values() if one.cls == cls)
        if held >= self.config.limits[cls].jobs:
            raise ServiceError("queue_full")
        super().accept(work)

    def unresolved_stop(self, job: Picture) -> None:
        self.unresolved[job.prompt_id] = Unresolved(cls=job.caller.cls, key=job.caller.key.label,
                                                    scope=job.caller.scope)

    def read_queue(self) -> None:
        """A job was taken: the queue is read at once."""
        self._read_queue.set()

    def _hear(self, message: dict[str, Any], epoch: int) -> None:
        data = message.get("data")
        prompt_id = data.get("prompt_id") if isinstance(data, dict) else None
        job = self.jobs.get(prompt_id) if isinstance(prompt_id, str) else None
        if job is not None:
            job.hear(message, epoch)

    async def _watch_queue(self) -> None:
        """Read ComfyUI's queue at every submit and once a second while any job waits, one read for all of them."""
        loop = asyncio.get_running_loop()
        while True:
            waiting = any(job.phase == "pending" for job in self.jobs.values())
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(QUEUE_S if waiting else None):
                    await self._read_queue.wait()
            self._read_queue.clear()
            if not any(job.phase == "pending" for job in self.jobs.values()):
                continue
            began = loop.time()
            try:
                queue = await self.comfy.queue()
            except ComfyError:
                continue
            self.queue_view = QueueView(began=began, done=loop.time(), queue=queue)
            for job in list(self.jobs.values()):
                job.wake()

    # State.

    def state(self, *, control: bool) -> dict[str, Any]:
        state: dict[str, Any] = {"contract": CONTRACT, "lane": LANE, "boot_id": self.boot_id, "status": self.status,
                                 "model": self.config.alias, "drain_generation": self.drain_generation}
        if control:
            active, waiting = dict.fromkeys(CLASSES, 0), dict.fromkeys(CLASSES, 0)
            for job in self.jobs.values():
                (waiting if job.phase in WAITING else active)[job.caller.cls] += 1
            for stop in self.unresolved.values():
                active[stop.cls] += 1
            state.update(sleep_requested=self.sleep_requested, active=active, waiting=waiting,
                         versions=self._versions())
        return state

    def engine_facts(self) -> dict[str, Any]:
        """`GET /v1/pictures/engine`: the engine as this boot verified it, never a request of its own to ComfyUI."""
        seen = self.fingerprint
        if seen is None:
            raise ServiceError("starting")
        return {"comfyui": seen.comfyui, "pytorch": seen.pytorch, "card": seen.card, "triton": seen.triton,
                "kitchen_attention": self.kitchen_attention}

    # Health.

    async def _watch_engine(self) -> None:
        await self._watch_checks(self._check_engine)

    async def _check_engine(self) -> None:
        """ComfyUI's fingerprint (section 6): before the first ready it must have the expected versions and Triton;
        after it, it must be the one this boot kept. Each check also looks again at the stops not yet confirmed, and
        erases the uploads whose ten minutes are over."""
        if self._sweep is None or self._sweep.done():
            self._sweep = asyncio.create_task(self.uploads.sweep())
        seen = Fingerprint.of(await self.comfy.system_stats())
        await self._resolve_stops()
        if self.fingerprint is None:
            expect = self.config.expect
            if (seen.comfyui, seen.pytorch, seen.triton) != (expect.comfyui, expect.pytorch, expect.triton):
                return  # not verified: the service stays starting
            with contextlib.suppress(ComfyError):
                self.kitchen_attention = await self.comfy.attention_offered()
            self.fingerprint = seen
            self._first_verified()
        elif seen != self.fingerprint:
            self._engine_changes()
            return
        self._engine_ready(lane=LANE)

    async def _resolve_stops(self) -> None:
        """Each stop not confirmed, asked again: resolved once the job's record is there, which is then deleted, or
        once neither the queue nor the history has held the job twice in a row."""
        for prompt_id, stop in list(self.unresolved.items()):
            with contextlib.suppress(ComfyError):
                await self.comfy.cancel(prompt_id)
                if (await self.comfy.queue()).holds(prompt_id):
                    stop.missing = 0
                    continue
                recorded = await self.comfy.history(prompt_id) is not None
                stop.missing += 1
                if recorded or stop.missing > 1:
                    if recorded:
                        self.track(self.comfy.forget(prompt_id))
                    del self.unresolved[prompt_id]
                    log.row("stop_resolved", lane=LANE, **{"class": stop.cls})

    def _socket_opened(self) -> None:
        """A socket that opens anew may face a ComfyUI started again: once the engine is verified, its attention offer
        is read again."""
        if self.fingerprint is not None:
            self.track(self._read_attention())

    async def _read_attention(self) -> None:
        self.kitchen_attention = await self.comfy.attention_offered()

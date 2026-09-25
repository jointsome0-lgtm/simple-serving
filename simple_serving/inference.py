"""A generation or a count after its checks, until its terminal event (contract sections 4, 5, 7, 8, 9 and 11).

Once its checks have passed the gateway accepts the request: its wall time starts, its measurements count from here,
and the service holds it as work that a drain must see end. The work runs in a task of its own, so that a client that
leaves, a wall time that runs out or a drain can stop it at any await: while it waits for a count place or is counted,
while it waits for a place, while the engine reads the prompt and while it streams. The client is watched on `receive`
the whole time, since nothing is sent while the prompt is read. After the task has ended, its engine request is
closed, which is the abort, and only then is its place freed: a local end, not a confirmation from the engine.

The terminal message goes out after that: the usage chunk and `[DONE]`, the count, or an error. When the client does
not read, that send waits without an engine request or a place, and not past a stop: once the work is stopped, a send
that waits for the client is given up, and the server closes the connection.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from .admission import Ticket
from .asgi import Exchange
from .engine import EngineError, EngineStream
from .errors import ServiceError
from .policy import Caller
from .stream import Translator, first_error
from .validation import ChatRequest

if TYPE_CHECKING:
    from .service import Service

# What the client is told when its request is stopped. A client that left is told nothing.
STOP_CODES = {"timeout": "timeout", "draining": "draining", "engine_changed": "engine_unavailable"}


class Work:
    """The part of a request that may be stopped from outside."""

    def __init__(self, service: Service, exchange: Exchange, caller: Caller, request: ChatRequest) -> None:
        self.service = service
        self.exchange = exchange
        self.caller = caller
        self.request = request
        self.accepted_at: float | None = None
        self.stop_reason: str | None = None
        self._stopped_at: float | None = None  # the event loop's time of the stop
        self._task: asyncio.Task[None] | None = None
        self._terminal_send: asyncio.Timeout | None = None  # while the terminal message is sent; a stop cuts it
        self._count_place: Ticket | None = None

    @property
    def active(self) -> bool:
        """Whether the engine has the request's main call (section 7). A drain lets our active work finish."""
        raise NotImplementedError

    @property
    def waiting(self) -> bool:
        """Whether the task runs and the engine does not have the main call yet: the work is counted or waits for a
        place. A drain stops it."""
        return self._task is not None and not self._task.done() and not self.active

    def stop(self, reason: str) -> None:
        """Stop the work: "disconnect", "timeout", "draining", "engine_changed" or "shutdown". Only the first stop
        counts. It cancels the task, whose cleanup runs once it has stopped, and from then on a send of the terminal
        message that waits for the client is given up."""
        if self._task is None or self.stop_reason is not None:
            return
        self.stop_reason = reason
        self._stopped_at = asyncio.get_running_loop().time()
        if not self._task.done():
            self._stopping()
            self._task.cancel()
        elif self._terminal_send is not None:
            self._terminal_send.reschedule(self._stopped_at)

    async def serve(self) -> None:
        """Run the request, which the service has accepted, to its terminal event."""
        self.accepted_at = time.monotonic()
        wall_s = self.service.config.limits[self.caller.cls].wall_s
        wall_timer = asyncio.get_running_loop().call_later(wall_s, self.stop, "timeout")
        watcher = asyncio.create_task(self._watch_client())
        try:
            task = await self._run()
            await self._send_terminal(task)
        finally:
            watcher.cancel()
            wall_timer.cancel()
            self.service.release(self)

    async def run(self) -> None:
        raise NotImplementedError

    async def answer(self) -> None:
        """The answer of work that ended well, sent after its task and its cleanup."""

    async def cleanup(self) -> None:
        """Runs after the task has ended, however it ended."""

    def _stopping(self) -> None:
        """Runs as the work is stopped, before its task is cancelled."""

    async def count(self) -> int:
        """The engine's count of the prompt, taken in a count place (section 5)."""
        places = self.service.count_places
        self._count_place = places.enter(self.caller.cls, self.caller.outside_key)
        try:
            await self._count_place.granted
            return await self.service.engine.count(self.service.config.alias, self.request)
        finally:
            places.leave(self._count_place)  # the engine call has ended: it returned, failed or was cancelled

    async def _run(self) -> asyncio.Task[None]:
        """Run the work's task to its end, then clean up."""
        task = self._task = asyncio.create_task(self.run())
        try:
            await asyncio.wait({task})
        finally:
            if not task.done():  # the server shuts down and cancelled this handler
                self.stop("shutdown")
                await asyncio.wait({task})
            await self.cleanup()
        return task

    async def _watch_client(self) -> None:
        await self.exchange.disconnected()
        self.stop("disconnect")

    async def _send_terminal(self, task: asyncio.Task[None]) -> None:
        """Send the terminal message of the task's outcome. After a stop it goes out only if the connection takes it
        at once: a client that does not read is given up."""
        send = self._terminal_send = asyncio.timeout_at(self._stopped_at)
        try:
            async with send:
                await self._answer_outcome(task)
        except TimeoutError:
            if not send.expired():
                raise
            self.exchange.record.cancelled = True
        finally:
            self._terminal_send = None

    async def _answer_outcome(self, task: asyncio.Task[None]) -> None:
        record = self.exchange.record
        if task.cancelled():
            record.cancelled = True
            code = STOP_CODES.get(self.stop_reason or "")
            if code is not None:
                await self.exchange.send_error(code)
            return
        error = task.exception()
        if error is None:
            await self.answer()
        elif isinstance(error, ServiceError):
            await self.exchange.send_error(error.code)
        else:
            record.exception = type(error).__name__  # the class only: an exception's text may quote the request
            await self.exchange.send_error("internal_error")


class Count(Work):
    """`POST /v1/chat/completions/input_tokens`: no class limits and no context (section 5)."""

    @property
    def active(self) -> bool:
        return self._count_place is not None and self._count_place.active

    async def run(self) -> None:
        self.exchange.record.input_tokens = await self.count()

    async def answer(self) -> None:
        await self.exchange.send_json(200, {"input_tokens": self.exchange.record.input_tokens})


class Generation(Work):
    """`POST /v1/chat/completions`: counted, checked against the limits, then a place and the engine's stream."""

    def __init__(self, service: Service, exchange: Exchange, caller: Caller, request: ChatRequest) -> None:
        super().__init__(service, exchange, caller, request)
        self.ticket: Ticket | None = None
        self.stream: EngineStream | None = None
        self._usage_chunk: dict[str, Any] | None = None  # the last chunk, once the engine's stream has ended well
        self._handed_at: float | None = None
        self._first_token_at: float | None = None

    @property
    def active(self) -> bool:
        return self.ticket is not None and self.ticket.active

    async def run(self) -> None:
        record = self.exchange.record
        record.input_tokens = await self.count()
        self.service.check_input(self.caller, record.input_tokens, self.request.max_tokens)
        self.ticket = self.service.admission.enter(self.caller.cls, self.caller.outside_key)
        await self.ticket.granted
        self._handed_at = time.monotonic()
        self.stream = await self.service.engine.generate(self.service.payload(self.caller, self.request))
        await self._relay(self.stream)

    async def answer(self) -> None:
        assert self._usage_chunk is not None
        await self.exchange.send_event(self._usage_chunk)
        await self.exchange.send_event("[DONE]", last=True)

    def _stopping(self) -> None:
        if self.ticket is not None and self.ticket.waiting:
            self.service.admission.leave(self.ticket)  # it holds no engine request, so its place in line frees at once

    async def cleanup(self) -> None:
        try:
            if self.stream is not None:
                await self.stream.aclose()  # the abort: once it returns, our engine request has ended locally
        except Exception as error:  # noqa: BLE001 - httpx does not map close errors; the client is answered anyway
            self.exchange.record.exception = type(error).__name__
        finally:
            if self.ticket is not None:
                self.service.admission.leave(self.ticket)
        if self.exchange.record.total_ms is None:
            self._record_times(self._measure(time.monotonic()))  # a stopped or failed request, for its log row

    async def _relay(self, stream: EngineStream) -> None:
        """Send 200 with the engine's first event, then each event as it comes. At the engine's end the usage chunk is
        built, which the answer sends with [DONE]."""
        translator = Translator(self.service.config.alias)
        event = await stream.next_event()
        if event is None:  # [DONE] before any event
            raise EngineError("engine_unavailable")
        if "error" in event:
            raise first_error(event["error"])
        chunks = translator.chunks(event)
        await self.exchange.start_stream()
        while True:
            if self._first_token_at is None and (translator.generated or translator.finish_reason is not None):
                self._first_token_at = time.monotonic()
            for chunk in chunks:
                await self.exchange.send_event(chunk)
            event = await stream.next_event()
            if event is None:
                break
            chunks = translator.chunks(event)
        usage = translator.end()
        record = self.exchange.record
        record.finish = translator.finish_reason
        record.output_tokens = usage.completion_tokens
        record.cached_tokens = usage.cached_tokens
        record.count_matches = usage.prompt_tokens == record.input_tokens
        times = self._measure(time.monotonic())
        self._record_times(times)
        self._usage_chunk = translator.usage_chunk(usage, times)

    def _measure(self, end: float) -> dict[str, int]:
        """The measurements of section 11, in whole milliseconds from acceptance. A stream without any token measures
        its first token when its finish arrived; for a request stopped earlier, the steps it did not reach end at
        `end`."""
        assert self.accepted_at is not None
        accepted = self.accepted_at

        def ms(moment: float | None) -> int:
            return int(((end if moment is None else moment) - accepted) * 1000)

        return {"wait_ms": ms(self._handed_at), "first_token_ms": ms(self._first_token_at), "total_ms": ms(end)}

    def _record_times(self, times: dict[str, int]) -> None:
        record = self.exchange.record
        record.wait_ms, record.first_token_ms, record.total_ms = (
            times["wait_ms"], times["first_token_ms"], times["total_ms"])

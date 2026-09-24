"""The service's own state: its boot, status and drain (contract sections 6 and 8), the engine's health, the checks
that every inference request passes (sections 4, 5 and 7), and the work it has accepted."""

from __future__ import annotations

import asyncio
import platform
import secrets
from importlib import metadata
from typing import TYPE_CHECKING, Any

from . import CONTRACT, log
from .admission import Admission, CountPlaces
from .asgi import Exchange
from .config import Config, Key
from .engine import EngineClient, generation_payload
from .errors import ServiceError
from .policy import CLASS_HEADER, SCOPE_HEADER, Caller, cache_salt, find_key, resolve
from .validation import ChatRequest

if TYPE_CHECKING:
    from .inference import Work

STARTING_RETRY_S = 1.0
PINNED_PACKAGES = ("fastapi", "starlette", "uvicorn", "httpx")


class Service:
    def __init__(self, config: Config, engine: EngineClient) -> None:
        self.config = config
        self.engine = engine
        self.boot_id = secrets.token_hex(16)
        self._salt_secret = secrets.token_bytes(32)  # made at every start, never shown
        self.drain_generation = 0
        self.draining = False
        self.engine_status = "starting"  # then "ready", and "failed" while checks fail or once the engine has changed
        self.context_tokens: int | None = None
        self._engine_length: int | None = None  # the engine's own context length, as this boot verified it
        self._engine_changed = False
        self.admission = Admission(config)
        self.count_places = CountPlaces(config)
        self.work: set[Work] = set()  # accepted generations and counts that have not answered yet
        self._deadline: asyncio.TimerHandle | None = None
        self._watch: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._watch = asyncio.create_task(self._watch_engine())

    async def close(self) -> None:
        if self._deadline is not None:
            self._deadline.cancel()
        if self._watch is not None:
            self._watch.cancel()
            await asyncio.wait({self._watch})
        await self.engine.aclose()

    @property
    def status(self) -> str:
        if self.draining:  # whatever the engine does
            return "draining" if self.work else "drained"  # the gateway has no work; the GPU may not be idle yet
        return self.engine_status

    # The checks of an inference request, in the order of section 4.

    def authenticate(self, exchange: Exchange) -> Key:
        key = find_key(self.config, exchange.headers.get("authorization"))
        if key is None:
            raise ServiceError("unauthorized")
        exchange.record.key = key.label
        return key

    def authorize(self, exchange: Exchange) -> Caller:
        """The key, the service status, the class and the scope of an inference request."""
        key = self.authenticate(exchange)
        self.check_serving()
        caller = resolve(key, exchange.header(CLASS_HEADER), exchange.header(SCOPE_HEADER))
        exchange.record.cls, exchange.record.scope = caller.cls, caller.scope_kind
        return caller

    def check_serving(self) -> None:
        status = self.status
        if status != "ready":
            raise ServiceError("engine_unavailable" if status == "failed" else status)

    def check_max_tokens(self, caller: Caller, request: ChatRequest) -> None:
        if request.max_tokens > self.config.limits[caller.cls].max_tokens:
            raise ServiceError("limit_exceeded")

    def accept(self, work: Work) -> None:
        """Take on a checked request. A drain that began while its body was read refuses it here."""
        self.check_serving()
        self.work.add(work)

    def release(self, work: Work) -> None:
        """The work has answered."""
        self.work.discard(work)

    def check_input(self, caller: Caller, input_tokens: int, max_tokens: int) -> None:
        limit = self.config.limits[caller.cls].input_tokens
        if limit is not None and input_tokens > limit:
            raise ServiceError("limit_exceeded")  # our quota
        assert self.context_tokens is not None  # known once the engine was verified
        if input_tokens + max_tokens > self.context_tokens:
            raise ServiceError("context_limit")  # the model's size

    def payload(self, caller: Caller, request: ChatRequest) -> dict[str, Any]:
        return generation_payload(self.config.alias, request, priority=self.config.engine_priority[caller.cls],
                                  cache_salt=cache_salt(self._salt_secret, caller))

    # Models, state and the control routes.

    def models(self) -> dict[str, Any]:
        if self.context_tokens is None:
            raise ServiceError("starting")
        return {"object": "list",
                "data": [{"id": self.config.alias, "object": "model", "max_model_len": self.context_tokens}]}

    def state(self, *, control: bool) -> dict[str, Any]:
        state: dict[str, Any] = {"contract": CONTRACT, "boot_id": self.boot_id, "status": self.status,
                                 "model": self.config.alias, "context_tokens": self.context_tokens,
                                 "drain_generation": self.drain_generation}
        if control:
            state.update(active=self.admission.active_counts(), waiting=self.admission.waiting_counts(),
                         versions=self._versions())
        return state

    def drain(self, boot_id: str) -> dict[str, Any]:
        """Refuse new work, stop waiting work and outside work, and let the rest of our work finish until the
        deadline. With no work at all the drain is complete before it answers."""
        self._check_boot(boot_id)
        if self.draining:  # a repeated drain answers the current state
            return self._control_answer()
        self.draining = True
        self.drain_generation += 1
        log.row("drain", drain_generation=self.drain_generation)
        for work in list(self.work):
            if work.waiting or work.caller.key.outside:
                work.stop("draining")
        self._deadline = asyncio.get_running_loop().call_later(self.config.drain_deadline_s, self._drain_deadline)
        return self._control_answer()

    def open(self, boot_id: str, drain_generation: int) -> dict[str, Any]:
        self._check_boot(boot_id)
        if drain_generation != self.drain_generation:
            raise ServiceError("stale_generation")  # a late open cannot undo a newer drain
        if self.draining:
            self.draining = False
            if self._deadline is not None:
                self._deadline.cancel()
                self._deadline = None
            log.row("open", drain_generation=self.drain_generation)
        return self._control_answer()

    def _drain_deadline(self) -> None:
        self._deadline = None
        log.row("drain_deadline", drain_generation=self.drain_generation)
        for work in list(self.work):
            work.stop("draining")

    def _check_boot(self, boot_id: str) -> None:
        if boot_id != self.boot_id:
            raise ServiceError("stale_boot")

    def _control_answer(self) -> dict[str, Any]:
        return {"status": self.status, "boot_id": self.boot_id, "drain_generation": self.drain_generation}

    def _versions(self) -> dict[str, str]:
        """The pinned versions of section 12: the gateway's own packages, and the engine's from the configuration."""
        versions = {name: metadata.version(name) for name in PINNED_PACKAGES}
        versions["python"] = platform.python_version()
        versions.update(self.config.versions)
        return versions

    async def _watch_engine(self) -> None:
        while True:
            try:
                length = await self.engine.context_length(self.config.alias)
            except Exception as error:
                self._check_failed(error)
            else:
                self._engine_answered(length)
            interval = self.config.health_interval_s
            await asyncio.sleep(min(interval, STARTING_RETRY_S) if self.engine_status == "starting" else interval)

    def _check_failed(self, error: Exception) -> None:
        """A failed check makes a ready service failed, and new work is refused. Accepted work goes on within its wall
        time: the engine may only be busy, and an error from the engine ends its own request."""
        if self.engine_status == "ready":
            self.engine_status = "failed"
            log.row("engine", service_status="failed", exception=type(error).__name__)

    def _engine_answered(self, length: int | None) -> None:
        """The engine answered a check with the alias's context length, or with None: it serves another model."""
        if self._engine_length is None:
            if length is None:
                return  # not verified: the service stays starting
            self._engine_length = length
            limit = self.config.context_tokens
            self.context_tokens = min(length, limit) if limit is not None else length
        elif length != self._engine_length:
            self._engine_changes()
            return
        self._engine_changed = False
        if self.engine_status != "ready":
            self.engine_status = "ready"
            log.row("engine", service_status="ready", context_tokens=self.context_tokens)

    def _engine_changes(self) -> None:
        """The engine serves another model or another context than this boot verified. The work in flight was checked
        against an engine that is gone: the service fails, and all its work ends with engine_unavailable. It is ready
        again once the engine serves the verified model and context again."""
        if self._engine_changed:
            return
        self._engine_changed = True
        self.engine_status = "failed"
        log.row("engine_changed", service_status="failed")
        for work in list(self.work):
            work.stop("engine_changed")

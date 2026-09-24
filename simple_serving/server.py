"""Running the gateway: the service and two uvicorn servers, public and control, in one process and one event loop."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import sys
from collections.abc import Iterator
from types import TracebackType
from typing import Self

import uvicorn
from starlette.types import ASGIApp

from . import log
from .app import control_app, public_app
from .config import ENV_VAR, Config, ConfigError, Listener, load
from .engine import EngineClient
from .service import Service

GRACEFUL_SHUTDOWN_S = 5


class Server(uvicorn.Server):
    """A uvicorn server that leaves signals to its owner, which stops several servers together."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def bind(listener: Listener) -> socket.socket:
    family = socket.AF_INET6 if ":" in listener.host else socket.AF_INET
    # IPPROTO_TCP, not the default 0: asyncio sets TCP_NODELAY only on sockets that name TCP, and accepted
    # connections inherit it. Without it every streamed chunk waits for a delayed ACK, about 40 ms.
    sock = socket.socket(family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((listener.host, listener.port))
    except OSError:
        sock.close()
        raise
    return sock


class Servers:
    """ASGI applications served by uvicorn on sockets that are already bound, started and stopped together."""

    def __init__(self, apps: list[tuple[ASGIApp, socket.socket]], *, graceful_s: int = GRACEFUL_SHUTDOWN_S) -> None:
        # No access log: the gateway writes its own rows. No lifespan: the owner starts and stops the service.
        self.servers = [Server(uvicorn.Config(app, http="h11", ws="none", lifespan="off", log_config=None,
                                              access_log=False, proxy_headers=False, server_header=False,
                                              timeout_graceful_shutdown=graceful_s)) for app, _ in apps]
        self._sockets = [sock for _, sock in apps]
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(server.serve(sockets=[sock]))
                       for server, sock in zip(self.servers, self._sockets, strict=True)]
        while not all(server.started for server in self.servers):
            for task in self._tasks:
                if task.done():
                    await self.stop()
                    raise RuntimeError("a server stopped while it started")
            await asyncio.sleep(0.005)

    async def stop(self) -> None:
        for server in self.servers:
            server.should_exit = True
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for sock in self._sockets:
            sock.close()


class Gateway:
    """The service and its two listeners. `async with Gateway(config)` starts them; leaving the block stops them."""

    def __init__(self, config: Config, *, graceful_s: int = GRACEFUL_SHUTDOWN_S) -> None:
        self.config = config
        self.service = Service(config, EngineClient(config.engine_url))
        self.ports: dict[str, int] = {}
        self._graceful_s = graceful_s
        self._servers: Servers | None = None

    async def __aenter__(self) -> Self:
        sockets = {"public": bind(self.config.public), "control": bind(self.config.control)}
        self.ports = {name: sock.getsockname()[1] for name, sock in sockets.items()}
        public, control = public_app(self.service), control_app(self.service)
        self._servers = Servers([(public, sockets["public"]), (control, sockets["control"])],
                                graceful_s=self._graceful_s)
        public_server = self._servers.servers[0]
        public.open_connections = lambda: len(public_server.server_state.connections)
        log.row("boot", boot_id=self.service.boot_id)
        self.service.start()
        try:
            await self._servers.start()
        except BaseException:
            await self.service.close()
            raise
        for name, port in self.ports.items():
            log.row("listening", listener=name, port=port)
        return self

    async def __aexit__(self, kind: type[BaseException] | None, error: BaseException | None,
                        traceback: TracebackType | None) -> None:
        if self._servers is not None:
            await self._servers.stop()
        await self.service.close()


def main() -> None:
    log.setup()
    path = os.environ.get(ENV_VAR)
    if not path:
        sys.exit(f"simple-serving: {ENV_VAR} must name the configuration file")
    try:
        config = load(path)
    except ConfigError as error:
        sys.exit(f"simple-serving: {error}")
    asyncio.run(serve(config))


async def serve(config: Config) -> None:
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(log.loop_exception)
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    async with Gateway(config):
        await stop.wait()

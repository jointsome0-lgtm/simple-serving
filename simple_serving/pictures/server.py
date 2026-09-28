"""Running the picture gateway (`python -m simple_serving.pictures`): its service and two listeners in one process,
as the text gateway runs (`simple_serving/server.py`), from the configuration file SIMPLE_SERVING_CONFIG names."""

from __future__ import annotations

import asyncio
import os
import signal
import sys

from .. import log, vast
from ..app import Guard
from ..config import ENV_VAR, ConfigError
from ..server import GRACEFUL_SHUTDOWN_S, Lane
from . import LANE
from .app import control_app, public_app
from .config import PictureConfig, load
from .service import PictureService


class PictureGateway(Lane):
    """The picture lane's service and its two listeners. `async with PictureGateway(config, stop=...)` starts them;
    leaving the block stops them."""

    def __init__(self, config: PictureConfig, *, stop: vast.Stop, graceful_s: int = GRACEFUL_SHUTDOWN_S) -> None:
        self.config = config
        self.service = PictureService(config, stop)
        super().__init__(self.service, config.public, config.control, graceful_s=graceful_s, lane=LANE)

    def _apps(self) -> tuple[Guard, Guard]:
        return public_app(self.service), control_app(self.service)


def main() -> None:
    log.setup()
    path = os.environ.get(ENV_VAR)
    if not path:
        sys.exit(f"simple-serving pictures: {ENV_VAR} must name the configuration file")
    try:
        config = load(path)
    except ConfigError as error:
        sys.exit(f"simple-serving pictures: {error}")
    asyncio.run(serve(config))


async def serve(config: PictureConfig) -> None:
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(log.loop_exception)
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    async with PictureGateway(config, stop=vast.from_environment()):
        await stop.wait()

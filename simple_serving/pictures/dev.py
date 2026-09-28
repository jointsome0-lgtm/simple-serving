"""The picture lane's dev launcher: the picture gateway on loopback, in front of a ComfyUI the caller runs, such as
simple-story-chat's fake (`--engine-url`), or of this package's fake (`--engine-port`), for a run of a real client
against the real gateway.

    uv run python -m simple_serving.pictures.dev --config keys.json \
        --engine-url http://127.0.0.1:8188 --public-port 8084 --control-port 8085

The configuration file is a block of the picture gateway's configuration whose keys are test keys in the clear,
`{"keys": {"<key>": {"label": …, "classes": […], "default": …, "scopes": …, "control": …}}}`. Fields it leaves out
take the provisional limits of docs/pictures.md, section 7, the phase 1 allowlist and the bot card's engine facts,
which the engine must show before the gateway is ready. As on the card, the gateway falls asleep after an idle
interval without our work, 780 seconds unless the file sets another; it has no instance to stop, and stays drained
until the launcher starts again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from typing import Any

from .. import log, vast
from ..config import CLASSES, ConfigError, Listener
from ..server import Servers, bind
from .config import PictureConfig, from_block
from .fake_comfy import FakeComfy
from .server import PictureGateway

HOST = "127.0.0.1"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.pictures.dev", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="a JSON file with a block of the configuration and test keys")
    engine = parser.add_mutually_exclusive_group(required=True)
    engine.add_argument("--engine-url", help="a ComfyUI of the caller's on loopback")
    engine.add_argument("--engine-port", type=int, help="the loopback port for this package's fake ComfyUI")
    parser.add_argument("--public-port", type=int, required=True, help="the gateway's public listener")
    parser.add_argument("--control-port", type=int, required=True, help="the gateway's control listener")
    args = parser.parse_args(argv)
    engine_url = args.engine_url or f"http://{HOST}:{args.engine_port}"
    try:
        with open(args.config, encoding="utf-8") as file:
            block: Any = json.load(file)
        config = from_block(block, engine_url=engine_url, public=Listener(HOST, args.public_port),
                            control=Listener(HOST, args.control_port))
    except (OSError, ValueError, ConfigError) as error:
        sys.exit(f"simple_serving.pictures.dev: {error}" if isinstance(error, ConfigError)
                 else f"simple_serving.pictures.dev: cannot read {args.config} ({type(error).__name__})")
    log.setup()
    asyncio.run(run(config, args.engine_port))


async def run(config: PictureConfig, engine_port: int | None) -> None:
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(log.loop_exception)
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    engine = None
    if engine_port is not None:
        engine = Servers([(FakeComfy(), bind(Listener(HOST, engine_port)))], ws="websockets-sansio")
        await engine.start()
    try:
        async with PictureGateway(config, stop=vast.unconfigured) as gateway:
            while gateway.service.status != "ready":
                await asyncio.sleep(0.01)
            print(summary(config, fake=engine is not None), flush=True)
            await stop.wait()
    finally:
        if engine is not None:
            await engine.stop()


def summary(config: PictureConfig, *, fake: bool) -> str:
    """What the launcher prints once the gateway is ready: addresses, the engine and key labels, never a key."""
    keys = []
    for key in config.keys:
        rights = [cls for cls in CLASSES if cls in key.classes]
        rights += ["scopes"] * key.scopes + ["control"] * key.control
        keys.append(f"{key.label} ({', '.join(rights)})")
    expect = config.expect
    return "\n".join([
        "simple-serving pictures dev launcher: ready",
        f"  engine       {config.engine_url}{' (the fake)' if fake else ''}",
        f"  public       http://{HOST}:{config.public.port}",
        f"  control      http://{HOST}:{config.control.port}",
        f"  model        {config.alias}; ComfyUI {expect.comfyui}, PyTorch {expect.pytorch}, Triton {expect.triton}",
        f"  keys         {'; '.join(keys)}",
        "Ctrl-C stops it.",
    ])


if __name__ == "__main__":
    main()

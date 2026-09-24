"""The dev launcher: the fake engine and the gateway on loopback, for a run of a real client against the real gateway.

    uv run python -m simple_serving.dev --config contract/cases-v2.json \
        --engine-port 8200 --public-port 8201 --control-port 8202

The configuration file is a `service` block in the shape of contract/cases-v2.json, with its test keys in the clear,
or a file that holds one under "service", such as the cases file itself. Fields it leaves out take the provisional
values of contract section 7. The fake engine answers every request: the count is ceil(characters of all message
contents / 4), and a generation streams a fixed synthetic sentence and ends with stop. It answers in milliseconds,
unless --first-event-delay-ms stands in for a long prompt and --event-delay-ms slows the stream, so that requests
overlap and wait for places.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from dataclasses import dataclass
from typing import Any

from . import log
from .config import CLASSES, Config, ConfigError, Listener, from_service_block
from .fake_engine import FakeEngine, Script
from .server import Gateway, Servers, bind

HOST = "127.0.0.1"
PROVISIONAL: dict[str, Any] = {
    "alias": "dev-model",
    "context_tokens": 65536,
    "body_limit_bytes": 2_000_000,
    "limits": {
        "reader": {"active": 4, "waiting": 8, "input_tokens": None, "max_tokens": 8192, "wall_s": 300},
        "agent": {"active": 1, "waiting": 2, "input_tokens": None, "max_tokens": 8192, "wall_s": 900},
        "internal": {"active": 2, "waiting": 8, "input_tokens": None, "max_tokens": 8192, "wall_s": 900},
        "external": {"active": 2, "waiting": 4, "input_tokens": 8192, "max_tokens": 1024, "wall_s": 120},
        "external_per_key": {"active": 1, "waiting": 2},
        "shared": {"active": 4},
    },
    "count_limits": {"active": 8, "per_outside_key": 2},
    "engine_priority": {"reader": 0, "agent": 1, "internal": 2, "external": 3},
    "drain_deadline_s": 60,
}


@dataclass(frozen=True)
class Delays:
    """The fake engine's pauses in a generation, in milliseconds."""

    first_event_ms: int = 0  # after the status and headers, before the first event
    event_ms: int = 0  # after each event

    def script(self) -> Script:
        """The default answers at this pace."""
        return Script(first_delay_s=self.first_event_ms / 1000, delay_s=self.event_ms / 1000)


def milliseconds(text: str) -> int:
    """A pause on the command line: a whole number of milliseconds, 0 or more."""
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError("must be a whole number of milliseconds, 0 or more")
    return int(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.dev", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="a JSON file with a service block and its test keys")
    parser.add_argument("--engine-port", type=int, required=True, help="the fake engine's loopback port")
    parser.add_argument("--public-port", type=int, required=True, help="the gateway's public listener")
    parser.add_argument("--control-port", type=int, required=True, help="the gateway's control listener")
    parser.add_argument("--first-event-delay-ms", type=milliseconds, default=0, metavar="N",
                        help="the fake engine's pause before the first event of a generation, as for a long prompt")
    parser.add_argument("--event-delay-ms", type=milliseconds, default=0, metavar="N",
                        help="the fake engine's pause after each event of a generation")
    args = parser.parse_args(argv)
    delays = Delays(first_event_ms=args.first_event_delay_ms, event_ms=args.event_delay_ms)
    try:
        block = service_block(args.config)
        config = from_service_block(block, engine_url=f"http://{HOST}:{args.engine_port}",
                                    public=Listener(HOST, args.public_port), control=Listener(HOST, args.control_port))
    except (OSError, ValueError, ConfigError) as error:
        sys.exit(f"simple_serving.dev: {error}" if isinstance(error, ConfigError)
                 else f"simple_serving.dev: cannot read {args.config} ({type(error).__name__})")
    log.setup()
    asyncio.run(run(config, args.engine_port, delays))


def service_block(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ConfigError("the configuration file must hold an object")
    block = data.get("service", data)
    if not isinstance(block, dict):
        raise ConfigError("service must be an object")
    limits = {**PROVISIONAL["limits"], **block.get("limits", {})}
    return {**PROVISIONAL, **block, "limits": limits}


async def run(config: Config, engine_port: int, delays: Delays) -> None:
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(log.loop_exception)
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    fake = FakeEngine(config.alias, config.context_tokens or 65536)
    fake.script = delays.script()
    engine = Servers([(fake, bind(Listener(HOST, engine_port)))])
    await engine.start()
    try:
        async with Gateway(config) as gateway:
            while gateway.service.status != "ready":
                await asyncio.sleep(0.01)
            print(summary(config, engine_port, delays), flush=True)
            await stop.wait()
    finally:
        await engine.stop()


def summary(config: Config, engine_port: int, delays: Delays) -> str:
    """What the launcher prints once the gateway is ready: addresses, model and key labels, never a key."""
    keys = []
    for key in config.keys:
        rights = [cls for cls in CLASSES if cls in key.classes]
        rights += ["scopes"] * key.scopes + ["control"] * key.control
        keys.append(f"{key.label} ({', '.join(rights)})")
    return "\n".join([
        "simple-serving dev launcher: ready",
        f"  fake engine  http://{HOST}:{engine_port}",
        f"  delays       {delays.first_event_ms} ms before the first event, {delays.event_ms} ms after each event",
        f"  public       http://{HOST}:{config.public.port}",
        f"  control      http://{HOST}:{config.control.port}",
        f"  model        {config.alias}, context {config.context_tokens} tokens",
        f"  keys         {'; '.join(keys)}",
        "Ctrl-C stops both.",
    ])


if __name__ == "__main__":
    main()

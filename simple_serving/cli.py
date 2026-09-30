"""simple-serving's command on the owner's machine: it starts the card, holds the tunnel and asks for sleep
(contract section 8).

    python -m simple_serving.cli [--config PATH] [--lane pictures] up | sleep | status
    python -m simple_serving.cli [--config PATH] keys [--metrics-file PATH]
    python -m simple_serving.cli [--config PATH] [--lane pictures] trial --ssh-host HOST

`up` waits out a stop in flight, resumes the instance once if it is stopped, and waits for it to run. Then it opens
the tunnel, local 8080 to the gateway's public listener and 8081 to its control listener, waits for the gateway to be
ready, and holds the tunnel in the foreground. Ctrl+C closes it and sends nothing, so the card stays up until its idle
interval runs out. A lost tunnel is opened again a few times while the instance runs, never by a resume and never by
starting a service; once the card has stopped, `up` ends.

`sleep` asks the gateway to sleep, through a short control-only forward of its own, and reads `stopped` back from
Vast. Without the gateway it stops nothing, since a stop through Vast would skip the drain. `status` tells the
instance's state in Vast apart from what the gateway answers. Nothing of an answer is printed as it came: only the
contract's codes and statuses, numbers, and the manifest's alias. `keys` creates the client key, the control key and
the metrics key, never over existing ones, and prints only the SHA-256 of each, for the card's preparation:

    python -m simple_serving.cli keys | ssh <card> bash /workspace/simple-serving/card/bootstrap.sh

With `--metrics-file` it also writes the metrics key alone into that file, open to its owner only, for Prometheus on
the owner's machine (README, "Monitoring"); the metrics key reads the metrics and nothing else.

The configuration is the owner's alone: a JSON object in `~/.config/simple-serving/config.json`, or the file that
`--config` names, open to its owner only. `keys` writes `client_key`, `control_key` and `metrics_key` into it. The
owner adds `instance_id`, `vast_api_key`, a Vast key allowed GET and PUT on that instance alone, and `ssh_host`, a host
of ~/.ssh/config whose host key is known. The owner copies the client key into the bot's model profile.

On a trial rental, `trial` writes the id and the host instead: it reads the card's instance id over SSH from the file
card/onstart.sh keeps, checks its form, and writes it with the host it read it through. The key stays the owner's:
Vast refuses the card's own key from outside the card.

`--lane pictures` does the same for the picture card (docs/pictures.md, section 10), whose instance is the object
`pictures` in the configuration, with its own `instance_id`, `vast_api_key` and `ssh_host`, which `trial --lane
pictures` writes. The keys are the same three. Its tunnel is local 8084 to the picture gateway's public listener and
8085 to its control listener, its remote command the picture launcher's `--hold`, and its up has a lock of its own, so
that both lanes may be up at once.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import socket
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Container, Iterator, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from . import card, vast
from .config import CLASSES, IDLE_TIMEOUT_S
from .errors import STATUS

CONFIG = Path.home() / ".config/simple-serving/config.json"
KEYS = ("client_key", "control_key", "metrics_key")
LOCAL = {"public": 8080, "control": 8081}  # the tunnel's ports on the owner's machine (contract section 3)
SSH_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SSH = ("ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ExitOnForwardFailure=yes",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=yes",
       "-o", "ControlPath=none")
HOLD = "cd /workspace/simple-serving && /workspace/simple-serving-card/gateway/bin/python -m simple_serving.card --hold"
HOLD_PICTURES = ("cd /workspace/simple-serving && /workspace/simple-serving-pictures-card/gateway/bin/python "
                 "-m simple_serving.pictures.card --hold")
# Prints the instance's id, which card/onstart.sh keeps in a file without a newline, as one line.
READ_TRIAL = "cat /root/.simple-chat-instance-id || exit 3; echo"
TRIAL_BYTES = 4096  # the most of the card's answer to READ_TRIAL that `trial` reads
SSH_FAILED = 255  # ssh's own exit code; any other before the first line comes from the card's shell
POLL_S = 10  # between two reads of Vast, and two tries of SSH
STOP_WAIT_S = 600  # for a stop in flight to end: a drain of 120 seconds at most, then Vast's own time
START_WAIT_S = 600  # for a resumed instance to run, and then for SSH to connect
READY_POLL_S = 5
READY_MARGIN_S = 60  # past the card's load deadline, by when the launcher has ended a load that is not ready
CONNECT_S = 30  # from starting ssh to the remote command's first line
CLOSE_S = 5  # from ssh's SIGTERM to its SIGKILL
CALL_S = 10  # one call to the gateway, from connecting to the end of its answer
ANSWER_BYTES = 16384  # the most of one answer of the gateway that the command reads
RECONNECTS = 3  # tunnels in a row that may end before the gateway is ready again
STATUSES = ("starting", "ready", "draining", "drained", "failed")  # contract section 6
NOT_PREPARED = "the card is not prepared: run its preparation first (README, 'The card')"
GIVEN_UP = (f"the card has given up loading the model and stops itself {IDLE_TIMEOUT_S // 60} minutes after its start. "
            "The launcher's --retry on the card loads it again, within seconds while the launcher still waits and "
            "otherwise at the next start (README, 'The card')")
UNCONFIRMED = ("the card's stop is not confirmed: Vast has not taken it, and costs may go on. Delete the trial "
               "instance with simple-story-chat's npm run gpu:rent -- --destroy ID (README, 'The first rental')")
REFUSALS = {card.NOT_PREPARED: NOT_PREPARED, card.GAVE_UP: GIVEN_UP, card.STOP_UNCONFIRMED: UNCONFIRMED}  # of --hold


@dataclass(frozen=True)
class Lane:
    """What `--lane` changes: where the configuration keeps the lane's instance, the lane's manifest, the tunnel's
    local ports and remote command, and up's lock."""

    name: str | None  # the configuration's object for the lane, or None for its top level, the text lane's
    folder: str  # of the lane's manifest.env, in the checkout
    local: Mapping[str, int]
    hold: str
    lock: str  # beside the configuration


LANES = {"text": Lane(None, "card", LOCAL, HOLD, "up.lock"),
         "pictures": Lane("pictures", "card/pictures", {"public": 8084, "control": 8085}, HOLD_PICTURES,
                          "up-pictures.lock")}

now = time.monotonic  # the command's own clock, which the tests replace
pause = asyncio.sleep


class Refusal(Exception):
    """Why the command stops, in words for the owner. It never holds a key or an answer of Vast."""


class GatewayError(Exception):
    """An answer of the gateway that is an error, or of another shape: its HTTP status and a fixed code."""


class NoAnswer(GatewayError):
    """A call to the gateway that got no answer."""


class NoTunnel(Exception):
    """ssh ended before the remote command's first line, with this exit code."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class Tunnel:
    """An SSH connection to the card with loopback forwards, whose remote command holds while the card's pair runs
    and starts nothing."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process

    @classmethod
    async def open(cls, host: str, forwards: Mapping[int, int], hold: str = HOLD) -> Tunnel:
        """Return once the remote command has printed its first line, which it does only after ssh has bound every
        forward; raise NoTunnel when ssh ends or does not get that far within CONNECT_S."""
        options = [arg for local, remote in forwards.items() for arg in ("-L", f"127.0.0.1:{local}:127.0.0.1:{remote}")]
        process = await asyncio.create_subprocess_exec(*SSH, *options, host, hold, stdin=asyncio.subprocess.DEVNULL,
                                                       stdout=asyncio.subprocess.PIPE)
        tunnel = cls(process)
        try:
            if await tunnel.held():
                return tunnel
        except BaseException:
            await tunnel.close()
            raise
        raise NoTunnel(await tunnel.close())

    async def held(self) -> bool:
        assert self.process.stdout is not None
        with contextlib.suppress(TimeoutError, ValueError):  # ValueError: a line past the reader's limit
            async with asyncio.timeout(CONNECT_S):
                while line := await self.process.stdout.readline():  # a shell may print a line of its own first
                    if line.decode(errors="replace").strip() == card.HOLDING:
                        return True
        return False

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    async def wait(self) -> int:
        return await self.process.wait()

    async def close(self) -> int:
        """End ssh, the command's own child: SIGTERM, then SIGKILL after CLOSE_S."""
        with contextlib.suppress(ProcessLookupError):
            self.process.terminate()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(CLOSE_S):
                return await self.process.wait()
        with contextlib.suppress(ProcessLookupError):
            self.process.kill()
        return await self.process.wait()


@dataclass(frozen=True)
class Setup:
    """What the command works with. The tests build one with a fake Vast and a fake SSH."""

    path: Path  # the configuration; up's lock lives beside it
    control_key: str = field(repr=False)
    alias: str  # the model the gateway must serve
    ready_s: float
    card_ports: Mapping[str, int]  # the gateway's listeners on the card
    show: Callable[[], Awaitable[str]] = field(repr=False)  # the instance's state in Vast, with the owner's key
    resume: Callable[[], Awaitable[None]] = field(repr=False)
    connect: Callable[[Mapping[int, int]], Awaitable[Tunnel]]
    local: Mapping[str, int] = field(default_factory=lambda: dict(LOCAL))
    lane: str | None = None  # what the gateway's state must name, as the configuration does
    lock: str = "up.lock"

    def control_url(self, port: int | None = None) -> str:
        return f"http://127.0.0.1:{port or self.local['control']}"


def load(path: Path, lane: Lane = LANES["text"]) -> Setup:
    config = read_config(path)
    block = config if lane.name is None else config.get(lane.name, {})
    if not isinstance(block, dict):
        raise Refusal(f"{path} must hold a JSON object as {lane.name}")
    prefix = "" if lane.name is None else f"{lane.name}."
    missing = [prefix + name for name in ("instance_id", "vast_api_key", "ssh_host")
               if not isinstance(block.get(name), str) or not block[name]]
    missing += [name for name in ("control_key",) if not isinstance(config.get(name), str) or not config[name]]
    if missing:
        raise Refusal(f"{path} has no {', '.join(missing)}")
    instance, key, host = block["instance_id"], block["vast_api_key"], block["ssh_host"]
    if not vast.INSTANCE.fullmatch(instance) or "\r" in key or "\n" in key or not SSH_HOST.fullmatch(host):
        raise Refusal(f"{path} has an {prefix}instance_id, vast_api_key or ssh_host of the wrong form")
    manifest = card.read_manifest(folder=lane.folder)
    return Setup(path, config["control_key"], manifest["MODEL_ALIAS"],
                 int(manifest["LOAD_DEADLINE_S"]) + READY_MARGIN_S,
                 {name: int(manifest[f"{name.upper()}_PORT"]) for name in LOCAL},
                 partial(vast.show, instance, key), partial(vast.resume, instance, key),
                 partial(Tunnel.open, host, hold=lane.hold), dict(lane.local), lane.name, lane.lock)


def read_config(path: Path) -> dict[str, Any]:
    """The configuration, or nothing when there is none yet. Others may not read it."""
    try:
        if path.stat().st_mode & 0o077:
            raise Refusal(f"{path} must be open to its owner only: chmod 600 it")
        config = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        raise Refusal(f"cannot read {path} as JSON") from None
    if not isinstance(config, dict):
        raise Refusal(f"{path} must hold a JSON object")
    return config


def write_config(path: Path, config: Mapping[str, Any]) -> None:
    """Replace the configuration in one step, open to its owner only."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    new = path.with_name(path.name + ".new")
    new.unlink(missing_ok=True)
    with os.fdopen(os.open(new, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        json.dump(config, file, indent=2)
    new.replace(path)


def keys(path: Path, metrics_file: Path | None = None) -> int:
    """Create the keys that are missing, keep the others, and print the SHA-256 of each. With `metrics_file`, write
    the metrics key into it as well, for Prometheus's credentials_file."""
    if metrics_file is not None:
        config_paths = {path.resolve(), path.with_name(path.name + ".new").resolve()}
        secret_paths = {metrics_file.resolve(), metrics_file.with_name(metrics_file.name + ".new").resolve()}
        if config_paths & secret_paths:
            raise Refusal("--metrics-file and its temporary file must be separate from --config")
    config = read_config(path)
    created = [name for name in KEYS if name not in config]
    for name in created:
        config[name] = secrets.token_urlsafe(32)
    if not all(isinstance(config[name], str) and config[name] for name in KEYS):
        raise Refusal(f"{path} has a {', '.join(KEYS[:-1])} or {KEYS[-1]} that is not a string")
    if created:
        write_config(path, config)
    if metrics_file is not None:
        write_secret(metrics_file, config["metrics_key"])
    print(*(hashlib.sha256(config[name].encode()).hexdigest() for name in KEYS), sep="\n")
    print(f"keys: {', '.join(created) + ' created' if created else 'all kept'} in {path}", file=sys.stderr)
    return 0


def write_secret(path: Path, secret: str) -> None:
    """Replace a file with one secret and nothing else, open to its owner only, in a directory open to its owner."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    new = path.with_name(path.name + ".new")
    new.unlink(missing_ok=True)
    with os.fdopen(os.open(new, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as file:
        file.write(secret)
    new.replace(path)


async def trial(path: Path, host: str, lane: Lane = LANES["text"]) -> int:
    """Write the trial instance's id, which card/onstart.sh keeps on the card, and the SSH host it was read from into
    the configuration, or into the lane's object there, keeping what else it holds, `vast_api_key` included. Nothing
    read from the card is printed, not even when it is refused."""
    if not SSH_HOST.fullmatch(host):
        raise Refusal("the SSH host has the wrong form")
    config = read_config(path)
    block = config if lane.name is None else config.get(lane.name, {})
    if not isinstance(block, dict):
        raise Refusal(f"{path} must hold a JSON object as {lane.name}")
    # Standard error is dropped unread: whatever the card prints there could hold its key.
    process = await asyncio.create_subprocess_exec(*SSH, host, READ_TRIAL, stdin=asyncio.subprocess.DEVNULL,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    assert process.stdout is not None
    answer = b""
    try:
        async with asyncio.timeout(CONNECT_S):
            while len(answer) <= TRIAL_BYTES and (chunk := await process.stdout.read(TRIAL_BYTES)):
                answer += chunk
            code = await process.wait() if len(answer) <= TRIAL_BYTES else None
    except TimeoutError:
        await Tunnel(process).close()
        raise Refusal(f"the card did not give the instance's id within {CONNECT_S} seconds") from None
    if code is None:  # an answer longer than an id
        await Tunnel(process).close()
    elif code:
        raise Refusal(f"could not read the instance's id from the card: ssh ended with {code}")
    try:
        instance, rest = answer.decode().split("\n")
    except ValueError:  # a byte that is not UTF-8, or other than one line
        instance, rest = "", "?"
    if code is None or rest or not vast.INSTANCE.fullmatch(instance):
        raise Refusal("the card's instance id is missing or of the wrong form")
    found = block | {"instance_id": instance, "ssh_host": host}
    write_config(path, found if lane.name is None else config | {lane.name: found})
    say(f"trial: the instance's id and the SSH host are in {path}")
    return 0


async def up(setup: Setup) -> int:
    with up_lock(setup.path, setup.lock):
        if busy := [port for port in setup.local.values() if card.taken(port)]:
            raise Refusal(f"local port {busy[0]} is taken, and up leaves another listener alone")
        await start(setup)
        await hold(setup)
    return 0


async def start(setup: Setup) -> None:
    """Make the instance run: wait out a stop in flight, then resume it once if it is stopped. So it never writes
    `running` against a stop."""
    state = await await_state(setup, {"stopped", "starting", "running"}, STOP_WAIT_S, "the stop in flight to end")
    if state == "stopped":
        say("vast: stopped; resuming the instance")
        try:
            async with asyncio.timeout(vast.ATTEMPT_S):
                await setup.resume()
        except (vast.VastError, TimeoutError) as error:
            raise Refusal(f"Vast did not take the resume ({describe(error)})") from None
    # Right after a resume Vast may still report the stop; only a stop it means anew ends the wait.
    if await await_state(setup, {"running", "stopping"}, START_WAIT_S, "the instance to run") != "running":
        raise Refusal("the instance is being stopped")


async def hold(setup: Setup) -> None:
    """Hold the tunnel while the instance runs, and open it again when it ends. Once the card has stopped, return."""
    ended = 0  # tunnels in a row that ended without the gateway ready
    while (tunnel := await connect(setup)) is not None:
        try:
            if await ready(setup, tunnel):
                ended = 0
                say(f"ready: {setup.alias} at http://127.0.0.1:{setup.local['public']}/v1, control at "
                    f"{setup.control_url()}. Ctrl+C closes the tunnel and sends nothing.")
            code = await tunnel.wait()
        finally:
            await tunnel.close()
        if code in (card.GAVE_UP, card.STOP_UNCONFIRMED):
            raise Refusal(REFUSALS[code])
        ended += 1
        if ended > RECONNECTS:
            raise Refusal("the tunnel keeps ending before the gateway is ready")
        say("tunnel: closed; opening it again while the instance runs")
        await pause(POLL_S)
    say("the card has stopped")


async def connect(setup: Setup) -> Tunnel | None:
    """The tunnel, opened while the instance runs. SSH may come up only after Vast reports `running`, so it is tried
    every POLL_S for START_WAIT_S. None once Vast means to stop the instance."""
    deadline = now() + START_WAIT_S
    forwards = {setup.local[name]: setup.card_ports[name] for name in LOCAL}
    while (state := await vast_state(setup)) not in ("stopping", "stopped"):
        if state == "running":
            try:
                return await setup.connect(forwards)
            except NoTunnel as error:
                if error.code != SSH_FAILED:
                    raise Refusal(REFUSALS.get(error.code, NOT_PREPARED)) from None
        if now() >= deadline:
            raise Refusal(f"the card does not take SSH; Vast says {state}")
        await pause(POLL_S)
    return None


async def ready(setup: Setup, tunnel: Tunnel) -> bool:
    """Wait for the gateway to be ready: True once it is, False if the tunnel ends first. A gateway of another
    contract or model, or one that has begun to sleep, is a refusal. A technical drain is left as it is."""
    deadline, shown = now() + setup.ready_s, None
    while tunnel.returncode is None:
        try:
            state = await call(setup.control_url(), setup.control_key, "GET", "/v1/state")
        except NoAnswer:
            state = {}  # the gateway may not listen yet
        except GatewayError as error:
            raise Refusal(f"the gateway answers {error}") from None
        if state:
            if (state.get("contract"), state.get("lane"), state.get("model")) != ("2", setup.lane, setup.alias):
                raise Refusal("the gateway on the card serves another contract, lane or model")
            if state.get("sleep_requested"):
                raise Refusal("the card is falling asleep: run up again once it has stopped")
            if state["status"] == "ready":
                return True
            if state["status"] != shown:
                shown = state["status"]
                say(f"gateway: {shown}")
        if now() >= deadline:
            raise Refusal("the gateway is not ready in time")
        await pause(READY_POLL_S)
    return False


async def sleep(setup: Setup) -> int:
    if await vast_state(setup) not in ("stopping", "stopped"):
        try:
            async with control(setup) as base:
                say(f"gateway: {await ask_sleep(base, setup.control_key)}, falling asleep")
        except (NoTunnel, GatewayError) as error:
            raise Refusal(f"the gateway did not take the sleep ({describe(error)}), and this command never stops "
                          "the card itself") from None
    await await_state(setup, {"stopped"}, STOP_WAIT_S, "the instance to stop")
    say("vast: stopped")
    return 0


async def ask_sleep(base: str, key: str) -> str:
    """Ask the gateway to sleep, and return its status. A sleep may be asked again, so a lost answer or a new boot
    gets a second try."""
    error = GatewayError()
    for _ in range(2):
        try:
            state = await call(base, key, "GET", "/v1/state")
            return str((await call(base, key, "POST", "/v1/control/sleep", {"boot_id": state["boot_id"]}))["status"])
        except GatewayError as failed:
            error = failed
    raise error


async def status(setup: Setup) -> int:
    state = await vast_state(setup)
    print(f"vast: {state}")
    if state in ("stopping", "stopped"):
        return 0
    try:
        async with control(setup) as base:
            view = await call(base, setup.control_key, "GET", "/v1/state")
    except (NoTunnel, GatewayError) as error:
        print(f"gateway: {describe(error)}")
        return 0
    counts = {name: {cls: n for cls, n in view.get(name, {}).items() if n} for name in ("active", "waiting")}
    serves = (view.get("contract"), view.get("lane"), view.get("model")) == ("2", setup.lane, setup.alias)
    lane = f"lane {setup.lane}, " if setup.lane else ""
    print(f"gateway: {view['status']}{', falling asleep' if view.get('sleep_requested') else ''}; "
          f"{f'contract 2, {lane}model {setup.alias}' if serves else 'another contract or model'}; "
          f"active {counts['active'] or 'none'}, waiting {counts['waiting'] or 'none'}")
    return 0


@contextlib.asynccontextmanager
async def control(setup: Setup) -> AsyncIterator[str]:
    """The gateway's control listener, through a short control-only forward of this command's own on a free loopback
    port. The key goes there only after the remote command's first line, once ssh has bound the port; never to up's
    ports, since up's lock does not show that its forwards are bound."""
    port = free_port()
    tunnel = await setup.connect({port: setup.card_ports["control"]})
    try:
        yield setup.control_url(port)
    finally:
        await tunnel.close()


async def call(base: str, key: str, method: str, path: str, body: Any = None) -> dict[str, Any]:
    """One call to the gateway within CALL_S, and its answer, read up to ANSWER_BYTES. Never through a proxy: the key
    goes to loopback only. An error keeps its HTTP status and a code of contract section 9, or `error`, and an answer
    whose fields the command reads have other shapes is `malformed`, so nothing of an answer is printed as it came."""
    try:
        async with (asyncio.timeout(CALL_S), httpx.AsyncClient(timeout=None, trust_env=False) as client,
                    client.stream(method, base + path, json=body,
                                  headers={"Authorization": f"Bearer {key}"}) as response):
            data = b""
            async for chunk in response.aiter_bytes():
                data += chunk
                if len(data) > ANSWER_BYTES:
                    raise ValueError("too long")
        answer = json.loads(data)
    except (httpx.HTTPError, ValueError, RecursionError, TimeoutError):
        raise NoAnswer("no answer") from None
    if not response.is_success or not isinstance(answer, dict):
        error = answer.get("error") if isinstance(answer, dict) else None
        code = error.get("code") if isinstance(error, dict) else None
        raise GatewayError(f"{response.status_code} {code if isinstance(code, str) and code in STATUS else 'error'}")
    if not well_formed(answer):
        raise GatewayError(f"{response.status_code} malformed")
    return answer


def well_formed(answer: dict[str, Any]) -> bool:
    """Whether the fields that the command reads have the shapes of contract sections 6 and 8: the status and the boot
    of a control answer, and in a control view also the sleep and the counts by class."""
    counts = [answer.get(name, {}) for name in ("active", "waiting")]
    return (answer.get("status") in STATUSES and isinstance(answer.get("boot_id"), str)
            and isinstance(answer.get("sleep_requested", False), bool)
            and all(isinstance(by_class, dict) and all(name in CLASSES and type(n) is int and n >= 0
                                                       for name, n in by_class.items()) for by_class in counts))


async def vast_state(setup: Setup) -> str:
    """The instance's state, or why it is unknown while Vast does not answer. A key that Vast refuses is a refusal:
    it does not mend itself."""
    try:
        async with asyncio.timeout(vast.ATTEMPT_S):
            return await setup.show()
    except (vast.VastError, TimeoutError) as error:
        if isinstance(error, vast.VastError) and error.code == "forbidden":
            raise Refusal(f"Vast refuses the key ({describe(error)})") from None
        return f"unknown ({describe(error)})"


async def await_state(setup: Setup, wanted: Container[str], within_s: float, what: str) -> str:
    deadline, shown = now() + within_s, None
    while (state := await vast_state(setup)) not in wanted:
        if now() >= deadline:
            raise Refusal(f"waited {within_s // 60:.0f} minutes for {what}; Vast says {state}")
        if state != shown:
            shown = state
            say(f"vast: {state}; waiting for {what}")
        await pause(POLL_S)
    return state


def describe(error: Exception) -> str:
    if isinstance(error, vast.VastError):
        return error.code if error.status is None else f"{error.code} {error.status}"
    if isinstance(error, NoTunnel):
        return REFUSALS.get(error.code, f"ssh ended with {error.code}")
    return str(error) or "timeout"


@contextlib.contextmanager
def up_lock(path: Path, name: str = "up.lock") -> Iterator[None]:
    """One up of a lane at a time for the configuration's directory: the lock guards the fixed local ports of the
    lane's tunnel, not a card."""
    lock = os.open(path.with_name(name), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock)
        raise Refusal("another up runs") from None
    try:
        yield
    finally:
        os.close(lock)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


def say(text: str) -> None:
    print(f"simple-serving: {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.cli", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG, help="the configuration file (default: %(default)s)")
    parser.add_argument("--ssh-host", help="trial's card: a host of ~/.ssh/config whose host key is known")
    parser.add_argument("--lane", choices=tuple(LANES), default="text", help="the card's lane (default: %(default)s)")
    parser.add_argument("--metrics-file", type=Path, help="keys: also write the metrics key into this file")
    parser.add_argument("command", choices=("up", "sleep", "status", "keys", "trial"))
    args = parser.parse_args(argv)
    if (args.command == "trial") != (args.ssh_host is not None):
        parser.error("--ssh-host goes with trial, and trial needs it")
    if args.metrics_file is not None and args.command != "keys":
        parser.error("--metrics-file goes with keys")
    commands = {"up": up, "sleep": sleep, "status": status}
    lane = LANES[args.lane]
    try:
        if args.command == "keys":  # the same three for every lane
            return keys(args.config, args.metrics_file)
        if args.command == "trial":
            return asyncio.run(trial(args.config, args.ssh_host, lane))
        return asyncio.run(commands[args.command](load(args.config, lane)))
    except Refusal as error:
        print(f"simple-serving: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

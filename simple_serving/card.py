"""The card's launcher: vLLM and the gateway as one pair, started at every start of the container (contract section 8).

    python -m simple_serving.card [--hold | --stop | --retry | --dry-run]

card/onstart.sh runs it from the checkout with the gateway's venv, and card/bootstrap.sh prepares the card once per
rental. Without an option it checks the card, starts the launcher in the background and returns 0, also when one
already runs. Otherwise it returns 3 when the card is not prepared (bootstrap.sh has not run for this manifest and
these locks, or the keys or the instance's credential are missing), 6 when the card has given up, and the file
`given-up` says why, and 7 when a port is taken. `--retry` removes `given-up` and starts; when a launcher runs already,
it says that the retry is left to it or deferred to the next start. `--stop` ends the pair and leaves the instance
running. `--hold` prints one line, then waits while the pair runs and returns 0 once it has ended, 6 once the card has
given up, or 8 once its stop is not confirmed; on such a card, or one that is not prepared, it returns 8, 6 or 3 at once
and prints nothing. It is the remote command of the tunnel, and it never owns the pair. `--dry-run` prints the two
commands and checks, and starts nothing.

The pair runs once, with no restarts. It ends on SIGTERM, when a process exits, or when the gateway is not ready by
the manifest's load deadline. Unless a signal ended it, the launcher then stops the instance as the gateway does when
it falls asleep (`vast.py`). When the pair never became ready, it first leaves `given-up`, and the card has given up:
a later start, a resume included, loads nothing. Its launcher waits the idle interval for the owner's `--retry`, and
without one stops the instance again, so that no card stays up with nothing to stop it. It looks for the retry every
POLL_S and then runs the pair; a retry in the interval's last moment may lose to its end, and the next start loads.

An attempt to stop the instance that fails, the launcher's or the gateway's, leaves `stop-unconfirmed` until the
launcher's next start: costs may go on. The attempts go on too, but they bound nothing while Vast refuses the key.

The logs are in `logs/` of the state directory, with one older file beside each. The launcher makes a log 0600 when it
opens it; one that it moves aside keeps its mode. `card.jsonl` holds the launcher's rows and, of vLLM's output, only a
few numbers and fixed categories of failure and warning. `gateway.jsonl` holds the gateway's rows that pass
`gateway_row`.
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
import resource
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from . import log, vast
from .config import IDLE_TIMEOUT_S, PROVISIONAL

CODE = Path(__file__).resolve().parent.parent  # the checkout
STATE = Path(os.environ.get("SIMPLE_SERVING_CARD_DIR", "/workspace/simple-serving-card"))
ROOT = Path(os.environ.get("SIMPLE_SERVING_CARD_ROOT", "/root"))  # where onstart.sh keeps the instance's id and key
STAMPED = ("manifest.env", "gateway-requirements.txt", "vllm-requirements.txt")  # the files in card/ that were prepared
NOT_PREPARED, GAVE_UP, PORT_TAKEN, STOP_UNCONFIRMED = 3, 6, 7, 8
GIVEN_UP, UNCONFIRMED = "given-up", "stop-unconfirmed"  # the card's markers
HOLDING = "simple-serving card: holding"  # the first line of --hold
LINE_BYTES = 8192  # a longer line of output reaches the filter cut to this length
CARD_LOG_BYTES, GATEWAY_LOG_BYTES = 1 << 20, 8 << 20
GATEWAY_GRACE_S, ENGINE_GRACE_S = 10, 30  # from SIGTERM to SIGKILL
POLL_S, HOLD_START_S, STOP_WAIT_S = 2, 60, 60
ENGINE_ENV = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# vLLM's wording of the numbers kept, as in the source of vLLM 0.30.0, and provisional until the first rental checks
# each against its output. A field's last word names its unit, and SCALE turns vLLM's GiB, seconds and factors into it.
MEASUREMENTS = tuple(re.compile(pattern) for pattern in (
    r"Model loading took (?P<weights_mib>\d+(?:\.\d+)?) GiB(?: memory)? and (?P<weights_load_ms>\d+(?:\.\d+)?) s",
    r"Available KV cache memory: (?P<kv_cache_mib>-?\d+(?:\.\d+)?) GiB",  # below 0 when the weights leave no room
    r"GPU KV cache size: (?P<kv_cache_tokens>\d[\d,]*) tokens",
    r"Maximum concurrency for [\d,]+ tokens per request: (?P<concurrency_x100>\d+(?:\.\d+)?)x",
))
SCALE = {"mib": 1024, "ms": 1000, "tokens": 1, "x100": 100}
# The categories of failure, each with words of a lowercased line, provisional as well. A line gets the first that fits.
FAILURES = (
    ("out_of_memory", ("out of memory", "outofmemoryerror")),
    ("kv_cache_too_small", ("larger than the available kv cache memory", "no available memory for the cache blocks")),
    ("memory_taken", ("free memory on device",)),  # another process holds a part of the card's memory at the start
    ("memory_profiling_failed", ("error in memory profiling",)),  # another process freed memory while vLLM profiled
    ("cuda_error", ("cuda error", "cudaerror", "cublas_status", "nccl error")),
    ("bind_failed", ("address already in use",)),
    ("argument_error", ("unrecognized arguments", "error: argument")),
    ("engine_dead", ("enginedeaderror", "engine core initialization failed", "frontend process failed")),
    ("traceback", ("traceback (most recent call last)",)),
)
# And of warnings, by the same rule. NVFP4 weights that vLLM fuses, q/k/v or gate/up, should share one global scale;
# where they do not, it takes the largest, which "will likely reduce accuracy" (modelopt.py in vLLM 0.30.0).
WARNINGS = (("fused_scales_differ", ("scale differs across parallel layers",)),)
GATEWAY_KEYS = log.FIELDS | {"time", "logger", "level"}


@dataclass(frozen=True)
class Command:
    argv: list[str]
    env: dict[str, str] = field(repr=False)
    cwd: Path | None = None


@dataclass(frozen=True)
class Card:
    code: Path  # the checkout, with card/manifest.env
    state: Path
    manifest: Mapping[str, str]
    credential: Mapping[str, str] = field(repr=False)  # CONTAINER_ID and CONTAINER_API_KEY, where known

    @classmethod
    def load(cls, code: Path = CODE, state: Path = STATE, environ: Mapping[str, str] = os.environ,
             root: Path = ROOT) -> Card:
        return cls(code, state, read_manifest(code), credential(environ, root))

    @property
    def ports(self) -> dict[str, int]:
        return {name: int(self.manifest[f"{name.upper()}_PORT"]) for name in ("public", "control", "engine")}

    def engine(self) -> Command:
        m = self.manifest
        return Command([
            str(self.state / "vllm/bin/vllm"), "serve", str(self.state / "models" / m["MODEL_REVISION"]),
            "--served-model-name", m["MODEL_ALIAS"], "--host", "127.0.0.1", "--port", m["ENGINE_PORT"],
            "--tokenizer", str(self.state / "tokenizer"), "--max-model-len", m["CONTEXT_TOKENS"],
            "--max-num-seqs", m["MAX_NUM_SEQS"], "--max-num-batched-tokens", m["MAX_NUM_BATCHED_TOKENS"],
            "--gpu-memory-utilization", m["GPU_MEMORY_UTILIZATION"], "--kv-cache-dtype", m["KV_CACHE_DTYPE"],
            "--scheduling-policy", "priority", "--enable-prefix-caching", "--enable-prompt-tokens-details",
            "--reasoning-parser", "gemma4",  # the model's thinking in the reasoning field, apart from the answer
            "--language-model-only",  # text only: the vision tower stays off the GPU
            # Sampling defaults from vLLM: the model's generation_config.json is pinned for its stop ids alone.
            "--generation-config", "vllm",
            "--disable-uvicorn-access-log", "--uvicorn-log-level", "warning",
        ], {name: value for name, value in os.environ.items() if name != "CONTAINER_API_KEY"} | ENGINE_ENV)

    def gateway(self) -> Command:
        return Command([str(self.state / "gateway/bin/python"), "-m", "simple_serving"],
                       {**os.environ, **self.credential, "SIMPLE_SERVING_CONFIG": str(self.state / "gateway.json")},
                       self.code)

    def config(self) -> dict[str, Any]:
        """The gateway's configuration: the provisional limits, the manifest's model and ports, and the hashes of the
        two keys."""
        m, keys = self.manifest, json.loads((self.state / "keys.json").read_text())
        return PROVISIONAL | {
            "alias": m["MODEL_ALIAS"],
            "context_tokens": int(m["CONTEXT_TOKENS"]),
            "engine_url": f"http://127.0.0.1:{m['ENGINE_PORT']}",
            "listen": {name: {"host": "127.0.0.1", "port": self.ports[name]} for name in ("public", "control")},
            "keys": [
                {"sha256": keys["client"], "label": "client", "classes": ["reader", "agent", "internal"],
                 "default": "internal", "scopes": True, "control": False},
                {"sha256": keys["control"], "label": "control", "classes": [], "default": None, "scopes": False,
                 "control": True},
            ],
            "versions": {"vllm": m["VLLM_VERSION"], "model_revision": m["MODEL_REVISION"],
                         "tokenizer_revision": m["TOKENIZER_REVISION"]},
        }

    def prepared(self) -> bool:
        """Whether bootstrap.sh prepared the card for this manifest and these locks, and the gateway will be able to
        stop the instance."""
        try:
            stamp = hashlib.sha256(b"".join((self.code / "card" / name).read_bytes() for name in STAMPED))
            return ((self.state / "prepared").read_text().strip() == stamp.hexdigest()
                    and (self.state / "keys.json").is_file()
                    and vast.from_environment(self.credential) is not vast.unconfigured)
        except OSError:
            return False


def read_manifest(code: Path = CODE) -> dict[str, str]:
    lines = (line.strip() for line in (code / "card/manifest.env").read_text().splitlines())
    return {name: value for name, _, value in (line.partition("=") for line in lines)
            if name and not name.startswith("#")}


def credential(environ: Mapping[str, str], root: Path) -> dict[str, str]:
    """The instance's id and key: from the environment Vast gives the container, or, in an SSH session, which lacks
    it, from the files onstart.sh writes."""
    files = {"CONTAINER_ID": ".simple-chat-instance-id", "CONTAINER_API_KEY": ".simple-chat-instance-api-key"}
    found = {}
    for name, file in files.items():
        with contextlib.suppress(OSError):
            found[name] = environ.get(name) or (root / file).read_text()
    return found


def launcher(state: Path) -> int | None:
    """The pid of the running launcher, if one runs."""
    try:
        pid = int((state / "launcher.pid").read_text())
        running = b"simple_serving.card\0--run" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except (OSError, ValueError):
        return None
    return pid if running else None


def taken(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
        return False


def start(card: Card, *, dry_run: bool = False) -> int:
    """Start the launcher in the background, unless something stands in the way. A card that has given up gets its
    launcher all the same, since that is what stops the instance, and 6 says so. A dry run prints the two commands and
    starts nothing."""
    if dry_run:
        print(shlex.join(card.engine().argv), shlex.join(card.gateway().argv), sep="\n")
    given_up = (card.state / GIVEN_UP).exists()
    if launcher(card.state) is not None:
        return GAVE_UP if given_up else 0
    if not given_up and not card.prepared():
        return NOT_PREPARED
    if not given_up and any(taken(port) for port in card.ports.values()):
        return PORT_TAKEN
    if not dry_run:
        subprocess.Popen([sys.executable, "-m", "simple_serving.card", "--run"], cwd=card.code,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    return GAVE_UP if given_up else 0


def hold(card: Card, pause: Callable[[float], object] = time.sleep) -> int:
    """Wait while the launcher runs. One that onstart has not started yet is waited for HOLD_START_S. The first line
    out tells the command that its forwards are up, since ssh runs its remote command only once it has bound them. A
    card with nothing to hold gets no line: one that a marker ends, or one that is not prepared and runs no launcher."""
    if code := marked(card.state):
        return code
    if launcher(card.state) is None and not card.prepared():
        return NOT_PREPARED
    print(HOLDING, flush=True)
    waited, seen = 0, False
    while not (code := marked(card.state)):
        if launcher(card.state) is not None:
            seen = True
        elif seen or waited >= HOLD_START_S:
            return 0
        pause(POLL_S)
        waited += POLL_S
    return code


def marked(state: Path) -> int:
    """What the card's markers say, the stop first: STOP_UNCONFIRMED, GAVE_UP, or 0."""
    return STOP_UNCONFIRMED if (state / UNCONFIRMED).exists() else GAVE_UP if (state / GIVEN_UP).exists() else 0


def stop_pair(state: Path, pause: Callable[[float], object] = time.sleep) -> int:
    """End the pair through its launcher, and wait for it. 1 when the launcher still runs after STOP_WAIT_S."""
    pid = launcher(state)
    if pid is not None:
        os.kill(pid, signal.SIGTERM)
    for _ in range(STOP_WAIT_S // POLL_S):
        if launcher(state) is None:
            return 0
        pause(POLL_S)
    return 1


def run(card: Card) -> int:
    """The launcher itself, in the background: one pair under the lock, then what its end calls for."""
    lock = os.open(card.state / "launcher.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(lock)
        return 0  # another launcher runs the pair
    os.umask(0o077)
    (card.state / UNCONFIRMED).unlink(missing_ok=True)  # a new start, whose stop has not failed yet
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # a core dump of either process could hold prompts
    (card.state / "logs").mkdir(exist_ok=True)
    log.setup(JsonLines(card.state / "logs/card.jsonl", CARD_LOG_BYTES))
    pid = card.state / "launcher.pid"
    pid.write_text(f"{os.getpid()}\n")
    try:
        config = json.dumps(card.config())
        with os.fdopen(private(card.state / "gateway.json", os.O_WRONLY | os.O_TRUNC), "w") as file:
            file.write(config)
        asyncio.run(serve(card))
    finally:
        pid.unlink(missing_ok=True)
    return 0


async def serve(card: Card) -> None:
    ended = asyncio.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(signum, ended.set)
    await launch(card, ended, vast.from_environment(card.credential))


async def launch(card: Card, ended: asyncio.Event, stop: vast.Stop) -> str:
    """Run the pair once, then stop the instance unless a signal ended it. Returns how the pair ended, or `given_up`
    when the card had given up and no retry came."""
    pair = Pair(card.state)
    outcome = await retried(card.state / GIVEN_UP, ended)
    if outcome is None:
        try:
            outcome = await pair.run(card.engine(), card.gateway(), float(card.manifest["LOAD_DEADLINE_S"]), ended)
        except OSError:  # a process that could not start
            outcome = "spawn_failed"
        if outcome != "signal" and pair.asleep:
            outcome = "asleep"  # the gateway had begun to stop the instance
    log.row("pair_end", reason=outcome)
    if outcome != "signal":
        if outcome not in ("asleep", "given_up") and not pair.ready.is_set():
            (card.state / GIVEN_UP).write_text(f"{outcome}\n")
        await first(vast.stop_until_accepted(stop, partial(unconfirmed, card.state)), ended.wait())
    return outcome


async def retried(marker: Path, ended: asyncio.Event) -> str | None:
    """None when the pair may run: at once on a card that has not given up, or once the owner's --retry has removed
    `marker` within the idle interval. Otherwise `given_up` at the interval's end, or `signal`."""
    if not marker.exists():  # noqa: ASYNC240 - one stat of a local file, in the launcher's own process
        return None
    log.row("wait_for_retry")
    return ("signal", "given_up", None)[await first(ended.wait(), asyncio.sleep(IDLE_TIMEOUT_S), removed(marker))]


async def removed(path: Path) -> None:
    while path.exists():  # noqa: ASYNC240 - as above, every POLL_S
        await asyncio.sleep(POLL_S)


def unconfirmed(state: Path, failure: Mapping[str, Any]) -> None:
    """Note a failed attempt to stop the instance, the launcher's or the gateway's: until the launcher's next start,
    the card's stop is not confirmed and costs may go on."""
    if not (state / UNCONFIRMED).exists():
        (state / UNCONFIRMED).write_text(f"{failure.get('code', 'exception')}\n")
        kept = ("code", "status", "exception")  # vast.py's fields of a failure
        log.row("stop_unconfirmed", **{name: failure[name] for name in kept if name in failure})


class Pair:
    """vLLM and the gateway, each in a process group of its own, their output read to the end."""

    def __init__(self, state: Path) -> None:
        self.state = state
        self.gateway_log = JsonLines(state / "logs/gateway.jsonl", GATEWAY_LOG_BYTES)
        self.ready = asyncio.Event()
        self.asleep = False
        self.noted: set[str] = set()  # the categories logged so far, each once

    async def run(self, engine: Command, gateway: Command, deadline_s: float, ended: asyncio.Event) -> str:
        """Run until a signal, the exit of either process or the load deadline, whichever comes first, and return
        which; then end both, the gateway first, and read the rest of their output."""
        started = asyncio.get_running_loop().time()
        processes: dict[str, asyncio.subprocess.Process] = {}
        readers = []
        try:
            for name, command, handle in (("engine", engine, self.engine_line),
                                          ("gateway", gateway, self.gateway_line)):
                processes[name] = await spawn(command)
                readers.append(asyncio.create_task(read(processes[name], handle)))
            which = await first(ended.wait(), processes["engine"].wait(), processes["gateway"].wait(),
                                self.load(deadline_s, started))
            return ("signal", "engine_exit", "gateway_exit", "load_deadline")[which]
        finally:
            for name, grace_s in (("gateway", GATEWAY_GRACE_S), ("engine", ENGINE_GRACE_S)):
                if name in processes:
                    await end(processes[name], grace_s)
                    log.row("exit", process=name, exit_code=processes[name].returncode)
            for reader in readers:  # the rest of the output, such as the lines before an exit
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(reader, GATEWAY_GRACE_S)

    async def load(self, deadline_s: float, started: float) -> None:
        """Return at the deadline if the gateway is not ready by then. Once it is ready, never return."""
        try:
            async with asyncio.timeout(deadline_s):
                await self.ready.wait()
        except TimeoutError:
            return
        log.row("ready", load_ms=round((asyncio.get_running_loop().time() - started) * 1000))
        await asyncio.Future()

    def engine_line(self, line: bytes) -> None:
        """A line of vLLM's output gives a few numbers, a category of failure or of warning, or nothing."""
        text = line.decode(errors="replace")
        fields = {name: value for pattern in MEASUREMENTS if (found := pattern.search(text))
                  for name, value in found.groupdict().items()}  # vLLM prints some of them on one line
        if fields:
            log.row("engine_measure", **{name: round(float(value.replace(",", "")) * SCALE[name.split("_")[-1]])
                                         for name, value in fields.items()})
            return
        lowered = text.lower()
        for event, categories in (("engine_failure", FAILURES), ("engine_warning", WARNINGS)):
            category = next((name for name, words in categories if any(word in lowered for word in words)), None)
            if category is not None and category not in self.noted:
                self.noted.add(category)
                log.row(event, code=category)

    def gateway_line(self, line: bytes) -> None:
        row = gateway_row(line)
        if row is None:
            log.row("gateway_output")
            return
        with contextlib.suppress(OSError):  # a full disk must not stop the reading, or the gateway would block
            self.gateway_log.write(line.decode().rstrip("\n") + "\n")
        if row.get("event") == "engine" and row.get("service_status") == "ready":
            self.ready.set()
        if row.get("event") == "stop_failed":  # the gateway's own stop, once it has fallen asleep
            unconfirmed(self.state, row)
        self.asleep = self.asleep or row.get("event") == "sleep"


def gateway_row(line: bytes) -> dict[str, Any] | None:
    """A row the gateway's log wrote, or None. The gateway's other output, such as a traceback, may quote a request."""
    try:
        row = json.loads(line.decode())
    except (ValueError, RecursionError):
        return None
    if not isinstance(row, dict) or not row.keys() <= GATEWAY_KEYS:
        return None
    return row if all(value is None or isinstance(value, str | int | float) for value in row.values()) else None


async def spawn(command: Command) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *command.argv, cwd=command.cwd, env=command.env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, start_new_session=True, limit=LINE_BYTES)


async def read(process: asyncio.subprocess.Process, handle: Callable[[bytes], None]) -> None:
    """Read a process's output to its end, a line at a time. Of a line over LINE_BYTES, `handle` gets the first
    LINE_BYTES, and the rest is dropped."""
    assert process.stdout is not None
    starts = True  # whether the next piece read starts a line
    while True:
        try:
            piece = await process.stdout.readuntil(b"\n")
        except asyncio.IncompleteReadError as error:  # the last line, without its newline, or nothing
            piece = error.partial
        except asyncio.LimitOverrunError as error:  # a part of a long line, as much as has arrived
            piece = await process.stdout.read(error.consumed)
        if not piece:
            return
        if starts:
            handle(piece[:LINE_BYTES])
        starts = piece.endswith(b"\n")


async def end(process: asyncio.subprocess.Process, grace_s: float) -> None:
    """End a process and its group, which may outlive it: SIGTERM, then SIGKILL after `grace_s`."""
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        async with asyncio.timeout(grace_s):
            await process.wait()
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def first(*awaitables: Awaitable[Any]) -> int:
    """Wait for the first of several to end, cancel the others, and return the index of the first."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        return next(index for index, task in enumerate(tasks) if task in done)
    finally:
        for task in tasks:
            task.cancel()


class JsonLines:
    """A log file of the card, 0600 once written to. Before a line would take it past `max_bytes`, it becomes
    `<name>.1`, whose mode stays as it was."""

    def __init__(self, path: Path, max_bytes: int) -> None:
        self.path = path
        self.max_bytes = max_bytes

    def write(self, text: str) -> None:
        data = text.encode()
        with contextlib.suppress(FileNotFoundError):
            if self.path.stat().st_size + len(data) > self.max_bytes:
                self.path.replace(self.path.with_name(self.path.name + ".1"))
        descriptor = private(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, data)
        finally:
            os.close(descriptor)

    def flush(self) -> None:
        pass


def private(path: Path, flags: int) -> int:
    """Open a file that its owner alone may read. The mode is set on an older file too, which umask leaves as it was."""
    descriptor = os.open(path, flags | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    os.fchmod(descriptor, 0o600)
    return descriptor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m simple_serving.card", description=__doc__.split("\n\n")[0])
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--hold", action="store_true", help="wait while the pair runs: the remote command of a tunnel")
    options.add_argument("--stop", action="store_true", help="end the pair, and leave the instance running")
    options.add_argument("--retry", action="store_true", help="remove given-up and start")
    options.add_argument("--dry-run", action="store_true", help="print the two commands and check; start nothing")
    options.add_argument("--run", action="store_true", help=argparse.SUPPRESS)  # the launcher that start() spawns
    args = parser.parse_args(argv)
    if args.stop:  # before the manifest is read, which ending the pair does not need
        return stop_pair(STATE)
    card = Card.load()
    if args.hold:
        return hold(card)
    if args.run:
        return run(card)
    if args.retry:
        (card.state / GIVEN_UP).unlink(missing_ok=True)
        if launcher(card.state) is not None:
            print(f"simple-serving card: a launcher runs. If it still waits for a retry, it runs the pair within "
                  f"{POLL_S} seconds; if it already stops the instance, the retry is deferred to the next start. up "
                  "tells which.", flush=True)
            return 0
    return start(card, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

"""The card's launcher and scripts (contract section 8, "Starting the card"): one pair with no restarts, how it ends,
what it keeps of the output, and a bootstrap and an onstart that change nothing when they run again. Stand-ins take
the place of vLLM, the gateway, pip, curl and flock; the clock and Vast are fakes, and nothing reaches the network."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from simple_serving import card
from simple_serving.card import Card

from .support import FakeClock, FakeVast, until

ENGINE, GATEWAY = "vllm/bin/vllm", "gateway/bin/python"
SECRET = "synthetic-secret-7f3a9c"  # stands in for a prompt, which no log may hold
CREDENTIAL = {"CONTAINER_ID": "1234", "CONTAINER_API_KEY": "synthetic-container-key"}
CLIENT, CONTROL = (hashlib.sha256(key).hexdigest() for key in (b"synthetic-client-key", b"synthetic-control-key"))
READY = json.dumps({"time": "2026-09-24T12:00:00.000Z", "event": "engine", "service_status": "ready",
                    "context_tokens": 65536})
SLEEP = json.dumps({"time": "2026-09-24T12:13:00.000Z", "event": "sleep", "reason": "idle"})
STAND_IN = """#!{python}
import json, os, sys, time
with open(sys.argv[0] + ".plan") as file:
    plan = json.load(file)
with open(sys.argv[0] + ".ran", "w") as file:
    json.dump({{"argv": sys.argv[1:], "env": sorted(os.environ)}}, file)
for line in plan["lines"]:
    print(line, flush=True)
if plan["exit"] is None:
    time.sleep(600)
sys.exit(plan["exit"])
"""


def stand_in(path: Path, *lines: str, exit_code: int | None = None) -> None:
    """A stand-in for vLLM or the gateway's Python: it notes its arguments and the names in its environment, prints
    `lines`, then exits with `exit_code` or waits to be ended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STAND_IN.format(python=sys.executable))
    path.chmod(0o700)
    Path(f"{path}.plan").write_text(json.dumps({"lines": lines, "exit": exit_code}))


def ran(path: Path) -> dict[str, Any]:
    return json.loads(Path(f"{path}.ran").read_text())


def rows(logged: io.StringIO | str, event: str) -> list[dict[str, Any]]:
    text = logged if isinstance(logged, str) else logged.getvalue()
    found = (json.loads(line) for line in text.splitlines())
    return [{name: value for name, value in row.items() if name != "time"}
            for row in found if row.get("event") == event]


def exits(logged: io.StringIO | str) -> dict[str, int]:
    return {row["process"]: row["exit_code"] for row in rows(logged, "exit")}


@pytest.fixture(autouse=True)
def no_vast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever path a test takes, it never reaches Vast: every stop that is not a fake fails at once."""
    async def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a test tried to reach Vast")

    monkeypatch.setattr("simple_serving.vast.stop", refuse)


@pytest.fixture
def state(tmp_path: Path) -> Path:
    (tmp_path / "state/logs").mkdir(parents=True)
    return tmp_path / "state"


def card_at(state: Path, **manifest: str) -> Card:
    loaded = Card.load(state=state, environ=CREDENTIAL, root=state)
    return replace(loaded, manifest={**loaded.manifest, **manifest})


def eventually(condition: Callable[[], bool]) -> None:
    """Wait for another process to get somewhere, which takes it milliseconds."""
    deadline = time.monotonic() + 10
    while not condition():
        assert time.monotonic() < deadline, "the condition did not come true in time"
        time.sleep(0.01)


@pytest.mark.anyio
async def test_a_signal_ends_the_pair_and_leaves_the_instance_running(state: Path, logged: io.StringIO,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in CREDENTIAL.items():  # as Vast gives them to the container, and so to onstart
        monkeypatch.setenv(name, value)
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY, READY)
    ended, vast = asyncio.Event(), FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state), ended, vast))
    await until(lambda: bool(rows(logged, "ready")))
    ended.set()
    assert await launch == "signal"
    assert (vast.attempts, (state / "given-up").exists()) == ([], False)
    assert exits(logged) == {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}
    assert (state / "logs/gateway.jsonl").read_text() == READY + "\n"
    engine, gateway = ran(state / ENGINE), ran(state / GATEWAY)
    argv = engine["argv"]
    assert argv[:2] == ["serve", str(state / "models" / card_at(state).manifest["MODEL_FILE"])]
    assert (argv[argv.index("--host") + 1], argv[argv.index("--port") + 1]) == ("127.0.0.1", "8092")
    assert argv[argv.index("--scheduling-policy") + 1] == "priority"
    assert "CONTAINER_API_KEY" not in engine["env"] and "VLLM_NO_USAGE_STATS" in engine["env"]
    assert gateway["argv"] == ["-m", "simple_serving"]
    assert {"CONTAINER_ID", "CONTAINER_API_KEY", "SIMPLE_SERVING_CONFIG"} <= set(gateway["env"])


@pytest.mark.anyio
async def test_a_load_that_fails_gives_up_and_stops_the_instance(state: Path, logged: io.StringIO) -> None:
    stand_in(state / ENGINE, "Traceback (most recent call last):", f'  File "{SECRET}.py", line 1, in load',
             "x" * 3 * card.LINE_BYTES + SECRET, f"torch.OutOfMemoryError: CUDA out of memory. {SECRET}", exit_code=1)
    stand_in(state / GATEWAY)
    vast = FakeVast()
    assert await card.launch(card_at(state), asyncio.Event(), vast) == "engine_exit"
    assert (state / "given-up").read_text() == "engine_exit\n"
    assert vast.stopped
    assert [row["code"] for row in rows(logged, "engine_failure")] == ["traceback", "out_of_memory"]
    assert exits(logged) == {"gateway": -signal.SIGTERM, "engine": 1}
    assert SECRET not in logged.getvalue()


@pytest.mark.anyio
async def test_a_load_that_is_not_ready_by_the_deadline_gives_up(state: Path, logged: io.StringIO,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY)
    vast = FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state, LOAD_DEADLINE_S="900"), asyncio.Event(), vast))
    await until(lambda: Path(f"{state / GATEWAY}.ran").exists())
    await clock.advance(899)
    assert not launch.done()
    await clock.advance(1)
    assert await launch == "load_deadline"
    assert (state / "given-up").read_text() == "load_deadline\n"
    assert vast.stopped


@pytest.mark.anyio
@pytest.mark.parametrize(("lines", "exit_code", "outcome"), [
    ((READY, SLEEP), 0, "asleep"),  # the gateway had begun to stop the instance
    ((READY,), 1, "gateway_exit"),
])
async def test_an_end_after_ready_stops_the_instance_without_giving_up(state: Path, logged: io.StringIO,
                                                                       lines: tuple[str, ...], exit_code: int,
                                                                       outcome: str) -> None:
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY, *lines, exit_code=exit_code)
    vast = FakeVast()
    assert await card.launch(card_at(state), asyncio.Event(), vast) == outcome
    assert vast.stopped
    assert not (state / "given-up").exists()
    assert exits(logged) == {"gateway": exit_code, "engine": -signal.SIGTERM}


@pytest.mark.anyio
async def test_a_gateway_that_cannot_start_ends_the_engine(state: Path, logged: io.StringIO) -> None:
    stand_in(state / ENGINE)
    vast = FakeVast()
    assert await card.launch(card_at(state), asyncio.Event(), vast) == "spawn_failed"
    assert exits(logged) == {"engine": -signal.SIGTERM}
    assert (state / "given-up").read_text() == "spawn_failed\n"
    assert vast.stopped


def test_the_engine_output_leaves_a_few_numbers_and_categories(tmp_path: Path, logged: io.StringIO) -> None:
    pair = card.Pair(tmp_path)
    for line in (
        "INFO 09-24 12:00:00 [gpu_model_runner.py:2007] Model loading took 23.5000 GiB memory and 41.250000 seconds",
        "INFO 09-24 12:00:10 [gpu_worker.py:298] Available KV cache memory: 4.25 GiB",
        "INFO 09-24 12:00:10 [kv_cache_utils.py:1087] GPU KV cache size: 123,456 tokens",
        "INFO 09-24 12:00:10 [kv_cache_utils.py:1091] Maximum concurrency for 65,536 tokens per request: 3.45x",
        f"INFO 09-24 12:01:00 [logger.py:43] Received request chatcmpl-1: prompt: '{SECRET}'",
        f"ERROR 09-24 12:02:00 [core.py:710] RuntimeError: CUDA error: an illegal memory access, {SECRET}",
        f"ERROR 09-24 12:02:00 [core.py:710] RuntimeError: CUDA error: {SECRET}",
        ("ValueError: To serve at least one request with the models's max seq len (65536), (8.00 GiB KV cache is "
         "needed, which is larger than the available KV cache memory (4.25 GiB)."),
    ):
        pair.engine_line(line.encode() + b"\n")
    assert rows(logged, "engine_measure") == [
        {"event": "engine_measure", "weights_mib": 24064, "weights_load_ms": 41250},
        {"event": "engine_measure", "kv_cache_mib": 4352},
        {"event": "engine_measure", "kv_cache_tokens": 123456},
        {"event": "engine_measure", "concurrency_x100": 345},
    ]
    assert [row["code"] for row in rows(logged, "engine_failure")] == ["cuda_error", "kv_cache_too_small"]
    assert SECRET not in logged.getvalue()


def test_the_gateway_output_keeps_the_gateways_own_rows_only(tmp_path: Path, logged: io.StringIO) -> None:
    pair = card.Pair(tmp_path)
    lines = [
        READY.encode(),
        f"Traceback (most recent call last): {SECRET}".encode(),
        json.dumps({"event": "request", "prompt": SECRET}).encode(),  # a field outside log.FIELDS
        json.dumps({"event": "request", "code": [SECRET]}).encode(),  # a value that is not a scalar
        b"[" * 5000,
        b'{"event": "\xff"}',
        b"",  # as read() hands on a line that was too long
    ]
    for line in lines:
        pair.gateway_line(line + b"\n")
    assert (tmp_path / "gateway.jsonl").read_text() == READY + "\n"
    assert len(rows(logged, "gateway_output")) == len(lines) - 1
    assert pair.ready.is_set() and not pair.asleep
    assert SECRET not in logged.getvalue()


def test_a_log_file_moves_aside_before_it_grows_past_its_size(tmp_path: Path) -> None:
    log_file = card.JsonLines(tmp_path / "card.jsonl", 100)
    for index in range(5):
        log_file.write(json.dumps({"event": "synthetic", "index": index, "padding": "x" * 20}) + "\n")
    current, older = tmp_path / "card.jsonl", tmp_path / "card.jsonl.1"
    assert current.stat().st_size <= 100 and older.stat().st_size <= 100
    assert json.loads(current.read_text().splitlines()[-1])["index"] == 4
    assert {current.stat().st_mode & 0o777, older.stat().st_mode & 0o777} == {0o600}


def test_the_launcher_runs_one_pair_until_sigterm(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    stand_in(state / ENGINE, "INFO 09-24 12:00:10 [kv_cache_utils.py:1087] GPU KV cache size: 123,456 tokens")
    stand_in(state / GATEWAY, READY)
    environ = {name: value for name, value in os.environ.items() if not name.startswith("CONTAINER_")}
    environ |= {"SIMPLE_SERVING_CARD_DIR": str(state), "SIMPLE_SERVING_CARD_ROOT": str(tmp_path)}
    command = [sys.executable, "-m", "simple_serving.card", "--run"]
    launcher = subprocess.Popen(command, cwd=card.CODE, env=environ)
    card_log = state / "logs/card.jsonl"
    try:
        eventually(lambda: card_log.exists() and bool(rows(card_log.read_text(), "ready")))
        assert card.launcher(state) == launcher.pid
        assert subprocess.run(command, cwd=card.CODE, env=environ, timeout=30, check=False).returncode == 0  # the lock
    finally:
        launcher.terminate()  # a launcher that is killed leaves its pair running
        assert launcher.wait(timeout=30) == 0
    assert card.launcher(state) is None
    assert not (state / "given-up").exists()
    text = card_log.read_text()
    assert rows(text, "engine_measure") == [{"event": "engine_measure", "kv_cache_tokens": 123456}]
    assert exits(text) == {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}
    assert rows(text, "pair_end") == [{"event": "pair_end", "reason": "signal"}]
    config = json.loads((state / "gateway.json").read_text())
    assert [(key["label"], key["sha256"]) for key in config["keys"]] == [("client", CLIENT), ("control", CONTROL)]
    assert (config["engine_url"], config["listen"]["control"]) == ("http://127.0.0.1:8092",
                                                                   {"host": "127.0.0.1", "port": 8091})
    assert {path.stat().st_mode & 0o777 for path in (state / "gateway.json", card_log)} == {0o600}


def test_a_second_launcher_leaves_the_pair_to_the_first(state: Path) -> None:
    with open(state / "launcher.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert card.run(card_at(state)) == 0
    assert not (state / "launcher.pid").exists()


def prepared(tmp_path: Path) -> Card:
    """A card that bootstrap.sh has prepared, on free loopback ports."""
    code, state = tmp_path / "code", tmp_path / "state"
    (code / "card").mkdir(parents=True)
    state.mkdir()
    ports = []
    for name in ("PUBLIC", "CONTROL", "ENGINE"):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(f"{name}_PORT={sock.getsockname()[1]}\n")
    (code / "card/manifest.env").write_text((card.CODE / "card/manifest.env").read_text() + "".join(ports))
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    stamp = hashlib.sha256(b"".join((code / "card" / name).read_bytes() for name in card.STAMPED))
    (state / "prepared").write_text(stamp.hexdigest() + "\n")
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    return Card.load(code=code, state=state, environ=CREDENTIAL, root=state)


def test_start_says_what_stands_in_the_way(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ready = prepared(tmp_path)
    assert card.start(ready, dry_run=True) == 0
    engine, gateway = capsys.readouterr().out.splitlines()
    assert engine.startswith(f"{ready.state}/vllm/bin/vllm serve {ready.state}/models/")
    assert gateway == f"{ready.state}/gateway/bin/python -m simple_serving"
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", ready.ports["control"]))
        busy.listen()
        assert card.start(ready, dry_run=True) == card.PORT_TAKEN
    assert card.start(replace(ready, credential={}), dry_run=True) == card.NOT_PREPARED
    manifest = ready.code / "card/manifest.env"
    manifest.write_text(manifest.read_text() + "MAX_NUM_SEQS=4\n")  # a change that bootstrap.sh has not prepared
    assert card.start(ready, dry_run=True) == card.NOT_PREPARED
    (ready.state / "given-up").write_text("load_deadline\n")
    assert card.start(ready, dry_run=True) == card.GAVE_UP


def launcher_stand_in(state: Path) -> subprocess.Popen[bytes]:
    """A process that passes for a running launcher."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "simple_serving.card", "--run"])
    (state / "launcher.pid").write_text(f"{process.pid}\n")
    return process


def test_hold_lasts_as_long_as_the_launcher(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ready = prepared(tmp_path)
    state = ready.state
    pauses: list[float] = []
    assert card.hold(ready, pauses.append) == 0  # none started: it waits a while for onstart
    assert sum(pauses) == card.HOLD_START_S
    assert capsys.readouterr().out == f"{card.HOLDING}\n"  # the command's sign that the tunnel's forwards are up
    assert card.hold(replace(ready, credential={}), pauses.append) == card.NOT_PREPARED  # and none will start

    process, pauses = launcher_stand_in(state), []

    def end_on_the_third(seconds: float) -> None:
        pauses.append(seconds)
        if len(pauses) == 3:
            process.kill()
            process.wait()

    assert card.hold(ready, end_on_the_third) == 0
    assert len(pauses) == 3

    process = launcher_stand_in(state)
    try:
        assert card.hold(ready, lambda _: (state / "given-up").write_text("engine_exit\n")) == card.GAVE_UP
    finally:
        process.kill()
        process.wait()


def test_stop_ends_the_launcher(state: Path) -> None:
    process = launcher_stand_in(state)
    assert card.stop_pair(state, lambda _: process.wait()) == 0
    assert process.wait() == -signal.SIGTERM


def recorder(path: Path, calls: Path) -> None:
    """A stand-in command that notes each of its calls in `calls`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "${{0##*/}} $*" >> {calls}\n')
    path.chmod(0o700)


def test_bootstrap_and_onstart_change_nothing_when_they_run_again(tmp_path: Path) -> None:
    code, state, root, calls = tmp_path / "code", tmp_path / "state", tmp_path / "root", tmp_path / "calls"
    (code / "card").mkdir(parents=True)
    root.mkdir()
    for name in ("bootstrap.sh", "onstart.sh"):
        shutil.copy(card.CODE / "card" / name, code / "card" / name)
    manifest = (card.CODE / "card/manifest.env").read_text()
    (code / "card/manifest.env").write_text(manifest)
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    weights, tokenizer = b"synthetic weights", b'{"synthetic": true}'
    model_file = Card.load(code=code, state=state, environ={}, root=root).manifest["MODEL_FILE"]
    (state / "models").mkdir(parents=True)
    (state / "models" / model_file).write_bytes(weights)  # in place, so nothing is fetched
    (state / "tokenizer").mkdir()
    (state / "tokenizer/tokenizer.json").write_bytes(tokenizer)
    for path in (state / "gateway/bin/pip", state / "vllm/bin/pip", state / GATEWAY, tmp_path / "bin/curl",
                 tmp_path / "bin/flock"):
        recorder(path, calls)
    environ = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "SIMPLE_SERVING_CARD_DIR": str(state),
               "SIMPLE_SERVING_CARD_ROOT": str(root), **CREDENTIAL}

    def bootstrap(stdin: str) -> int:
        done = subprocess.run(["bash", str(code / "card/bootstrap.sh")], input=stdin, env=environ, capture_output=True,
                              text=True, timeout=60, check=False)
        assert CLIENT not in done.stdout + done.stderr and CONTROL not in done.stdout + done.stderr
        return done.returncode

    keys = f"{CLIENT}\n{CONTROL}\n"
    assert bootstrap(keys) == 3  # the pins are empty
    assert not (state / "keys.json").exists()
    pins = {"VLLM_VERSION": "0.0.1", "MODEL_SHA256": hashlib.sha256(weights).hexdigest(),
            "TOKENIZER_REPO": "synthetic/tokenizer", "TOKENIZER_REVISION": "0" * 40,
            "TOKENIZER_FILES": f"tokenizer.json:{hashlib.sha256(tokenizer).hexdigest()}"}
    (code / "card/manifest.env").write_text(manifest + "".join(f"{name}={value}\n" for name, value in pins.items()))
    assert bootstrap(keys) == 0
    written = [state / "keys.json", state / "prepared", root / "onstart.sh", *root.glob(".simple-chat-*")]
    first = {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written}
    assert bootstrap(keys) == 0
    assert bootstrap("") == 0  # the card holds the keys
    assert bootstrap(f"{hashlib.sha256(b'other').hexdigest()}\n{CONTROL}\n") == 5
    assert bootstrap(f"{CLIENT}\n") == 4
    assert bootstrap(f"{CLIENT}\n{CLIENT}\n") == 4
    assert {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written} == first
    assert {mode for _, mode in first.values()} == {0o600}
    assert json.loads((state / "keys.json").read_text()) == {"client": CLIENT, "control": CONTROL}
    assert (root / "onstart.sh").read_text() == f"bash {code}/card/onstart.sh\n"
    assert (root / ".simple-chat-instance-id").read_text() == CREDENTIAL["CONTAINER_ID"]
    noted = calls.read_text().splitlines()
    assert noted.count("python -m simple_serving.card") == 3
    assert not any(line.startswith(("curl", "flock")) for line in noted)  # nothing to fetch, and no guard
    assert not list(root.glob(".simple-chat-trial-*"))
    assert all("--require-hashes" in line for line in noted if line.startswith("pip"))
    assert Card.load(code=code, state=state, environ=environ, root=root).prepared()

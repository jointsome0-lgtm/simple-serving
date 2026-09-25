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
from types import SimpleNamespace
from typing import Any, cast

import pytest

from simple_serving import card
from simple_serving.card import Card
from simple_serving.vast import RETRY_S, VastError

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
    assert argv[:2] == ["serve", str(state / "models" / card_at(state).manifest["MODEL_REVISION"])]
    assert (argv[argv.index("--host") + 1], argv[argv.index("--port") + 1]) == ("127.0.0.1", "8092")
    assert argv[argv.index("--scheduling-policy") + 1] == "priority"
    assert argv[argv.index("--reasoning-parser") + 1] == "gemma4" and "--language-model-only" in argv
    assert "CONTAINER_API_KEY" not in engine["env"] and "VLLM_NO_USAGE_STATS" in engine["env"]
    assert gateway["argv"] == ["-m", "simple_serving"]
    assert {"CONTAINER_ID", "CONTAINER_API_KEY", "SIMPLE_SERVING_CONFIG"} <= set(gateway["env"])


FAILED_LOAD = ("Traceback (most recent call last):", f'  File "{SECRET}.py", line 1, in load',
               "x" * 3 * card.LINE_BYTES + SECRET, f"torch.OutOfMemoryError: CUDA out of memory. {SECRET}")
WAITS = ((), None)  # a stand-in that prints nothing and waits to be ended


@pytest.mark.anyio
@pytest.mark.parametrize(("engine", "gateway", "outcome", "exit_codes"), [
    ((FAILED_LOAD, 1), WAITS, "engine_exit", {"gateway": -signal.SIGTERM, "engine": 1}),
    (WAITS, WAITS, "load_deadline", {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}),
    (WAITS, None, "spawn_failed", {"engine": -signal.SIGTERM}),  # a gateway that cannot start
    (WAITS, ((READY, SLEEP), 0), "asleep", {"gateway": 0, "engine": -signal.SIGTERM}),  # the gateway's own stop began
    (WAITS, ((READY,), 1), "gateway_exit", {"gateway": 1, "engine": -signal.SIGTERM}),
])
async def test_a_pair_that_ends_stops_the_instance_and_gives_up_only_before_ready(
        state: Path, logged: io.StringIO, monkeypatch: pytest.MonkeyPatch, engine: tuple[tuple[str, ...], int | None],
        gateway: tuple[tuple[str, ...], int | None] | None, outcome: str, exit_codes: dict[str, int]) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE, *engine[0], exit_code=engine[1])
    if gateway is not None:
        stand_in(state / GATEWAY, *gateway[0], exit_code=gateway[1])
    vast = FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state, LOAD_DEADLINE_S="900"), asyncio.Event(), vast))
    if outcome == "load_deadline":
        await until(lambda: Path(f"{state / GATEWAY}.ran").exists())
        await clock.advance(899)
        assert not launch.done()
        await clock.advance(1)
    assert await launch == outcome
    if outcome in ("asleep", "gateway_exit"):  # an end after ready gives nothing up
        assert not (state / "given-up").exists()
    else:
        assert (state / "given-up").read_text() == f"{outcome}\n"
    assert vast.stopped and exits(logged) == exit_codes
    failures = [row["code"] for row in rows(logged, "engine_failure")]
    assert failures == (["traceback", "out_of_memory"] if outcome == "engine_exit" else [])
    assert SECRET not in logged.getvalue()


@pytest.mark.anyio
async def test_a_card_that_gave_up_loads_nothing_at_a_resume_and_stops_again(state: Path, logged: io.StringIO,
                                                                             monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE, exit_code=1)
    stand_in(state / GATEWAY)
    failed = FakeVast()
    assert await card.launch(card_at(state), asyncio.Event(), failed) == "engine_exit"
    assert failed.stopped
    resumed = FakeVast(VastError("forbidden", 403))  # the owner resumes the instance, and onstart starts the launcher
    launch = asyncio.create_task(card.launch(card_at(state), asyncio.Event(), resumed))
    await until(lambda: bool(rows(logged, "wait_for_retry")))
    await clock.advance(card.IDLE_TIMEOUT_S - 1)
    assert not launch.done() and resumed.attempts == []
    await clock.advance(1)
    await until(lambda: bool(rows(logged, "stop_unconfirmed")))  # Vast refuses the first stop
    assert (state / "stop-unconfirmed").read_text() == "forbidden\n"
    await clock.advance(RETRY_S)
    assert await launch == "given_up"
    assert resumed.stopped and (state / "given-up").read_text() == "engine_exit\n"
    assert rows(logged, "stop_unconfirmed") == [{"event": "stop_unconfirmed", "code": "forbidden", "status": 403}]
    assert [row["reason"] for row in rows(logged, "pair_end")] == ["engine_exit", "given_up"]
    assert len(rows(logged, "exit")) == 2  # the processes of the failed load, and none since


@pytest.mark.anyio
async def test_a_retry_within_the_interval_runs_the_pair_and_stops_nothing(state: Path, logged: io.StringIO,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    stand_in(state / ENGINE)
    stand_in(state / GATEWAY, READY)
    (state / "given-up").write_text("load_deadline\n")
    ended, vast = asyncio.Event(), FakeVast()
    launch = asyncio.create_task(card.launch(card_at(state), ended, vast))
    await until(lambda: bool(rows(logged, "wait_for_retry")))
    await clock.advance(card.IDLE_TIMEOUT_S - 60)
    (state / "given-up").unlink()  # what --retry does, and its start then finds this launcher running
    await clock.advance(card.POLL_S)
    await until(lambda: bool(rows(logged, "ready")))
    await clock.advance(60)  # past the end of the interval, which no longer counts
    ended.set()
    assert await launch == "signal"
    assert vast.attempts == []


def test_the_output_leaves_a_few_numbers_and_categories_of_vllm_and_the_gateways_own_rows(state: Path,
                                                                                          logged: io.StringIO) -> None:
    pair = card.Pair(state)
    for text in (
        "INFO 09-24 12:00:00 [gpu_model_runner.py:2007] Model loading took 23.5000 GiB memory and 41.250000 seconds",
        "INFO 09-24 12:00:10 [gpu_worker.py:298] Available KV cache memory: 4.25 GiB",
        ("(EngineCore pid=7) INFO 09-24 12:00:10 [kv_cache_utils.py:2395] GPU KV cache size: 123,456 tokens, Maximum "
         "concurrency for 65,536 tokens per request: 3.45x"),
        "INFO 09-24 12:00:10 [gpu_worker.py:298] Available KV cache memory: -1.25 GiB",
        f"INFO 09-24 12:01:00 [logger.py:43] Received request chatcmpl-1: prompt: '{SECRET}'",
        f"ERROR 09-24 12:02:00 [core.py:710] RuntimeError: CUDA error: an illegal memory access, {SECRET}",
        f"ERROR 09-24 12:02:00 [core.py:710] RuntimeError: CUDA error: {SECRET}",
        ("ValueError: To serve at least one request with the models's max seq len (65536), (8.00 GiB KV cache is "
         "needed, which is larger than the available KV cache memory (4.25 GiB)."),
        ("ValueError: Free memory on device cuda:0 (28.1/31.84 GiB) on startup is less than desired GPU memory "
         "utilization (0.92, 29.3 GiB)."),
        "AssertionError: Error in memory profiling. Initial free memory 29.1 GiB, current free memory 29.6 GiB.",
        "RuntimeError: Frontend process failed during engine core initialization. See root cause above.",
        ("(EngineCore pid=7) WARNING 09-24 12:00:05 [modelopt.py:1981] In NVFP4 linear, the global weight scale "
         "differs across parallel layers (e.g. q_proj, k_proj, v_proj). This will likely reduce accuracy."),
    ):
        pair.engine_line(text.encode() + b"\n")
    assert rows(logged, "engine_measure") == [
        {"event": "engine_measure", "weights_mib": 24064, "weights_load_ms": 41250},
        {"event": "engine_measure", "kv_cache_mib": 4352},
        {"event": "engine_measure", "kv_cache_tokens": 123456, "concurrency_x100": 345},
        {"event": "engine_measure", "kv_cache_mib": -1280},
    ]
    assert [row["code"] for row in rows(logged, "engine_failure")] == [
        "cuda_error", "kv_cache_too_small", "memory_taken", "memory_profiling_failed", "engine_dead"]
    assert rows(logged, "engine_warning") == [{"event": "engine_warning", "code": "fused_scales_differ"}]
    refused = json.dumps({"event": "stop_failed", "code": "forbidden", "status": 403})  # the gateway's own stop
    lines = [
        READY.encode(),
        refused.encode(),
        f"Traceback (most recent call last): {SECRET}".encode(),
        json.dumps({"event": "request", "prompt": SECRET}).encode(),  # a field outside log.FIELDS
        json.dumps({"event": "request", "code": [SECRET]}).encode(),  # a value that is not a scalar
        b"[" * 5000,
        b'{"event": "\xff"}',
        b"",  # a blank line
    ]
    for line in lines:
        pair.gateway_line(line + b"\n")
    assert (state / "logs/gateway.jsonl").read_text() == f"{READY}\n{refused}\n"
    assert len(rows(logged, "gateway_output")) == len(lines) - 2
    assert pair.ready.is_set() and not pair.asleep and card.marked(state) == card.STOP_UNCONFIRMED
    assert SECRET not in logged.getvalue()


@pytest.mark.anyio
async def test_of_a_line_over_line_bytes_only_the_head_is_read() -> None:
    output = asyncio.StreamReader(limit=card.LINE_BYTES)
    output.feed_data(b"x" * 3 * card.LINE_BYTES + b"CUDA error\nshort\nlast")
    output.feed_eof()
    lines: list[bytes] = []
    await card.read(cast(Any, SimpleNamespace(stdout=output)), lines.append)  # a process, as far as read() needs one
    assert lines == [b"x" * card.LINE_BYTES, b"short\n", b"last"]


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
    for older in (state / "gateway.json", state / "logs/card.jsonl"):  # left wide, as by another hand
        older.parent.mkdir(exist_ok=True)
        older.write_text('{"event": "synthetic"}\n')
        older.chmod(0o644)
    (state / "stop-unconfirmed").write_text("forbidden\n")  # of the start before
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
    assert not (state / "given-up").exists() and not (state / "stop-unconfirmed").exists()
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


def test_start_says_what_stands_in_the_way(tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                           monkeypatch: pytest.MonkeyPatch) -> None:
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
    spawned: list[list[str]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **options: spawned.append(argv))
    assert card.start(ready) == card.GAVE_UP
    assert spawned == [[sys.executable, "-m", "simple_serving.card", "--run"]]  # the launcher that will stop it


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
    assert capsys.readouterr().out == ""

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
        capsys.readouterr()
        assert card.hold(ready, pauses.append) == card.GAVE_UP
        (state / "stop-unconfirmed").write_text("forbidden\n")
        assert card.hold(ready, pauses.append) == card.STOP_UNCONFIRMED  # the stop first
        assert capsys.readouterr().out == ""
    finally:
        process.kill()
        process.wait()


def test_a_retry_leaves_the_pair_to_a_running_launcher_and_stop_ends_it(state: Path, monkeypatch: pytest.MonkeyPatch,
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    loaded, process = card_at(state), launcher_stand_in(state)
    (state / "given-up").write_text("engine_exit\n")
    monkeypatch.setattr(Card, "load", lambda: loaded)
    assert card.main(["--retry"]) == 0 and not (state / "given-up").exists()
    assert "deferred to the next start" in capsys.readouterr().out  # a 0 that does not read as ready
    assert card.stop_pair(state, lambda _: process.wait()) == 0
    assert process.wait() == -signal.SIGTERM

    def unreadable(*args: Any, **kwargs: Any) -> Card:
        raise OSError("a checkout whose manifest cannot be read")

    monkeypatch.setattr(Card, "load", unreadable)
    monkeypatch.setattr(card, "stop_pair", lambda at: 0 if at == card.STATE else 1)
    assert card.main(["--stop"]) == 0


def recorder(path: Path, calls: Path) -> None:
    """A stand-in command that notes each of its calls in `calls`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\necho "${{0##*/}} $*" >> {calls}\n')
    path.chmod(0o700)


def test_bootstrap_changes_nothing_when_it_runs_again_and_the_rentals_onstart_starts_the_card(tmp_path: Path) -> None:
    code, state, root, calls = tmp_path / "code", tmp_path / "state", tmp_path / "root", tmp_path / "calls"
    (code / "card").mkdir(parents=True)
    root.mkdir()
    environ = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "SIMPLE_SERVING_CARD_DIR": str(state),
               "SIMPLE_SERVING_CARD_ROOT": str(root), **CREDENTIAL}
    # The rental's own onstart, which ends with the README's line. The platform may write it anew at every start.
    line = next(line for line in (card.CODE / "README.md").read_text().splitlines() if line.startswith("if [[ -f /"))
    rental = f"#!/usr/bin/env bash\n{line.replace('/workspace/simple-serving', str(code))}\n"

    def start_container() -> int:
        (root / "onstart.sh").write_text(rental)
        return subprocess.run(["bash", str(root / "onstart.sh")], env=environ, capture_output=True, timeout=60,
                              check=False).returncode

    assert start_container() == 0 and not calls.exists()  # the first start, before the checkout is on the disk
    for name in ("bootstrap.sh", "onstart.sh", "gateway-requirements.txt", "vllm-requirements.txt"):
        shutil.copy(card.CODE / "card" / name, code / "card" / name)
    manifest = (card.CODE / "card/manifest.env").read_text()
    (code / "card/manifest.env").write_text(manifest)
    weights, tokenizer = b"synthetic weights", b'{"synthetic": true}'
    model = state / "models" / Card.load(code=code, state=state, environ={}, root=root).manifest["MODEL_REVISION"]
    model.mkdir(parents=True)
    for name in ("model.safetensors", "other.safetensors"):  # the pinned one in place, so nothing is fetched
        (model / name).write_bytes(weights)
    (state / "tokenizer").mkdir()
    (state / "tokenizer/tokenizer.json").write_bytes(tokenizer)
    for path in (state / "gateway/bin/pip", state / "vllm/bin/pip", state / GATEWAY, tmp_path / "bin/curl",
                 tmp_path / "bin/flock"):
        recorder(path, calls)

    def bootstrap(stdin: str) -> int:
        done = subprocess.run(["bash", str(code / "card/bootstrap.sh")], input=stdin, env=environ, capture_output=True,
                              text=True, timeout=60, check=False)
        assert CLIENT not in done.stdout + done.stderr and CONTROL not in done.stdout + done.stderr
        return done.returncode

    keys = f"{CLIENT}\n{CONTROL}\n"
    assert bootstrap("") == 4  # the real pins and locks pass, and the card holds no keys yet
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"{name}==0.0.1 \\\n    --hash=sha256:{'0' * 64}\n")
    assert bootstrap(keys) == 3  # the lock is not the pinned vLLM's
    assert not (state / "keys.json").exists()
    for older, text in ((state / "keys.json", json.dumps({"client": CLIENT, "control": CONTROL})),
                        (root / ".simple-chat-instance-api-key", "synthetic-older-key")):
        older.write_text(text + "\n")  # left wide, as by another hand
        older.chmod(0o644)
    pins = {"VLLM_VERSION": "0.0.1", "MODEL_FILES": f"model.safetensors:{hashlib.sha256(weights).hexdigest()}",
            "TOKENIZER_FILES": f"tokenizer.json:{hashlib.sha256(tokenizer).hexdigest()}"}
    (code / "card/manifest.env").write_text(manifest + "".join(f"{name}={value}\n" for name, value in pins.items()))
    assert bootstrap(keys) == 0
    assert [path.name for path in model.iterdir()] == ["model.safetensors"]  # vLLM would load any other
    written = [state / "keys.json", state / "prepared", *root.glob(".simple-chat-*")]
    first = {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written}
    assert bootstrap(keys) == 0
    assert bootstrap("") == 0  # the card holds the keys
    assert bootstrap(f"{hashlib.sha256(b'other').hexdigest()}\n{CONTROL}\n") == 5
    assert bootstrap(f"{CLIENT}\n") == 4
    assert bootstrap(f"{CLIENT}\n{CLIENT}\n") == 4
    assert {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written} == first
    assert {mode for _, mode in first.values()} == {0o600}
    assert json.loads((state / "keys.json").read_text()) == {"client": CLIENT, "control": CONTROL}
    assert (root / "onstart.sh").read_text() == rental  # the preparation changes no onstart
    kept = [(root / f".simple-chat-instance-{name}").read_text() for name in ("id", "api-key")]
    assert kept == list(CREDENTIAL.values())
    assert start_container() == 0  # a resume: the rental's own onstart, as it was, starts the card
    noted = calls.read_text().splitlines()
    assert noted.count("python -m simple_serving.card") == 4  # the three preparations, then the resume
    assert not any(line.startswith(("curl", "flock")) for line in noted)  # nothing to fetch, and no guard
    assert not list(root.glob(".simple-chat-trial-*"))
    assert all("--require-hashes --only-binary :all:" in line for line in noted if line.startswith("pip"))
    assert Card.load(code=code, state=state, environ=environ, root=root).prepared()

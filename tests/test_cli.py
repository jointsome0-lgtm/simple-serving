"""The command on the owner's machine (contract section 8), against a fake Vast, a fake SSH and a stand-in for the
gateway's control listener; `sleep` once against the gateway in front of the fake engine. The command's clock moves
only when the command waits, so no test waits its minutes. Keys and hosts are synthetic."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import signal
import stat
import subprocess
from collections.abc import AsyncIterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from simple_serving import card, cli
from tests.support import ALIAS, CONTROL, FakeVast, running, until

VIEW = {"contract": "2", "boot_id": "synthetic-boot", "status": "ready", "model": ALIAS, "context_tokens": 4096,
        "drain_generation": 0, "sleep_requested": False, "active": {"reader": 1, "agent": 0}, "waiting": {}}


class Clock:
    def __init__(self) -> None:
        self.time = 0.0

    def now(self) -> float:
        return self.time

    async def pause(self, seconds: float) -> None:
        self.time += seconds
        await asyncio.sleep(0.001)  # the gateway and the forwards go on meanwhile, in real milliseconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(cli, "now", clock.now)
    monkeypatch.setattr(cli, "pause", clock.pause)
    return clock


class Instance:
    """The instance as the owner's key sees it in Vast. The reads return `states` in turn and then the last one for
    good. With `card_stop`, the instance is stopped once the card has stopped it."""

    def __init__(self, *states: str, card_stop: FakeVast | None = None) -> None:
        self.states = list(states)
        self.card_stop = card_stop
        self.reads = self.resumes = 0
        self.last = ""

    async def show(self) -> str:
        self.reads += 1
        if self.card_stop is not None and self.card_stop.stopped:
            self.states = ["stopped"]
        self.last = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return self.last

    async def resume(self) -> None:
        assert self.last == "stopped", "a resume against a stop"
        self.resumes += 1
        self.states = ["stopped", "starting", "running"]  # Vast may report the stop once more


class FakeTunnel(cli.Tunnel):
    def __init__(self, servers: list[asyncio.Server]) -> None:
        self.servers = servers
        self.end: asyncio.Future[int] = asyncio.get_running_loop().create_future()  # ssh's exit code, once it ends
        self.closed = self.waited = False

    @property
    def returncode(self) -> int | None:
        return self.end.result() if self.end.done() else None

    async def wait(self) -> int:
        self.waited = True
        return await asyncio.shield(self.end)

    async def close(self) -> int:
        self.closed = True
        for server in self.servers:
            server.close()
        if not self.end.done():
            self.end.set_result(-signal.SIGTERM)
        return self.end.result()


class Ssh:
    """ssh as the command starts it. A tunnel forwards its local ports to `port`, which stands for the card's
    listeners, until the test ends it or the command closes it. `refuse` holds exit codes of the next connections that
    end before the remote command's first line, `end` those that end right after it."""

    def __init__(self, port: int = 0, *, refuse: tuple[int, ...] = (), end: tuple[int, ...] = ()) -> None:
        self.port = port
        self.refuse, self.end = list(refuse), list(end)
        self.opened: list[dict[int, int]] = []
        self.tunnels: list[FakeTunnel] = []

    async def __call__(self, forwards: Mapping[int, int]) -> cli.Tunnel:
        self.opened.append(dict(forwards))
        if self.refuse:
            raise cli.NoTunnel(self.refuse.pop(0))
        if self.end:
            tunnel = FakeTunnel([])
            tunnel.end.set_result(self.end.pop(0))
        else:
            tunnel = FakeTunnel([await asyncio.start_server(partial(forward, self.port), "127.0.0.1", local)
                                 for local in forwards])
        self.tunnels.append(tunnel)
        return tunnel

    def holding(self, count: int = 1) -> bool:
        """Whether `up` holds its `count`th tunnel, with the gateway ready."""
        return len(self.tunnels) == count and self.tunnels[-1].waited


async def forward(port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """One connection through a fake tunnel, carried both ways."""
    upstream_reader, upstream_writer = await asyncio.open_connection("127.0.0.1", port)

    async def copy(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
        with contextlib.suppress(ConnectionError):
            while data := await source.read(1 << 16):
                sink.write(data)
                await sink.drain()
        sink.close()

    await asyncio.gather(copy(reader, upstream_writer), copy(upstream_reader, writer))


@contextlib.asynccontextmanager
async def gateway(status: int = 200, body: Any = VIEW, drip_s: float = 0,
                  port: int = 0) -> AsyncIterator[tuple[int, list[bytes]]]:
    """A stand-in for the gateway's control listener, on `port` or a free one. It notes the head of each request and
    answers every one with `status` and `body`, a byte at a time every `drip_s` if that is set."""
    heads: list[bytes] = []

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        heads.append(await reader.readuntil(b"\r\n\r\n"))
        data = json.dumps(body).encode()
        message = b"HTTP/1.1 %d Synthetic\r\nContent-Length: %d\r\n\r\n%b" % (status, len(data), data)
        with contextlib.suppress(ConnectionError):
            for part in [message[i:i + 1] for i in range(len(message))] if drip_s else [message]:
                writer.write(part)
                await writer.drain()
                await asyncio.sleep(drip_s)
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", port)
    try:
        yield server.sockets[0].getsockname()[1], heads
    finally:
        server.close()


def port_of(url: str) -> int:
    return int(url.rsplit(":", 1)[1])


def setup_for(tmp_path: Path, instance: Instance, ssh: Ssh) -> cli.Setup:
    return cli.Setup(tmp_path / "config.json", CONTROL, ALIAS, 960, {"public": 8090, "control": 8091}, instance.show,
                     instance.resume, ssh, local={"public": cli.free_port(), "control": cli.free_port()})


# keys

def test_keys_are_made_once_and_only_their_hashes_are_printed(tmp_path: Path,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "simple-serving" / "config.json"
    assert cli.main(["--config", str(path), "keys"]) == 0
    printed = capsys.readouterr()
    config = json.loads(path.read_text())
    hashes = [hashlib.sha256(config[name].encode()).hexdigest() for name in cli.KEYS]
    assert printed.out.split() == hashes and hashes[0] != hashes[1]  # what bootstrap.sh reads on its stdin
    assert not any(config[name] in printed.out + printed.err for name in cli.KEYS)
    assert (stat.S_IMODE(path.stat().st_mode), stat.S_IMODE(path.parent.stat().st_mode)) == (0o600, 0o700)
    before = path.read_bytes()
    assert cli.main(["--config", str(path), "keys"]) == 0
    assert capsys.readouterr().out.split() == hashes and path.read_bytes() == before


def test_keys_keep_what_the_file_holds_and_it_stays_the_owners(tmp_path: Path,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"instance_id": "123", "client_key": "synthetic-client-key"}))
    path.chmod(0o600)
    assert cli.main(["--config", str(path), "keys"]) == 0
    config = json.loads(path.read_text())
    assert (config["instance_id"], config["client_key"]) == ("123", "synthetic-client-key") and config["control_key"]
    path.chmod(0o640)
    assert cli.main(["--config", str(path), "keys"]) == 1
    assert "open to its owner only" in capsys.readouterr().err


def test_the_configuration_names_the_instance_its_key_and_the_host(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}")
    path.chmod(0o600)
    with pytest.raises(cli.Refusal, match="has no instance_id, vast_api_key, ssh_host, control_key"):
        cli.load(path)
    config = {"instance_id": "123", "vast_api_key": "synthetic-vast-key", "ssh_host": "card", "control_key": "c"}
    for field, value in (("instance_id", "12/"), ("vast_api_key", "a\nb"), ("ssh_host", "-oProxyCommand=x")):
        path.write_text(json.dumps(config | {field: value}))
        with pytest.raises(cli.Refusal, match="wrong form"):
            cli.load(path)
    path.write_text(json.dumps(config))
    setup = cli.load(path)
    assert (setup.alias, setup.card_ports) == ("gemma-4-31b-heretic-nvfp4", {"public": 8090, "control": 8091})
    assert "synthetic-vast-key" not in repr(setup)


# trial

TRIAL_STAND_IN = """#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["SSH_ARGS"], "w") as file:
    json.dump(sys.argv[1:], file)
with open(os.environ["SSH_ANSWER"], "rb") as file:
    sys.stdout.buffer.write(file.read())
sys.stdout.flush()
if os.environ.get("SSH_HANG"):
    time.sleep(600)
sys.exit(int(os.environ.get("SSH_EXIT", "0")))
"""
INSTANCE, KEY = "31415926", "synthetic-container-key"


@pytest.fixture
def card_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The file whose bytes a stand-in for ssh prints as the card's answer; it keeps its arguments in args.json."""
    ssh = tmp_path / "ssh"
    ssh.write_text(TRIAL_STAND_IN)
    ssh.chmod(0o700)
    monkeypatch.setattr(cli, "SSH", (str(ssh), *cli.SSH[1:]))
    monkeypatch.setenv("SSH_ARGS", str(tmp_path / "args.json"))
    monkeypatch.setenv("SSH_ANSWER", str(tmp_path / "answer"))
    return tmp_path / "answer"


def test_trial_writes_the_cards_id_and_key_beside_the_keys_and_prints_neither(
        tmp_path: Path, card_answer: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "simple-serving" / "config.json"
    assert cli.main(["--config", str(path), "keys"]) == 0
    before = json.loads(path.read_text())
    card_answer.write_bytes(f"{INSTANCE}\n{KEY}\n".encode())
    capsys.readouterr()
    assert cli.main(["--config", str(path), "--ssh-host", "card", "trial"]) == 0
    printed = capsys.readouterr()
    assert not any(value in printed.out + printed.err for value in (INSTANCE, KEY))
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[-2:] == ["card", cli.READ_TRIAL] and {"BatchMode=yes", "StrictHostKeyChecking=yes"} <= set(args)
    assert json.loads(path.read_text()) == before | {"instance_id": INSTANCE, "vast_api_key": KEY, "ssh_host": "card"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600 and KEY not in repr(cli.load(path))
    for argv in (["trial"], ["--ssh-host", "card", "status"]):  # the host goes with trial alone
        with pytest.raises(SystemExit):
            cli.main(["--config", str(path), *argv])


@pytest.mark.parametrize(("host", "said", "env", "words"), [
    ("-oProxyCommand=x", b"", {}, "SSH host has the wrong form"),
    ("card", f"{INSTANCE}\n".encode(), {"SSH_EXIT": "3"}, "ssh ended with 3"),  # the key's file is missing
    ("card", b"", {"SSH_EXIT": "255"}, "ssh ended with 255"),
    ("card", f"{INSTANCE}\n{KEY}\n".encode(), {"SSH_HANG": "1"}, "within 0.5 seconds"),
    ("card", f"{INSTANCE}\n".encode(), {}, "wrong form"),
    ("card", f"3141/926\n{KEY}\n".encode(), {}, "wrong form"),
    ("card", f"{INSTANCE}\n\n".encode(), {}, "wrong form"),
    ("card", f"{INSTANCE}\n{KEY}\r\n".encode(), {}, "wrong form"),
    ("card", f"{INSTANCE}\n{KEY}\nmore".encode(), {}, "wrong form"),  # a third line
    ("card", f"\xff\n{KEY}\n".encode("latin-1"), {}, "wrong form"),
    ("card", f"{INSTANCE}\n{KEY * 200}\n".encode(), {}, "wrong form"),  # past TRIAL_BYTES
])
def test_trial_refuses_a_wrong_host_ssh_or_answer_and_leaves_the_configuration_alone(
        tmp_path: Path, card_answer: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
        host: str, said: bytes, env: dict[str, str], words: str) -> None:
    monkeypatch.setattr(cli, "CONNECT_S", 0.5)
    monkeypatch.setattr(cli, "CLOSE_S", 0.1)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"client_key": "synthetic-client-key"}))
    path.chmod(0o600)
    card_answer.write_bytes(said)
    assert cli.main(["--config", str(path), f"--ssh-host={host}", "trial"]) == 1
    printed = capsys.readouterr()
    assert words in printed.err and KEY not in printed.out + printed.err
    assert json.loads(path.read_text()) == {"client_key": "synthetic-client-key"}
    assert (tmp_path / "args.json").exists() == (host == "card")  # a host of the wrong form never reaches ssh


def test_the_trial_read_prints_the_files_of_onstart_one_line_apiece(tmp_path: Path) -> None:
    command = cli.READ_TRIAL.replace("/root/", f"{tmp_path}/")
    (tmp_path / ".simple-chat-instance-id").write_text(INSTANCE)  # without a newline, as card/onstart.sh writes it
    run = subprocess.run(["bash", "-c", command], capture_output=True, check=False)
    assert (run.returncode, run.stdout) == (3, f"{INSTANCE}\n".encode())
    (tmp_path / ".simple-chat-instance-api-key").write_text(KEY)
    run = subprocess.run(["bash", "-c", command], capture_output=True, check=False)
    assert (run.returncode, run.stdout) == (0, f"{INSTANCE}\n{KEY}\n".encode())


# up

@pytest.mark.anyio
async def test_up_resumes_once_runs_alone_opens_a_lost_tunnel_again_and_ctrl_c_sends_nothing(tmp_path: Path,
                                                                                             clock: Clock) -> None:
    async with gateway() as (port, heads):
        instance, ssh = Instance("stopping", "stopping", "stopping", "stopped"), Ssh(port)
        setup = setup_for(tmp_path, instance, ssh)
        task = asyncio.create_task(cli.up(setup))
        await until(ssh.holding)
        assert instance.resumes == 1 and clock.time >= 3 * cli.POLL_S
        with pytest.raises(cli.Refusal, match="another up runs"):
            await cli.up(setup)
        ssh.tunnels[0].end.set_result(cli.SSH_FAILED)  # the connection drops
        await until(lambda: ssh.holding(2))
        task.cancel()  # as asyncio.run does on Ctrl+C
        with pytest.raises(asyncio.CancelledError):
            await task
    assert instance.resumes == 1 and len(ssh.opened) == 2 and all(tunnel.closed for tunnel in ssh.tunnels)
    assert all(head.startswith(b"GET /v1/state ") for head in heads)  # it read the state, and sent nothing
    with cli.up_lock(setup.path):  # which up let go of
        pass


@pytest.mark.anyio
async def test_up_never_resumes_against_a_stop_that_does_not_end(tmp_path: Path, clock: Clock) -> None:
    instance, ssh = Instance("stopping"), Ssh()
    with pytest.raises(cli.Refusal, match="the stop in flight to end; Vast says stopping"):
        await cli.up(setup_for(tmp_path, instance, ssh))
    assert (instance.resumes, ssh.opened) == (0, []) and clock.time >= cli.STOP_WAIT_S


@pytest.mark.anyio
async def test_up_refuses_a_busy_local_port_and_leaves_its_listener_alone(tmp_path: Path, clock: Clock) -> None:
    instance, ssh = Instance("running"), Ssh()
    setup = setup_for(tmp_path, instance, ssh)
    async with gateway(port=setup.local["control"]) as (_, heads):
        with pytest.raises(cli.Refusal, match=f"local port {setup.local['control']} is taken"):
            await cli.up(setup)
        assert (heads, instance.reads, ssh.opened) == ([], 0, [])
        assert await cli.call(setup.control_url(), CONTROL, "GET", "/v1/state") == VIEW  # it still answers


@pytest.mark.anyio
@pytest.mark.parametrize(("refuse", "end", "words"), [
    ((1,), (), "not prepared"),  # the card's shell found no checkout or no venv
    ((card.NOT_PREPARED,), (), "not prepared"),
    ((card.GAVE_UP,), (), "given up"),  # as at a resume after a failed load
    ((), (card.GAVE_UP,), "given up"),  # a load that fails while up waits
    ((card.STOP_UNCONFIRMED,), (), "stop is not confirmed"),
    ((), (card.STOP_UNCONFIRMED,), "stop is not confirmed"),  # a stop that Vast refuses while up holds
    ((), (0,) * (cli.RECONNECTS + 1), "keeps ending"),  # --hold finds no pair to hold, and up never resumes
])
async def test_up_says_why_the_card_does_not_serve(tmp_path: Path, clock: Clock, refuse: tuple[int, ...],
                                                   end: tuple[int, ...], words: str) -> None:
    instance, ssh = Instance("running"), Ssh(refuse=refuse, end=end)
    with pytest.raises(cli.Refusal, match=words):
        await cli.up(setup_for(tmp_path, instance, ssh))
    assert len(ssh.opened) == len(refuse + end)  # a refusal comes at once, and tunnels that end only so often


# sleep and status

@pytest.mark.anyio
async def test_sleep_during_up_drains_through_a_forward_of_its_own_and_up_ends_with_the_stop(tmp_path: Path,
                                                                                            clock: Clock) -> None:
    async with running() as stack:
        instance, ssh = Instance("running", card_stop=stack.vast), Ssh(port_of(stack.control))
        setup = setup_for(tmp_path, instance, ssh)
        holding = asyncio.create_task(cli.up(setup))
        await until(ssh.holding)
        assert await cli.sleep(setup) == 0
        ((local, remote),) = ssh.opened[1].items()  # the control listener alone, on a port of its own
        assert remote == 8091 and local not in setup.local.values() and ssh.tunnels[1].closed
        assert stack.service.sleep_requested and stack.vast.stopped and instance.last == "stopped"
        ssh.tunnels[0].end.set_result(cli.SSH_FAILED)  # the stop takes up's connection down
        assert await holding == 0 and instance.resumes == 0


@pytest.mark.anyio
async def test_sleep_stops_nothing_when_the_gateway_cannot_be_reached(tmp_path: Path, clock: Clock) -> None:
    instance, ssh = Instance("running"), Ssh(refuse=(cli.SSH_FAILED,))
    with pytest.raises(cli.Refusal, match="never stops the card itself"):
        await cli.sleep(setup_for(tmp_path, instance, ssh))
    assert (instance.reads, instance.resumes) == (1, 0)  # and neither the command nor its Vast can stop


@pytest.mark.anyio
@pytest.mark.parametrize(("vast", "refuse", "answer", "line"), [
    ("running", (), {}, f"ready; contract 2, model {ALIAS}; active {{'reader': 1}}, waiting none"),
    ("running", (), {"body": VIEW | {"model": CONTROL, "sleep_requested": True}},
     "ready, falling asleep; another contract or model; active {'reader': 1}, waiting none"),
    ("running", (), {"status": 401, "body": {"error": {"code": CONTROL}}}, "401 error"),  # the key, echoed
    ("running", (), {"body": VIEW | {"status": CONTROL}}, "200 malformed"),
    ("running", (), {"body": VIEW | {"active": [CONTROL]}}, "200 malformed"),
    ("running", (), {"body": VIEW | {"padding": "x" * cli.ANSWER_BYTES}}, "no answer"),
    ("running", (), {"drip_s": 0.01}, "no answer"),  # a byte every 10 ms: only the bound of the whole call ends it
    ("running", (cli.SSH_FAILED,), {}, "ssh ended with 255"),
    ("running", (card.GAVE_UP,), {}, cli.GIVEN_UP),
    ("running", (card.STOP_UNCONFIRMED,), {}, cli.UNCONFIRMED),
    ("stopped", (), {}, None),
])
async def test_status_tells_vast_apart_from_the_gateway_and_prints_nothing_as_it_came(
        tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], vast: str,
        refuse: tuple[int, ...], answer: dict[str, Any], line: str | None) -> None:
    monkeypatch.setattr(cli, "CALL_S", 0.2)
    async with gateway(**answer) as (port, _):
        ssh = Ssh(port, refuse=refuse)
        setup = setup_for(tmp_path, Instance(vast), ssh)
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{setup.local['control']}")
        # Another process owns up's control port while an up holds the lock, and it is a proxy as well.
        async with gateway(port=setup.local["control"]) as (_, strangers):
            with cli.up_lock(setup.path):
                assert await cli.status(setup) == 0
    printed = capsys.readouterr().out
    assert printed.splitlines() == [f"vast: {vast}", *([f"gateway: {line}"] if line else [])]
    assert CONTROL not in printed and strangers == [] and len(ssh.opened) == (vast == "running")


# the real tunnel, with a stand-in for ssh

STAND_IN = """#!/usr/bin/env python3
import json, os, signal, sys, time
with open(os.environ["SSH_ARGS"], "w") as file:
    json.dump(sys.argv[1:], file)
print("a line of the remote shell's own", flush=True)
if os.environ.get("SSH_EXIT"):
    sys.exit(int(os.environ["SSH_EXIT"]))
if os.environ.get("SSH_IGNORE_TERM"):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(os.environ["SSH_LINE"], flush=True)
time.sleep(600)
"""


@pytest.mark.anyio
@pytest.mark.parametrize(("env", "ended"), [
    ({}, -signal.SIGTERM),
    ({"SSH_IGNORE_TERM": "1"}, -signal.SIGKILL),  # the command's own child, killed after the grace
])
async def test_the_tunnel_is_ssh_with_loopback_forwards_and_the_hold_command(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch,
                                                                            env: dict[str, str], ended: int) -> None:
    ssh = tmp_path / "ssh"
    ssh.write_text(STAND_IN)
    ssh.chmod(0o700)
    monkeypatch.setattr(cli, "SSH", (str(ssh), *cli.SSH[1:]))
    monkeypatch.setattr(cli, "CLOSE_S", 0.1)
    for name, value in {"SSH_ARGS": str(tmp_path / "args.json"), "SSH_LINE": card.HOLDING, **env}.items():
        monkeypatch.setenv(name, value)
    tunnel = await cli.Tunnel.open("card", {18080: 8090, 18081: 8091})
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[-2:] == ["card", cli.HOLD] and tunnel.returncode is None
    assert {"127.0.0.1:18080:127.0.0.1:8090", "127.0.0.1:18081:127.0.0.1:8091", "BatchMode=yes",
            "ExitOnForwardFailure=yes", "StrictHostKeyChecking=yes"} <= set(args)
    assert await asyncio.wait_for(tunnel.close(), 5) == ended  # a close that never ends fails here
    monkeypatch.setenv("SSH_EXIT", "255")
    with pytest.raises(cli.NoTunnel) as caught:
        await cli.Tunnel.open("card", {18081: 8091})
    assert caught.value.code == 255

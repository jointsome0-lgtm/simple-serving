"""The command on the owner's machine (contract section 8), against a fake Vast and a fake SSH that forwards to the
gateway in front of the fake engine. The command's clock moves only when the command waits, so no test waits its
minutes. Keys and hosts are synthetic."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import signal
import stat
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from simple_serving import card, cli
from tests.support import ALIAS, CONTROL, FakeVast, Stack, running, until


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
    """ssh as the command starts it. A tunnel forwards its local ports to the gateway's listeners, standing for the
    card's, until the test ends it or the command closes it. `refuse` holds exit codes of the next connections that
    end before the remote command's first line, `end` those that end right after it."""

    def __init__(self, stack: Stack | None = None, *, refuse: tuple[int, ...] = (), end: tuple[int, ...] = ()) -> None:
        self.targets = {} if stack is None else {8090: port_of(stack.public), 8091: port_of(stack.control)}
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
            tunnel = FakeTunnel([await asyncio.start_server(partial(forward, self.targets[remote]), "127.0.0.1", local)
                                 for local, remote in forwards.items()])
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
    path.chmod(0o644)
    assert cli.main(["--config", str(path), "keys"]) == 1
    assert "chmod 600" in capsys.readouterr().err


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
    assert (setup.alias, setup.card_ports) == ("gemma-4-31b-heretic-q6k", {"public": 8090, "control": 8091})
    assert "synthetic-vast-key" not in repr(setup)


# up

@pytest.mark.anyio
async def test_up_waits_out_a_stop_in_flight_and_resumes_once(tmp_path: Path, clock: Clock) -> None:
    async with running() as stack:
        instance, ssh = Instance("stopping", "stopping", "stopping", "stopped"), Ssh(stack)
        task = asyncio.create_task(cli.up(setup_for(tmp_path, instance, ssh)))
        await until(ssh.holding)
        assert instance.resumes == 1 and clock.time >= 3 * cli.POLL_S
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


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
    other = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", setup.local["control"])
    try:
        with pytest.raises(cli.Refusal, match=f"local port {setup.local['control']} is taken"):
            await cli.up(setup)
        assert other.is_serving() and (instance.reads, ssh.opened) == (0, [])
        _, writer = await asyncio.open_connection("127.0.0.1", setup.local["control"])
        writer.close()
    finally:
        other.close()


@pytest.mark.anyio
async def test_ctrl_c_closes_the_tunnel_and_sends_nothing(tmp_path: Path, clock: Clock) -> None:
    async with running() as stack:
        instance, ssh = Instance("running"), Ssh(stack)
        task = asyncio.create_task(cli.up(setup_for(tmp_path, instance, ssh)))
        await until(ssh.holding)
        task.cancel()  # as asyncio.run does on Ctrl+C
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ssh.tunnels[0].closed and len(ssh.opened) == 1 and instance.resumes == 0
        assert (stack.service.status, stack.service.sleep_requested, stack.service.drain_generation) == ("ready",
                                                                                                        False, 0)
        assert stack.vast.attempts == [] and not cli.up_runs(tmp_path / "config.json")


@pytest.mark.anyio
async def test_a_lost_tunnel_is_opened_again_without_a_resume_until_the_card_stops(tmp_path: Path,
                                                                                   clock: Clock) -> None:
    async with running() as stack:
        instance, ssh = Instance("running"), Ssh(stack)
        task = asyncio.create_task(cli.up(setup_for(tmp_path, instance, ssh)))
        await until(ssh.holding)
        ssh.tunnels[0].end.set_result(cli.SSH_FAILED)  # the connection drops
        await until(lambda: ssh.holding(2))
        instance.states = ["stopped"]  # the card fell asleep, and its stop dropped the connection
        ssh.tunnels[1].end.set_result(cli.SSH_FAILED)
        assert await task == 0
        assert instance.resumes == 0 and len(ssh.opened) == 2 and all(tunnel.closed for tunnel in ssh.tunnels)


@pytest.mark.anyio
async def test_tunnels_that_keep_ending_before_ready_are_given_up(tmp_path: Path, clock: Clock) -> None:
    instance, ssh = Instance("running"), Ssh(end=(0,) * 10)  # --hold finds no pair to hold
    with pytest.raises(cli.Refusal, match="keeps ending"):
        await cli.up(setup_for(tmp_path, instance, ssh))
    assert len(ssh.opened) == cli.RECONNECTS + 1 and instance.resumes == 0


@pytest.mark.anyio
@pytest.mark.parametrize(("refuse", "end", "words"), [
    ((1,), (), "not prepared"),  # the card's shell found no checkout or no venv
    ((), (card.NOT_PREPARED,), "not prepared"),
    ((), (card.GAVE_UP,), "gave up"),
])
async def test_up_says_why_the_card_does_not_serve(tmp_path: Path, clock: Clock, refuse: tuple[int, ...],
                                                   end: tuple[int, ...], words: str) -> None:
    instance, ssh = Instance("running"), Ssh(refuse=refuse, end=end)
    with pytest.raises(cli.Refusal, match=words):
        await cli.up(setup_for(tmp_path, instance, ssh))
    assert len(ssh.opened) == 1


# sleep and status

@pytest.mark.anyio
async def test_sleep_without_a_tunnel_drains_through_a_forward_of_its_own(tmp_path: Path, clock: Clock) -> None:
    async with running() as stack:
        instance, ssh = Instance("running", card_stop=stack.vast), Ssh(stack)
        setup = setup_for(tmp_path, instance, ssh)
        assert await cli.sleep(setup) == 0
        ((local, remote),) = ssh.opened[0].items()  # the control listener alone, on a port of its own
        assert remote == 8091 and local not in setup.local.values() and ssh.tunnels[0].closed
        assert stack.service.sleep_requested and stack.vast.stopped and instance.last == "stopped"


@pytest.mark.anyio
async def test_sleep_uses_the_tunnel_that_up_holds(tmp_path: Path, clock: Clock) -> None:
    async with running() as stack:
        instance, ssh = Instance("running", card_stop=stack.vast), Ssh(stack)
        setup = setup_for(tmp_path, instance, ssh)
        holding = asyncio.create_task(cli.up(setup))
        await until(ssh.holding)
        assert await cli.sleep(setup) == 0
        assert len(ssh.opened) == 1 and stack.vast.stopped
        ssh.tunnels[0].end.set_result(cli.SSH_FAILED)  # the stop takes the connection down
        assert await holding == 0 and instance.resumes == 0


@pytest.mark.anyio
async def test_sleep_stops_nothing_when_the_gateway_cannot_be_reached(tmp_path: Path, clock: Clock) -> None:
    instance, ssh = Instance("running"), Ssh(refuse=(cli.SSH_FAILED,))
    with pytest.raises(cli.Refusal, match="so nothing stops"):
        await cli.sleep(setup_for(tmp_path, instance, ssh))
    assert (instance.reads, instance.resumes) == (1, 0)  # and neither the command nor its Vast can stop


@pytest.mark.anyio
async def test_status_tells_vast_apart_from_the_gateway(tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    proxied: list[bool] = []

    def trap(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        proxied.append(True)
        writer.close()

    proxy = await asyncio.start_server(trap, "127.0.0.1", 0)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.sockets[0].getsockname()[1]}")
    async with running() as stack:
        assert await cli.status(setup_for(tmp_path, Instance("running"), Ssh(stack))) == 0
        assert capsys.readouterr().out.splitlines() == [
            "vast: running", f"gateway: ready; contract 2, model {ALIAS}; active none, waiting none"]
    proxy.close()
    assert proxied == []  # the control key goes to loopback only, never to a proxy
    await cli.status(setup_for(tmp_path, Instance("running"), Ssh(refuse=(cli.SSH_FAILED,))))
    assert capsys.readouterr().out.splitlines() == ["vast: running", "gateway: ssh ended with 255"]
    ssh = Ssh()
    await cli.status(setup_for(tmp_path, Instance("stopped"), ssh))
    assert capsys.readouterr().out == "vast: stopped\n" and ssh.opened == []


# the real tunnel, with a stand-in for ssh

STAND_IN = """#!/usr/bin/env python3
import json, os, sys, time
with open(os.environ["SSH_ARGS"], "w") as file:
    json.dump(sys.argv[1:], file)
print("a line of the remote shell's own", flush=True)
if os.environ.get("SSH_EXIT"):
    sys.exit(int(os.environ["SSH_EXIT"]))
print(os.environ["SSH_LINE"], flush=True)
time.sleep(600)
"""


@pytest.mark.anyio
async def test_the_tunnel_is_ssh_with_loopback_forwards_and_the_hold_command(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    ssh = tmp_path / "ssh"
    ssh.write_text(STAND_IN)
    ssh.chmod(0o700)
    monkeypatch.setattr(cli, "SSH", (str(ssh), *cli.SSH[1:]))
    monkeypatch.setenv("SSH_ARGS", str(tmp_path / "args.json"))
    monkeypatch.setenv("SSH_LINE", card.HOLDING)
    tunnel = await cli.Tunnel.open("card", {18080: 8090, 18081: 8091})
    args = json.loads((tmp_path / "args.json").read_text())
    assert args[-2:] == ["card", cli.HOLD] and tunnel.returncode is None
    assert {"127.0.0.1:18080:127.0.0.1:8090", "127.0.0.1:18081:127.0.0.1:8091", "BatchMode=yes",
            "ExitOnForwardFailure=yes", "StrictHostKeyChecking=yes"} <= set(args)
    assert await tunnel.close() == -signal.SIGTERM
    monkeypatch.setenv("SSH_EXIT", "255")
    with pytest.raises(cli.NoTunnel) as caught:
        await cli.Tunnel.open("card", {18081: 8091})
    assert caught.value.code == 255

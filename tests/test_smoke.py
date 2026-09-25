"""The smoke of the first rental (simple_serving/smoke.py): the reading of a stream, the command line, the card's report
over SSH, and the dry run: the whole smoke through the dev launcher, as the README's "The smoke" describes it, with a
directory that stands for the card."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import io
import json
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving import card, cli, smoke
from simple_serving.fake_engine import SENTENCE, THOUGHT

from .support import ALIAS, BOT, CONTROL, ROOT, SERVICE, call
from .test_card import LAUNCHER, fake_process
from .test_dev import Launch, free_ports, launched

pytestmark = pytest.mark.anyio
MARKER = "0123456789abcdef" * 2


# The stream as the gateway sends it, event by event; its reading, and the rules that a changed event breaks.

def chunk(delta: dict[str, str], finish: str | None = None) -> dict[str, Any]:
    return {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": ALIAS,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


USAGE: dict[str, Any] = {
    "id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": ALIAS, "choices": [],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "prompt_tokens_details": {"cached_tokens": 0},
              "simple_serving": {"wait_ms": 1, "first_token_ms": 20, "total_ms": 30}}}
EVENTS: list[Any] = [chunk({"role": "assistant"}), chunk({"reasoning_content": "Calm."}), chunk({"content": "Lit."}),
                     chunk({}, "stop"), USAGE, "[DONE]"]


def sse(events: list[Any]) -> bytes:
    return b"".join(b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n"
                    for event in events)


async def read(events: list[Any], status: int = 200, content_type: str = "text/event-stream; charset=utf-8",
               raw: bytes | None = None) -> smoke.Stream:
    body = raw if raw is not None else sse(events)
    return await smoke.read_stream(httpx.Response(status, headers={"content-type": content_type}, content=body), ALIAS)


async def test_a_stream_by_the_rules_is_read_and_gives_numbers_only() -> None:
    stream = await read(EVENTS)
    assert stream.ok, stream.broken
    assert (stream.content, stream.reasoning, stream.finish) == ("Lit.", "Calm.", "stop")
    assert stream.numbers() == {"chunks": 5, "content_chars": 4, "reasoning_chars": 5, "prompt_tokens": 12,
                                "completion_tokens": 3, "wait_ms": 1, "first_token_ms": 20, "total_ms": 30,
                                "cached_tokens": 0}
    no_cache = copy.deepcopy(USAGE)
    del no_cache["usage"]["prompt_tokens_details"]
    assert "cached_tokens" not in (await read([*EVENTS[:4], no_cache, "[DONE]"])).numbers()


def other_model(event: dict[str, Any]) -> dict[str, Any]:
    return {**event, "model": "another-model"}


@pytest.mark.parametrize(("events", "rule"), [
    ([other_model(EVENTS[0]), *EVENTS[1:]], "model"),
    ([*EVENTS[:4], other_model(USAGE), "[DONE]"], "model"),
    ([{**EVENTS[0], "object": "chat.completion"}, *EVENTS[1:]], "object"),
    ([{**EVENTS[0], "choices": EVENTS[0]["choices"] * 2}, *EVENTS[1:]], "choices"),
    ([chunk({"role": "assistant", "content": ""}), *EVENTS[1:]], "delta"),
    ([chunk({"tool_calls": "x"}), *EVENTS[1:]], "delta"),
    ([*EVENTS[:4], chunk({"content": "More."}), USAGE, "[DONE]"], "after_finish"),
    ([*EVENTS[:4], chunk({}, "stop"), USAGE, "[DONE]"], "after_finish"),
    ([*EVENTS[:3], chunk({}, "abort"), USAGE, "[DONE]"], "finish"),
    ([*EVENTS[:3], USAGE, "[DONE]"], "usage_chunk"),
    ([*EVENTS[:3], USAGE, "[DONE]"], "finish"),
    ([*EVENTS[:4], "[DONE]"], "usage"),
    ([*EVENTS[:4], {**USAGE, "choices": EVENTS[0]["choices"]}, "[DONE]"], "usage_chunk"),
    ([*EVENTS[:4], {**USAGE, "usage": {**USAGE["usage"], "total_tokens": 15}}, "[DONE]"], "usage_fields"),
    ([*EVENTS[:4], {**USAGE, "usage": {**USAGE["usage"], "prompt_tokens": "12"}}, "[DONE]"], "usage_fields"),
    ([*EVENTS[:4], USAGE, USAGE, "[DONE]"], "after_usage"),
    (EVENTS[:5], "done"),
    ([*EVENTS, chunk({"content": "Late."})], "after_end"),
])
async def test_a_stream_that_breaks_a_rule_fails_with_its_name(events: list[Any], rule: str) -> None:
    stream = await read(events)
    assert not stream.ok
    assert rule in stream.broken


async def test_what_is_not_an_event_breaks_the_stream() -> None:
    assert "event_form" in (await read([], raw=b"data: [DONE]\n\nevent: x\n\n")).broken
    assert "event_form" in (await read([], raw=b'data: {"half":')).broken
    assert "json" in (await read([], raw=b"data: {not json}\n\ndata: [DONE]\n\n")).broken
    assert "content_type" in (await read(EVENTS, content_type="application/json")).broken


async def test_an_error_event_or_a_refusal_gives_the_code_and_nothing_else() -> None:
    stream = await read([*EVENTS[:3], {"error": {"code": "draining"}}])
    assert (stream.ok, stream.code, stream.broken) == (False, "draining", [])
    refusal = await read([], status=400, content_type="application/json",
                         raw=json.dumps({"error": {"code": "invalid_request"}}).encode())
    assert (refusal.ok, refusal.status, refusal.code) == (False, 400, "invalid_request")
    # A code outside section 9, or an error of another shape, is never printed as it came.
    odd = await read([], status=400, content_type="application/json",
                     raw=json.dumps({"error": {"code": "the prompt was: keeper"}}).encode())
    assert odd.code == "error"
    assert (await read([*EVENTS[:3], {"error": "quoted text"}])).code == "error"


def test_an_answers_numbers_are_read_exactly() -> None:
    schema = next(entry.schema for entry in smoke.bot_schemas.load() if entry.name == "walk_cross")
    answer = ('{"checks": [{"finding": FINDING, "confirmed": true, "note": "no reason"}, '
              '{"finding": 2, "confirmed": false, "note": "he is there"}]}')
    for finding, problems in (("1", []), ("1.0", []), ("1e0", []), ("1.0000000000000001", ["type"]),
                              ("1e400", ["maximum"]), ("NaN", ["json"]), ("Infinity", ["json"])):
        assert smoke.answer_problems(answer.replace("FINDING", finding), schema) == problems, finding
    # A float would round 1.0000000000000001 to 1.0, an integer.
    assert json.loads(answer.replace("FINDING", "1.0000000000000001"))["checks"][0]["finding"] == 1.0


# The command line.

def keys_file(tmp_path: Path, **keys: str) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(keys))
    path.chmod(0o600)
    return path


def run_smoke(*args: str) -> tuple[int, str, str]:
    """The smoke's main in this process, with its exit code and what it printed."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = smoke.main(list(args))
        except SystemExit as exit_:
            code = exit_.code if isinstance(exit_.code, int) else 2
    return code, out.getvalue(), err.getvalue()


def test_only_names_known_probes_and_keeps_their_order() -> None:
    assert smoke.probe_names("counts,state") == ["state", "counts"]
    for names in ("state,nope", "lifecycle"):  # the lifecycle comes with --before or --after alone
        code, out, err = run_smoke("--only", names)
        assert (code, out) == (2, "")
        assert "no probe" in err


def test_the_smoke_needs_both_keys_and_prints_neither(tmp_path: Path) -> None:
    code, out, err = run_smoke("--config", str(keys_file(tmp_path, client_key=BOT)), "--public-port", "1",
                               "--control-port", "2")
    assert (code, out) == (2, "")
    assert "has no control_key" in err
    assert BOT not in err
    path = keys_file(tmp_path, client_key=BOT, control_key=CONTROL)
    path.chmod(0o644)
    code, _, err = run_smoke("--config", str(path), "--public-port", "1", "--control-port", "2")
    assert code == 2
    assert "chmod 600" in err


def test_fake_is_never_run_against_the_forward_of_up(tmp_path: Path) -> None:
    config = str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL))
    for ports in (("--public-port", "8080", "--control-port", "18081"), ("--control-port", "8081")):
        code, out, err = run_smoke("--config", config, "--fake", *ports)
        assert (code, out) == (2, "")
        assert "--fake is for the dev launcher" in err


@pytest.mark.parametrize(("keys", "args", "said"), [
    ({}, (), "has no ssh_host of the right form, and abort, privacy read the card's files over SSH"),
    ({"ssh_host": "-oProxyCommand=sh"}, ("--only", "abort"), "has no ssh_host of the right form, and abort read"),
    ({"ssh_host": "card"}, ("--card-dir", "card"), "--card-dir stands for the card in the dry run, with --fake alone"),
    ({"ssh_host": "card"}, ("--fake", "--only", "state,privacy"), "--fake reads the card's files from --card-dir"),
    ({}, ("--only", "state", "--after", "none.json"), "has no ssh_host of the right form, and lifecycle read"),
])
def test_the_card_is_read_over_ssh_or_in_the_dry_run_from_card_dir_alone(
        tmp_path: Path, keys: dict[str, str], args: tuple[str, ...], said: str) -> None:
    config = str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL, **keys))
    code, out, err = run_smoke("--config", config, "--public-port", "1", "--control-port", "2", *args)
    assert (code, out) == (2, "")
    assert said in err


def test_before_writes_a_new_file_and_after_reads_one_that_before_wrote(tmp_path: Path) -> None:
    config = str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL, ssh_host="card"))
    written = tmp_path / "lifecycle.json"
    written.write_text(json.dumps({"boot": "0" * 64}))
    for args, said in ((("--before", str(written)), "exists, and --before writes a new file"),
                       (("--after", str(written)), "is not a file of --before"),
                       (("--after", str(tmp_path / "none.json")), "cannot read"),
                       (("--before", "a", "--after", "b"), "not allowed with argument")):
        code, out, err = run_smoke("--config", config, "--public-port", "1", "--control-port", "2", "--only", "state",
                                   *args)
        assert (code, out) == (2, "")
        assert said in err


def test_nothing_answers_and_the_first_probe_that_fails_ends_the_smoke(tmp_path: Path) -> None:
    public, control = free_ports(2)
    code, out, err = run_smoke("--config", str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL)),
                               "--public-port", str(public), "--control-port", str(control), "--only",
                               "state,completion")
    assert code == 1
    assert [json.loads(line) for line in out.splitlines()] == [{"probe": "state", "ok": False, "failed": ["no_answer"]}]
    assert err == "simple-serving smoke: 0 of 2 probes passed; state failed, and the smoke stopped there\n"


async def test_a_probe_that_breaks_or_outlasts_its_bound_fails_and_prints_no_text(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    async def breaks(_: smoke.Smoke) -> dict[str, Any]:
        raise ValueError(smoke.SHORT)

    async def hangs(_: smoke.Smoke) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {"ok": True}

    monkeypatch.setitem(smoke.PROBES, "state", breaks)
    monkeypatch.setitem(smoke.PROBES, "completion", hangs)
    monkeypatch.setitem(smoke.BOUND_S, "completion", 0.05)
    async with httpx.AsyncClient() as client:
        probe = smoke.Smoke(client, "", "", BOT, CONTROL, ALIAS, 4096)
        codes = [await smoke.run(probe, ["state", "completion"]), await smoke.run(probe, ["completion", "state"])]
    out, err = capsys.readouterr()
    assert codes == [1, 1]
    assert [json.loads(line) for line in out.splitlines()] == [
        {"probe": "state", "ok": False, "failed": ["smoke_error"], "error": "ValueError"},
        {"probe": "completion", "ok": False, "failed": ["time_bound"], "bound_s": 0.05}]
    assert err == ("simple-serving smoke: 0 of 2 probes passed; state failed, and the smoke stopped there\n"
                   "simple-serving smoke: 0 of 2 probes passed; completion failed, and the smoke stopped there\n")
    assert smoke.SHORT not in out


@pytest.mark.parametrize(("ends", "undo", "line"), [
    ("passes", "confirmed", {"ok": True}),
    ("passes", "unconfirmed", {"ok": False, "failed": ["left_drained"]}),
    ("late", "slow", {"ok": True}),
    ("outlasts", "confirmed", {"ok": False, "failed": ["time_bound"], "bound_s": 0.5}),
    ("outlasts", "unconfirmed", {"ok": False, "failed": ["time_bound", "left_drained"], "bound_s": 0.5}),
    ("outlasts", "hangs", {"ok": False, "failed": ["time_bound", "left_drained"], "bound_s": 0.5}),
    ("outlasts", "breaks", {"ok": False, "failed": ["time_bound", "left_drained"], "bound_s": 0.5}),
])
async def test_an_undo_runs_once_its_probe_ends_within_the_time_the_bound_keeps_for_it(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], ends: str, undo: str,
        line: dict[str, Any]) -> None:
    monkeypatch.setitem(smoke.BOUND_S, "abort", 0.5)
    monkeypatch.setitem(smoke.UNDO_S, "abort", 0.25)
    loop = asyncio.get_running_loop()
    started, undone = loop.time(), []

    async def reopen() -> bool:
        undone.append(loop.time() - started)
        if undo == "hangs":
            await asyncio.sleep(10)
        if undo == "slow":
            await asyncio.sleep(0.1)  # longer than the probe's own time has left
        if undo == "breaks":
            raise smoke.NoAnswer
        return undo in ("confirmed", "slow")

    async def drains(probe: smoke.Smoke) -> dict[str, Any]:
        probe.undo = reopen
        if ends == "outlasts":
            await asyncio.sleep(10)  # the drain's answer never comes
        if ends == "late":
            await asyncio.sleep(0.2)  # it passes near the end of its own time
        return {"ok": True}

    async def passes(_: smoke.Smoke) -> dict[str, Any]:
        return {"ok": True}

    monkeypatch.setitem(smoke.PROBES, "abort", drains)
    monkeypatch.setitem(smoke.PROBES, "privacy", passes)
    async with httpx.AsyncClient() as client:
        code = await smoke.run(smoke.Smoke(client, "", "", BOT, CONTROL, ALIAS, 4096), ["abort", "privacy"])
    took = loop.time() - started
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0] == {"probe": "abort", **line}
    assert (code, lines[1:]) == ((0, [{"probe": "privacy", "ok": True}]) if line["ok"] else (1, []))
    assert len(undone) == 1
    assert took < 0.5 + 0.1  # the bound, the undo included
    if ends == "outlasts":
        assert undone[0] >= 0.25  # once the probe's own part of the bound had passed


# The card's report over SSH, as up reaches the card.

SSH_STAND_IN = """#!{python}
import json, os, sys
with open(os.environ["SSH_ARGS"], "w") as file:
    json.dump(sys.argv[1:], file)
if os.environ.get("SSH_PRINT"):
    print(os.environ["SSH_PRINT"], flush=True)
    sys.exit(int(os.environ.get("SSH_EXIT", "0")))
os.execv(sys.executable, [sys.executable, "-m", "simple_serving.card", "--inspect"])
"""


def card_dir(tmp_path: Path, deadline: int | None = None) -> Path:
    """A directory that stands for the card: its state's logs/, /root with the guard's deadline, and /proc with one
    launcher and its pair."""
    directory = tmp_path / "card"
    (directory / "logs").mkdir(parents=True)
    fake_process(directory / "proc", 100, 1, *LAUNCHER)
    fake_process(directory / "proc", 101, 100, "vllm", "serve")
    fake_process(directory / "proc", 102, 100, sys.executable, "-m", "simple_serving")
    (directory / card.DEADLINE).write_text(str(deadline or int(time.time()) + 3 * 3600))
    return directory


async def test_the_report_comes_over_ssh_with_up_s_options_and_host(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    ssh, directory = tmp_path / "ssh", card_dir(tmp_path)
    ssh.write_text(SSH_STAND_IN.format(python=sys.executable))
    ssh.chmod(0o700)
    (directory / "logs/gateway.jsonl").write_text('{"event": "request", "route": "/v1/chat/completions"}\n')
    monkeypatch.setattr(cli, "SSH", (str(ssh), *cli.SSH[1:]))
    places = {"DIR": directory, "ROOT": directory, "PROC": directory / "proc"}
    for name, value in {"SSH_ARGS": tmp_path / "args.json",
                        **{f"SIMPLE_SERVING_CARD_{name}": path for name, path in places.items()}}.items():
        monkeypatch.setenv(name, str(value))
    report = await smoke.CardFiles("card").inspect(marker=MARKER, since=0)
    assert json.loads((tmp_path / "args.json").read_text()) == [*cli.SSH[1:], "card", smoke.INSPECT]
    assert cli.HOLD.replace("--hold", "--inspect") == smoke.INSPECT
    assert (report["launchers"], report["children"], report["cancelled"]) == (1, 2, [False])
    assert report["logs"]["gateway.jsonl"] == {"rows": 1, "found": 0}
    for printed, exit_code in (("", "255"), ('{"launchers": 1}', "0"), ("not json", "0")):
        monkeypatch.setenv("SSH_PRINT", printed or " ")
        monkeypatch.setenv("SSH_EXIT", exit_code)
        with pytest.raises(smoke.NoReport):
            await smoke.CardFiles("card").inspect()
    with pytest.raises(smoke.NoReport):
        await smoke.CardFiles().inspect()  # neither a host nor a directory


# The privacy probe against a gateway that logs a body where it should not.

def gateway_that_logs(directory: Path, body_to: str | None, row: bool = True) -> httpx.MockTransport:
    """A gateway, as far as the privacy probe sees one: it answers with EVENTS, writes the request's row to the card's
    gateway.jsonl, and with `body_to` also the request's body to that log."""
    def handle(request: httpx.Request) -> httpx.Response:
        if row:
            with (directory / "logs/gateway.jsonl").open("a") as file:
                file.write(json.dumps({"event": "request", "route": smoke.CHAT, "cancelled": False}) + "\n")
        if body_to is not None:
            (directory / "logs" / body_to).parent.mkdir(parents=True, exist_ok=True)
            with (directory / "logs" / body_to).open("ab") as file:
                file.write(request.content + b"\n")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(EVENTS))

    return httpx.MockTransport(handle)


@pytest.mark.parametrize("body_to", [None, "gateway.jsonl", "card.jsonl.1", "vllm/output.log"])
async def test_the_privacy_probe_fails_on_its_marker_in_any_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                body_to: str | None) -> None:
    directory = card_dir(tmp_path)
    monkeypatch.setattr(smoke, "new_marker", lambda: MARKER)
    async with httpx.AsyncClient(transport=gateway_that_logs(directory, body_to)) as client:
        line = await smoke.probe_privacy(smoke.Smoke(client, "http://gateway", "http://gateway", BOT, CONTROL, ALIAS,
                                                     4096, card=smoke.CardFiles(directory=directory)))
    found = dict.fromkeys((*card.LOGS, "other"), 0)
    if body_to is not None:
        found[body_to if body_to in card.LOGS else "other"] = 1
    assert line == {"ok": body_to is None, "status": 200, **({"failed": ["marker"]} if body_to else {}),
                    "found": found, "logged": 1, "echoed": 0}
    assert MARKER not in json.dumps(line)


async def test_the_privacy_probe_fails_when_the_gateway_logs_no_row(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    directory = card_dir(tmp_path)
    monkeypatch.setattr(smoke, "ROWS_S", 0.5)
    async with httpx.AsyncClient(transport=gateway_that_logs(directory, None, row=False)) as client:
        line = await smoke.probe_privacy(smoke.Smoke(client, "http://gateway", "http://gateway", BOT, CONTROL, ALIAS,
                                                     4096, card=smoke.CardFiles(directory=directory)))
    assert (line["ok"], line["failed"], line["logged"]) == (False, ["unlogged"], 0)  # a silent log proves nothing


# The dry run: the dev launcher configured as the card's gateway is, whose rows go to a directory that stands for the
# card, and the whole smoke through it.

@pytest.fixture
def dry_run(tmp_path: Path) -> tuple[str, str, Path]:
    """The launcher's configuration, with the manifest's model, context and pins and the cases' two test keys; the
    smoke's configuration with those keys; and the directory that stands for the card."""
    manifest = card.read_manifest()
    keys = {key: SERVICE["keys"][key] for key in (BOT, CONTROL)}
    block = {"alias": manifest["MODEL_ALIAS"], "context_tokens": int(manifest["CONTEXT_TOKENS"]),
             "versions": card.pinned_versions(manifest), "keys": keys}
    launcher = tmp_path / "dev.json"
    launcher.write_text(json.dumps(block))
    return str(launcher), str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL)), card_dir(tmp_path)


async def smoke_process(config: str, launch: Launch, *options: str) -> tuple[int, list[Any], str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "simple_serving.smoke", "--config", config, "--public-port",
        launch.public.rsplit(":", 1)[1], "--control-port", launch.control.rsplit(":", 1)[1], *options, cwd=ROOT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(process.communicate(), 120)
    assert process.returncode is not None
    printed = out + err
    # Never a key, a prompt, the fake's answer or its thought.
    for secret in (BOT, CONTROL, smoke.SHORT, smoke.FILLER.strip(), smoke.COUNTING, smoke.ECHO, "keeper", *SENTENCE,
                   *THOUGHT):
        assert secret.encode() not in printed
    return process.returncode, [json.loads(line) for line in out.decode().splitlines()], err.decode()


async def test_the_dry_run_passes_every_probe_and_names_what_the_fake_cannot_verify(
        dry_run: tuple[str, str, Path]) -> None:
    launcher, config, directory = dry_run
    context = int(card.read_manifest()["CONTEXT_TOKENS"])
    # The pause after each event keeps a stream open while the abort probe reads the state.
    async with launched("--event-delay-ms", "40", config=launcher, rows=directory / "logs/gateway.jsonl") as launch:
        code, lines, err = await smoke_process(config, launch, "--fake", "--card-dir", str(directory))
        async with httpx.AsyncClient(timeout=10) as client:
            state = (await call(client, "GET", launch.control + "/v1/state", key=CONTROL)).json()
    assert code == 0, lines
    assert err == "simple-serving smoke: 10 of 10 probes passed\n"
    assert [line["probe"] for line in lines] == list(smoke.CHOSEN)
    assert all(line["ok"] for line in lines)
    by_probe = {line.pop("probe"): line for line in lines}
    assert {name: line["not_verifiable"] for name, line in by_probe.items() if "not_verifiable" in line} == {
        "fields": ["answer", "seed_repeats"], "schemas": ["answer"], "counts": ["equal"]}
    assert by_probe["state"]["versions"]["vllm"] == card.read_manifest()["VLLM_VERSION"]
    assert by_probe["reasoning"]["on"]["reasoning_chars"] > 0 == by_probe["reasoning"]["off"]["reasoning_chars"]
    assert by_probe["finish"]["length"]["completion_tokens"] == 8
    assert (by_probe["refusal"]["status"], by_probe["refusal"]["code"]) == (400, "invalid_request")
    assert (by_probe["abort"]["held"], by_probe["abort"]["cancelled"]) == (1, True)
    assert by_probe["schemas"]["passed"] == by_probe["schemas"]["of"] == 12
    cells = {name: part for name, part in by_probe["counts"].items() if isinstance(part, dict)}
    assert len(cells) == 7
    assert all(part["input_tokens"] == part["prompt_tokens"] for part in cells.values())
    near = cells["near_context"]
    assert near["input_tokens"] + near["max_tokens"] == context
    assert context - smoke.NEAR_BAND <= near["input_tokens"] < context
    assert by_probe["privacy"] == {"ok": True, "status": 200, "found": dict.fromkeys((*card.LOGS, "other"), 0),
                                   "logged": 1, "echoed": 0}
    # The abort probe's drain was undone, and nothing of the smoke is left.
    assert (state["status"], smoke.total(state["active"]), smoke.total(state["waiting"])) == ("ready", 0, 0)


async def test_without_fake_the_first_answer_that_fails_its_schema_ends_the_smoke(
        dry_run: tuple[str, str, Path]) -> None:
    launcher, config, _ = dry_run
    async with launched(config=launcher) as launch:
        fields_run = await smoke_process(config, launch, "--only", "fields,schemas")
        schemas_run = await smoke_process(config, launch, "--only", "schemas")
    code, lines, err = fields_run
    assert code == 1
    assert err == "simple-serving smoke: 0 of 2 probes passed; fields failed, and the smoke stopped there\n"
    (fields,) = lines
    names = list(smoke.FIELDS)
    reached = names[:names.index("response_format") + 1]  # the parts up to the first that failed
    assert [name for name, part in fields.items() if isinstance(part, dict)] == reached
    assert (fields["response_format"]["failed"], fields["response_format"]["answer"]) == (["answer"], ["json"])
    assert not fields["ok"] and "seed_repeats" not in fields and "not_verifiable" not in fields
    code, lines, _ = schemas_run
    (schemas,) = lines
    parts = [part for part in schemas.values() if isinstance(part, dict)]
    assert (code, schemas["passed"], schemas["of"], len(parts)) == (1, 0, 12, 1)
    assert (parts[0]["failed"], parts[0]["answer"]) == (["answer"], ["json"])


async def test_the_lifecycle_needs_a_new_boot_one_pair_and_the_guards_deadline_unchanged(
        dry_run: tuple[str, str, Path], tmp_path: Path) -> None:
    launcher, config, directory = dry_run
    record, rows, deadline = tmp_path / "lifecycle.json", directory / "logs/gateway.jsonl", int(time.time()) + 10800
    (directory / card.DEADLINE).write_text(str(deadline))
    options = ("--fake", "--card-dir", str(directory))
    async with launched(config=launcher, rows=rows) as first:
        before = await smoke_process(config, first, *options, "--only", "state,completion", "--before", str(record))
        same_boot = await smoke_process(config, first, *options, "--only", "state", "--after", str(record))
        async with httpx.AsyncClient(timeout=10) as client:
            boot = (await call(client, "GET", first.control + "/v1/state", key=CONTROL)).json()["boot_id"]
    async with launched(config=launcher, rows=rows) as second:  # the resume: a new boot
        after = await smoke_process(config, second, *options, "--only", "state", "--after", str(record))
        (directory / card.DEADLINE).write_text(str(deadline - 60))
        fake_process(directory / "proc", 200, 1, *LAUNCHER)  # a second launcher, with no pair
        moved = await smoke_process(config, second, *options, "--only", "state", "--after", str(record))

    def lifecycle(run: tuple[int, list[Any], str]) -> dict[str, Any]:
        line: dict[str, Any] = run[1][0]
        assert line.pop("probe") == "lifecycle"
        assert 10800 - 120 < line.pop("deadline_left_s") <= 10800
        return line

    assert (before[0], before[2]) == (0, "simple-serving smoke: 3 of 3 probes passed\n")
    assert lifecycle(before) == {"ok": True, "launchers": 1, "children": 2, "recorded": True}
    assert json.loads(record.read_text()) == {"boot": smoke.digest(boot), "deadline": deadline}
    assert record.stat().st_mode & 0o777 == 0o600
    assert (after[0], after[2]) == (0, "simple-serving smoke: 2 of 2 probes passed\n")
    assert lifecycle(after) == {"ok": True, "launchers": 1, "children": 2, "boot_new": True,
                                "deadline_unchanged": True}
    for run, failed, launchers, boot_new, unchanged in ((same_boot, ["boot"], 1, False, True),
                                                        (moved, ["pair", "deadline"], 2, True, False)):
        assert (run[0], len(run[1]), run[2]) == (
            1, 1, "simple-serving smoke: 0 of 2 probes passed; lifecycle failed, and the smoke stopped there\n")
        assert lifecycle(run) == {"ok": False, "failed": failed, "launchers": launchers, "children": 2,
                                  "boot_new": boot_new, "deadline_unchanged": unchanged}


# The abort probe's two ways to pass without a cancel, against the fake: neither passes.

@asynccontextmanager
async def relay_that_hides_the_leave(url: str) -> AsyncIterator[str]:
    """A relay in front of the public listener that goes on reading the gateway's answer after its client has left,
    so that the gateway never learns of the leave: to the smoke, a gateway that does not cancel."""
    port, upstreams, readers = int(url.rsplit(":", 1)[1]), [], []

    async def relay(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        upstreams.append(writer)

        async def down() -> None:
            while data := await reader.read(65536):
                if not client_writer.is_closing():
                    client_writer.write(data)

        readers.append(asyncio.ensure_future(down()))
        with contextlib.suppress(ConnectionError):
            while data := await client_reader.read(65536):
                writer.write(data)
        client_writer.close()  # the client left; the connection to the gateway stays open

    server = await asyncio.start_server(relay, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
    finally:
        for task in readers:
            task.cancel()
        for writer in upstreams:
            writer.close()
        server.close()


class LateLook(smoke.Smoke):
    """A smoke that looks at the state a second after the stream's first event, when a short stream has ended."""

    looks = 0

    async def view(self) -> dict[str, Any]:
        self.looks += 1
        if self.looks == 2:  # the look for the place the stream holds
            await asyncio.sleep(1)
        return await super().view()


async def test_abort_is_inconclusive_when_the_request_ended_by_itself(dry_run: tuple[str, str, Path]) -> None:
    launcher, _, directory = dry_run
    alias, context = card.read_manifest()["MODEL_ALIAS"], int(card.read_manifest()["CONTEXT_TOKENS"])
    async with (launched("--event-delay-ms", "40", config=launcher, rows=directory / "logs/gateway.jsonl") as launch,
                httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10), trust_env=False) as client):
        files = smoke.CardFiles(directory=directory)
        async with relay_that_hides_the_leave(launch.public) as public:
            unseen_leave = await smoke.one(smoke.Smoke(client, public, launch.control, BOT, CONTROL, alias, context,
                                                       card=files), "abort")
        ended_first = await smoke.one(LateLook(client, launch.public, launch.control, BOT, CONTROL, alias, context,
                                               card=files), "abort")
        state = (await call(client, "GET", launch.control + "/v1/state", key=CONTROL)).json()
    assert {name: unseen_leave[name] for name in ("ok", "failed", "held", "cancelled")} == {
        "ok": False, "failed": ["inconclusive"], "held": 1, "cancelled": False}
    assert {name: ended_first[name] for name in ("ok", "failed", "held", "cancelled")} == {
        "ok": False, "failed": ["inconclusive"], "held": 0, "cancelled": False}
    assert state["status"] == "ready"  # neither came as far as the drain

"""The smoke of the first rental (simple_serving/smoke.py): the reading of a stream, the command line, and the dry
run: the whole smoke through the dev launcher, as the README's "The smoke" describes it."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import io
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving import card, smoke
from simple_serving.fake_engine import SENTENCE, THOUGHT

from .support import ALIAS, BOT, CONTROL, ROOT, SERVICE, call
from .test_dev import free_ports, launched

pytestmark = pytest.mark.anyio


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


async def read(events: list[Any], status: int = 200, content_type: str = "text/event-stream; charset=utf-8",
               raw: bytes | None = None) -> smoke.Stream:
    body = raw if raw is not None else b"".join(
        b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n"
        for event in events)
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
    code, out, err = run_smoke("--only", "state,nope")
    assert (code, out) == (2, "")
    assert "no probe 'nope'" in err


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


def test_nothing_answers_and_every_probe_fails_without_an_answer(tmp_path: Path) -> None:
    public, control = free_ports(2)
    code, out, _ = run_smoke("--config", str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL)),
                             "--public-port", str(public), "--control-port", str(control), "--only", "state,abort")
    assert code == 1
    assert [json.loads(line) for line in out.splitlines()] == [
        {"probe": "state", "ok": False, "failed": ["no_answer"]},
        {"probe": "abort", "ok": False, "failed": ["no_answer"]}]


async def test_a_probe_that_breaks_or_outlasts_its_bound_fails_alone_and_prints_no_text(
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
        code = await smoke.run(smoke.Smoke(client, "", "", BOT, CONTROL, ALIAS, 4096), ["state", "completion"])
    out, err = capsys.readouterr()
    assert code == 1
    assert [json.loads(line) for line in out.splitlines()] == [
        {"probe": "state", "ok": False, "failed": ["smoke_error"], "error": "ValueError"},
        {"probe": "completion", "ok": False, "failed": ["time_bound"], "bound_s": 0.05}]
    assert err == "simple-serving smoke: 0 of 2 probes passed\n"
    assert smoke.SHORT not in out


# The dry run: the dev launcher configured as the card's gateway is, and the whole smoke through it.

@pytest.fixture
def dry_run(tmp_path: Path) -> tuple[str, str]:
    """The launcher's configuration, with the manifest's model, context and pins and the cases' two test keys, and
    the smoke's configuration with those keys."""
    manifest = card.read_manifest()
    keys = {key: SERVICE["keys"][key] for key in (BOT, CONTROL)}
    block = {"alias": manifest["MODEL_ALIAS"], "context_tokens": int(manifest["CONTEXT_TOKENS"]),
             "versions": card.pinned_versions(manifest), "keys": keys}
    launcher = tmp_path / "dev.json"
    launcher.write_text(json.dumps(block))
    return str(launcher), str(keys_file(tmp_path, client_key=BOT, control_key=CONTROL))


async def smoke_process(config: str, public: str, control: str, *options: str) -> tuple[int, list[Any], str]:
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "simple_serving.smoke", "--config", config, "--public-port", public.rsplit(":", 1)[1],
        "--control-port", control.rsplit(":", 1)[1], *options, cwd=ROOT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(process.communicate(), 120)
    assert process.returncode is not None
    printed = out + err
    # Never a key, a prompt, the fake's answer or its thought.
    for secret in (BOT, CONTROL, smoke.SHORT, smoke.FILLER.strip(), "keeper", *SENTENCE, *THOUGHT):
        assert secret.encode() not in printed
    return process.returncode, [json.loads(line) for line in out.decode().splitlines()], err.decode()


async def test_the_dry_run_passes_every_probe_and_names_what_the_fake_cannot_verify(dry_run: tuple[str, str]) -> None:
    launcher, config = dry_run
    context = int(card.read_manifest()["CONTEXT_TOKENS"])
    # The pause after each event keeps a stream open while the abort probe reads the state.
    async with launched("--event-delay-ms", "40", config=launcher) as launch:
        code, lines, err = await smoke_process(config, launch.public, launch.control, "--fake")
        async with httpx.AsyncClient(timeout=10) as client:
            state = (await call(client, "GET", launch.control + "/v1/state", key=CONTROL)).json()
    assert code == 0, lines
    assert err == "simple-serving smoke: 9 of 9 probes passed\n"
    assert [line["probe"] for line in lines] == list(smoke.PROBES)
    assert all(line["ok"] for line in lines)
    by_probe = {line.pop("probe"): line for line in lines}
    assert {name: line["not_verifiable"] for name, line in by_probe.items() if "not_verifiable" in line} == {
        "fields": ["answer", "seed_repeats"], "schemas": ["answer"], "counts": ["equal"]}
    assert by_probe["state"]["versions"]["vllm"] == card.read_manifest()["VLLM_VERSION"]
    assert by_probe["reasoning"]["on"]["reasoning_chars"] > 0 == by_probe["reasoning"]["off"]["reasoning_chars"]
    assert by_probe["finish"]["length"]["completion_tokens"] == 8
    assert (by_probe["refusal"]["status"], by_probe["refusal"]["code"]) == (400, "invalid_request")
    assert by_probe["abort"]["held"] == 1
    assert by_probe["schemas"]["passed"] == by_probe["schemas"]["of"] == 12
    cells = {name: part for name, part in by_probe["counts"].items() if isinstance(part, dict)}
    assert len(cells) == 7
    assert all(part["input_tokens"] == part["prompt_tokens"] for part in cells.values())
    near = cells["near_context"]
    assert near["input_tokens"] + near["max_tokens"] == context
    assert context - smoke.NEAR_BAND <= near["input_tokens"] < context
    # The abort probe's drain was undone, and nothing of the smoke is left.
    assert (state["status"], smoke.total(state["active"]), smoke.total(state["waiting"])) == ("ready", 0, 0)


async def test_without_fake_the_answers_of_the_fake_engine_fail_their_schemas(dry_run: tuple[str, str]) -> None:
    launcher, config = dry_run
    async with launched(config=launcher) as launch:
        code, lines, err = await smoke_process(config, launch.public, launch.control, "--only", "fields,schemas")
    assert code == 1
    assert err == "simple-serving smoke: 0 of 2 probes passed\n"
    fields, schemas = lines
    assert fields["response_format"]["failed"] == ["answer"]
    assert fields["response_format"]["answer"] == ["json"]
    assert [name for name, part in fields.items() if isinstance(part, dict) and not part["ok"]] == ["response_format"]
    assert schemas["passed"] == 0
    assert all(part["failed"] == ["answer"] and part["answer"] == ["json"]
               for part in schemas.values() if isinstance(part, dict))
    assert "not_verifiable" not in fields

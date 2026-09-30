"""The smoke of the first rental (simple_serving/smoke.py), where a fault of its own would pass unseen on the card: the
privacy probe's verdict, and the whole smoke through the dev launcher, whose output holds no key, no prompt and nothing
of the model's text."""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from simple_serving import card, smoke
from simple_serving.fake_engine import SENTENCE, THOUGHT

from .support import ALIAS, BOT, CONTROL, ROOT, SERVICE
from .test_dev import launched

pytestmark = pytest.mark.anyio
MARKER = "0123456789abcdef" * 2


def chunk(delta: dict[str, str], finish: str | None = None) -> dict[str, Any]:
    return {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": ALIAS,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


# An answer as the gateway streams one: its role, its text, its finish, its usage, and [DONE].
ANSWER = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in (
    chunk({"role": "assistant"}), chunk({"content": "Lit."}), chunk({}, "stop"),
    {**chunk({}), "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3,
                                           "prompt_tokens_details": {"cached_tokens": 0},
                                           "simple_serving": {"wait_ms": 1, "first_token_ms": 20, "total_ms": 30}}},
)) + b"data: [DONE]\n\n"


@pytest.mark.parametrize(("body_to", "row"), [(None, True), ("gateway.jsonl", True), ("card.jsonl.1", True),
                                              ("vllm/output.log", True), (None, False)])
async def test_the_privacy_probe_fails_on_its_marker_in_any_log_and_on_a_request_without_a_row(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body_to: str | None, row: bool) -> None:
    logs = tmp_path / "logs"  # of the card's state, which tmp_path stands for
    logs.mkdir()

    def gateway(request: httpx.Request) -> httpx.Response:
        """A gateway, as far as the probe sees one: it answers, writes its row of the request unless `row` is false,
        and with `body_to` also the request's body to that log, the marker across the end of the first MiB, which the
        card's report reads at once."""
        if row:
            with (logs / "gateway.jsonl").open("a") as file:
                file.write(json.dumps({"event": "request", "route": smoke.CHAT, "cancelled": False}) + "\n")
        if body_to is not None:
            (logs / body_to).parent.mkdir(exist_ok=True)
            with (logs / body_to).open("ab") as file:
                file.write(b"x" * ((1 << 20) - 16 - file.tell() - request.content.index(MARKER.encode()))
                           + request.content + b"\n")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=ANSWER)

    monkeypatch.setattr(smoke, "new_marker", lambda: MARKER)
    monkeypatch.setattr(smoke, "ROWS_S", 0.5)
    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as client:
        line = await smoke.probe_privacy(smoke.Smoke(client, "http://gateway", "http://gateway", BOT, CONTROL, ALIAS,
                                                     4096, card=smoke.CardFiles(directory=tmp_path)))
    found = dict.fromkeys((*card.LOGS, "other"), 0)
    if body_to is not None:
        found[body_to if body_to in card.LOGS else "other"] = 1
    failed = ([] if row else ["unlogged"]) + (["marker"] if body_to else [])  # a silent log proves nothing
    assert line == {"ok": not failed, "status": 200, **({"failed": failed} if failed else {}), "found": found,
                    "logged": int(row), "echoed": 0}
    assert MARKER not in json.dumps(line)


async def test_the_dry_run_passes_every_probe_and_prints_no_key_no_prompt_and_no_answer(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The runbook's dry run: the dev launcher configured as the card's gateway, with the manifest's model,
    context and pins, whose rows go to a directory that stands for the card, and the whole smoke through it."""
    manifest = card.read_manifest()
    launcher, config, logs = tmp_path / "dev.json", tmp_path / "config.json", tmp_path / "card/logs"
    launcher.write_text(json.dumps({"alias": manifest["MODEL_ALIAS"],
                                    "context_tokens": int(manifest["CONTEXT_TOKENS"]),
                                    "versions": card.pinned_versions(manifest),
                                    "keys": {key: SERVICE["keys"][key] for key in (BOT, CONTROL)}}))
    config.write_text(json.dumps({"client_key": BOT, "control_key": CONTROL}))
    config.chmod(0o600)
    logs.mkdir(parents=True)
    # The pause after each event keeps a stream open while the abort probe reads the state.
    async with launched("--event-delay-ms", "40", config=str(launcher), rows=logs / "gateway.jsonl") as launch:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "simple_serving.smoke", "--config", str(config), "--fake", "--card-dir",
            str(logs.parent), "--public-port", launch.public.rsplit(":", 1)[1], "--control-port",
            launch.control.rsplit(":", 1)[1], cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(process.communicate(), 120)
    assert (process.returncode, err) == (0, b"simple-serving smoke: 11 of 11 probes passed\n")  # each one printed
    lines = {line["probe"]: line for line in map(json.loads, out.splitlines())}
    min_p, budget = lines["fields"]["min_p"], lines["fields"]["thinking_token_budget"]
    drafting = int(manifest["MTP_SPECULATIVE_TOKENS"]) > 0  # the gateway refuses the min_p that vLLM would
    assert (min_p["status"], min_p.get("code")) == ((400, "min_p_unavailable") if drafting else (200, None))
    held = int(manifest["THINKING_BUDGET"]) > 0  # off, the gateway refuses the budget
    assert not held  # integration keeps the unmeasured GPU feature off
    assert (budget["status"], budget.get("code")) == ((200, None) if held else (400, "thinking_budget_unavailable"))
    assert held or lines["budget"]["code"] == "thinking_budget_unavailable"
    assert lines["budget"]["enforcement"] == "disabled"

    # The enabled route is checked on a second fake gateway without changing the committed manifest.
    enabled = manifest | {"THINKING_BUDGET": "1"}
    setup = json.loads(launcher.read_text())
    setup["versions"] = card.pinned_versions(enabled)
    launcher.write_text(json.dumps(setup))
    async with launched(config=str(launcher)) as launch, httpx.AsyncClient(timeout=None, trust_env=False) as client:
        trial = smoke.Smoke(client, launch.public, launch.control, BOT, CONTROL, manifest["MODEL_ALIAS"],
                            int(manifest["CONTEXT_TOKENS"]), fake=True, mtp=drafting, budget=True)
        assert await smoke.run(trial, ["fields", "budget"]) == 0
    printed = capsys.readouterr()
    enabled_lines = {line["probe"]: line for line in map(json.loads, printed.out.splitlines())}
    assert enabled_lines["fields"]["thinking_token_budget"]["status"] == 200
    assert enabled_lines["budget"]["ok"] is True
    assert enabled_lines["budget"]["enforcement"] == "not_verifiable"
    assert enabled_lines["budget"]["bound_counts"] == dict.fromkeys(map(str, smoke.BUDGETS), 0)
    assert "enforcement" in enabled_lines["budget"]["not_verifiable"]
    out += printed.out.encode() + printed.err.encode()
    for secret in (BOT, CONTROL, smoke.SHORT, smoke.PUZZLE, smoke.FILLER.strip(), smoke.COUNTING, smoke.ECHO, "keeper",
                   *SENTENCE, *THOUGHT):
        assert secret.encode() not in out


async def test_budget_enforcement_needs_binding_baselines_and_complete_bounded_private_streams(
        monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [replace(entry, schema=smoke.PUZZLE_SCHEMA) for entry in smoke.bot_schemas.load()]
    monkeypatch.setattr(smoke.bot_schemas, "load", lambda: entries)
    content = json.dumps(smoke.SOLVED)
    cases = (("held", 512), ("unbound", 11), ("within_margin", 308), ("overflow", 512),
             ("negative_count", 512), ("overcount", 512), ("stream_error", 512), ("missing_done", 512),
             ("timeout", 512))
    for fault, baseline in cases:
        generations = 0
        interrupted = False
        monkeypatch.setitem(smoke.BOUND_S, "budget", 0.03 if fault == "timeout" else 2)

        async def gateway(request: httpx.Request, *, fault: str = fault, baseline: int = baseline) -> httpx.Response:
            nonlocal generations, interrupted
            body = json.loads(request.content)
            if request.url.path == smoke.COUNT:
                empty = body["messages"][0]["content"] == smoke.COUNTED
                counted = 20 if empty else 19 if fault == "negative_count" else 9999 if fault == "overcount" else 45
                return httpx.Response(200, json={"input_tokens": counted})
            assert request.url.path == smoke.CHAT and body["max_tokens"] == smoke.BUDGET_TOKENS
            generations += 1
            if fault == "timeout":
                try:
                    await asyncio.Event().wait()
                finally:
                    interrupted = True
            budget = body.get("thinking_token_budget")
            thinking = body["chat_template_kwargs"]["enable_thinking"]
            thought = min(baseline, budget) if budget is not None else baseline if thinking else 0
            if fault == "overflow" and budget is not None:
                thought = budget + smoke.BUDGET_MARGIN + 1
            events = [chunk({"role": "assistant"}), chunk({"reasoning_content": MARKER} if thinking else {}),
                      chunk({"content": content}), chunk({}, "stop"),
                      {**chunk({}), "choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 25 + thought,
                          "simple_serving": {"wait_ms": 0, "first_token_ms": 1, "total_ms": 2}}}]
            broken = budget == 128
            if fault == "stream_error" and broken:
                events = [{"error": {"code": "engine_unavailable", "message": MARKER}}]
            data = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
            if not (broken and fault in ("stream_error", "missing_done")):
                data += b"data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=data)

        async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as client:
            trial = smoke.Smoke(client, "http://gateway", "http://gateway", BOT, CONTROL, ALIAS, 65536, budget=True)
            line = await smoke.one(trial, "budget")
        assert generations == (1 if fault == "timeout" else 15)
        assert line["ok"] is (fault == "held"), fault
        if fault == "held":
            assert line["enforcement"] == "observed" and line["bound_counts"] == {"128": 3, "300": 3, "1024": 0}
        elif fault in ("unbound", "within_margin"):
            assert line["enforcement"] == "inconclusive" and line["failed"] == ["budget_not_bound"]
        elif fault == "timeout":
            assert interrupted and line["failed"] == ["time_bound"]
        else:
            assert line["enforcement"] == "failed"
            expected = {"overflow": "within", "negative_count": "token_count", "overcount": "token_count",
                        "missing_done": "done"}.get(fault)
            part = line["puzzle_128"]
            assert (part.get("code") == "engine_unavailable" if expected is None else expected in part["failed"])
        assert all(secret not in json.dumps(line) for secret in (BOT, CONTROL, MARKER, content, smoke.PUZZLE))

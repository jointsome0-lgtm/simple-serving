"""The smoke of the first rental (simple_serving/smoke.py), where a fault of its own would pass unseen on the card: the
privacy probe's verdict, and the whole smoke through the dev launcher, whose output holds no key, no prompt and nothing
of the model's text."""

from __future__ import annotations

import asyncio
import json
import sys
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


async def test_the_dry_run_passes_every_probe_and_prints_no_key_no_prompt_and_no_answer(tmp_path: Path) -> None:
    """The dry run of the README: the dev launcher configured as the card's gateway, with the manifest's model,
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
    assert (process.returncode, err) == (0, b"simple-serving smoke: 10 of 10 probes passed\n")  # each one printed
    for secret in (BOT, CONTROL, smoke.SHORT, smoke.FILLER.strip(), smoke.COUNTING, smoke.ECHO, "keeper", *SENTENCE,
                   *THOUGHT):
        assert secret.encode() not in out

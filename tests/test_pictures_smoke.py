"""Synthetic loopback smoke: the real picture gateway, complete Qwen graph, CLI and correlated Comfy observer.
The fake models protocol and cleanup only; none of these checks establishes GPU inference quality or capacity.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
from starlette.types import Message, Receive, Scope, Send

from simple_serving import card
from simple_serving.config import Listener
from simple_serving.pictures import fake_comfy, smoke
from simple_serving.pictures.app import public_app
from simple_serving.pictures.fake_comfy import Script
from simple_serving.pictures.graphs import PNG_SIGNATURE
from simple_serving.server import Servers, bind

from .support import BOT, CONTROL, until
from .test_pictures import KEYS, Stack, running

pytestmark = pytest.mark.anyio
ROOT = Path(__file__).resolve().parent.parent
MARKER = "synthetic-private-picture-71bcfe"
REPORT_KEYS = {"ok", "mode", "phase", "code", "gpu", "ready", "versions_match", "engine_match", "accepted",
               "observer_correlated", "stream_complete", "png_received", "images", "image_bytes", "events",
               "stop_requested", "stop_confirmed", "history_deleted", "cleanup_confirmed", "elapsed_ms"}


@asynccontextmanager
async def fixture(script: Script | None = None, **config: Any) -> AsyncIterator[Stack]:
    """Reuse the existing gateway harness and also reap the fake's independent worker between scenarios."""
    fake: fake_comfy.FakeComfy | None = None
    setup = {"alias": card.read_manifest(folder="card/pictures")["MODEL_ALIAS"],
             "keys": {**KEYS, BOT: {**KEYS[BOT], "classes": ["internal"], "default": "internal", "scopes": False}},
             "versions": smoke.expected_versions(), **config}
    try:
        async with running(script or Script(job_s=0.3, marker=MARKER), **setup) as stack:
            fake = stack.fake
            yield stack
    finally:
        if fake is not None and fake._worker is not None:
            fake._worker.cancel()
            await asyncio.gather(fake._worker, return_exceptions=True)


def target(stack: Stack) -> smoke.Target:
    public = urlsplit(stack.public).port
    control = urlsplit(stack.control).port
    engine = urlsplit(stack.service.config.engine_url).port
    assert public is not None and control is not None and engine is not None
    return smoke.Target(public, control, engine, BOT, CONTROL)


async def invoke(tmp_path: Path, destination: smoke.Target, *, wall_s: float = 3,
                 cleanup_s: float = 1) -> tuple[int, dict[str, Any], str]:
    config = tmp_path / "synthetic-client.json"
    config.write_text(json.dumps({"client_key": BOT, "control_key": CONTROL}))
    config.chmod(0o600)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "simple_serving.pictures.smoke", "--fake", "--config", str(config),
        "--public-port", str(destination.public_port), "--control-port", str(destination.control_port),
        "--engine-port", str(destination.engine_port), "--wall-s", str(wall_s), "--cleanup-s", str(cleanup_s),
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), wall_s + cleanup_s + 5)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert process.returncode is not None
    shown = (stdout + stderr).decode()
    report = json.loads(stdout)
    assert isinstance(report, dict) and report.keys() <= REPORT_KEYS
    assert report["mode"] == "fake" and report["gpu"] == "not_verifiable"
    for name, value in report.items():
        if name in {"mode", "phase", "code", "gpu"}:
            assert isinstance(value, str) and re.fullmatch(r"[a-z_]+", value)
        else:
            assert type(value) in {bool, int}
    assert not re.search(r"\b[0-9a-f]{24,}\b|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", shown, re.IGNORECASE)
    for private in (BOT, CONTROL, MARKER, str(tmp_path), str(ROOT), "class_type", "UNETLoader", "safetensors", "iVBOR",
                    "qwen_image_2.1", "Traceback"):
        assert private not in shown
    return process.returncode, report, shown


class Wire:
    """Alter only the response wire around the actual gateway, leaving job submission and erasure intact."""

    def __init__(self, stack: Stack, fault: str) -> None:
        self.app = public_app(stack.service)
        self.fault = fault

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def changed(message: Message) -> None:
            body = message.get("body", b"")
            if scope["path"] == "/v1/pictures" and body.startswith(b"data: "):
                data = body[6:].strip()
                event = json.loads(data) if data != b"[DONE]" else None
                name = event.get("event") if isinstance(event, dict) else None
                if self.fault == "sse" and name == "started":
                    body = b""
                elif self.fault == "framing" and data == b"[DONE]":
                    body = b"data: [DONE]\n"
                elif self.fault in {"png", "private_png"} and name == "image":
                    assert isinstance(event, dict)
                    png = PNG_SIGNATURE + b"broken" if self.fault == "png" else fake_comfy._png(MARKER)
                    event["png"] = base64.b64encode(png).decode()
                    body = b"data: " + json.dumps(event).encode() + b"\n\n"
                elif self.fault == "drop" and name != "accepted":
                    body = b""
                message = {**message, "body": body}
            await send(message)

        await self.app(scope, receive, changed)


@asynccontextmanager
async def altered(stack: Stack, fault: str) -> AsyncIterator[smoke.Target]:
    sock = bind(Listener("127.0.0.1", 0))
    port = sock.getsockname()[1]
    servers = Servers([(Wire(stack, fault), sock)], graceful_s=1)
    await servers.start()
    try:
        yield replace(target(stack), public_port=port)
    finally:
        await servers.stop()


async def test_cli_submits_the_complete_synthetic_recipe_and_confirms_its_own_job_is_erased(tmp_path: Path) -> None:
    async with fixture() as stack:
        code, report, shown = await invoke(tmp_path, target(stack))
        assert code == 0 and report["ok"] is True, report
        assert all(report[name] is True for name in ("ready", "versions_match", "engine_match", "accepted",
                                                     "observer_correlated", "stream_complete", "png_received",
                                                     "history_deleted", "cleanup_confirmed"))
        assert report["images"] == 1 and report["image_bytes"] > 0 and report["events"] >= 5
        assert report["stop_requested"] is False
        await until(lambda: not stack.fake.history and not stack.service.jobs)
        (job,) = stack.fake.jobs
        assert job.id not in shown and job.client_id not in shown
        assert stack.fake.running is None and stack.fake.pending == []
        assert ("GET", f"/history/{job.id}") in stack.fake.calls and ("POST", "/history") in stack.fake.calls
        original = json.loads((ROOT / "card/pictures/image-workflow-qwen.json").read_text())
        graph = job.graph
        assert set(graph) == {str(number) for number in range(1, 10)}
        assert all(graph[node] == original[node] for node in ("1", "2", "3", "7"))
        assert graph["4"]["inputs"]["clip"] == ["2", 0] and graph["4"]["inputs"]["prompt"]
        assert graph["5"]["inputs"] == {"width": 512, "height": 512, "batch_size": 1}
        assert graph["6"]["inputs"]["model"] == ["9", 0] and 4 <= graph["6"]["inputs"]["steps"] <= 6
        assert graph["6"]["inputs"]["latent_image"] == ["5", 0]
        assert graph["8"]["class_type"] == "PreviewImage" and graph["8"]["inputs"]["images"] == ["7", 0]
        assert graph["9"] == {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["1", 0], "attention": "comfy kitchen attention"}}
        assert MARKER.encode() in next(iter(stack.fake.files.values()))  # Raw fake PNG metadata was private.


async def test_bad_preflight_stream_framing_and_private_or_invalid_png_fail_without_printing_payloads(
        tmp_path: Path) -> None:
    for preflight in ("ready", "pins"):
        config: dict[str, Any] = (
            {"versions": smoke.expected_versions() | {"qwen_revision": MARKER}} if preflight == "pins" else {})
        async with fixture(**config) as stack:
            if preflight == "ready":
                stack.service.drain(stack.service.boot_id)
            code, report, _ = await invoke(tmp_path, target(stack))
            assert code != 0 and report["ok"] is False and report["accepted"] is False
            assert not stack.fake.jobs
    async with fixture() as stack, fixture() as unrelated:
        destination = replace(target(stack), control_port=target(unrelated).control_port)
        code, report, _ = await invoke(tmp_path, destination)
        assert code != 0 and report["code"] == "state_mismatch" and report["accepted"] is False
        assert not stack.fake.jobs and not unrelated.fake.jobs
    for fault in ("sse", "framing", "png", "private_png"):
        async with fixture() as stack, altered(stack, fault) as destination:
            code, report, shown = await invoke(tmp_path, destination)
            assert code != 0 and report["ok"] is False and report["accepted"] is True, report
            assert report["observer_correlated"] is True and report["cleanup_confirmed"] is True
            assert all(job.id not in shown for job in stack.fake.jobs)
            await until(lambda: not stack.fake.history and not stack.service.jobs)


async def test_an_accepted_job_is_cleaned_after_deadline_or_drop_and_an_unrelated_observer_cannot_confirm_it(
        tmp_path: Path) -> None:
    async with fixture(Script(outcomes={1: "hang"}, marker=MARKER)) as stack:
        code, report, shown = await invoke(tmp_path, target(stack), wall_s=0.4)
        assert code != 0 and report["ok"] is False and report["accepted"] is True, report
        assert report["observer_correlated"] is True and report["stop_requested"] is True
        assert report["cleanup_confirmed"] is True and report["history_deleted"] is True
        await until(lambda: stack.fake.running is None and not stack.fake.pending and not stack.fake.history
                    and not stack.service.jobs)
        assert stack.fake.jobs[0].id not in shown
    async with fixture() as stack, altered(stack, "drop") as destination:
        code, report, _ = await invoke(tmp_path, destination)
        assert code != 0 and report["accepted"] is True and report["stream_complete"] is False
        assert report["stop_requested"] is True and report["cleanup_confirmed"] is True
    async with fixture() as stack, fixture() as unrelated:
        destination = replace(target(stack), engine_port=target(unrelated).engine_port)
        code, report, shown = await invoke(tmp_path, destination)
        assert code != 0 and report["ok"] is False and report["accepted"] is True
        assert report["observer_correlated"] is False and report["cleanup_confirmed"] is False
        assert report["history_deleted"] is False and not unrelated.fake.jobs
        await until(lambda: stack.fake.running is None and not stack.fake.pending and not stack.service.jobs)
        assert stack.fake.jobs[0].id not in shown

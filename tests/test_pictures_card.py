"""The picture card's launcher and scripts (docs/pictures.md, section 9): the copies of simple-story-chat's scripts as
SOURCE names them, ComfyUI under the old start's lock with the manifest's settings, what the launcher keeps of its
output, the picture gateway's configuration, and a preparation that adopts a whole install and changes nothing when it
runs again. Stand-ins take the place of ComfyUI's start and install, the gateway, pip and git; nothing reaches the
network, and no weights are fetched."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from simple_serving import card
from simple_serving.pictures import config as pictures_config
from simple_serving.pictures.card import COPIES, LOCK, SETTINGS, PictureCard, PicturePair

from .support import FakeVast, until
from .test_card import CLIENT, CONTROL, CREDENTIAL, METRICS, SECRET, exits, ran, recorder, rows, stand_in
from .test_dev import free_ports

HERE = card.CODE / "card/pictures"
READY = json.dumps({"time": "2026-09-28T12:00:00.000Z", "event": "engine", "service_status": "ready",
                    "lane": "pictures"})
SERVE = f"""#!/usr/bin/env bash
env -0 > "$0.env"
echo "loaded partially; 21000.00 MB usable, 10000.00 MB loaded, 2000.00 MB offloaded"
echo "torch.OutOfMemoryError: Allocation on device 0. {SECRET}"
echo '{{"event":"swept","files":3,"failedFiles":0,"records":2,"path":"{SECRET}"}}'
exec sleep 600
"""
# simple-story-chat's image-bootstrap.sh as far as the preparation sees it: it notes its settings and each run, then
# fails, or leaves the install whole: the ready mark, the three Qwen files at their pinned sizes, holes for their bytes,
# image-verified.txt and the two Qwen graphs.
INSTALL = """#!/usr/bin/env bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
env | grep '^SIMPLE_CHAT_' | sort > "$0.env"
echo run >> "$0.runs"
[[ $(< "$0.outcome") == whole ]] || exit 1
source "$here/image-manifest.env"
comfy=$SIMPLE_CHAT_GPU_DIR/ComfyUI
mkdir -p "$comfy/.venv-cu130" "$comfy/models/diffusion_models" "$comfy/models/text_encoders" "$comfy/models/vae"
touch "$comfy/.venv-cu130/simple-chat-ready"
truncate -s "$IMAGE_QWEN_MODEL_BYTES" "$comfy/models/diffusion_models/$IMAGE_QWEN_MODEL_FILE"
truncate -s "$IMAGE_QWEN_ENCODER_BYTES" "$comfy/models/text_encoders/$IMAGE_QWEN_ENCODER_FILE"
truncate -s "$IMAGE_QWEN_VAE_BYTES" "$comfy/models/vae/$IMAGE_QWEN_VAE_FILE"
printf '%s\\n' "revision $COMFYUI_REVISION" "$IMAGE_QWEN_MODEL_SHA256  $IMAGE_QWEN_MODEL_FILE" \\
  "$IMAGE_QWEN_ENCODER_SHA256  $IMAGE_QWEN_ENCODER_FILE" "$IMAGE_QWEN_VAE_SHA256  $IMAGE_QWEN_VAE_FILE" \\
  > "$SIMPLE_CHAT_GPU_DIR/image-verified.txt"
cp "$here/$IMAGE_QWEN_WORKFLOW" "$here/$IMAGE_QWEN_EDIT_WORKFLOW" "$SIMPLE_CHAT_GPU_DIR/"
"""


@pytest.fixture(autouse=True)
def no_vast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whatever path a test takes, it never reaches Vast: every stop that is not a fake fails at once."""
    async def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test tried to reach Vast")

    monkeypatch.setattr("simple_serving.vast.stop", refuse)


def requirements(path: Path) -> dict[str, tuple[str, set[str]]]:
    exported = re.findall(r"^(\S+)==(\S+) \\\n((?: +--hash=.*\n)+)", path.read_text(), re.MULTILINE)
    return {name: (version, set(re.findall(r"sha256:\w+", hashes))) for name, version, hashes in exported}


def test_the_copies_are_the_commits_and_the_lists_of_the_lane_hold_to_each_other() -> None:
    source = (HERE / "SOURCE").read_text()
    assert re.search(r"^commit [0-9a-f]{40}$", source, re.MULTILINE)
    assert re.findall(r"^[0-9a-f]{64}  (\S+)$", source, re.MULTILINE) == list(COPIES)
    for name in COPIES:  # what bootstrap.sh checks on the card, held here before a copy is committed
        assert f"{hashlib.sha256((HERE / name).read_bytes()).hexdigest()}  {name}\n" in source
    manifest = card.read_manifest(folder="card/pictures")
    assert tuple(manifest["READER_NODES"].split(",")) == pictures_config.READER_NODES
    for name in ("image-workflow.json", "image-workflow-qwen.json", "image-workflow-qwen-edit.json"):
        nodes = {node["class_type"] for node in json.loads((HERE / name).read_text()).values()}
        assert "SaveImage" in nodes  # which the bot turns into a preview, the one node a reader may use to show one
        assert nodes - {"SaveImage"} | {"PreviewImage"} <= set(pictures_config.READER_NODES)
    assert "SaveImage" not in pictures_config.READER_NODES
    # The picture gateway's lock is the text gateway's with websockets, as uv.lock pins it.
    lock = {package["name"]: package for package in tomllib.loads((card.CODE / "uv.lock").read_text())["package"]}
    websockets = lock["websockets"]
    text = requirements(card.CODE / "card/gateway-requirements.txt")
    assert requirements(HERE / "gateway-requirements.txt") == text | {
        "websockets": (websockets["version"], {file["hash"] for file in [websockets["sdist"], *websockets["wheels"]]})}


@pytest.mark.anyio
async def test_the_text_gateway_runs_without_websockets() -> None:
    """The text card's venv has no websockets: nothing on the text gateway's path may import it."""
    engine, public, control = free_ports(3)
    launcher = ("import runpy, sys; sys.modules['websockets'] = None; "
                "runpy.run_module('simple_serving.dev', run_name='__main__')")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", launcher, "--config", "contract/cases-v2.json", "--engine-port", str(engine),
        "--public-port", str(public), "--control-port", str(control), cwd=card.CODE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL)
    assert process.stdout is not None
    try:
        await asyncio.wait_for(process.stdout.readuntil(b"Ctrl-C stops both.\n"), 15)
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
        assert await asyncio.wait_for(process.wait(), 15) == 0


def picture_card(tmp_path: Path, serve: str) -> PictureCard:
    """A picture card whose image-serve.sh is `serve`, with the checkout's manifest and the test's credential."""
    code, state, root = tmp_path / "code", tmp_path / "state", tmp_path / "root"
    for path in (code / "card/pictures", state / "logs", root):
        path.mkdir(parents=True)
    shutil.copy(HERE / "manifest.env", code / "card/pictures/manifest.env")
    (code / "card/pictures/image-serve.sh").write_text(serve)
    (state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL}))
    return PictureCard.load(code=code, state=state, environ=CREDENTIAL, root=root)


@pytest.mark.anyio
async def test_comfyui_runs_under_the_old_starts_lock_with_the_manifests_settings_and_its_output_is_filtered(
        tmp_path: Path, logged: io.StringIO, monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {**CREDENTIAL, "SIMPLE_CHAT_IMAGE_VIGGLE": "true",
                        "SIMPLE_CHAT_TELEGRAM_TOKEN": "synthetic-telegram-token"}.items():
        monkeypatch.setenv(name, value)  # what a shell on the card may hold, and none of it reaches ComfyUI
    ready = picture_card(tmp_path, SERVE)
    serve, lock = ready.code / "card/pictures/image-serve.sh", ready.root / LOCK
    stand_in(ready.state / "gateway/bin/python", READY)
    ended = asyncio.Event()
    launch = asyncio.create_task(card.launch(ready, ended, FakeVast()))
    await until(lambda: bool(rows(logged, "ready")) and noted(serve) is not None)
    with pytest.raises(BlockingIOError), holding(lock):  # the old start's lock, which ComfyUI holds while it runs
        pass
    ended.set()
    assert await launch == "signal"
    environ = noted(serve)
    assert environ is not None
    assert {name: value for name, value in environ.items() if name.startswith("SIMPLE_CHAT_")} == {
        name: ready.manifest[setting] for name, setting in SETTINGS.items()} | {"SIMPLE_CHAT_IMAGE_VIGGLE": "false"}
    assert "CONTAINER_API_KEY" not in environ and environ["SIMPLE_CHAT_IMAGE_PORT"] == "8188"
    gateway = ran(ready.state / "gateway/bin/python")
    assert gateway["argv"] == ["-m", "simple_serving.pictures"]
    assert {"CONTAINER_ID", "CONTAINER_API_KEY", "SIMPLE_SERVING_CONFIG"} <= set(gateway["env"])
    assert exits(logged) == {"gateway": -signal.SIGTERM, "engine": -signal.SIGTERM}
    assert [row["code"] for row in rows(logged, "engine_warning")] == ["partial_load"]
    assert [row["code"] for row in rows(logged, "engine_failure")] == ["out_of_memory"]
    assert rows(logged, "swept") == [{"event": "swept", "files": 3, "failed_files": 0, "records": 2}]
    # With the lock free once ComfyUI has ended, and held as by the old start, a second ComfyUI never starts: the pair
    # ends, and the card stops.
    with holding(lock):
        noted(serve, forget=True)
        vast = FakeVast()
        assert await card.launch(ready, asyncio.Event(), vast) == "engine_exit"
    assert vast.stopped and noted(serve) is None and exits(logged)["engine"] == 1
    assert SECRET not in logged.getvalue()


@contextlib.contextmanager
def holding(path: Path) -> Iterator[None]:
    """The lock on `path`, taken as `flock -n` takes it, and held for the block."""
    with open(path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def noted(serve: Path, *, forget: bool = False) -> dict[str, str] | None:
    """The environment the stand-in for image-serve.sh ran with, if it ran; with `forget`, the note is removed."""
    path = Path(f"{serve}.env")
    if forget:
        path.unlink()
    if not path.exists():
        return None
    return dict(entry.split("=", 1) for entry in path.read_text().split("\0") if entry)


def test_of_comfyuis_output_each_category_is_kept_once_and_of_the_sweepers_rows_their_counts_and_codes(
        tmp_path: Path, logged: io.StringIO) -> None:
    (tmp_path / "logs").mkdir()
    pair = PicturePair(tmp_path)
    for text in (
        f"torch.OutOfMemoryError: Allocation on device 0 would exceed allowed memory. {SECRET}",
        f"RuntimeError: CUDA error: an illegal memory access was encountered {SECRET}",
        "[Errno 98] error while attempting to bind on address ('127.0.0.1', 8188): address already in use",
        f"!!! Exception during processing !!! {SECRET}",
        f"Traceback (most recent call last): {SECRET}",
        "No whole cu130 environment; run image-bootstrap.sh with SIMPLE_CHAT_IMAGE_TORCH=cu130 first.",
        "/dev/shm is not a writable tmpfs, and the pictures of scenes would land on the disk. Refusing to start.",
        f"loaded partially; 21000.00 MB usable, 10000.00 MB loaded {SECRET}",
        f"comfy kitchen attention is unavailable; using pytorch attention {SECRET}",
        f"CUDA out of memory. Tried to allocate 2.00 GiB {SECRET}",  # a category once logged is not logged again
        f"Prompt executed in 12.50 seconds {SECRET}",
    ):
        pair.engine_line(text.encode() + b"\n")
    for line in (
        '{"event":"sweeper_started","graceSeconds":60,"fileCapSeconds":600,"historyCapSeconds":600}',
        '{"event":"swept","files":3,"failedFiles":1,"records":2}',
        '{"event":"records_undeleted","code":"http_500","count":2}',
        '{"event":"history_unreadable","code":"connection_refused"}',
        '{"event":"sweeper_stopped","reason":"server_gone"}',
        # A line that forges the sweeper's: only what the sweeper could have written passes.
        f'{{"event":"swept","files":-1,"failedFiles":true,"records":1e12,"path":"{SECRET}"}}',
        '{"event":"swept","files":NaN,"records":2.4}',
        f'{{"event":"sweep_failed","code":"{SECRET}"}}',
        f'{{"event":"{SECRET}","files":1}}',
    ):
        pair.engine_line(line.encode() + b"\n")
    assert [row["code"] for row in rows(logged, "engine_failure")] == [
        "out_of_memory", "cuda_error", "bind_failed", "exception", "traceback", "install_incomplete", "temp_not_tmpfs"]
    assert [row["code"] for row in rows(logged, "engine_warning")] == ["partial_load", "attention_fallback"]
    kept = [{name: value for name, value in json.loads(line).items() if name != "time"}
            for line in logged.getvalue().splitlines()]
    assert [row for row in kept if not row["event"].startswith("engine_")] == [
        {"event": "sweeper_started", "grace_s": 60, "file_cap_s": 600, "history_cap_s": 600},
        {"event": "swept", "files": 3, "failed_files": 1, "records": 2},
        {"event": "records_undeleted", "code": "http_500", "count": 2},
        {"event": "history_unreadable", "code": "connection_refused"},
        {"event": "sweeper_stopped", "reason": "server_gone"},
        {"event": "swept"},
        {"event": "swept", "records": 2},
        {"event": "sweep_failed"},
    ]
    assert SECRET not in logged.getvalue()


def test_the_picture_gateways_configuration_is_the_manifests_and_the_pins_and_loads(tmp_path: Path) -> None:
    ready = replace(picture_card(tmp_path, SERVE), code=card.CODE)  # the checkout's copies and pins
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(ready.config()))
    loaded = pictures_config.load(str(path))
    pins = card.read_manifest(folder="card/pictures", name="image-manifest.env")
    assert (loaded.alias, loaded.engine_url, loaded.public.port, loaded.control.port) == (
        "qwen-image-2.1", "http://127.0.0.1:8188", 8094, 8095)
    assert asdict(loaded.expect) == pictures_config.CARD_ENGINE  # what the dev launcher and the fake stand for
    assert (loaded.expect.comfyui, loaded.expect.pytorch) == (pins["COMFYUI_VERSION"], pins["TORCH_CU130_VERSION"])
    commit = re.findall(r"^commit (\S+)$", (HERE / "SOURCE").read_text(), re.MULTILINE)[0]
    assert dict(loaded.versions) == {"comfyui_revision": pins["COMFYUI_REVISION"],
                                     "qwen_revision": pins["IMAGE_QWEN_REVISION"], "scripts_commit": commit}
    assert [key.label for key in loaded.keys] == ["client", "control"]
    assert loaded.keys[0].digest == bytes.fromhex(CLIENT) and loaded.reader_nodes == set(pictures_config.READER_NODES)
    (ready.state / "keys.json").write_text(json.dumps({"client": CLIENT, "control": CONTROL, "metrics": METRICS}))
    path.write_text(json.dumps(ready.config()))
    loaded = pictures_config.load(str(path))
    metric_key = loaded.keys[-1]
    assert (metric_key.label, metric_key.digest, metric_key.metrics, metric_key.control, metric_key.classes) == (
        "metrics", bytes.fromhex(METRICS), True, False, frozenset())


def test_bootstrap_checks_the_copies_adopts_a_whole_install_and_changes_nothing_when_it_runs_again(
        tmp_path: Path) -> None:
    code, state, root, gpu, calls = (tmp_path / name for name in ("code", "state", "root", "gpu", "calls"))
    here = code / "card/pictures"
    shutil.copytree(HERE, here)
    root.mkdir()
    (here / "manifest.env").write_text((here / "manifest.env").read_text() + f"GPU_DIR={gpu}\n")
    install = here / "image-bootstrap.sh"
    install.write_text(INSTALL)  # in SOURCE under its own hash, as a copy of a commit would be
    digest = hashlib.sha256(INSTALL.encode()).hexdigest()
    source = here / "SOURCE"
    source.write_text(re.sub(r"^\S+  image-bootstrap\.sh$", f"{digest}  image-bootstrap.sh", source.read_text(),
                             flags=re.MULTILINE))
    pins = card.read_manifest(folder="card/pictures", name="image-manifest.env")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/git").write_text(f"#!/bin/sh\necho {pins['COMFYUI_REVISION']}\n")  # ComfyUI's checkout
    (tmp_path / "bin/git").chmod(0o700)
    for path in (state / "gateway/bin/pip", state / "gateway/bin/python"):
        recorder(path, calls)
    environ = {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "SIMPLE_SERVING_PICTURES_CARD_DIR": str(state),
               "SIMPLE_SERVING_CARD_ROOT": str(root), **CREDENTIAL}

    def run(script: str, stdin: str = "") -> int:
        done = subprocess.run(["bash", str(here / script)], input=stdin, env=environ, capture_output=True, text=True,
                              timeout=60, check=False)
        assert not any(digest in done.stdout + done.stderr for digest in (CLIENT, CONTROL, METRICS))
        return done.returncode

    def installs() -> int:
        return len(Path(f"{install}.runs").read_text().splitlines())

    keys = f"{CLIENT}\n{CONTROL}\n"
    assert run("onstart.sh") == 0 and not calls.exists()  # a start before the preparation starts nothing
    assert [(root / f".simple-chat-instance-{name}").read_text() for name in ("id", "api-key")] == list(
        CREDENTIAL.values())
    assert run("bootstrap.sh") == 4  # the copies are SOURCE's, and the card holds no keys yet
    serve = (here / "image-serve.sh").read_bytes()
    (here / "image-serve.sh").write_bytes(serve + b"\n")
    assert run("bootstrap.sh", keys) == 3 and not (state / "keys.json").exists()  # a copy that is not SOURCE's
    (here / "image-serve.sh").write_bytes(serve)
    Path(f"{install}.outcome").write_text("fails\n")
    assert run("bootstrap.sh", keys) == 1 and not (state / "prepared").exists()
    # The bot card's engine, without Viggle and the prompt enhancer.
    assert Path(f"{install}.env").read_text().splitlines() == [
        f"SIMPLE_CHAT_GPU_DIR={gpu}", "SIMPLE_CHAT_IMAGE_QWEN=only", "SIMPLE_CHAT_IMAGE_QWEN_PE=false",
        "SIMPLE_CHAT_IMAGE_TORCH=cu130", "SIMPLE_CHAT_IMAGE_VIGGLE=false"]
    Path(f"{install}.outcome").write_text("whole\n")
    assert run("bootstrap.sh", keys) == 0 and installs() == 2
    assert PictureCard.load(code=code, state=state, environ=environ, root=root).prepared()
    written = [state / "keys.json", state / "prepared", *root.glob(".simple-chat-*")]
    first = {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written}
    (state / "prepared").unlink()  # a card whose install the old scripts left: it is adopted as it is
    assert run("bootstrap.sh", keys) == 0 and run("bootstrap.sh") == 0 and installs() == 2
    assert run("bootstrap.sh", f"{hashlib.sha256(b'other').hexdigest()}\n{CONTROL}\n") == 5
    assert run("bootstrap.sh", f"{CLIENT}\n") == 4
    assert run("bootstrap.sh", f"{CLIENT}\n{CLIENT}\n") == 4
    assert {path: (path.read_bytes(), path.stat().st_mode & 0o777) for path in written} == first
    assert {mode for _, mode in first.values()} == {0o600}
    vae = gpu / "ComfyUI/models/vae" / pins["IMAGE_QWEN_VAE_FILE"]
    os.truncate(vae, int(pins["IMAGE_QWEN_VAE_BYTES"]) - 1)  # a file cut short is not whole: the install runs again
    assert run("bootstrap.sh") == 0 and installs() == 3 and vae.stat().st_size == int(pins["IMAGE_QWEN_VAE_BYTES"])
    noted = calls.read_text().splitlines()
    assert noted.count("python -m simple_serving.pictures.card") == 4  # the start at the end of each preparation
    installed = [line for line in noted if line.startswith("pip install")]
    assert installed and all("--require-hashes --only-binary :all:" in line and line.endswith(
        "card/pictures/gateway-requirements.txt") for line in installed)
    # Preparing a previously two-key card with the optional metrics key preserves its original rights.
    for wrong in (CLIENT, CONTROL, "not-a-digest"):
        assert run("bootstrap.sh", f"{keys}{wrong}\n") == 4
    assert run("bootstrap.sh", f"{keys}{METRICS}\n") == 0 and installs() == 3
    three = {"client": CLIENT, "control": CONTROL, "metrics": METRICS}
    assert json.loads((state / "keys.json").read_text()) == three
    assert run("bootstrap.sh", f"{keys}{METRICS}\n") == 0 and run("bootstrap.sh") == 0
    assert run("bootstrap.sh", keys) == 5  # no downgrade or replacement of installed keys
    assert (state / "keys.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((state / "keys.json").read_text()) == three

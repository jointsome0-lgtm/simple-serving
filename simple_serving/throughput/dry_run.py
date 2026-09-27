"""The throughput harness's dry run on this machine (docs/speed.md, the runbook's "Before any card", step 2):

    python -m simple_serving.throughput dry-run

It prepares a card in a temporary directory as bootstrap.sh would leave it (this checkout's manifest on free loopback
ports, the prepared stamp, synthetic key hashes and a synthetic instance credential), with stand-ins for its venvs:
`vllm` is fake_vllm.py, which prints vLLM's lines from its arguments and serves the fake engine, and both Pythons are
this one. On that card it runs the harness's own code: the reference with the identity set and every workload, a
change of MAX_NUM_BATCHED_TOKENS on the other side of the set, the synthetic start with a small timing sweep, and the
microbenchmarks' plans. Beside the reference, a reader sends a request with the bot's key every second, so that some
of its passes must be flagged. The workloads come from a synthetic pack in the public pack's form, each text of which
holds a marker; at the end no file the harness wrote may hold the marker or the credential, and each result must have
passed the check of counts, times and fixed words. It prints each run's report and one line of what it ran.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from .. import card, smoke
from ..card import Card
from . import cli, microbench, report, workloads

SCENES, CHARS = 16, 1200  # a scenario of the synthetic pack: enough for a context of 16000 tokens at 4 a token
CONTEXT = 2000  # the passes' target context, small so that the fake engine's run is quick
CREDENTIAL_ID = "1234"


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def script(path: Path, command: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{command}\n")
    path.chmod(0o700)


def prepared_card(root: Path, credential: dict[str, str], client_key: str) -> Card:
    """A card that bootstrap.sh has prepared, as far as the harness looks, with stand-ins for vLLM and the venvs."""
    code, state = root / "code", root / "state"
    (code / "card").mkdir(parents=True)
    state.mkdir(mode=0o700)
    ports = dict(zip(("PUBLIC_PORT", "CONTROL_PORT", "ENGINE_PORT"), free_ports(3), strict=True))
    manifest = (card.CODE / "card/manifest.env").read_text()
    (code / "card/manifest.env").write_text(manifest + "".join(f"{name}={port}\n" for name, port in ports.items()))
    for name in ("gateway", "vllm"):
        (code / f"card/{name}-requirements.txt").write_text(f"# the dry run's stand-in for {name}'s lock\n")
    stamp = hashlib.sha256(b"".join((code / "card" / name).read_bytes() for name in card.STAMPED)).hexdigest()
    (state / "prepared").write_text(stamp + "\n")
    keys = {"client": hashlib.sha256(client_key.encode()).hexdigest(),
            "control": hashlib.sha256(secrets.token_bytes(32)).hexdigest()}
    (state / "keys.json").write_text(json.dumps(keys))
    python, checkout = sys.executable, card.CODE
    fake = "simple_serving.throughput.fake_vllm"
    script(state / "vllm/bin/vllm", f'PYTHONPATH="{checkout}" exec "{python}" -m {fake} "$@"')
    script(state / "vllm/bin/python", f'exec "{python}" "$@"')
    script(state / "gateway/bin/python", f'PYTHONPATH="{checkout}" exec "{python}" "$@"')
    (state / "tokenizer").mkdir()
    return Card.load(code=code, state=state, environ=credential, root=state)


def synthetic_pack(root: Path, marker: str) -> Path:
    """The public eval pack's form: five scenarios, each a seed and a chain of scenes, every text with the marker."""
    for index, name in enumerate(workloads.SCENARIOS):
        nodes, parent = {}, None
        for number in range(SCENES):
            node = f"n{index}{number:02d}"
            nodes[node] = {"id": node, "parent": parent, "input": f"{marker} The reader acts, turn {number}.",
                           "text": f"{marker} " + "The tide comes in over the causeway. " * (CHARS // 37),
                           "time": "", "truncated": False}
            parent = node
        state = {"seeds": {"s1": {"id": "s1", "title": f"Story {index}", "startTime": "evening",
                                  "text": f"{marker} A lighthouse keeper and a stranger."}},
                 "stories": {"h1": {"id": "h1", "seedId": "s1", "nodes": nodes}}}
        (root / name).mkdir(parents=True)
        (root / name / "frozen.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    return root


def leaks(root: Path, secrets_: list[str]) -> list[str]:
    """The files under `root` that hold one of `secrets_`, by their names relative to it."""
    return [str(path.relative_to(root)) for path in sorted(root.rglob("*"))
            if path.is_file() and any(secret.encode() in path.read_bytes() for secret in secrets_)]


async def reader(base: Card, key: str, marker: str, stop: asyncio.Event) -> int:
    """A reader of the bot beside the harness: a short turn every second until `stop`, and how many were answered."""
    turn = [{"role": "user", "content": f"{marker} A reader's turn."}]
    body = workloads.body(base.manifest["MODEL_ALIAS"], turn, 16, 0.8)
    headers = {"Authorization": f"Bearer {key}", "X-Simple-Serving-Class": "reader",
               "X-Simple-Serving-Scope": workloads.scope()}
    answered = 0
    async with httpx.AsyncClient(timeout=10) as client:
        while not stop.is_set():
            with contextlib.suppress(httpx.HTTPError):  # the gateway is not up yet, or no longer
                async with client.stream("POST", f"http://127.0.0.1:{base.ports['public']}{smoke.CHAT}", json=body,
                                         headers=headers) as response:
                    await response.aread()
                    answered += response.status_code == 200
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), 1.0)
    return answered


async def runs(base: Card, pack: Path, client_key: str,
               marker: str) -> tuple[list[tuple[int, dict[str, Any] | None]], int]:
    quick = cli.Options(pack=pack, context=CONTEXT, turns=2, gap_s=0.1, sample_s=0.05)
    steps = [
        replace(quick, change="batched-tokens=512", workloads=("scenes", "compactions"), at=(2,), identity="change"),
        cli.Options(change="synthetic=0.8,0.6,0.4", sweep=True, contexts=(4000,), unshared=(16000,), sweep_at=(1, 2),
                    rounds=1, sample_s=0.05),
    ]
    stop = asyncio.Event()
    beside = asyncio.create_task(reader(base, client_key, marker, stop))
    try:
        first = await cli.run(base, replace(quick, change="reference", workloads=workloads.NAMES, at=(1, 2),
                                            identity="reference"))
    finally:
        stop.set()
    answered = await beside
    return [first, *[await cli.run(base, options) for options in steps]], answered


def plans(base: Card) -> list[dict[str, Any]]:
    """Each microbenchmark's plan, from its script run as the harness runs it, with --dry-run."""
    found = []
    for kind in microbench.KINDS:
        argv = [*cli.microbench_argv(base, kind, None), "--dry-run"]
        done = subprocess.run(argv, capture_output=True, check=True, text=True, stdin=subprocess.DEVNULL)
        found.append(json.loads(done.stdout))
    return found


def main(keep: Path | None = None) -> int:
    marker, client_key = secrets.token_hex(16), secrets.token_urlsafe(32)
    credential = {"CONTAINER_ID": CREDENTIAL_ID, "CONTAINER_API_KEY": secrets.token_hex(16)}
    os.umask(0o077)  # as the harness's command runs
    with tempfile.TemporaryDirectory(prefix="throughput-dry-run.") as temporary:
        root = keep or Path(temporary)
        base = prepared_card(root, credential, client_key)
        pack = synthetic_pack(root / "pack", marker)
        results, answered = asyncio.run(runs(base, pack, client_key, marker))
        benches = plans(base)
        problems = [f"{options} exited {code}" for options, (code, _) in zip(("reference", "change", "synthetic"),
                                                                              results, strict=True) if code]
        flagged = (results[0][1] or {}).get("flagged", 0)
        if not answered or not flagged:
            problems.append(f"{answered} reader requests answered and {flagged} passes flagged beside them")
        for _, result in results:
            if result is not None:
                print(report.text(result))
        written = [path for path in (root / "state").rglob("*") if path.is_file()]
        if leaked := leaks(root / "state", [marker, credential["CONTAINER_API_KEY"], client_key]):
            problems.append(f"{len(leaked)} files hold the pack's marker or the credential")
        if problems:
            print("dry run: failed: " + "; ".join(problems))
            return 1
        finished = [result for _, result in results if result is not None]
        identity = [result["identity"] for result in finished if "identity" in result]
        print(f"dry run: ok: {len(finished)} runs, {sum(len(result.get('passes', ())) for result in finished)} passes, "
              f"{sum(len(result.get('sweep', ())) for result in finished)} sweep cells, {flagged} passes flagged "
              f"beside {answered} reader requests, identity "
              f"{' and '.join(f'{verdict['same']} of {verdict['of']}' for verdict in identity)}, "
              f"{len(benches[0]['shapes'])} GEMM shapes at {len(benches[0]['rows'])} row counts and "
              f"{len(benches[1]['cases'])} attention cases planned, {len(written)} files without the marker")
    return 0

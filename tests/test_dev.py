"""The dev launcher as the integration run starts it: a child process with the cases' service block, whose fake engine
gives its default answers to any request."""

from __future__ import annotations

import asyncio
import math
import signal
import socket
import sys

import httpx
import pytest

from simple_serving.fake_engine import SENTENCE

from .support import BOT, CONTROL, OUTSIDE_A, OUTSIDE_B, ROOT, call, chat_body, reader

pytestmark = pytest.mark.anyio
READY = b"simple-serving dev launcher: ready\n"


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


async def test_the_dev_launcher_serves_a_client_with_the_test_keys() -> None:
    engine, public, control = free_ports(3)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "simple_serving.dev", "--config", "contract/cases-v1.json",
        "--engine-port", str(engine), "--public-port", str(public), "--control-port", str(control),
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    log = asyncio.ensure_future(process.stderr.read())
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 15) == READY
        async with httpx.AsyncClient(timeout=10) as client:
            body = chat_body()
            counted = await call(client, "POST", f"http://127.0.0.1:{public}/v1/chat/completions/input_tokens",
                                 body=body)
            characters = sum(len(message["content"]) for message in body["messages"])
            assert counted.json() == {"input_tokens": math.ceil(characters / 4)}
            answer = await call(client, "POST", f"http://127.0.0.1:{public}/v1/chat/completions", headers=reader(),
                                body=body)
            assert answer.done
            chunks = [event for event in answer.events()[:-1] if event["choices"]]
            assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "".join(SENTENCE)
            assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
            state = await call(client, "GET", f"http://127.0.0.1:{control}/v1/state", key=CONTROL)
            assert state.json()["status"] == "ready"
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)  # Ctrl-C, as the launcher says
        await asyncio.wait_for(process.wait(), 15)
    printed = READY + await process.stdout.read()
    assert process.returncode == 0
    assert f"public       http://127.0.0.1:{public}".encode() in printed
    for key in (BOT, CONTROL, OUTSIDE_A, OUTSIDE_B):
        assert key.encode() not in printed + await log

"""The dev launcher as the integration run starts it: a child process with the cases' service block, whose fake engine
gives its default answers to any request, at once or paced by its delay options; and those default answers."""

from __future__ import annotations

import asyncio
import math
import signal
import socket
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from simple_serving import dev
from simple_serving.fake_engine import SENTENCE, THOUGHT, FakeEngine

from .support import (
    ALIAS,
    BOT,
    CONTROL,
    OUTSIDE_A,
    OUTSIDE_B,
    ROOT,
    SERVICE,
    Answer,
    call,
    chat_body,
    generate,
    reader,
    running,
)

pytestmark = pytest.mark.anyio
CONFIG = "contract/cases-v2.json"
CHAT, COUNT = "/v1/chat/completions", "/v1/chat/completions/input_tokens"


@dataclass
class Launch:
    public: str
    control: str
    banner: str  # what the launcher printed once it was ready


def free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


@asynccontextmanager
async def launched(*options: str) -> AsyncIterator[Launch]:
    """The launcher with the cases' service block and these options, stopped with Ctrl-C, as its banner says. It must
    exit cleanly, and nothing it printed may hold a key."""
    engine, public, control = free_ports(3)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "simple_serving.dev", "--config", CONFIG, "--engine-port", str(engine),
        "--public-port", str(public), "--control-port", str(control), *options,
        cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    log = asyncio.ensure_future(process.stderr.read())
    try:
        banner = await asyncio.wait_for(process.stdout.readuntil(b"Ctrl-C stops both.\n"), 15)
        yield Launch(f"http://127.0.0.1:{public}", f"http://127.0.0.1:{control}", banner.decode())
    finally:
        if process.returncode is None:
            process.send_signal(signal.SIGINT)
        await asyncio.wait_for(process.wait(), 15)
    printed = banner + await process.stdout.read() + await log
    assert process.returncode == 0
    for key in (BOT, CONTROL, OUTSIDE_A, OUTSIDE_B):
        assert key.encode() not in printed


def measurements(answer: Answer) -> dict[str, int]:
    """`usage.simple_serving` of a stream that ended with [DONE]."""
    assert answer.done, answer.body[-200:]
    usage: dict[str, int] = answer.events()[-2]["usage"]["simple_serving"]
    return usage


async def test_the_dev_launcher_serves_a_client_with_the_test_keys() -> None:
    async with launched() as launch, httpx.AsyncClient(timeout=10) as client:
        assert launch.banner.startswith("simple-serving dev launcher: ready\n")
        assert f"  public       {launch.public}\n" in launch.banner
        assert "  delays       0 ms before the first event, 0 ms after each event\n" in launch.banner
        body = chat_body()
        counted = await call(client, "POST", launch.public + COUNT, body=body)
        characters = sum(len(message["content"]) for message in body["messages"])
        assert counted.json() == {"input_tokens": math.ceil(characters / 4)}
        answer = await call(client, "POST", launch.public + CHAT, headers=reader(), body=body)
        measurements(answer)
        chunks = [event for event in answer.events()[:-1] if event["choices"]]
        assert "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks) == "".join(SENTENCE)
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        state = await call(client, "GET", launch.control + "/v1/state", key=CONTROL)
        assert state.json()["status"] == "ready"


async def test_with_delays_concurrent_readers_fill_their_places_and_wait() -> None:
    places = SERVICE["limits"]["reader"]["active"]
    async with (launched("--first-event-delay-ms", "300", "--event-delay-ms", "20") as launch,
                httpx.AsyncClient(timeout=10) as client):
        assert "  delays       300 ms before the first event, 20 ms after each event\n" in launch.banner
        turns = [asyncio.ensure_future(call(client, "POST", launch.public + CHAT, headers=reader(), body=chat_body()))
                 for _ in range(places + 1)]
        for _ in range(500):
            state = (await call(client, "GET", launch.control + "/v1/state", key=CONTROL)).json()
            if (state["active"]["reader"], state["waiting"]["reader"]) == (places, 1):
                break
            await asyncio.sleep(0.005)
        else:
            raise AssertionError("the readers never filled their places")
        times = [measurements(answer) for answer in await asyncio.gather(*turns)]
    assert all(time["first_token_ms"] >= 300 for time in times)
    assert max(time["wait_ms"] for time in times) >= 300  # one reader waited for a place
    # After the first token come three more contents and the finish, each followed by a pause.
    assert all(time["total_ms"] - time["first_token_ms"] >= 4 * 20 for time in times)


@pytest.mark.parametrize("value", ["-1", "1.5", "soon", "²"])
def test_a_delay_is_a_whole_number_of_milliseconds(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_:
        dev.main(["--config", CONFIG, "--engine-port", "1", "--public-port", "2", "--control-port", "3",
                  "--event-delay-ms", value])
    assert exit_.value.code == 2
    assert "--event-delay-ms" in capsys.readouterr().err


# The fake engine's default answers that the smoke's dry run needs: finish length at max_tokens, reasoning when
# thinking is on, and a refusal of a schema it cannot compile, before any stream.

def text_of(answer: Any, field: str) -> str:
    return "".join(event["choices"][0]["delta"].get(field, "") for event in answer.events()[:-1]
                   if isinstance(event, dict) and event["choices"])


async def test_the_fake_engine_cuts_at_max_tokens_and_thinks_when_asked() -> None:
    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        cut = await generate(client, stack, chat_body(max_tokens=4))
        thinking = await generate(client, stack, chat_body(chat_template_kwargs={"enable_thinking": True}))
        both_cut = await generate(client, stack,
                                  chat_body(max_tokens=5, chat_template_kwargs={"enable_thinking": True}))
    assert text_of(cut, "content") == "".join(SENTENCE)[:16]
    assert cut.events()[-2]["usage"]["completion_tokens"] == 4
    assert [event["choices"][0]["finish_reason"] for event in cut.events()[:-2]][-1] == "length"
    assert (text_of(thinking, "reasoning_content"), text_of(thinking, "content")) == ("".join(THOUGHT),
                                                                                     "".join(SENTENCE))
    assert thinking.events()[-2]["usage"]["completion_tokens"] == 24  # the thought's 41 characters and the sentence's
    assert (text_of(both_cut, "reasoning_content"), text_of(both_cut, "content")) == ("".join(THOUGHT)[:20], "")


async def test_the_fake_engine_refuses_a_pattern_that_does_not_compile_before_any_stream() -> None:
    def body(pattern: str) -> dict[str, Any]:
        schema = {"type": "object", "properties": {"id": {"type": "string", "pattern": pattern}}}
        return chat_body(response_format={"type": "json_schema",
                                          "json_schema": {"name": "id", "strict": True, "schema": schema}})

    async with running() as stack, httpx.AsyncClient(timeout=10) as client:
        refused = await generate(client, stack, body("(unclosed"))
        served = await generate(client, stack, body("^e[1-9][0-9]{0,3}$"))
    assert (refused.status, refused.json()) == (400, {"error": {"code": "invalid_request"}})
    assert served.done


def test_a_fake_engine_without_a_limit_or_thinking_gives_the_whole_sentence() -> None:
    events = FakeEngine(ALIAS, 4096)._default_events({"messages": []})
    assert [event.get("content") for event in events[1:-1]] == list(SENTENCE)
    assert events[-1] == {"finish_reason": "stop"}

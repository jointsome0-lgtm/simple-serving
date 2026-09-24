"""What keeps the service awake, and how it falls asleep and stops its instance (contract section 8). The clock is a
fake one, so no test waits for the idle interval or a deadline."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from simple_serving.fake_engine import Script
from simple_serving.routes import PREPARE_S

from .support import (
    OUTSIDE_A,
    FakeClock,
    RawClient,
    by_prompt,
    generate,
    reached,
    reader,
    running,
    stall_the_send_of,
    until,
    user_body,
)

pytestmark = pytest.mark.anyio
CHAT = "/v1/chat/completions"


async def test_our_request_counts_from_before_its_body_and_an_outside_one_never(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    async with running() as stack, httpx.AsyncClient(timeout=None) as client:
        gate = asyncio.Event()
        stack.fake.script = by_prompt({"outside": Script(hold_first=gate)})
        slow = await RawClient.post(stack.public, CHAT, user_body("ours"), headers=reader(), missing=10)
        await until(lambda: stack.service.ours_inflight == 1)  # its body has not all arrived
        outside = asyncio.ensure_future(generate(client, stack, user_body("outside"), key=OUTSIDE_A))
        await reached(stack, "outside")
        assert stack.service.ours_inflight == 1

        await clock.advance(PREPARE_S - 1)
        assert stack.service.ours_inflight == 1
        await clock.advance(1)
        await until(lambda: stack.service.ours_inflight == 0)
        answer = await slow.read_some(5)
        assert answer.startswith(b"HTTP/1.1 504") and answer.endswith(b'{"error": {"code": "timeout"}}')
        ended = stack.service.idle_since
        gate.set()
        assert (await outside).done
        assert (stack.service.ours_inflight, stack.service.idle_since) == (0, ended)


async def test_a_refusal_that_its_client_does_not_read_is_given_up_at_the_bound(
        monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock(monkeypatch)
    began = stall_the_send_of(monkeypatch, b'{"error"')
    async with running() as stack:
        await RawClient.post(stack.public, CHAT, user_body("refused", max_tokens=0), headers=reader())
        await began.wait()
        await clock.advance(PREPARE_S - 1)
        assert stack.service.ours_inflight == 1
        await clock.advance(1)
        await until(lambda: stack.service.ours_inflight == 0)

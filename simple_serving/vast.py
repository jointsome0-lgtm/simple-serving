"""The stop of the card's own instance through the Vast API (contract section 8).

Vast gives every container two values: CONTAINER_ID, its instance, and CONTAINER_API_KEY, a key for that instance
alone. With them the card stops itself: `PUT {"state": "stopped"}` to one fixed HTTPS endpoint, with the key in a
header, never in a URL, an argument or a log. The answer is checked as the bot checks it (`local/vast.ts` in
simple-chat): a 2xx status, then `"success": true`. Stop, not delete: the disk with the weights stays.

A log row about an attempt holds a fixed category and Vast's HTTP status, never Vast's answer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any

import httpx

from . import log

ENDPOINT = "https://console.vast.ai/api/v0/instances/{}/"
INSTANCE = re.compile(r"[1-9][0-9]*")
ATTEMPT_S = 20  # one attempt, from connecting to the end of the answer
RETRY_S = 30  # from a failed attempt to the next
MAX_ANSWER_BYTES = 200_000

Stop = Callable[[], Awaitable[None]]
"""One attempt to stop the instance. It returns once Vast has accepted the stop and raises otherwise."""


class VastError(Exception):
    """A failed attempt: a fixed category and, when Vast answered, its HTTP status. It never holds Vast's text."""

    def __init__(self, code: str, status: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def from_environment(environ: Mapping[str, str] = os.environ) -> Stop:
    """The stop of this container's instance. Without Vast's two values every attempt fails as `unconfigured`."""
    instance, key = environ.get("CONTAINER_ID", ""), environ.get("CONTAINER_API_KEY", "")
    if not INSTANCE.fullmatch(instance) or not key or "\r" in key or "\n" in key:
        return unconfigured
    return partial(stop, instance, key)


async def unconfigured() -> None:
    raise VastError("unconfigured")


async def stop(instance: str, key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
    """One attempt. A redirect is not followed: the endpoint is fixed, so it is a failure like any status but 2xx."""
    try:
        async with (httpx.AsyncClient(transport=transport, timeout=ATTEMPT_S) as client,
                    client.stream("PUT", ENDPOINT.format(instance), json={"state": "stopped"},
                                  headers={"Authorization": f"Bearer {key}"}) as response):
            if not response.is_success:
                code = "forbidden" if response.status_code in (401, 403) else "http"
                raise VastError(code, response.status_code)
            text = b""
            async for chunk in response.aiter_bytes():
                text += chunk
                if len(text) > MAX_ANSWER_BYTES:
                    raise VastError("answer")
    except httpx.TimeoutException:
        raise VastError("timeout") from None
    except httpx.HTTPError:
        raise VastError("network") from None
    try:
        answer = json.loads(text)
    except ValueError:
        raise VastError("answer") from None
    if not isinstance(answer, dict) or answer.get("success") is not True:
        raise VastError("answer")


async def stop_until_accepted(stop: Stop) -> None:
    """Stop the instance: one attempt at a time, each within ATTEMPT_S, RETRY_S apart, until Vast accepts one. Every
    failure is tried again, 401 and 403 included, which the log tells apart. Stopping an instance that already stops
    changes nothing, so an attempt repeated after a lost answer does no harm."""
    while True:
        try:
            async with asyncio.timeout(ATTEMPT_S):
                await stop()
        except Exception as error:  # noqa: BLE001 - whatever failed, the next attempt comes
            log.row("stop_failed", **_failure(error))
        else:
            log.row("stop_accepted")
            return
        await asyncio.sleep(RETRY_S)


def _failure(error: Exception) -> dict[str, Any]:
    if isinstance(error, VastError):
        return {"code": error.code} | ({} if error.status is None else {"status": error.status})
    return {"code": "timeout"} if isinstance(error, TimeoutError) else {"exception": type(error).__name__}

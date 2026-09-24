"""The instance through the Vast API, split by authority (contract sections 8 and 12).

On the card: Vast gives every container two values, CONTAINER_ID, its instance, and CONTAINER_API_KEY, a key for that
instance alone. With them the card stops itself: `PUT {"state": "stopped"}` to one fixed HTTPS endpoint, with the key
in a header, never in a URL, an argument or a log. The answer is checked as the bot checks it (`local/vast.ts` in
simple-chat): a 2xx status, then `"success": true`. Stop, not delete: the disk with the weights stays.

On the owner's machine: the command shows the instance and resumes it, with the owner's key restricted to GET and PUT
on that instance. It has no stop of its own: only the gateway stops the card, after its drain.

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
    """One attempt."""
    await _put(instance, key, "stopped", transport)


async def show(instance: str, key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> str:
    """The instance's state, as the bot reads it (`local/gpu.ts`): `stopped` only once Vast both means and reports it,
    `stopping` while it means it, `running` once it means and reports that, and `starting` otherwise. The rest of the
    answer, which can hold the instance's credentials, is dropped."""
    found = (await _call("GET", instance, key, None, transport)).get("instances")
    if not isinstance(found, dict) or str(found.get("id")) != instance:
        raise VastError("answer")
    actual, intended = found.get("actual_status"), found.get("intended_status")
    if intended == "stopped":
        return "stopped" if actual in ("stopped", "exited") else "stopping"
    return "running" if (actual, intended) == ("running", "running") else "starting"


async def resume(instance: str, key: str, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
    await _put(instance, key, "running", transport)


async def _put(instance: str, key: str, state: str, transport: httpx.AsyncBaseTransport | None) -> None:
    if (await _call("PUT", instance, key, {"state": state}, transport)).get("success") is not True:
        raise VastError("answer")


async def _call(method: str, instance: str, key: str, body: dict[str, str] | None,
                transport: httpx.AsyncBaseTransport | None) -> dict[str, Any]:
    """One request, its answer read within MAX_ANSWER_BYTES and parsed. A redirect is not followed: the endpoint is
    fixed, so it is a failure like any status but 2xx."""
    try:
        async with (httpx.AsyncClient(transport=transport, timeout=ATTEMPT_S) as client,
                    client.stream(method, ENDPOINT.format(instance), json=body,
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
    if not isinstance(answer, dict):
        raise VastError("answer")
    return answer


async def stop_until_accepted(stop: Stop, failed: Callable[[dict[str, Any]], None] = lambda failure: None) -> None:
    """Stop the instance: one attempt at a time, each within ATTEMPT_S, RETRY_S apart, until Vast accepts one. Every
    failure is tried again, 401 and 403 included, which the log tells apart, and `failed` hears of each. The attempts
    bound nothing: while Vast refuses the key they fail without end, and the instance runs on. Stopping an instance
    that already stops changes nothing, so an attempt repeated after a lost answer does no harm."""
    while True:
        try:
            async with asyncio.timeout(ATTEMPT_S):
                await stop()
        except Exception as error:  # noqa: BLE001 - whatever failed, the next attempt comes
            log.row("stop_failed", **(failure := _failure(error)))
            failed(failure)
        else:
            log.row("stop_accepted")
            return
        await asyncio.sleep(RETRY_S)


def _failure(error: Exception) -> dict[str, Any]:
    if isinstance(error, VastError):
        return {"code": error.code} | ({} if error.status is None else {"status": error.status})
    return {"code": "timeout"} if isinstance(error, TimeoutError) else {"exception": type(error).__name__}
